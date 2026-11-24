"""Batched grasp-stabilisation environment on MuJoCo Warp.

:class:`GraspEnvGPU` reproduces :class:`~manip_rubiks.environments.grasp_env.GraspEnv`
for ``num_envs`` worlds simulated in parallel on the GPU: same model, control
rate, action mapping, observations, tactile readings, wrist schedule, pushes,
reward, termination and success criterion. The task-independent machinery
(simulation, resets, sensing, observations) and the differences from the CPU
reference are described in :mod:`manip_rubiks.gpu.hand_cube_env`.

The actor observation follows
:data:`~manip_rubiks.environments.grasp_env.GRASP_OBS_KEYS` (see
:func:`flatten_observation`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from manip_rubiks.environments.grasp_env import GRASP_OBS_KEYS, STANDARD_GRAVITY, GraspConfig, GraspEnv
from manip_rubiks.gpu.hand_cube_env import (
    EVAL_BANK_SEEDS,
    MAX_QVEL,
    TRAIN_BANK_SEEDS,
    HandCubeEnvGPU,
    StepInfo,
    flatten_keys,
)
from manip_rubiks.gpu.reset_bank import ResetBank

__all__ = [
    "EVAL_BANK_SEEDS",
    "MAX_QVEL",
    "TRAIN_BANK_SEEDS",
    "GraspEnvGPU",
    "GraspStepInfo",
    "flatten_observation",
]


def flatten_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a CPU ``GraspEnv`` observation into the GPU actor layout."""
    return flatten_keys(obs, GRASP_OBS_KEYS)


@dataclass(kw_only=True)
class GraspStepInfo(StepInfo):
    rotation: torch.Tensor
    linear_speed: torch.Tensor
    angular_speed: torch.Tensor


def _min_jerk(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _rotate_towards(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Great-circle interpolation from unit ``a`` to unit ``b`` (as ``slerp_direction``)."""
    angle = torch.arccos((a * b).sum(-1).clamp(-1.0, 1.0))
    axis = torch.cross(a, b, dim=-1)
    norm = axis.norm(dim=-1, keepdim=True)
    ex = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a)
    ey = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a)
    fallback = torch.cross(a, torch.where(a[..., :1].abs() < 0.9, ex, ey), dim=-1)  # antipodal case
    axis = torch.where(norm < 1e-6, fallback, axis)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    phi = (s * angle)[..., None]
    k_cross_a = torch.cross(axis, a, dim=-1)
    k_dot_a = (axis * a).sum(-1, keepdim=True)
    out = a * torch.cos(phi) + k_cross_a * torch.sin(phi) + axis * k_dot_a * (1 - torch.cos(phi))
    return torch.where((angle < 1e-6)[..., None], b, out)


class GraspEnvGPU(HandCubeEnvGPU):
    """``num_envs`` grasp-stabilisation worlds stepped together on the GPU."""

    ref_cls = GraspEnv

    def __init__(
        self,
        num_envs: int,
        config: GraspConfig | None = None,
        device: str = "cuda:0",
        seed: int = 0,
        bank: ResetBank | None = None,
        bank_seeds: range = TRAIN_BANK_SEEDS,
        nconmax: int = 128,
        njmax: int = 600,
    ) -> None:
        super().__init__(num_envs, config or GraspConfig(), device, seed, bank, bank_seeds, nconmax, njmax)

    def observation_keys(self) -> tuple[str, ...]:
        return GRASP_OBS_KEYS

    def _init_task_state(self) -> None:
        n, dev, cfg = self.num_envs, self.device, self.cfg
        self.d_nominal = self._tensor(self.ref.d_nominal)
        horizon = self.max_steps * self.dt
        self.n_moves = int(np.ceil((horizon - cfg.start_delay[0]) / (cfg.move_duration[0] + cfg.dwell[0]))) + 1
        self.gravity_dir = self.d_nominal.expand(n, 3).clone()
        self.push_until = torch.full((n,), -1.0, device=dev)
        self.sched_t0 = torch.zeros(n, self.n_moves, device=dev)
        self.sched_t1 = torch.zeros_like(self.sched_t0)
        self.sched_from = torch.zeros(n, self.n_moves, 3, device=dev)
        self.sched_to = torch.zeros_like(self.sched_from)
        self.sched_valid = torch.zeros(n, self.n_moves, dtype=torch.bool, device=dev)
        self.r_cube0 = torch.eye(3, device=dev).expand(n, 3, 3).clone()

    def tilted_direction(self, tilt: torch.Tensor, azimuth: torch.Tensor) -> torch.Tensor:
        """Gravity direction after tilting the wrist (``GraspEnv.tilted_direction``)."""
        u = torch.stack([torch.zeros_like(azimuth), torch.sin(azimuth), torch.cos(azimuth)], dim=-1)
        return torch.cos(tilt)[..., None] * self.d_nominal + torch.sin(tilt)[..., None] * u

    # ----------------------------------------------------- wrist schedule
    def _sample_schedule(self, ids: torch.Tensor) -> None:
        cfg, k = self.cfg, len(ids)
        horizon = self.max_steps * self.dt
        t = self._uniform(k, *cfg.start_delay)
        d = self.d_nominal.expand(k, 3).clone()
        for j in range(self.n_moves):
            tilt = self._uniform(k, 0.0, cfg.max_tilt * self.difficulty)
            target = self.tilted_direction(tilt, self._uniform(k, 0.0, 2 * np.pi))
            t_end = t + self._uniform(k, *cfg.move_duration)
            valid = t < horizon
            self.sched_t0[ids, j], self.sched_t1[ids, j] = t, t_end
            self.sched_from[ids, j], self.sched_to[ids, j] = d, target
            self.sched_valid[ids, j] = valid
            d = torch.where(valid[:, None], target, d)
            t = t_end + self._uniform(k, *cfg.dwell)

    def gravity_direction_at(self, t: torch.Tensor) -> torch.Tensor:
        """(N, 3) gravity direction in the hand frame at per-world time ``t``."""
        started = (t[:, None] >= self.sched_t0) & self.sched_valid
        finished = (t[:, None] >= self.sched_t1) & self.sched_valid
        n_started, n_finished = started.sum(1), finished.sum(1)
        rows = torch.arange(self.num_envs, device=self.device)
        last = (n_finished - 1).clamp(min=0)
        d = torch.where((n_finished > 0)[:, None], self.sched_to[rows, last], self.d_nominal.expand(self.num_envs, 3))
        cur = (n_started - 1).clamp(min=0)
        moving = n_started > n_finished
        t0, t1 = self.sched_t0[rows, cur], self.sched_t1[rows, cur]
        s = _min_jerk((t - t0) / (t1 - t0))
        mid = _rotate_towards(self.sched_from[rows, cur], self.sched_to[rows, cur], s)
        return torch.where(moving[:, None], mid, d)

    def _set_gravity(self, d_hand: torch.Tensor) -> None:
        self.gravity_dir = d_hand / d_hand.norm(dim=-1, keepdim=True)
        self.gravity_world[:] = STANDARD_GRAVITY * self.gravity_dir @ self.r_mount.T

    # ------------------------------------------------------------- pushes
    def _update_push(self, t: torch.Tensor) -> None:
        cfg = self.cfg
        free = t >= self.push_until
        self.xfrc[free, self.core] = 0.0
        scale = cfg.push_force * self.difficulty * cfg.push_scale
        if scale <= 0:
            return
        start = free & (torch.rand(self.num_envs, generator=self.gen, device=self.device) < cfg.push_rate * self.dt)
        k = int(start.sum())
        if k == 0:
            return
        force = self._unit(k) * scale * self._uniform((k, 1), 0.3, 1.0)
        torque = self._unit(k) * cfg.push_torque * self.difficulty * cfg.push_scale * self._uniform((k, 1), 0.3, 1.0)
        ids = start.nonzero().squeeze(1)
        self.xfrc[ids, self.core] = torch.cat([force, torque], dim=1)
        self.push_until[ids] = t[ids] + self._uniform(k, *cfg.push_duration)

    # -------------------------------------------------------------- hooks
    def _pre_physics(self, t: torch.Tensor) -> None:
        self._set_gravity(self.gravity_direction_at(t))
        self._update_push(t)

    def _task_obs(self) -> list[torch.Tensor]:
        return [self.gravity_dir]

    def _privileged(self, cq: dict[str, torch.Tensor]) -> torch.Tensor:
        state = super()._privileged(cq)
        if not self.cfg.critic_wrench:
            return state
        norm = self._tensor(self.ref._wrench_norm)
        return torch.cat([state, self.xfrc[:, self.core] / norm], dim=1)

    def _task_reset_pre(self, ids: torch.Tensor) -> None:
        self.push_until[ids] = -1.0
        self._sample_schedule(ids)
        d_nom = self.d_nominal.expand(len(ids), 3)
        self.gravity_dir[ids] = d_nom
        self.gravity_world[ids] = STANDARD_GRAVITY * d_nom @ self.r_mount.T

    def _task_reset_post(self, ids: torch.Tensor, cq: dict[str, torch.Tensor]) -> None:
        self.r_cube0[ids] = cq["r_cube"][ids]

    def _task_info(self, cq: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> GraspStepInfo:
        ok = ~base["dropped"] & ~base["unstable"] & base["state_preserved"]
        return GraspStepInfo(
            **base,
            rotation=self.math.angle(self.r_cube0.transpose(-1, -2) @ cq["r_cube"]),
            linear_speed=cq["lin"].norm(dim=-1),
            angular_speed=cq["ang"].norm(dim=-1),
            is_success=ok & (base["misalignment"] < self.cfg.align_tolerance),
        )

    def _reward(self, action: torch.Tensor, info: GraspStepInfo) -> torch.Tensor:
        """Total reward; the signed per-term contributions are stored in ``info.reward_terms``."""
        w = self.cfg.reward
        effort = self.actuator_force[:, self.act_ids] / self.force_limit
        failed = info.dropped | info.unstable | ~info.state_preserved
        barrier = ((info.misalignment - w.align_onset) / (w.align_limit - w.align_onset)).clamp(0.0, 1.0)
        terms = {
            "alive": torch.full_like(info.displacement, w.alive),
            "displacement": -w.displacement * info.displacement / 0.01,
            "rotation": -w.rotation * info.rotation / 0.5,
            "alignment": -w.alignment * info.misalignment / 0.1,
            "velocity": -w.velocity * (info.linear_speed / 0.1 + info.angular_speed / 2.0),
            "effort": -w.effort * torch.nan_to_num(effort**2).mean(1),
            "action_rate": -w.action_rate * ((action - self.last_action) ** 2).mean(1),
            "failure": -w.failure * failed.float(),
        }
        if w.align_barrier > 0:
            terms["barrier"] = -w.align_barrier * barrier**2
        info.reward_terms = terms
        return torch.stack(list(terms.values())).sum(0)
