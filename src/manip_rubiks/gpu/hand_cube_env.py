"""Batched hand-cube environment on MuJoCo Warp: the task-independent part of the GPU environments.

:class:`HandCubeEnvGPU` steps ``num_envs`` copies of a CPU task environment
(a :class:`~manip_rubiks.environments.hand_cube_env.HandCubeEnv` subclass) in
one MuJoCo Warp simulation. It owns what every task shares: the compiled model
and captured CUDA graphs, resets from a
:class:`~manip_rubiks.gpu.reset_bank.ResetBank`, the joint-target action
mapping, tactile readings with noise, dropout and latency, the cube readout,
the base observation and the privileged critic state, drop and instability
detection, automatic resets, and episode bookkeeping.

A task subclass sets :attr:`ref_cls` and implements the hooks:

* :meth:`_task_reset_pre` / :meth:`_task_reset_post`: per-world task state
  before and after the reset forward pass;
* :meth:`_pre_physics`: per-step inputs applied before physics (gravity, pushes);
* :meth:`_task_obs`: task observation entries appended to the base observation;
* :meth:`_task_info`: the per-world :class:`StepInfo` after a step;
* :meth:`_reward` and :meth:`_task_terminated`.

Differences from the CPU reference:

* Physics runs in float32 (MuJoCo Warp) instead of float64.
* Initial states are drawn from the reset bank instead of being sampled per episode.
* Random numbers come from a seeded torch generator, so episodes follow the
  same distributions but not the same sequences as the CPU environment.
* Instability is detected from non-finite or runaway joint velocities, since
  MuJoCo Warp does not report solver warnings per world.

Observations are flat float32 tensors. The actor slice follows the task's
observation keys (:func:`observation_layout`), and the critic receives it
followed by the privileged cube state of :meth:`HandCubeEnv.privileged_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from manip_rubiks.environments.hand_cube_env import HandCubeEnv
from manip_rubiks.gpu.cube_math import CubeMath
from manip_rubiks.gpu.reset_bank import ResetBank, load_reset_bank

MAX_QVEL = 1.0e3  # joint speed treated as a numerical blow-up [rad/s or m/s]
TRAIN_BANK_SEEDS = range(0, 8192)
EVAL_BANK_SEEDS = range(10_000, 11_024)  # same seed block as the CPU in-training evaluation


def observation_layout(env: HandCubeEnv, keys: tuple[str, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """(key, shape) of each actor observation entry, in flattening order."""
    return [(k, tuple(env.observation_space[k].shape)) for k in keys]


def flatten_keys(obs: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Flatten a CPU observation into the GPU actor layout of ``keys``."""
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys])


@dataclass(kw_only=True)
class StepInfo:
    """Per-world quantities after a control step (before automatic resets)."""

    displacement: torch.Tensor
    misalignment: torch.Tensor
    dropped: torch.Tensor
    unstable: torch.Tensor
    state_preserved: torch.Tensor
    is_success: torch.Tensor
    episode_return: torch.Tensor  # undiscounted return so far, including this step
    episode_length: torch.Tensor  # control steps so far, including this step
    reward_terms: dict[str, torch.Tensor] = field(default_factory=dict)  # signed contribution of each reward term


class HandCubeEnvGPU:
    """``num_envs`` worlds of a hand-cube task stepped together on the GPU."""

    ref_cls: ClassVar[type[HandCubeEnv]]

    def __init__(
        self,
        num_envs: int,
        config: Any,
        device: str = "cuda:0",
        seed: int = 0,
        bank: ResetBank | None = None,
        bank_seeds: range = TRAIN_BANK_SEEDS,
        nconmax: int = 128,
        njmax: int = 600,
    ) -> None:
        self.cfg = config
        self.num_envs = n = int(num_envs)
        self.device = torch.device(device)
        self.ref = ref = self.ref_cls(config)
        ref.reset(seed=0)
        mjm = ref.model
        self.mjm = mjm
        self.n_substeps = ref.n_substeps
        self.dt = ref.cfg.control_dt
        self.max_steps = ref.cfg.max_episode_steps
        self.difficulty = float(config.difficulty)
        self.bank = bank if bank is not None else load_reset_bank(config, bank_seeds, self.ref_cls)
        self.obs_keys = tuple(self.observation_keys())
        self.layout = observation_layout(ref, self.obs_keys)
        self.obs_dim = sum(int(np.prod(s)) for _, s in self.layout)
        self.privileged_dim = int(ref.privileged_state().size)

        # ---------------------------------------------------------- warp sim
        wp.init()
        self.wp_device = wp.get_device(str(self.device))
        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(mjm)
            self.d = mjw.put_data(mjm, ref.data, nworld=n, nconmax=nconmax, njmax=njmax)
            g = np.tile(mjm.opt.gravity.astype(np.float32), (n, 1))
            self.m.opt.gravity = wp.array(g, dtype=wp.vec3)  # one gravity vector per world
            self._contact_ids = wp.array(np.arange(self.d.naconmax, dtype=np.int32), dtype=wp.int32)
            self._contact_force = wp.zeros(self.d.naconmax, dtype=wp.spatial_vector)
            mjw.step(self.m, self.d)  # compile kernels before capture
            mjw.forward(self.m, self.d)
            wp.synchronize_device()
            with wp.ScopedCapture() as cap:
                for _ in range(self.n_substeps):
                    mjw.step(self.m, self.d)
            self._step_graph = cap.graph
            with wp.ScopedCapture() as cap:
                mjw.forward(self.m, self.d)
            self._forward_graph = cap.graph

        t = wp.to_torch
        d = self.d
        self.qpos, self.qvel, self.ctrl = t(d.qpos), t(d.qvel), t(d.ctrl)
        self.act, self.qacc_warmstart = t(d.act), t(d.qacc_warmstart)
        self.actuator_force = t(d.actuator_force)
        self.xpos, self.xmat, self.xquat = t(d.xpos), t(d.xmat), t(d.xquat)
        self.xipos, self.cvel, self.subtree_com = t(d.xipos), t(d.cvel), t(d.subtree_com)
        self.site_xmat = t(d.site_xmat)
        self.xfrc = t(d.xfrc_applied)
        self.gravity_world = t(self.m.opt.gravity)
        self.contact_geom, self.contact_world = t(d.contact.geom), t(d.contact.worldid)
        self.nacon = t(d.nacon)
        self.contact_force = t(self._contact_force)
        self._extra_zero = [t(getattr(d, f)) for f in ("cqacc_warmstart",) if hasattr(d, f)]

        # --------------------------------------------------- static layouts
        dev, f32 = self.device, torch.float32

        def tensor(x, dtype=f32):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)

        self._tensor = tensor
        r = ref.robot
        self.act_ids = tensor(r.actuator_ids, torch.long)
        self.q_adr, self.v_adr = tensor(r.qpos_adr, torch.long), tensor(r.dof_adr, torch.long)
        self.ctrl_min, self.ctrl_max = tensor(r.ctrl_min), tensor(r.ctrl_max)
        self.force_limit = tensor(r.force_limit)
        self.n_act = r.n
        cube = ref.cube
        self.core = int(cube.core_body)
        self.core_root = int(mjm.body_rootid[self.core])
        self.cubies = tensor(cube.cubie_bodies, torch.long)
        self.rest_pos = tensor(ref.rest_pos)
        self.r_mount = tensor(ref.scene.r_mount)
        self.gravity0 = tensor(mjm.opt.gravity)
        ts = ref.tactile
        self.tactile_cfg = ts.cfg
        self.tactile_shape = ts.shape
        self.n_regions = len(ts.regions)
        self.geom_region = tensor(ts._geom_region, torch.long)
        self.region_site = tensor([max(rg.site, 0) for rg in ts.regions], torch.long)
        self.region_body = tensor([rg.body for rg in ts.regions], torch.long)
        self.region_has_site = tensor([rg.site >= 0 for rg in ts.regions], torch.bool)
        self.math = CubeMath(dev)
        self.bank_t = {
            k: tensor(getattr(self.bank, k))
            for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act")
            if getattr(self.bank, k).size
        }
        self.bank_canonical = self.math.canonical(tensor(self.bank.cube_state, torch.long))

        # ------------------------------------------------------ world state
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed))
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_action = torch.zeros(n, self.n_act, device=dev)
        self.last_action = torch.zeros_like(self.prev_action)
        self.ref_canonical = torch.zeros(n, 8, dtype=torch.long, device=dev)
        latency = self.tactile_cfg.latency
        self.tactile_hist = torch.zeros(latency + 1, n, *self.tactile_shape, device=dev)
        self.tactile_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.tactile_reading = torch.zeros(n, *self.tactile_shape, device=dev)
        self.episode_return = torch.zeros(n, device=dev)
        self._init_task_state()

    # ------------------------------------------------------------ hooks
    def observation_keys(self) -> tuple[str, ...]:
        """Actor observation keys, in flattening order."""
        raise NotImplementedError

    def _init_task_state(self) -> None:
        """Allocate per-world task buffers (called at the end of ``__init__``)."""

    def _task_reset_pre(self, ids: torch.Tensor) -> None:
        """Reset task state of worlds ``ids`` before the reset forward pass."""

    def _task_reset_post(self, ids: torch.Tensor, cq: dict[str, torch.Tensor]) -> None:
        """Reset task state of worlds ``ids`` that depends on the reset cube state."""

    def _pre_physics(self, t: torch.Tensor) -> None:
        """Apply per-step task inputs at per-world time ``t``, before the action and physics."""

    def _task_obs(self) -> list[torch.Tensor]:
        """Task observation entries appended to the base observation, as (N, k) tensors."""
        return []

    def _task_info(self, cq: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> StepInfo:
        raise NotImplementedError

    def _reward(self, action: torch.Tensor, info: StepInfo) -> torch.Tensor:
        raise NotImplementedError

    def _task_terminated(self, info: StepInfo) -> torch.Tensor:
        return ~info.state_preserved

    @property
    def failure_penalty(self) -> float:
        return float(self.cfg.reward.failure)

    # ------------------------------------------------------------ helpers
    def _uniform(self, shape, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, generator=self.gen, device=self.device)

    def _unit(self, k: int) -> torch.Tensor:
        v = torch.randn(k, 3, generator=self.gen, device=self.device)
        return v / v.norm(dim=-1, keepdim=True)

    def set_difficulty(self, difficulty: float) -> None:
        """Set the curriculum difficulty; takes effect at each world's next reset."""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ------------------------------------------------------------ sensing
    def _region_forces(self) -> torch.Tensor:
        """(N, R, 3) net contact force on each tactile region, world frame (``TactileSensors.region_forces``)."""
        with wp.ScopedDevice(self.wp_device):
            mjw.contact_force(self.m, self.d, self._contact_ids, True, self._contact_force)
        wp.synchronize_device()
        n_c = int(self.nacon[0])
        out = torch.zeros(self.num_envs * self.n_regions, 3, device=self.device)
        if n_c == 0:
            return out.view(self.num_envs, self.n_regions, 3)
        geom = self.contact_geom[:n_c].long()
        world = self.contact_world[:n_c].long()
        f = self.contact_force[:n_c, :3]  # force exerted on geom2 by geom1
        r1, r2 = self.geom_region[geom[:, 0]], self.geom_region[geom[:, 1]]
        on2, on1 = r2 >= 0, r1 >= 0
        out.index_add_(0, (world * self.n_regions + r2)[on2], f[on2])
        out.index_add_(0, (world * self.n_regions + r1)[on1], -f[on1])
        return out.view(self.num_envs, self.n_regions, 3)

    def _measure_tactile(self) -> torch.Tensor:
        cfg = self.tactile_cfg
        forces = self._region_forces()
        if cfg.mode == "contact":
            return (forces.norm(dim=-1) > cfg.contact_threshold).float()
        rots = torch.where(
            self.region_has_site[None, :, None, None],
            self.site_xmat[:, self.region_site],
            self.xmat[:, self.region_body],
        )
        local = torch.einsum("nrji,nrj->nri", rots, forces)  # R^T f
        return local.clamp(-cfg.max_force, cfg.max_force)

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.tactile_cfg
        if cfg.mode == "force":
            if cfg.noise_rel > 0:
                x = x * (1.0 + cfg.noise_rel * torch.randn(x.shape, generator=self.gen, device=self.device))
            if cfg.noise_std > 0:
                x = x + cfg.noise_std * torch.randn(x.shape, generator=self.gen, device=self.device)
            x = x.clamp(-cfg.max_force, cfg.max_force)
        if cfg.dropout > 0:
            drop = torch.rand(self.num_envs, self.n_regions, generator=self.gen, device=self.device) < cfg.dropout
            x = x * (~drop).reshape(drop.shape + (1,) * (x.dim() - 2))
        return x

    def _read_tactile(self, ids: torch.Tensor | None = None) -> None:
        """Advance the delayed tactile reading (``TactileSensors.read``) for worlds ``ids`` (default: all)."""
        reading = self._corrupt(self._measure_tactile())
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        size = self.tactile_hist.shape[0]
        k = self.tactile_count[ids]
        self.tactile_hist[k % size, ids] = reading[ids]
        latency = size - 1
        self.tactile_reading[ids] = self.tactile_hist[(k - latency).clamp(min=0) % size, ids]
        self.tactile_count[ids] = k + 1

    def _cube_quantities(self) -> dict[str, torch.Tensor]:
        xmat = torch.nan_to_num(self.xmat)
        r_core = xmat[:, self.core]
        rel = r_core.transpose(-1, -2)[:, None] @ xmat[:, self.cubies]
        reading = self.math.read(rel)
        # mj_objectVelocity(body, world orientation): cvel moved from the subtree COM to xipos.
        cvel = torch.nan_to_num(self.cvel[:, self.core])
        ang, lin_com = cvel[:, :3], cvel[:, 3:]
        offset = torch.nan_to_num(self.xipos[:, self.core] - self.subtree_com[:, self.core_root])
        lin = lin_com + torch.cross(ang, offset, dim=-1)
        return {
            "pos": torch.nan_to_num(self.xpos[:, self.core]),
            "quat": torch.nan_to_num(self.xquat[:, self.core]),
            "r_cube": r_core @ reading.frame,
            "rel": rel,
            "lin": lin,
            "ang": ang,
            "reading": reading,
        }

    # ---------------------------------------------------- observations
    def _actor_obs(self) -> torch.Tensor:
        parts = [
            self.qpos[:, self.q_adr],
            self.qvel[:, self.v_adr],
            self.actuator_force[:, self.act_ids],
            self.ctrl[:, self.act_ids],
            self.tactile_reading.reshape(self.num_envs, -1),
            self.prev_action,
            *self._task_obs(),
        ]
        return torch.nan_to_num(torch.cat(parts, dim=1))

    def _privileged(self, cq: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [cq["pos"] - self.rest_pos, cq["quat"], cq["lin"], cq["ang"], cq["rel"].reshape(self.num_envs, 72)]
        return torch.cat(parts, dim=1)

    def _base_info(self, cq: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reading = cq["reading"]
        displacement = (cq["pos"] - self.rest_pos).norm(dim=-1)
        finite = torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1)
        return {
            "displacement": displacement,
            "misalignment": reading.max_error,
            "dropped": displacement > self.ref.cfg.drop_distance,
            "unstable": ~finite | (torch.nan_to_num(self.qvel, nan=0.0).abs().amax(1) > MAX_QVEL),
            "state_preserved": self.math.same_configuration(reading.indices, self.ref_canonical),
            "episode_return": self.episode_return.clone(),
            "episode_length": self.steps.clone(),
        }

    # ----------------------------------------------------------- episode
    def _reset_worlds(self, ids: torch.Tensor, bank_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        for k, v in self.bank_t.items():
            getattr(self, k)[ids] = v[bank_idx]
        for buf in self._extra_zero:
            buf[ids] = 0
        self.xfrc[ids] = 0.0
        self.steps[ids] = 0
        self.prev_action[ids] = 0.0
        self.last_action[ids] = 0.0
        self.episode_return[ids] = 0.0
        self.ref_canonical[ids] = self.bank_canonical[bank_idx]
        self.gravity_world[ids] = self.gravity0
        self._task_reset_pre(ids)
        self.tactile_count[ids] = 0
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(self.wp_device):
            wp.capture_launch(self._forward_graph)
        wp.synchronize_device()
        self._read_tactile(ids)
        cq = self._cube_quantities()
        self._task_reset_post(ids, cq)
        return cq

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset every world, using bank entries ``0..num_envs-1`` in order (deterministic for a seed)."""
        if seed is not None:
            self.gen.manual_seed(int(seed))
        ids = torch.arange(self.num_envs, device=self.device)
        cq = self._reset_worlds(ids, ids % len(self.bank))
        actor_obs = self._actor_obs()
        return actor_obs, torch.cat([actor_obs, self._privileged(cq)], dim=1)

    def step(self, action: torch.Tensor):
        """Advance all worlds one control step.

        Returns ``(actor_obs, critic_obs, reward, terminated, truncated, info, terminal_critic_obs)``.
        ``info`` describes the state reached by the step, before automatic resets;
        ``terminal_critic_obs`` holds the critic observation of that state, which
        is needed to bootstrap truncated episodes.
        """
        a = action.to(self.device, torch.float32).clamp(-1.0, 1.0)
        t = self.steps.float() * self.dt
        self._pre_physics(t)
        ids = self.act_ids
        self.ctrl[:, ids] = torch.minimum(
            torch.maximum(self.ctrl[:, ids] + self.ref.cfg.k_a * a, self.ctrl_min), self.ctrl_max
        )
        self.last_action = self.prev_action
        self.prev_action = a
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(self.wp_device):
            wp.capture_launch(self._step_graph)
        wp.synchronize_device()
        self.steps += 1
        self._read_tactile()

        cq = self._cube_quantities()
        info = self._task_info(cq, self._base_info(cq))
        reward = torch.nan_to_num(self._reward(a, info), nan=-self.failure_penalty)
        self.episode_return += reward
        info.episode_return += reward
        terminated = info.dropped | info.unstable | self._task_terminated(info)
        truncated = ~terminated & (self.steps >= self.max_steps)
        actor_obs = self._actor_obs()
        critic_obs = torch.cat([actor_obs, self._privileged(cq)], dim=1)
        terminal_critic_obs = critic_obs.clone()

        done = terminated | truncated
        if bool(done.any()):
            ids = done.nonzero().squeeze(1)
            bank_idx = torch.randint(len(self.bank), (len(ids),), generator=self.gen, device=self.device)
            cq = self._reset_worlds(ids, bank_idx)
            actor_obs = self._actor_obs()
            critic_obs = torch.cat([actor_obs, self._privileged(cq)], dim=1)
        return actor_obs, critic_obs, reward, terminated, truncated, info, terminal_critic_obs

    def close(self) -> None:
        self.ref.close()
