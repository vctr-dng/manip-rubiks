"""Batched quarter-turn reorientation environment on MuJoCo Warp.

:class:`RotationEnvGPU` reproduces
:class:`~manip_rubiks.environments.rotation_env.RotationEnv` for ``num_envs``
worlds: same goal chain, tolerance curriculum, observations (including the
privileged ``teacher`` entry), reward, termination and success criterion. The
shared machinery and the differences from the CPU reference are described in
:mod:`manip_rubiks.gpu.hand_cube_env`. Commands are drawn from the seeded torch
generator; the CPU ``reset(options={"commands": ...})`` queue has no GPU
counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from manip_rubiks.environments.rotation_env import (
    N_COMMANDS,
    TEACHER_BOUND,
    RotationConfig,
    RotationEnv,
    command_rotation,
)
from manip_rubiks.gpu.hand_cube_env import TRAIN_BANK_SEEDS, HandCubeEnvGPU, StepInfo, flatten_keys
from manip_rubiks.gpu.reset_bank import ResetBank


@dataclass(kw_only=True)
class RotationStepInfo(StepInfo):
    theta: torch.Tensor  # angle between the cube and its goal after the step [rad]
    timed_out: torch.Tensor
    goal_reached: torch.Tensor  # a goal was reached at this step
    goals_reached: torch.Tensor  # goals reached so far in the episode, including this step
    command: torch.Tensor  # command that was active during the step (0..5)


def flatten_rotation_observation(obs: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Flatten a CPU ``RotationEnv`` observation into the GPU actor layout."""
    return flatten_keys(obs, keys)


class RotationEnvGPU(HandCubeEnvGPU):
    """``num_envs`` reorientation worlds stepped together on the GPU."""

    ref_cls = RotationEnv

    def __init__(
        self,
        num_envs: int,
        config: RotationConfig | None = None,
        device: str = "cuda:0",
        seed: int = 0,
        bank: ResetBank | None = None,
        bank_seeds: range = TRAIN_BANK_SEEDS,
        nconmax: int = 128,
        njmax: int = 600,
    ) -> None:
        super().__init__(num_envs, config or RotationConfig(), device, seed, bank, bank_seeds, nconmax, njmax)

    def observation_keys(self) -> tuple[str, ...]:
        return self.ref.obs_keys

    def _init_task_state(self) -> None:
        n, dev = self.num_envs, self.device
        self.r_rest_hand = self._tensor(self.ref.r_rest_hand)
        self.command_rots = self._tensor(np.stack([command_rotation(c) for c in range(N_COMMANDS)]))
        self.allowed = self._tensor(self.ref.commands, torch.long)
        self.teacher_bound = self._tensor(TEACHER_BOUND)
        self.goal = torch.eye(3, device=dev).expand(n, 3, 3).clone()
        self.command = torch.zeros(n, dtype=torch.long, device=dev)
        self.goals_reached = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.hold = torch.zeros(n, dtype=torch.long, device=dev)
        self.theta_prev = torch.zeros(n, device=dev)
        self.r_hand = torch.eye(3, device=dev).expand(n, 3, 3).clone()
        self._cq: dict[str, torch.Tensor] | None = None  # cube quantities of the current state, all worlds

    @property
    def tolerance(self) -> float:
        return self.cfg.tolerance(self.difficulty)

    # -------------------------------------------------------------- goals
    def _hand_rotation(self, cq: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.r_mount.T @ cq["r_cube"]

    def _issue_goals(self, ids: torch.Tensor) -> None:
        pick = torch.randint(len(self.allowed), (len(ids),), generator=self.gen, device=self.device)
        self.command[ids] = self.allowed[pick]
        self.goal[ids] = self.command_rots[self.command[ids]] @ self.goal[ids]
        self.goal_steps[ids] = 0
        self.hold[ids] = 0
        self.theta_prev[ids] = self.math.angle(self.r_hand[ids].transpose(-1, -2) @ self.goal[ids])

    # -------------------------------------------------------------- hooks
    def _task_reset_post(self, ids: torch.Tensor, cq: dict[str, torch.Tensor]) -> None:
        self._cq = cq
        self.r_hand[ids] = self._hand_rotation(cq)[ids]
        k = self.math.snap(self.r_rest_hand.T @ self.r_hand[ids])
        self.goal[ids] = self.r_rest_hand @ self.math.rotations[k]
        self.goals_reached[ids] = 0
        self._issue_goals(ids)

    def teacher_state(self, cq: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """(N, 12) privileged teacher input (``RotationEnv.teacher_state``)."""
        cq = cq or self._cq or self._cube_quantities()
        err = self.goal @ self.r_hand.transpose(-1, -2)
        pos = (cq["pos"] - self.rest_pos) @ self.r_mount
        ang = cq["ang"] @ self.r_mount
        out = torch.cat([err[:, :, :2].reshape(self.num_envs, 6), pos, ang], dim=1)
        return torch.maximum(torch.minimum(out, self.teacher_bound), -self.teacher_bound)

    def _task_obs(self) -> list[torch.Tensor]:
        parts = [torch.nn.functional.one_hot(self.command, N_COMMANDS).float()]
        if self.cfg.teacher:
            parts.append(self.teacher_state())
        return parts

    def _task_info(self, cq: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> RotationStepInfo:
        self._cq = cq
        self.r_hand = self._hand_rotation(cq)
        theta = self.math.angle(self.r_hand.transpose(-1, -2) @ self.goal)
        self.goal_steps += 1
        self.hold = torch.where(theta < self.tolerance, self.hold + 1, torch.zeros_like(self.hold))
        timed_out = (self.goal_steps >= self.cfg.goal_timeout) & (self.hold < self.cfg.hold_steps)
        ok = ~base["dropped"] & ~base["unstable"] & base["state_preserved"] & ~timed_out
        return RotationStepInfo(
            **base,
            theta=theta,
            timed_out=timed_out,
            goal_reached=torch.zeros_like(timed_out),
            goals_reached=self.goals_reached.clone(),
            command=self.command.clone(),
            is_success=ok & (base["misalignment"] < self.cfg.align_tolerance),
        )

    def _task_terminated(self, info: RotationStepInfo) -> torch.Tensor:
        return ~info.state_preserved | info.timed_out

    def _reward(self, action: torch.Tensor, info: RotationStepInfo) -> torch.Tensor:
        """Total reward; also advances the goal chain of worlds that reached their goal."""
        w = self.cfg.reward
        effort = self.actuator_force[:, self.act_ids] / self.force_limit
        failed = info.dropped | info.unstable | ~info.state_preserved | info.timed_out
        reached = (self.hold >= self.cfg.hold_steps) & ~failed
        terms = {
            "progress": w.progress * (self.theta_prev - info.theta) / 0.1,
            "goal": w.goal * reached.float(),
            "displacement": -w.displacement * info.displacement / 0.01,
            "alignment": -w.alignment * info.misalignment / 0.1,
            "effort": -w.effort * torch.nan_to_num(effort**2).mean(1),
            "action_rate": -w.action_rate * ((action - self.last_action) ** 2).mean(1),
            "failure": -w.failure * failed.float(),
        }
        self.theta_prev = info.theta.clone()
        if bool(reached.any()):
            ids = reached.nonzero().squeeze(1)
            self.goals_reached[ids] += 1
            self._issue_goals(ids)
        info.goal_reached = reached
        info.goals_reached = self.goals_reached.clone()
        info.reward_terms = terms
        return torch.stack(list(terms.values())).sum(0)
