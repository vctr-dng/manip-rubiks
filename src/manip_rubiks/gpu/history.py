"""Frame stacking for the GPU environments.

:class:`HistoryEnvGPU` gives the actor its last ``k`` observation frames, newest first, so that the
policy can react to what a push changed (joint velocities, effort, tactile readings and the previous
action all shift when a push lands). The deployed policy stacks the same way
(:class:`~manip_rubiks.policies.torch_policy.TorchGraspPolicy`), so training and deployment match.

The newest frame comes first so every slice of the single-frame layout still addresses the current
observation. When a world ends, its stack is refilled with the first observation of the next episode.
The critic input is the stacked actor observation followed by the privileged state.
"""

from __future__ import annotations

import torch


class HistoryEnvGPU:
    """Wraps a :class:`~manip_rubiks.gpu.hand_cube_env.HandCubeEnvGPU` to stack ``k`` actor frames."""

    def __init__(self, env, k: int) -> None:
        if k < 1:
            raise ValueError("history must be at least 1")
        self.env = env
        self.history = k
        self.frame_dim = env.obs_dim
        self.obs_dim = k * env.obs_dim
        self._stack: torch.Tensor | None = None

    def __getattr__(self, name: str):
        # Only reached for attributes not set on the wrapper: everything else is the wrapped env's.
        return getattr(self.env, name)

    def _critic(self, stack: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([stack, critic_obs[:, self.frame_dim :]], dim=1)

    def reset(self, seed: int | None = None):
        obs, critic_obs = self.env.reset(seed=seed)
        self._stack = obs.repeat(1, self.history)
        return self._stack, self._critic(self._stack, critic_obs)

    def step(self, action: torch.Tensor):
        obs, critic_obs, reward, terminated, truncated, info, terminal_critic = self.env.step(action)
        fd = self.frame_dim
        previous = self._stack[:, : fd * (self.history - 1)]
        # The observation of the state a world ended in, stacked on the history it had.
        terminal_stack = torch.cat([terminal_critic[:, :fd], previous], dim=1)
        terminal = self._critic(terminal_stack, terminal_critic)
        stack = torch.cat([obs, previous], dim=1)
        done = terminated | truncated
        if bool(done.any()):  # autoreset: `obs` already belongs to the next episode
            stack[done] = obs[done].repeat(1, self.history)
        self._stack = stack
        return stack, self._critic(stack, critic_obs), reward, terminated, truncated, info, terminal
