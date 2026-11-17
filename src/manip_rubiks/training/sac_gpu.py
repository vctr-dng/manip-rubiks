"""Soft Actor-Critic with an asymmetric critic, entirely on the GPU (docs/project-draft.md sections 13 and 25).

Off-policy counterpart of :mod:`manip_rubiks.training.ppo_gpu` for the same
batched environments. The grasp simulation costs ~0.1-0.3 s per control step
for a thousand worlds while a gradient step costs a few milliseconds, so
reusing each transition many times from a replay buffer is cheap.

* **Actor**: tanh-squashed Gaussian on the non-privileged observation only,
  with its own running observation normaliser. Its deterministic action
  (``tanh`` of the mean) is what gets deployed.
* **Critic**: twin Q-networks (clipped double Q) with LayerNorm, reading the
  normalised actor observation, the normalised privileged cube state and the
  action, plus Polyak-averaged target copies. Critic and actor share no layers.
* **Entropy**: temperature tuned automatically towards ``target_entropy``.
* **Replay**: one ring buffer per world, stored as ``(time, world)`` on the
  device, holding raw observations so normalisation always uses the current
  statistics. Targets use n-step returns that stop at episode ends; truncated
  episodes bootstrap from their final state, terminated ones do not.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from manip_rubiks.training.ppo_gpu import RunningMeanStd

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def _mlp(in_dim: int, sizes: tuple[int, ...], layer_norm: bool = False) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    for size in sizes:
        linear = nn.Linear(in_dim, size)
        nn.init.orthogonal_(linear.weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        layers.append(nn.ELU())
        in_dim = size
    return nn.Sequential(*layers), in_dim


class SquashedGaussianActor(nn.Module):
    """``a = tanh(u)``, ``u ~ N(mu(o), sigma(o))``, on the normalised actor observation."""

    def __init__(self, obs_dim: int, act_dim: int, sizes: tuple[int, ...] = (256, 256), log_std_init: float = -1.0):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs_rms = RunningMeanStd(obs_dim)
        self.body, width = _mlp(obs_dim, sizes)
        self.mean = nn.Linear(width, act_dim)
        self.log_std = nn.Linear(width, act_dim)
        for head in (self.mean, self.log_std):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.log_std.bias, float(log_std_init))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(obs)

    def forward(self, obs_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs_n)
        return self.mean(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised action in ``[-1, 1]`` and its log-probability (tanh-corrected)."""
        mu, log_std = self(obs_n)
        u = mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        gauss = -0.5 * (((u - mu) / log_std.exp()) ** 2) - log_std - 0.5 * math.log(2 * math.pi)
        # log(1 - tanh(u)^2), written in a numerically stable form.
        correction = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return a, (gauss - correction).sum(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for raw observations (the deployed policy)."""
        return torch.tanh(self(self.normalize(obs))[0])


class TwinQ(nn.Module):
    """Two independent Q-networks on ``(normalised obs, normalised privileged state, action)``."""

    def __init__(self, obs_dim: int, privileged_dim: int, act_dim: int, sizes: tuple[int, ...], layer_norm: bool):
        super().__init__()
        in_dim = obs_dim + privileged_dim + act_dim
        self.nets = nn.ModuleList()
        for _ in range(2):
            body, width = _mlp(in_dim, sizes, layer_norm)
            head = nn.Linear(width, 1)
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
            self.nets.append(nn.Sequential(body, head))

    def forward(self, obs_n: torch.Tensor, priv_n: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = torch.cat([obs_n, priv_n, action], dim=1)
        return tuple(net(x).squeeze(-1) for net in self.nets)


@dataclass
class SACConfig:
    buffer_size: int = 1_000_000  # transitions, shared equally between worlds
    batch_size: int = 4096
    updates_per_step: int = 4  # gradient steps per batched environment step
    learning_starts: int = 8  # batched environment steps collected before the first update
    gamma: float = 0.99
    n_step: int = 3
    tau: float = 0.01  # Polyak coefficient of the target critics
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.01
    target_entropy_scale: float = 1.0  # target entropy = -scale * act_dim
    actor_sizes: tuple[int, ...] = (256, 256)
    critic_sizes: tuple[int, ...] = (512, 512, 256)
    layer_norm: bool = True
    log_std_init: float = -1.0
    max_grad_norm: float = 0.0  # 0 disables clipping


class ReplayBuffer:
    """``(time, world)`` ring buffer of raw transitions with n-step sampling."""

    def __init__(self, capacity: int, num_envs: int, critic_dim: int, act_dim: int, device: torch.device) -> None:
        self.cap = max(2, capacity // num_envs)
        self.num_envs, self.device = num_envs, device
        t, n = self.cap, num_envs
        self.obs = torch.zeros(t, n, critic_dim, device=device)
        self.next_obs = torch.zeros(t, n, critic_dim, device=device)
        self.actions = torch.zeros(t, n, act_dim, device=device)
        self.rewards = torch.zeros(t, n, device=device)
        self.terminated = torch.zeros(t, n, dtype=torch.bool, device=device)
        self.done = torch.zeros(t, n, dtype=torch.bool, device=device)
        self.ptr, self.size = 0, 0

    def __len__(self) -> int:
        return self.size * self.num_envs

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Store one batched step; ``next_obs`` is the state reached (before any automatic reset)."""
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i], self.next_obs[i] = obs, action, reward, next_obs
        self.terminated[i], self.done[i] = terminated, terminated | truncated
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, n_step: int, gamma: float, generator: torch.Generator | None = None) -> dict:
        """Random n-step transitions.

        Returns the start observation and action, the discounted reward sum
        over at most ``n_step`` steps (stopping at the first episode end), the
        observation to bootstrap from, and its discount ``gamma^m`` (0 when
        the episode terminated within those steps).
        """
        n_step = min(n_step, self.size)
        oldest = (self.ptr - self.size) % self.cap
        dev = self.device
        offset = torch.randint(self.size - n_step + 1, (batch,), generator=generator, device=dev)
        world = torch.randint(self.num_envs, (batch,), generator=generator, device=dev)
        start = (oldest + offset) % self.cap
        ret = torch.zeros(batch, device=dev)
        alive = torch.ones(batch, dtype=torch.bool, device=dev)
        steps = torch.zeros(batch, device=dev)
        last = start.clone()
        for k in range(n_step):
            row = (start + k) % self.cap
            ret = ret + alive * gamma**k * self.rewards[row, world]
            steps = steps + alive
            last = torch.where(alive, row, last)
            alive = alive & ~self.done[row, world]
        discount = gamma**steps * (~self.terminated[last, world])
        return {
            "obs": self.obs[start, world],
            "action": self.actions[start, world],
            "return": ret,
            "next_obs": self.next_obs[last, world],
            "discount": discount,
        }


class SAC:
    """Actor, twin critics, temperature, replay buffer and their updates for ``num_envs`` worlds."""

    def __init__(
        self, num_envs: int, obs_dim: int, privileged_dim: int, act_dim: int, cfg: SACConfig, device: torch.device
    ) -> None:
        self.cfg, self.device, self.num_envs = cfg, device, num_envs
        self.obs_dim, self.privileged_dim, self.act_dim = obs_dim, privileged_dim, act_dim
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.actor_sizes, cfg.log_std_init).to(device)
        self.critic = TwinQ(obs_dim, privileged_dim, act_dim, cfg.critic_sizes, cfg.layer_norm).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.privileged_rms = RunningMeanStd(privileged_dim).to(device)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=device, requires_grad=True)
        self.target_entropy = -cfg.target_entropy_scale * act_dim
        fused = device.type == "cuda"
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, fused=fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, fused=fused)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, num_envs, obs_dim + privileged_dim, act_dim, device)
        self.env_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.detach().exp()

    def _normalize(self, critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_n = self.actor.normalize(critic_obs[:, : self.obs_dim])
        return obs_n, self.privileged_rms.normalize(critic_obs[:, self.obs_dim :])

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def act(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Stochastic exploration action; also updates the observation statistics."""
        self.actor.obs_rms.update(actor_obs)
        self.privileged_rms.update(critic_obs[:, self.obs_dim :])
        action, _ = self.actor.sample(self.actor.normalize(actor_obs))
        return action

    @torch.no_grad()
    def record(
        self,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        terminal_critic_obs: torch.Tensor,
    ) -> None:
        self.buffer.add(critic_obs, action, reward, terminated, truncated, terminal_critic_obs)
        self.env_steps += 1

    @property
    def ready(self) -> bool:
        return self.env_steps >= max(self.cfg.learning_starts, self.cfg.n_step)

    # ------------------------------------------------------------- update
    def _step(self, opt: torch.optim.Optimizer, loss: torch.Tensor, params) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        opt.step()

    def update_once(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        b = self.buffer.sample(cfg.batch_size, cfg.n_step, cfg.gamma)
        obs_n, priv_n = self._normalize(b["obs"])
        with torch.no_grad():
            next_obs_n, next_priv_n = self._normalize(b["next_obs"])
            next_a, next_logp = self.actor.sample(next_obs_n)
            q1_t, q2_t = self.critic_target(next_obs_n, next_priv_n, next_a)
            v_next = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = b["return"] + b["discount"] * v_next
        q1, q2 = self.critic(obs_n, priv_n, b["action"])
        critic_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        self._step(self.critic_opt, critic_loss, self.critic.parameters())

        self.critic.requires_grad_(False)
        a, logp = self.actor.sample(obs_n)
        q_pi = torch.min(*self.critic(obs_n, priv_n, a))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self._step(self.actor_opt, actor_loss, self.actor.parameters())
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
                p_t.lerp_(p, cfg.tau)
        return {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "q": q1.detach().mean(),
            "target_q": target.mean(),
            "entropy": -logp.detach().mean(),
        }

    def update(self) -> dict[str, float]:
        """``updates_per_step`` gradient steps; returns their mean statistics."""
        if not self.ready:
            return {}
        stats: dict[str, list[torch.Tensor]] = {}
        for _ in range(self.cfg.updates_per_step):
            for k, v in self.update_once().items():
                stats.setdefault(k, []).append(v)
        out = {k: float(torch.stack(v).mean()) for k, v in stats.items()}
        out["alpha"] = float(self.alpha)
        return out

    # ------------------------------------------------------- checkpoints
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "privileged_rms": self.privileged_rms.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "env_steps": self.env_steps,
        }
