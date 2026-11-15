"""PPO with an asymmetric actor-critic, entirely on the GPU (docs/project-draft.md section 25).

The actor reads only the non-privileged observation; the critic reads it
followed by the privileged cube state. They share no layers, so privileged
values cannot influence an action. The design follows the Stable-Baselines3
setup used on the CPU (orthogonal initialisation, state-independent log
standard deviation, running observation and return normalisation clipped at
10, per-minibatch advantage normalisation, KL early stopping), adapted to
rollouts collected from thousands of parallel worlds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

NORM_CLIP = 10.0


class RunningMeanStd(nn.Module):
    """Running mean and variance (parallel Welford update), stored as buffers."""

    def __init__(self, shape: tuple[int, ...] | int, epsilon: float = 1e-4) -> None:
        super().__init__()
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape).float()
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        total = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / total).float()
        m2 = self.var * self.count.float() + batch_var * n + delta**2 * (self.count * n / total).float()
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total.float())
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor, clip: float = NORM_CLIP) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-clip, clip)


def _mlp(in_dim: int, sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    for size in sizes:
        linear = nn.Linear(in_dim, size)
        nn.init.orthogonal_(linear.weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ELU()]
        in_dim = size
    return nn.Sequential(*layers), in_dim


class GaussianActor(nn.Module):
    """Policy network on the actor observation, with its own observation normaliser."""

    def __init__(self, obs_dim: int, act_dim: int, sizes: tuple[int, ...] = (256, 256), log_std_init: float = -1.0):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs_rms = RunningMeanStd(obs_dim)
        self.body, width = _mlp(obs_dim, sizes)
        self.mean = nn.Linear(width, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(obs)

    def distribution(self, obs_n: torch.Tensor) -> Normal:
        return Normal(self.mean(self.body(obs_n)), self.log_std.exp().expand(obs_n.shape[0], -1))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the distribution mean) for raw observations."""
        return self.mean(self.body(self.normalize(obs)))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, privileged_dim: int, sizes: tuple[int, ...] = (512, 512)) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.privileged_rms = RunningMeanStd(privileged_dim)
        self.body, width = _mlp(obs_dim + privileged_dim, sizes)
        self.head = nn.Linear(width, 1)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def normalize_privileged(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.privileged_rms.normalize(critic_obs[:, self.obs_dim :])

    def forward(self, obs_n: torch.Tensor, priv_n: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([obs_n, priv_n], dim=1))).squeeze(-1)


@dataclass
class PPOConfig:
    n_steps: int = 32  # rollout length per world
    n_minibatches: int = 4
    n_epochs: int = 5
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    actor_sizes: tuple[int, ...] = (256, 256)
    critic_sizes: tuple[int, ...] = (512, 512)
    log_std_init: float = -1.0


class PPO:
    """Rollout storage, normalisation and clipped-surrogate updates for ``num_envs`` worlds."""

    def __init__(
        self, num_envs: int, obs_dim: int, privileged_dim: int, act_dim: int, cfg: PPOConfig, device: torch.device
    ) -> None:
        self.cfg, self.device, self.num_envs = cfg, device, num_envs
        self.obs_dim, self.privileged_dim = obs_dim, privileged_dim
        self.actor = GaussianActor(obs_dim, act_dim, cfg.actor_sizes, cfg.log_std_init).to(device)
        self.critic = Critic(obs_dim, privileged_dim, cfg.critic_sizes).to(device)
        self.return_rms = RunningMeanStd(()).to(device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
        self.params = params
        t, n = cfg.n_steps, num_envs
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.obs_n, self.priv_n = z(t, n, obs_dim), z(t, n, privileged_dim)
        self.actions, self.log_probs = z(t, n, act_dim), z(t, n)
        self.values, self.rewards, self.dones = z(t, n), z(t, n), z(t, n)
        self.discounted_return = z(n)

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def act(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, step: int) -> torch.Tensor:
        self.actor.obs_rms.update(actor_obs)
        self.critic.privileged_rms.update(critic_obs[:, self.obs_dim :])
        obs_n = self.actor.normalize(actor_obs)
        priv_n = self.critic.normalize_privileged(critic_obs)
        dist = self.actor.distribution(obs_n)
        action = dist.sample()
        self.obs_n[step], self.priv_n[step] = obs_n, priv_n
        self.actions[step] = action
        self.log_probs[step] = dist.log_prob(action).sum(-1)
        self.values[step] = self.critic(obs_n, priv_n)
        return action

    @torch.no_grad()
    def value_of(self, critic_obs: torch.Tensor) -> torch.Tensor:
        obs_n = self.actor.normalize(critic_obs[:, : self.obs_dim])
        return self.critic(obs_n, self.critic.normalize_privileged(critic_obs))

    @torch.no_grad()
    def record(
        self,
        step: int,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        terminal_critic_obs: torch.Tensor,
    ) -> None:
        """Store a scaled reward; truncated episodes are bootstrapped from their final state."""
        done = terminated | truncated
        self.discounted_return = self.discounted_return * self.cfg.gamma + reward
        self.return_rms.update(self.discounted_return)
        self.discounted_return = torch.where(done, torch.zeros_like(reward), self.discounted_return)
        scaled = (reward / torch.sqrt(self.return_rms.var + 1e-8)).clamp(-NORM_CLIP, NORM_CLIP)
        if bool(truncated.any()):
            scaled = scaled + self.cfg.gamma * truncated.float() * self.value_of(terminal_critic_obs)
        self.rewards[step], self.dones[step] = scaled, done.float()

    # ------------------------------------------------------------- update
    @torch.no_grad()
    def _advantages(self, last_critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        next_value = self.value_of(last_critic_obs)
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(cfg.n_steps)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + cfg.gamma * next_value * not_done - self.values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            adv[t] = gae
            next_value = self.values[t]
        return adv, adv + self.values

    def update(self, last_critic_obs: torch.Tensor) -> dict[str, float]:
        cfg = self.cfg
        adv, returns = self._advantages(last_critic_obs)
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        obs_n, priv_n, actions = flat(self.obs_n), flat(self.priv_n), flat(self.actions)
        old_logp, adv, returns = flat(self.log_probs), flat(adv), flat(returns)
        values_old = flat(self.values)
        total = obs_n.shape[0]
        mb = total // cfg.n_minibatches
        keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        stats: dict[str, list[float]] = {k: [] for k in keys}
        stop = False
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(total, device=self.device)
            for k in range(cfg.n_minibatches):
                idx = perm[k * mb : (k + 1) * mb]
                dist = self.actor.distribution(obs_n[idx])
                logp = dist.log_prob(actions[idx]).sum(-1)
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(a * ratio, a * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
                value = self.critic(obs_n[idx], priv_n[idx])
                v_loss = ((returns[idx] - value) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"].append(pg.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(((ratio - 1).abs() > cfg.clip_range).float().mean().item())
            if stop:
                break
        var_y = returns.var()
        out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
        out["explained_variance"] = float(1 - (returns - values_old).var() / var_y) if var_y > 0 else float("nan")
        out["std"] = float(self.actor.log_std.detach().exp().mean())
        out["early_stopped"] = float(stop)
        return out

    # ------------------------------------------------------- checkpoints
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "return_rms": self.return_rms.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
