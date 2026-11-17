"""GPU SAC: n-step replay targets, squashed-Gaussian likelihood, blindfolded actor and updates."""

import numpy as np
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from manip_rubiks.training.sac_gpu import SAC, ReplayBuffer, SACConfig, SquashedGaussianActor

CPU = torch.device("cpu")


def fill_buffer(rewards, terminated, truncated, cap=None):
    """Buffer whose observations encode (step, world) so samples can be traced back."""
    t, n = rewards.shape
    buf = ReplayBuffer((cap or t) * n, n, critic_dim=2, act_dim=1, device=CPU)
    for i in range(t):
        obs = torch.stack([torch.full((n,), float(i)), torch.arange(n, dtype=torch.float32)], 1)
        nxt = obs + torch.tensor([0.5, 0.0])  # "terminal" observation of step i
        buf.add(obs, torch.zeros(n, 1), rewards[i], terminated[i], truncated[i], nxt)
    return buf


def reference_nstep(rewards, terminated, truncated, i, w, n, gamma):
    ret, m = 0.0, 0
    for k in range(n):
        ret += gamma**k * rewards[i + k, w]
        m += 1
        if terminated[i + k, w] or truncated[i + k, w]:
            break
    last = i + m - 1
    return ret, last, 0.0 if terminated[last, w] else gamma**m


def test_nstep_sampling_matches_reference():
    g = torch.Generator().manual_seed(0)
    t, n, n_step, gamma = 12, 3, 3, 0.9
    rewards = torch.randn(t, n, generator=g)
    terminated = torch.zeros(t, n, dtype=torch.bool)
    truncated = torch.zeros(t, n, dtype=torch.bool)
    terminated[2, 0] = terminated[7, 1] = True
    truncated[4, 0] = truncated[5, 2] = truncated[6, 2] = True
    buf = fill_buffer(rewards, terminated, truncated)
    b = buf.sample(512, n_step, gamma, generator=g)
    starts = b["obs"][:, 0].long().numpy()
    worlds = b["obs"][:, 1].long().numpy()
    assert starts.max() <= t - n_step  # never runs past the newest step
    r, te, tr = rewards.numpy(), terminated.numpy(), truncated.numpy()
    for j in range(512):
        ret, last, disc = reference_nstep(r, te, tr, starts[j], worlds[j], n_step, gamma)
        assert np.isclose(b["return"][j].item(), ret, atol=1e-5)
        assert b["next_obs"][j, 0].item() == last + 0.5  # bootstraps from the state reached, not the reset
        assert np.isclose(b["discount"][j].item(), disc, atol=1e-6)


def test_sampling_respects_ring_buffer_wraparound():
    t, n = 10, 2
    rewards = torch.arange(t, dtype=torch.float32)[:, None].repeat(1, n)
    flags = torch.zeros(t, n, dtype=torch.bool)
    buf = fill_buffer(rewards, flags, flags, cap=4)  # keeps steps 6..9 only
    b = buf.sample(256, 2, 1.0)
    starts = b["obs"][:, 0]
    assert starts.min() >= 6 and starts.max() <= 8
    torch.testing.assert_close(b["return"], 2 * starts + 1)  # r_i + r_{i+1} with r_i = i


def test_squashed_log_prob_matches_torch_distribution():
    torch.manual_seed(0)
    actor = SquashedGaussianActor(4, 3, (16,), log_std_init=0.0)
    with torch.no_grad():
        actor.mean.weight.normal_()
        actor.log_std.weight.normal_(0, 0.3)
    obs = torch.randn(64, 4)
    torch.manual_seed(1)
    a, logp = actor.sample(obs)
    mu, log_std = actor(obs)
    torch.manual_seed(1)
    u = mu + log_std.exp() * torch.randn_like(mu)
    torch.testing.assert_close(a, torch.tanh(u))
    # Change of variables in float64: log N(u) - log(1 - tanh(u)^2).
    u64, mu64, std64 = u.double(), mu.double(), log_std.double().exp()
    expected = (Normal(mu64, std64).log_prob(u64) - torch.log1p(-(torch.tanh(u64) ** 2))).sum(-1)
    torch.testing.assert_close(logp.double(), expected, atol=1e-4, rtol=1e-5)
    # Agrees with torch's own transformed distribution away from saturation.
    dist = TransformedDistribution(Normal(mu, log_std.exp()), TanhTransform())
    ok = a.abs().max(-1).values < 0.99
    torch.testing.assert_close(logp[ok], dist.log_prob(a).sum(-1)[ok], atol=1e-3, rtol=1e-4)
    assert a.abs().max() < 1.0


def make_sac(**kw):
    cfg = SACConfig(
        buffer_size=64, batch_size=16, learning_starts=2, actor_sizes=(16,), critic_sizes=(16,), n_step=2, **kw
    )
    return SAC(4, obs_dim=5, privileged_dim=2, act_dim=3, cfg=cfg, device=CPU)


def test_actor_never_sees_privileged_state():
    sac = make_sac()
    assert sac.actor.body[0].in_features == sac.obs_dim
    assert {id(p) for p in sac.actor.parameters()}.isdisjoint(id(p) for p in sac.critic.parameters())
    assert not any("privileged" in k for k in sac.actor.state_dict())
    obs = torch.randn(4, 5)
    torch.manual_seed(0)
    a = make_sac()
    torch.manual_seed(0)
    b = make_sac()
    a.act(obs, torch.cat([obs, torch.zeros(4, 2)], 1))
    b.act(obs, torch.cat([obs, 1e3 * torch.randn(4, 2)], 1))
    assert torch.equal(a.actor.obs_rms.mean, b.actor.obs_rms.mean)
    assert torch.equal(a.actor.act(obs), b.actor.act(obs))


def test_update_runs_and_moves_targets_slowly():
    torch.manual_seed(0)
    sac = make_sac(tau=0.1, updates_per_step=1)
    assert sac.update() == {}  # nothing before learning_starts
    for _ in range(4):
        obs = torch.randn(4, 7)
        action = sac.act(obs[:, :5], obs)
        sac.record(obs, action, torch.randn(4), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), obs)
    before = [p.clone() for p in sac.critic_target.parameters()]
    stats = sac.update()
    assert {"critic_loss", "actor_loss", "alpha", "entropy"} <= stats.keys()
    assert all(np.isfinite(v) for v in stats.values())
    for b, p, c in zip(before, sac.critic_target.parameters(), sac.critic.parameters(), strict=True):
        torch.testing.assert_close(p, 0.9 * b + 0.1 * c)  # Polyak step towards the updated critic
    assert any(not torch.equal(b, p) for b, p in zip(before, sac.critic_target.parameters(), strict=True))
