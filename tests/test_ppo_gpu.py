"""GPU PPO: normalisation, advantages, truncation bootstrapping, blindfolded actor and deployment."""

import csv

import mlflow
import numpy as np
import pytest
import torch

from manip_rubiks.environments.grasp_env import GraspEnv
from manip_rubiks.training.asymmetric import PRIVILEGED_KEY
from manip_rubiks.training.ppo_gpu import PPO, PPOConfig, RunningMeanStd

CPU = torch.device("cpu")


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = [rng.normal(3.0, 2.0, (n, 4)) for n in (5, 17, 64)]
    rms = RunningMeanStd(4, epsilon=1e-12)
    for x in data:
        rms.update(torch.as_tensor(x, dtype=torch.float32))
    allx = np.concatenate(data)
    np.testing.assert_allclose(rms.mean.numpy(), allx.mean(0), rtol=1e-5)
    np.testing.assert_allclose(rms.var.numpy(), allx.var(0), rtol=1e-4)
    assert rms.normalize(torch.full((1, 4), 1e6)).max() == 10.0


def make_ppo(n=3, steps=4, **kw):
    return PPO(n, obs_dim=5, privileged_dim=2, act_dim=2, cfg=PPOConfig(n_steps=steps, **kw), device=CPU)


def test_actor_never_sees_privileged_state():
    ppo = make_ppo()
    assert ppo.actor.body[0].in_features == ppo.obs_dim
    assert ppo.critic.body[0].in_features == ppo.obs_dim + ppo.privileged_dim
    actor_params = {id(p) for p in ppo.actor.parameters()}
    assert actor_params.isdisjoint(id(p) for p in ppo.critic.parameters())
    assert not any("privileged" in k for k in ppo.actor.state_dict())
    # Collecting with wildly different privileged values leaves the actor's statistics identical.
    obs = torch.randn(3, 5)
    a, b = make_ppo(), make_ppo()
    a.act(obs, torch.cat([obs, torch.zeros(3, 2)], 1), 0)
    b.act(obs, torch.cat([obs, 1e3 * torch.randn(3, 2)], 1), 0)
    assert torch.equal(a.actor.obs_rms.mean, b.actor.obs_rms.mean)
    assert torch.equal(a.actor.obs_rms.var, b.actor.obs_rms.var)


def test_gae_matches_reference_loop():
    ppo = make_ppo(n=2, steps=5, gamma=0.9, gae_lambda=0.8)
    g = torch.Generator().manual_seed(0)
    ppo.rewards = torch.randn(5, 2, generator=g)
    ppo.values = torch.randn(5, 2, generator=g)
    ppo.dones = torch.tensor([[0, 0], [1, 0], [0, 0], [0, 1], [0, 0]], dtype=torch.float32)
    last_value = torch.tensor([0.5, -0.3])
    ppo.value_of = lambda _: last_value
    adv, ret = ppo._advantages(None)
    r, v, d = ppo.rewards.numpy(), ppo.values.numpy(), ppo.dones.numpy()
    expected = np.zeros((5, 2))
    for e in range(2):
        gae, nxt = 0.0, last_value[e].item()
        for t in reversed(range(5)):
            nd = 1.0 - d[t, e]
            delta = r[t, e] + 0.9 * nxt * nd - v[t, e]
            gae = delta + 0.9 * 0.8 * nd * gae
            expected[t, e] = gae
            nxt = v[t, e]
    np.testing.assert_allclose(adv.numpy(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ret.numpy(), expected + v, rtol=1e-5, atol=1e-6)


def test_truncated_episodes_are_bootstrapped():
    ppo = make_ppo(n=2, gamma=0.5)
    ppo.return_rms.var.fill_(4.0)  # return scale 2, with enough history that one update barely moves it
    ppo.return_rms.count.fill_(1e9)
    ppo.value_of = lambda _: torch.tensor([4.0, 4.0])
    reward = torch.tensor([1.0, 1.0])
    ppo.record(0, reward, torch.tensor([True, False]), torch.tensor([False, True]), torch.zeros(2, 7))
    np.testing.assert_allclose(ppo.rewards[0].numpy(), [0.5, 0.5 + 0.5 * 4.0], rtol=1e-4)
    assert ppo.dones[0].tolist() == [1.0, 1.0]
    assert ppo.discounted_return.tolist() == [0.0, 0.0]


def test_update_runs_and_reports_losses():
    torch.manual_seed(0)
    ppo = make_ppo(n=8, steps=6, n_minibatches=2, n_epochs=2)
    for t in range(6):
        obs = torch.randn(8, 5)
        ppo.act(obs, torch.cat([obs, torch.randn(8, 2)], 1), t)
        ppo.record(t, torch.randn(8), torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool), None)
    before = [p.clone() for p in ppo.actor.parameters()]
    stats = ppo.update(torch.randn(8, 7))
    assert {"policy_loss", "value_loss", "approx_kl", "explained_variance"} <= set(stats)
    assert any(not torch.equal(a, b) for a, b in zip(before, ppo.actor.parameters(), strict=True))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_training_cli_deploys_on_the_cpu_benchmark(tmp_path, monkeypatch):
    from manip_rubiks.evaluation.grasp_benchmark import build_policy, run_episode
    from manip_rubiks.training.train_grasp_gpu import main as train_main

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlruns = tmp_path / "mlruns"
    train_main(
        [
            "--name=t",
            f"--out={tmp_path}",
            f"--mlflow-uri=file://{mlruns}",
            "--experiment=test",
            "--timesteps=2048",
            "--num-envs=64",
            "--n-steps=16",
            "--eval-every=1024",
            "--eval-envs=8",
            "--bank-size=64",
            "--video-episodes=4",
        ]
    )
    run = tmp_path / "t"
    for f in ("policy.pt", "best/policy.pt", "config.json", "summary.json"):
        assert (run / f).exists(), f
    policy = build_policy(f"gpu={run}")
    env = GraspEnv()
    obs, _ = env.reset(seed=0)
    a = policy(obs)
    assert a.shape == (22,) and np.all(np.abs(a) <= 1.0)
    with pytest.raises(ValueError):
        policy({**obs, PRIVILEGED_KEY: np.zeros(85, np.float32)})
    result = run_episode(env, policy, "gpu", "reorient", 100_000)
    assert result.steps > 0
    mlflow.set_tracking_uri(f"file://{mlruns}")
    (logged,) = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert logged.info.status == "FINISHED" and logged.data.tags["backend"] == "mujoco_warp"
    assert "eval/success_rate" in logged.data.metrics and "time/fps" in logged.data.metrics
    client = mlflow.MlflowClient()

    def ls(path):
        return {a.path for a in client.list_artifacts(logged.info.run_id, path)}

    assert ls("checkpoints/latest") == {"checkpoints/latest/policy.pt"}
    assert ls("eval") == {"eval/000001024", "eval/000002048"}
    assert ls("eval/000001024/videos")  # the first evaluation records videos
    rows = list(csv.DictReader((run / "eval" / "000002048" / "episodes.csv").open()))
    assert len(rows) == 8 and [int(r["reset_seed"]) for r in rows] == list(range(10_000, 10_008))
    terms = [[float(v) for k, v in r.items() if k.startswith("reward/")] for r in rows]
    np.testing.assert_allclose([sum(t) for t in terms], [float(r["return"]) for r in rows], rtol=1e-3, atol=1e-2)
    latest = torch.load(run / "latest" / "policy.pt", weights_only=False)
    assert {"optimizer", "env_rng", "torch_rng", "best_score"} <= set(latest) and latest["timesteps"] == 2048
    mlflow.set_tracking_uri(None)
