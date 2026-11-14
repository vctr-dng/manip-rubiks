"""Asymmetric PPO: the actor must never depend on privileged state; deployment must stay blindfolded."""

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from manip_rubiks.environments import GRASP_OBS_KEYS, GraspEnv
from manip_rubiks.policies.learned import LearnedPolicy
from manip_rubiks.training.asymmetric import PRIVILEGED_KEY, AsymmetricActorCriticPolicy, PrivilegedObservation
from manip_rubiks.training.train_grasp import main as train_main

POLICY_KWARGS = {"net_arch": {"pi": [32, 32], "vf": [32, 32]}, "actor_keys": GRASP_OBS_KEYS}


@pytest.fixture(scope="module")
def model():
    venv = DummyVecEnv([lambda: PrivilegedObservation(GraspEnv())])
    m = PPO(AsymmetricActorCriticPolicy, venv, n_steps=16, batch_size=16, policy_kwargs=POLICY_KWARGS, device="cpu")
    yield m
    venv.close()


def obs_tensor(model, obs):
    t, _ = model.policy.obs_to_tensor(obs)
    return t


def test_wrapper_exposes_privileged_state():
    env = PrivilegedObservation(GraspEnv())
    obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(obs[PRIVILEGED_KEY], info[PRIVILEGED_KEY].astype(np.float32))
    assert env.observation_space.contains(obs)
    obs, *_ = env.step(np.zeros(22))
    assert set(obs) == {*GRASP_OBS_KEYS, PRIVILEGED_KEY}


def test_actor_ignores_privileged_state_but_critic_uses_it(model):
    obs, _ = PrivilegedObservation(GraspEnv()).reset(seed=0)
    other = {**obs, PRIVILEGED_KEY: np.random.default_rng(0).normal(size=obs[PRIVILEGED_KEY].shape).astype(np.float32)}
    with th.no_grad():
        a = model.policy.get_distribution(obs_tensor(model, obs)).distribution.mean
        b = model.policy.get_distribution(obs_tensor(model, other)).distribution.mean
        va = model.policy.predict_values(obs_tensor(model, obs))
        vb = model.policy.predict_values(obs_tensor(model, other))
    th.testing.assert_close(a, b, rtol=0, atol=0)
    assert not th.allclose(va, vb)


def test_actor_input_width_excludes_privileged_state(model):
    obs_space = model.observation_space
    actor_dim = sum(int(np.prod(obs_space[k].shape)) for k in GRASP_OBS_KEYS)
    assert model.policy.mlp_extractor.actor_dim == actor_dim
    assert model.policy.mlp_extractor.policy_net[0].in_features == actor_dim


def test_actor_keys_may_not_include_privileged_state(model):
    with pytest.raises(ValueError):
        AsymmetricActorCriticPolicy(
            model.observation_space,
            model.action_space,
            lambda _: 1e-4,
            actor_keys=(*GRASP_OBS_KEYS, PRIVILEGED_KEY),
        )


def test_saved_policy_deploys_without_privileged_state(model, tmp_path):
    model.save(tmp_path / "model.zip")
    norm = VecNormalize(DummyVecEnv([lambda: PrivilegedObservation(GraspEnv())]))
    norm.save(str(tmp_path / "vecnormalize.pkl"))
    policy = LearnedPolicy(tmp_path / "model.zip")
    env = GraspEnv()
    obs, _ = env.reset(seed=0)
    a1, a2 = policy(obs), policy(obs)
    assert a1.shape == (22,) and np.all(np.abs(a1) <= 1.0)
    np.testing.assert_array_equal(a1, a2)  # deterministic
    with pytest.raises(ValueError):
        policy({**obs, PRIVILEGED_KEY: np.zeros(policy.privileged_dim, np.float32)})


@pytest.mark.slow
def test_training_cli_writes_a_loadable_run(tmp_path):
    train_main(
        [
            "--name=t",
            f"--out={tmp_path}",
            "--timesteps=256",
            "--n-envs=2",
            "--n-eval-envs=1",
            "--n-steps=64",
            "--batch-size=64",
            "--eval-every=128",
            "--eval-episodes=1",
        ]
    )
    run = tmp_path / "t"
    for f in ("model.zip", "vecnormalize.pkl", "config.json", "summary.json", "best/best_model.zip"):
        assert (run / f).exists(), f
    policy = LearnedPolicy.from_run(run)
    obs, _ = GraspEnv().reset(seed=0)
    assert policy(obs).shape == (22,)
