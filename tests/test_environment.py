"""Episode logic of the base hand-cube environment: seeding, resets, termination."""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from manip_rubiks.environments import ENV_ID, EnvConfig, HandCubeEnv
from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.simulation.scene_builder import SceneConfig
from manip_rubiks.simulation.tactile_sensors import TactileConfig

NOISY = {"pose_noise": 0.003, "joint_noise": 0.05}


@pytest.fixture(scope="module")
def env():
    e = HandCubeEnv(EnvConfig(**NOISY))
    yield e
    e.close()


def assert_obs_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_passes_gymnasium_checker():
    env = HandCubeEnv(EnvConfig(scramble_moves=4, **NOISY))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=True)


def test_registered():
    env = gym.make(ENV_ID, config=EnvConfig(max_episode_steps=5))
    assert isinstance(env.unwrapped, HandCubeEnv)
    env.close()


def test_control_rate(env):
    assert env.n_substeps * env.model.opt.timestep == pytest.approx(env.cfg.control_dt)
    env.reset(seed=0)
    t0 = env.data.time
    env.step(np.zeros(env.robot.n))
    assert env.data.time - t0 == pytest.approx(env.cfg.control_dt)


def test_same_seed_same_episode_start(env):
    o1, i1 = env.reset(seed=11)
    o2, i2 = env.reset(seed=11)
    assert_obs_equal(o1, o2)
    np.testing.assert_array_equal(i1["privileged"], i2["privileged"])
    o3, _ = env.reset(seed=12)
    assert not np.array_equal(o1["q"], o3["q"])


def test_reset_after_rollout_matches_fresh_env(env):
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(40):
        env.step(rng.uniform(-1, 1, env.robot.n))
    obs, _ = env.reset(seed=4)
    fresh = HandCubeEnv(EnvConfig(**NOISY))
    obs_fresh, _ = fresh.reset(seed=4)
    assert_obs_equal(obs, obs_fresh)
    np.testing.assert_array_equal(env.data.qpos, fresh.data.qpos)
    np.testing.assert_array_equal(env.data.qvel, fresh.data.qvel)
    np.testing.assert_array_equal(env.data.ctrl, fresh.data.ctrl)
    assert env.data.time == fresh.data.time
    for k in range(5):  # stepping stays identical too
        a = np.full(env.robot.n, 0.3 * (-1) ** k)
        assert_obs_equal(env.step(a)[0], fresh.step(a)[0])


def test_initial_state_options(env):
    state = CubeState.solved().apply_sequence("R U2 F'")
    _, info = env.reset(seed=0, options={"state": state})
    assert info["cube_state"].same_configuration(state)
    _, info = env.reset(seed=0, options={"moves": "R U2 F'"})
    assert info["cube_state"].same_configuration(state)
    assert info["initial_state"].same_configuration(state)


def test_random_scramble_is_seeded_and_valid():
    env = HandCubeEnv(EnvConfig(scramble_moves=8))
    _, a = env.reset(seed=5)
    _, b = env.reset(seed=5)
    _, c = env.reset(seed=6)
    assert a["initial_state"] == b["initial_state"]
    assert a["initial_state"] != c["initial_state"]
    assert a["initial_state"].is_valid() and not a["initial_state"].is_solved()
    assert a["cube_state"].same_configuration(a["initial_state"])


def test_truncation_without_termination():
    env = HandCubeEnv(EnvConfig(max_episode_steps=6))
    env.reset(seed=0)
    flags = [env.step(np.zeros(env.robot.n))[2:4] for _ in range(6)]
    assert flags[:-1] == [(False, False)] * 5
    assert flags[-1] == (False, True)


def test_drop_terminates(env):
    env.reset(seed=0)
    pos, _ = env.cube.core_pose()
    env.cube.set_pose(pos + np.array([0.0, 0.2, 0.0]), env.data.xquat[env.cube.core_body].copy())
    _, _, terminated, truncated, info = env.step(np.zeros(env.robot.n))
    assert info["dropped"] and terminated and not truncated


def test_actions_are_clipped_and_reported(env):
    env.reset(seed=0)
    t0 = env.robot.target()
    a = np.zeros(env.robot.n)
    a[2] = 7.0
    obs, *_ = env.step(a)
    assert obs["prev_action"][2] == 1.0
    assert env.robot.target()[2] == pytest.approx(min(t0[2] + env.cfg.k_a, env.robot.ctrl_max[2]))


def test_random_rollout_stays_in_observation_space(env):
    obs, _ = env.reset(seed=1)
    env.action_space.seed(1)
    for _ in range(60):
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert not info["unstable"]
        if terminated or truncated:
            obs, _ = env.reset()


@pytest.mark.parametrize("mode", ["contact", "force"])
def test_tactile_configurations(mode):
    env = HandCubeEnv(EnvConfig(tactile=TactileConfig(mode=mode, coverage="hand")))
    obs, _ = env.reset(seed=0)
    assert obs["tactile"].shape == env.tactile.shape
    assert env.observation_space.contains(obs)
    assert np.abs(obs["tactile"]).sum() > 0  # the cube rests on the hand


def test_single_axis_variant_scrambles_u_and_d_only():
    env = HandCubeEnv(EnvConfig(scene=SceneConfig(variant="single_axis"), scramble_moves=3))
    assert env.scramble_faces == ("U", "D")
    _, info = env.reset(seed=2)
    assert info["cube_state"].same_configuration(info["initial_state"])
    env.step(np.zeros(env.robot.n))


def test_rgb_array_render():
    env = HandCubeEnv(render_mode="rgb_array")
    env.reset(seed=0)
    try:
        img = env.render()
    except Exception as exc:  # no OpenGL context available on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    finally:
        env.close()
    assert img.shape == (480, 640, 3)
    assert img.std() > 0
