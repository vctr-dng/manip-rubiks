"""Grasp-stabilisation environment: wrist schedule, pushes, rewards, success and termination."""

import warnings
from dataclasses import replace

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from manip_rubiks.environments import GRASP_ENV_ID, GRASP_OBS_KEYS, GraspConfig, GraspEnv
from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.policies import PassivePolicy, ScriptedGraspPolicy

PALM_DOWN = {"tilt": np.pi, "pushes": False}


@pytest.fixture(scope="module")
def env():
    e = GraspEnv()
    yield e
    e.close()


def run(env, policy, **reset_kwargs):
    obs, info = env.reset(**reset_kwargs)
    total = 0.0
    while True:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total += r
        if terminated or truncated:
            return total, terminated, truncated, info


def test_passes_gymnasium_checker():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(GraspEnv(), skip_render_check=True)


def test_registered():
    e = gym.make(GRASP_ENV_ID)
    assert isinstance(e.unwrapped, GraspEnv)
    e.close()


def test_observation_adds_only_the_gravity_direction(env):
    obs, info = env.reset(seed=0)
    assert set(obs) == set(GRASP_OBS_KEYS) == set(env.observation_space.spaces)
    np.testing.assert_allclose(np.linalg.norm(obs["gravity"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(obs["gravity"], env.d_nominal, atol=1e-6)
    assert "privileged" in info and "privileged" not in obs


def test_observations_do_not_depend_on_the_scramble():
    a, b = GraspEnv(), GraspEnv()
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1, options={"state": CubeState.solved().apply_sequence("R U' F2")})
    rng = np.random.default_rng(0)
    for _ in range(8):
        for k in GRASP_OBS_KEYS:
            np.testing.assert_allclose(oa[k], ob[k], atol=1e-4, err_msg=k)
        action = rng.uniform(-1, 1, a.robot.n)
        oa, *_ = a.step(action)
        ob, *_ = b.step(action)


def test_tilt_option_turns_the_palm_down(env):
    env.reset(seed=0, options=PALM_DOWN)
    (move,) = env.schedule
    steps = int(np.ceil(move.t_end / env.cfg.control_dt)) + 1
    for _ in range(steps):
        obs, *_ = env.step(np.zeros(env.robot.n))
    np.testing.assert_allclose(obs["gravity"], -env.d_nominal, atol=1e-6)
    assert env.tilt() == pytest.approx(np.pi)
    np.testing.assert_allclose(np.linalg.norm(env.model.opt.gravity), 9.81)


def test_tilt_directions_follow_the_hand_frame(env):
    # Towards the fingertips (hand +z) is world +x for the palm-up mount.
    d = env.r_mount @ env.tilted_direction(np.pi / 2, 0.0)
    np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-9)
    for az in np.linspace(0, 2 * np.pi, 7):
        t = env.tilted_direction(0.7, az)
        assert np.arccos(t @ env.d_nominal) == pytest.approx(0.7)


def test_schedule_is_smooth_and_seeded(env):
    env.reset(seed=4)
    first = list(env.schedule)
    dirs = np.array([env.gravity_direction_at(t) for t in np.arange(0, 10, 0.01)])
    assert np.max(np.linalg.norm(np.diff(dirs, axis=0), axis=1)) < 0.05
    env.reset(seed=4)
    assert [m.t_start for m in env.schedule] == [m.t_start for m in first]


def test_zero_difficulty_keeps_the_palm_up_without_pushes():
    e = GraspEnv(GraspConfig(difficulty=0.0, push_rate=50.0))
    e.reset(seed=0)
    assert all(np.allclose(m.d_to, e.d_nominal) for m in e.schedule)
    for _ in range(30):
        e.step(np.zeros(e.robot.n))
        assert not e.data.xfrc_applied.any()


def test_pushes_hit_the_cube_and_can_be_disabled():
    e = GraspEnv(replace(GraspConfig(), push_rate=50.0))
    e.reset(seed=0)
    pushed = []
    for _ in range(20):
        _, _, _, _, info = e.step(np.zeros(e.robot.n))
        pushed.append(info["pushing"])
    assert any(pushed)
    assert np.linalg.norm(e.data.xfrc_applied[e.cube.core_body, :3]) <= e.grasp_cfg.push_force
    e.reset(seed=0, options={"pushes": False})
    for _ in range(20):
        _, _, _, _, info = e.step(np.zeros(e.robot.n))
        assert not info["pushing"]


def _push_wrenches(options, steps=60):
    e = GraspEnv(replace(GraspConfig(), push_rate=50.0))
    e.reset(seed=0, options=options)
    wrenches = []
    for _ in range(steps):
        e.step(np.zeros(e.robot.n))
        wrenches.append(e.data.xfrc_applied[e.cube.core_body].copy())
    e.close()
    return np.array(wrenches)


def test_push_scale_and_kind_change_only_the_wrench():
    both = _push_wrenches({})
    assert np.abs(both[:, :3]).max() > 0 and np.abs(both[:, 3:]).max() > 0
    np.testing.assert_allclose(_push_wrenches({"push_scale": 0.5}), 0.5 * both)
    force, torque = _push_wrenches({"push_kind": "force"}), _push_wrenches({"push_kind": "torque"})
    np.testing.assert_allclose(force[:, :3], both[:, :3])
    assert not force[:, 3:].any()
    np.testing.assert_allclose(torque[:, 3:], both[:, 3:])
    assert not torque[:, :3].any()
    assert not _push_wrenches({"push_scale": 0.0}).any()


def test_push_kind_is_validated(env):
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"push_kind": "spin"})
    env.reset(seed=0)  # leaves the shared env in a valid state


def test_wrist_moving_follows_the_schedule(env):
    env.reset(seed=1, options={"tilt": np.pi, "pushes": False})
    move = env.schedule[0]
    assert not env.wrist_moving(0.0)
    assert env.wrist_moving(0.5 * (move.t_start + move.t_end))
    assert not env.wrist_moving(move.t_end + 0.1)


def test_reset_after_a_palm_down_episode_matches_a_fresh_env(env):
    env.reset(seed=2, options=PALM_DOWN)
    for _ in range(80):
        env.step(np.zeros(env.robot.n))
    obs, _ = env.reset(seed=5)
    fresh = GraspEnv()
    obs_fresh, _ = fresh.reset(seed=5)
    for k in GRASP_OBS_KEYS:
        np.testing.assert_array_equal(obs[k], obs_fresh[k], err_msg=k)
    np.testing.assert_array_equal(env.data.qpos, fresh.data.qpos)


def test_passive_hand_drops_the_cube_palm_down(env):
    total, terminated, truncated, info = run(env, PassivePolicy(), seed=0, options=PALM_DOWN)
    assert terminated and not truncated
    assert info["dropped"] and not info["is_success"]
    assert total < 0  # the failure penalty outweighs the alive bonus collected


def test_scripted_power_grasp_holds_palm_down(env):
    # The power grasp keeps the cube but shears the R and L layers by 5-17 degrees,
    # so only the hold itself is asserted here, not the alignment tolerance.
    total, terminated, truncated, info = run(env, ScriptedGraspPolicy(), seed=0, options=PALM_DOWN)
    assert truncated and not terminated
    assert not info["dropped"] and info["state_preserved"]
    assert info["displacement"] < 0.04
    assert total > 0.3 * env.cfg.max_episode_steps


def test_unintended_layer_turn_terminates(env):
    env.reset(seed=0)
    env.cube.set_state(CubeState.solved().apply("R"))
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, _, info = env.step(np.zeros(env.robot.n))
    assert terminated and not info["state_preserved"] and not info["is_success"]
    assert reward < -env.grasp_cfg.reward.failure / 2


def test_reward_terms_sum_to_the_reward(env):
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, r, terminated, truncated, info = env.step(rng.uniform(-1, 1, 22))
        terms = info["reward_terms"]
        assert set(terms) == {
            "alive",
            "displacement",
            "rotation",
            "alignment",
            "velocity",
            "effort",
            "action_rate",
            "failure",
        }
        np.testing.assert_allclose(sum(terms.values()), r, rtol=1e-12, atol=1e-12)
        assert terms["alive"] > 0 and all(v <= 0 for k, v in terms.items() if k != "alive")
        if terminated or truncated:
            break
