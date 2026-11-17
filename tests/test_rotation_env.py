"""Quarter-turn reorientation environment: goals, teacher observation, reward and termination."""

import warnings

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from manip_rubiks.environments import (
    ROTATION_ENV_ID,
    ROTATION_OBS_KEYS,
    ROTATION_STUDENT_OBS_KEYS,
    RotationConfig,
    RotationEnv,
)
from manip_rubiks.environments.rotation_env import allowed_commands, command_rotation
from manip_rubiks.planning.cube_state import ROTATIONS
from manip_rubiks.simulation.rotations import mat_to_quat, rotation_angle

ZERO = np.zeros(22)


@pytest.fixture(scope="module")
def env():
    e = RotationEnv()
    yield e
    e.close()


def teleport_to_goal(env: RotationEnv) -> None:
    """Rotate the cube in place onto the current goal (a spin about the palm normal keeps it on the palm)."""
    pos, _ = env.cube.core_pose()
    r_core = env.r_mount @ env.goal @ env.cube_reading.frame.T
    env.cube.set_pose(pos, mat_to_quat(r_core))
    mujoco.mj_forward(env.model, env.data)


def test_passes_gymnasium_checker():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(RotationEnv(), skip_render_check=True)
    assert gym.spec(ROTATION_ENV_ID) is not None


def test_commands_are_quarter_turns_within_the_24_orientations(env):
    for c in range(6):
        r = command_rotation(c)
        assert rotation_angle(r) == pytest.approx(np.pi / 2)
        cube_frame = env.r_rest_hand.T @ r @ env.r_rest_hand
        assert any(np.allclose(cube_frame, q) for q in ROTATIONS)
    np.testing.assert_array_equal(allowed_commands("xz"), [0, 1, 4, 5])
    with pytest.raises(ValueError):
        allowed_commands("")


def test_first_goal_is_a_quarter_turn_from_the_start(env):
    for seed in range(3):
        obs, info = env.reset(seed=seed)
        assert info["theta"] == pytest.approx(np.pi / 2, abs=0.1)
        assert obs["command"].sum() == 1.0 and obs["command"][info["command"]] == 1.0
        assert env.command in env.commands


def test_observation_keys_with_and_without_teacher():
    teacher, student = RotationEnv(), RotationEnv(RotationConfig(teacher=False))
    ot, _ = teacher.reset(seed=0)
    os_, _ = student.reset(seed=0)
    assert set(ot) == set(ROTATION_OBS_KEYS) == set(teacher.observation_space.spaces)
    assert set(os_) == set(ROTATION_STUDENT_OBS_KEYS) == set(student.observation_space.spaces)
    # The teacher's goal error is the identity rotation's first two columns once the cube is on the goal.
    teleport_to_goal(teacher)
    t = teacher.teacher_state()
    np.testing.assert_allclose(t[:6], np.eye(3)[:, :2].ravel(), atol=1e-6)
    teacher.close()
    student.close()


def test_reaching_a_goal_pays_progress_and_bonus_then_issues_the_next(env):
    env.reset(seed=0, options={"commands": [0, 4]})
    assert env.command == 0
    first_goal = env.goal.copy()
    teleport_to_goal(env)
    rewards = []
    for _ in range(env.rot_cfg.hold_steps):
        _, r, terminated, _, info = env.step(ZERO)
        rewards.append(info["reward_terms"])
        assert not terminated
    assert rewards[0]["progress"] > 5.0  # ~pi/2 of progress at once
    assert info["goal_reached"] and info["goals_reached"] == 1
    assert rewards[-1]["goal"] == env.rot_cfg.reward.goal
    assert env.command == 4
    assert rotation_angle(first_goal.T @ env.goal) == pytest.approx(np.pi / 2)
    assert info["theta"] == pytest.approx(np.pi / 2, abs=0.1)


def test_goal_timeout_terminates_with_failure_penalty():
    e = RotationEnv(RotationConfig(goal_timeout=10))
    e.reset(seed=0)
    steps = 0
    terminated = False
    while not terminated and steps < 20:
        _, _, terminated, _, info = e.step(ZERO)
        steps += 1
    assert terminated and steps == 10
    assert info["timed_out"] and not info["is_success"]
    assert info["reward_terms"]["timeout"] == -e.rot_cfg.reward.timeout
    assert info["reward_terms"]["failure"] == 0.0
    e.close()


def test_curriculum_tightens_the_tolerance(env):
    env.set_difficulty(0.0)
    assert env.tolerance == pytest.approx(env.rot_cfg.easy_tolerance)
    env.set_difficulty(1.0)
    assert env.tolerance == pytest.approx(env.rot_cfg.goal_tolerance)
