"""Sharpa Wave integration: loading, actuator mapping and action interface."""

import mujoco
import numpy as np
import pytest

from manip_rubiks.simulation.robot_interface import JOINTS
from manip_rubiks.simulation.scene_builder import HandCubeScene, SceneConfig, mount_rotation


@pytest.fixture(scope="module")
def hand_scene():
    return HandCubeScene(SceneConfig(variant=None))


def test_hand_loads_with_22_actuated_joints(hand_scene):
    m = hand_scene.model
    assert m.nu == 22
    assert hand_scene.robot.n == 22
    assert len(JOINTS) == 22
    assert hand_scene.cube is None


def test_mount_is_palm_up():
    r = mount_rotation(0.0)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 0, 1])  # palm normal up
    np.testing.assert_allclose(r @ [0, 0, 1], [1, 0, 0])  # fingers along +x
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_fingertips_above_palm_and_forward(hand_scene):
    hand_scene.reset()
    tips = hand_scene.robot.fingertip_positions()
    assert np.all(tips[:, 0] > 0.05)  # fingers extend along +x


def test_targets_are_clipped(hand_scene):
    robot = hand_scene.robot
    applied = robot.set_target(robot.ctrl_max + 1.0)
    np.testing.assert_allclose(applied, robot.ctrl_max)
    applied = robot.set_target(robot.ctrl_min - 1.0)
    np.testing.assert_allclose(applied, robot.ctrl_min)


def test_delta_action(hand_scene):
    robot = hand_scene.robot
    hand_scene.reset()
    t0 = robot.target()
    a = np.zeros(robot.n)
    a[0] = 1.0
    a[1] = -5.0  # clipped to -1
    t1 = robot.apply_action(a, k_a=0.01)
    assert t1[0] - t0[0] == pytest.approx(0.01)
    assert t1[1] - t0[1] == pytest.approx(-0.01)
    np.testing.assert_allclose(np.delete(t1, [0, 1]), np.delete(t0, [0, 1]))


@pytest.mark.parametrize("joint", ["index_PIP", "thumb_CMC_FE", "pinky_MCP_FE"])
def test_position_actuator_tracks_step(hand_scene, joint):
    m, d, robot = hand_scene.model, hand_scene.data, hand_scene.robot
    hand_scene.reset()
    # open hand: in the caged home pose the opposed thumb blocks finger flexion
    robot.reset(np.clip(np.zeros(robot.n), robot.q_min, robot.q_max))
    mujoco.mj_step(m, d, 150)
    j = JOINTS.index(joint)
    target = robot.target()
    target[j] = 0.5 * (robot.ctrl_min[j] + robot.ctrl_max[j])
    robot.set_target(target)
    mujoco.mj_step(m, d, 400)
    assert abs(robot.q()[j] - target[j]) < 0.06
    assert np.all(np.abs(robot.effort()) <= robot.force_limit + 1e-9)
