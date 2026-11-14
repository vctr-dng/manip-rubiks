"""Hand + cube scene: resets, reproducibility and short interaction rollouts."""

import numpy as np
import pytest

from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.simulation.scene_builder import HandCubeScene
from manip_rubiks.validation import m0


@pytest.fixture(scope="module")
def scene():
    return HandCubeScene()


def test_reset_places_cube_in_hand(scene):
    scene.reset(seed=0)
    pos, _ = scene.cube.core_pose()
    assert pos[2] > scene.cfg.wrist_height
    assert scene.cube.read().state.is_solved()
    assert scene.warnings() == 0


def test_reset_with_scrambled_state(scene):
    state = CubeState.solved().apply_sequence("R U R' F2")
    scene.reset(seed=1, state=state)
    assert scene.cube.read().state.same_configuration(state)


def test_reset_is_deterministic(scene):
    other = HandCubeScene()
    scene.reset(seed=5, pose_noise=0.004, joint_noise=0.05)
    scene.step(200)
    scene.reset(seed=5, pose_noise=0.004, joint_noise=0.05)
    other.reset(seed=5, pose_noise=0.004, joint_noise=0.05)
    assert np.array_equal(scene.data.qpos, other.data.qpos)
    assert np.array_equal(scene.data.qvel, other.data.qvel)


def test_random_finger_rollout_is_stable(scene):
    stats = m0.rollout(scene, seed=0, duration=1.5)
    assert stats["finite"]
    assert stats["warnings"] == 0
    assert stats["max_pen_in_hand_cube_internal_mm"] < 1.0
    assert stats["max_pen_in_hand_hand_cube_mm"] < 5.0
    assert stats["final_state_valid"]


def test_reproducibility_check():
    assert m0.check_reproducibility(duration=0.5).passed


@pytest.mark.slow
def test_gate_m0_quick():
    report = m0.run_all(quick=True)
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert report["passed"], failed
