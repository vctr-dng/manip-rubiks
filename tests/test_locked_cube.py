"""Locked-layer cube: the cubies stay welded together under a squeezing grasp."""

import numpy as np
import pytest

from manip_rubiks.environments import EnvConfig, HandCubeEnv
from manip_rubiks.policies import ScriptedGraspPolicy
from manip_rubiks.simulation.cube_model import cube_xml
from manip_rubiks.simulation.scene_builder import SceneConfig


def squeeze(lock: bool, steps: int = 50) -> float:
    """Largest cubie misalignment while the scripted power grasp squeezes the cube for ``steps`` (2 s)."""
    env = HandCubeEnv(EnvConfig(scene=SceneConfig(lock_layers=lock)))
    policy = ScriptedGraspPolicy(k_a=env.cfg.k_a)
    obs, _ = env.reset(seed=0)
    worst = 0.0
    for _ in range(steps):
        obs, _, terminated, _, _ = env.step(policy(obs))
        worst = max(worst, env.cube_reading.max_error)
        assert not terminated
    env.close()
    return worst


def test_locked_cube_stays_rigid_under_power_grasp():
    locked, free = squeeze(True), squeeze(False)
    assert locked < np.deg2rad(1.0), np.rad2deg(locked)
    assert free > 3 * locked  # the power grasp shears the free cube's layers


def test_lock_needs_full_variant_and_solved_cube():
    with pytest.raises(ValueError):
        cube_xml("single_axis", lock_layers=True)
    with pytest.raises(ValueError):
        HandCubeEnv(EnvConfig(scene=SceneConfig(lock_layers=True), scramble_moves=3))
    env = HandCubeEnv(EnvConfig(scene=SceneConfig(lock_layers=True)))
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"moves": "R"})
    env.close()
