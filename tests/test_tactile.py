"""Contact-derived tactile readings: layout, physical consistency and corruption."""

import mujoco
import numpy as np
import pytest

from manip_rubiks.simulation.robot_interface import FINGERTIPS
from manip_rubiks.simulation.scene_builder import HAND_PREFIX, HandCubeScene, SceneConfig
from manip_rubiks.simulation.tactile_sensors import TactileConfig, TactileSensors


@pytest.fixture(scope="module")
def settled():
    scene = HandCubeScene()
    scene.reset(seed=0)
    scene.step(500)
    return scene


def sensors(scene, **kw):
    return TactileSensors(scene.model, scene.data, prefix=HAND_PREFIX, config=TactileConfig(**kw))


def test_fingertip_layout(settled):
    t = sensors(settled)
    assert t.names == tuple(f"{f}_tip" for f in FINGERTIPS)
    assert t.measure().shape == t.shape == (5, 3)
    assert sensors(settled, mode="contact").measure().shape == (5,)


def test_hand_coverage_includes_every_hand_collision_geom(settled):
    m = settled.model
    t = sensors(settled, coverage="hand")
    covered = np.concatenate([r.geoms for r in t.regions])
    assert len(covered) == len(np.unique(covered))
    hand_bodies = [b for b in range(m.nbody) if m.body(b).name.startswith(HAND_PREFIX)]
    colliding = np.flatnonzero(np.isin(m.geom_bodyid, hand_bodies) & ((m.geom_contype > 0) | (m.geom_conaffinity > 0)))
    assert set(covered) == set(colliding)


def test_whole_hand_force_balances_cube_weight(settled):
    """Hand-hand contacts cancel, so the net force on the hand is the cube's weight."""
    t = sensors(settled, coverage="hand")
    total = t.region_forces().sum(axis=0)
    weight = settled.cfg.cube_params.mass * 9.81
    np.testing.assert_allclose(total, [0.0, 0.0, -weight], atol=0.02 * weight)


def test_local_frame_preserves_magnitude(settled):
    t = sensors(settled, coverage="hand")
    np.testing.assert_allclose(np.linalg.norm(t.measure(), axis=1), np.linalg.norm(t.region_forces(), axis=1))


def test_open_hand_reads_zero():
    scene = HandCubeScene(SceneConfig(variant=None))
    scene.reset()
    scene.robot.reset(np.clip(np.zeros(scene.robot.n), scene.robot.q_min, scene.robot.q_max))
    mujoco.mj_step(scene.model, scene.data, 200)
    assert not sensors(scene, mode="contact", coverage="hand").measure().any()


def test_contact_appears_when_cube_is_pressed_onto_a_fingertip():
    scene = HandCubeScene()
    scene.reset(seed=0)
    t = sensors(scene, mode="contact")
    middle = FINGERTIPS.index("middle")
    assert t.measure()[middle] == 0.0
    tip = scene.robot.fingertip_positions()[middle]
    quat = scene.data.qpos[scene.cube.free_qpos + 3 : scene.cube.free_qpos + 7].copy()
    # Lift the cube off the palm and drop it onto the middle fingertip.
    scene.cube.set_pose(tip + np.array([0.0, 0.0, 0.03]), quat)
    hits = 0
    for _ in range(100):
        scene.step(2)
        hits += t.measure()[middle]
    assert hits > 0


def test_latency(settled, monkeypatch):
    t = sensors(settled, latency=2)
    t.reset(seed=0)
    counter = iter(range(100))
    monkeypatch.setattr(t, "measure", lambda: np.full(t.shape, float(next(counter))))
    # Until the buffer fills, the oldest available reading is returned.
    assert [t.read()[0, 0] for _ in range(5)] == [0.0, 0.0, 0.0, 1.0, 2.0]
    t.reset(seed=0)
    assert t.read()[0, 0] == 5.0


def test_dropout_and_noise(settled):
    assert not sensors(settled, dropout=1.0).read().any()

    a, b = sensors(settled, noise_std=0.1), sensors(settled, noise_std=0.1)
    a.reset(seed=3)
    b.reset(seed=3)
    ra, rb = a.read(), b.read()
    np.testing.assert_array_equal(ra, rb)
    assert not np.array_equal(ra, a.measure())
