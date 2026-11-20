"""Contact-derived tactile readings: layout, physical consistency and corruption."""

import mujoco
import numpy as np
import pytest

from manip_rubiks.simulation.robot_interface import FINGERTIPS
from manip_rubiks.simulation.scene_builder import HAND_PREFIX, HandCubeScene, SceneConfig
from manip_rubiks.simulation.tactile_sensors import TactileConfig, TactileSensors
from manip_rubiks.simulation.taxels import COLS, ROWS


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


# ------------------------------------------------------------- configuration C (taxels)
@pytest.fixture(scope="module")
def taxel_scene():
    scene = HandCubeScene(SceneConfig(tactile_taxels=True))
    scene.reset(seed=0)
    return scene


def test_taxel_layout(taxel_scene):
    t = sensors(taxel_scene, mode="taxel")
    assert t.shape == (5, ROWS, COLS, 3)
    reading = t.measure()
    assert reading.shape == t.shape
    assert reading.min() >= 0.0 and reading.max() <= 1.0
    assert t.measure().size == 960


def test_taxel_skin_lies_on_the_pad_surface(taxel_scene):
    """Every skin vertex sits at the pad capsule radius, so depth is the indentation of the pad."""
    m, d = taxel_scene.model, taxel_scene.data
    mujoco.mj_forward(m, d)
    for f in FINGERTIPS:
        mesh, geom = m.mesh(f"{HAND_PREFIX}taxel_{f}"), m.geom(f"{HAND_PREFIX}taxel_{f}")
        pad = m.geom(f"{HAND_PREFIX}right_{f}_fingertip_collision")
        verts = m.mesh_vert[m.mesh_vertadr[mesh.id] : m.mesh_vertadr[mesh.id] + m.mesh_vertnum[mesh.id]]
        world = verts @ d.geom_xmat[geom.id].reshape(3, 3).T + d.geom_xpos[geom.id]
        rel = world - d.geom_xpos[pad.id]
        axis = d.geom_xmat[pad.id].reshape(3, 3)[:, 2]
        radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
        np.testing.assert_allclose(radial, pad.size[0], atol=1e-6)


def test_taxel_vertex_frames_are_normal_then_tangents(taxel_scene):
    """MuJoCo reads the tangential channels from the 2nd and 3rd vertex normals (see simulation/taxels.py)."""
    m = taxel_scene.model
    mesh = m.mesh(f"{HAND_PREFIX}taxel_index")
    adr = m.mesh_normaladr[mesh.id]
    frames = m.mesh_normal[adr : adr + 3 * m.mesh_vertnum[mesh.id]].reshape(-1, 3, 3)
    np.testing.assert_allclose(frames @ frames.transpose(0, 2, 1), np.broadcast_to(np.eye(3), frames.shape), atol=1e-6)
    np.testing.assert_allclose(frames[:, 1], [[0.0, 1.0, 0.0]] * len(frames), atol=1e-6)  # along the finger


def test_taxel_mode_needs_the_scene_flag(settled):
    with pytest.raises(ValueError, match="tactile_taxels"):
        sensors(settled, mode="taxel")
    with pytest.raises(ValueError, match="fingertip"):
        sensors(HandCubeScene(SceneConfig(tactile_taxels=True)), mode="taxel", coverage="hand")


def test_taxels_do_not_change_the_default_model(settled, taxel_scene):
    base, with_taxels = settled.model, taxel_scene.model
    assert base.nsensor == 0 and with_taxels.nsensor == 5
    assert with_taxels.ngeom == base.ngeom + 5
    assert with_taxels.nq == base.nq and with_taxels.nbody == base.nbody
    for i in range(base.ngeom):  # geom ids shift when geoms are added, so compare by name
        name = base.geom(i).name
        if name:
            assert with_taxels.geom(name).contype == base.geom(i).contype
            assert with_taxels.geom(name).conaffinity == base.geom(i).conaffinity
    assert (with_taxels.geom_contype > 0).sum() == (base.geom_contype > 0).sum()


def test_taxel_corruption(taxel_scene):
    t = sensors(taxel_scene, mode="taxel", noise_std=0.05, dropout=0.0)
    t.reset(seed=1)
    noisy = t.read()
    assert noisy.shape == t.shape and noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.array_equal(noisy, t.measure())
    assert not sensors(taxel_scene, mode="taxel", dropout=1.0).read().any()
    delayed = sensors(taxel_scene, mode="taxel", latency=1)
    assert delayed.read().shape == (5, ROWS, COLS, 3)


def test_pad_compliance_knob():
    pads = lambda m: [m.geom(f"{HAND_PREFIX}right_{f}_fingertip_collision") for f in FINGERTIPS]  # noqa: E731
    default = pads(HandCubeScene(SceneConfig()).model)
    stiff = pads(HandCubeScene(SceneConfig(pad_solref=(0.02, 1.0), pad_solimp=(0.9, 0.95, 0.001, 0.5, 2.0))).model)
    for a, b in zip(default, stiff, strict=True):
        np.testing.assert_allclose(a.solref, [0.06, 0.9])  # Menagerie's elastomer pads stay the default
        np.testing.assert_allclose(b.solref, [0.02, 1.0])
        np.testing.assert_allclose(b.solimp, [0.9, 0.95, 0.001, 0.5, 2.0])
