"""Physical cube models: structure, state I/O and externally driven legal moves."""

import mujoco
import numpy as np
import pytest

from manip_rubiks.planning.cube_state import CubeState, Move
from manip_rubiks.simulation.cube_interface import CubeInterface, LayerDriver
from manip_rubiks.simulation.cube_model import CubeParams, cube_xml, fixture_cube_xml
from manip_rubiks.simulation.rotations import axis_angle_matrix, log_rotation, mat_to_quat, twist_angle


def make_fixture(variant):
    m = mujoco.MjModel.from_xml_string(fixture_cube_xml(variant))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, CubeInterface(m, d, variant)


@pytest.mark.parametrize(("variant", "nv"), [("full", 30), ("single_axis", 7)])
def test_free_cube_structure(variant, nv):
    m = mujoco.MjModel.from_xml_string(cube_xml(variant))
    assert m.nv == nv
    assert m.body("cube_core").id == 1
    p = CubeParams()
    assert m.body_subtreemass[1] == pytest.approx(p.mass if variant == "full" else p.mass - p.core_mass)
    # free joint must not be damped
    assert np.all(m.dof_damping[:6] == 0)


def test_free_full_cube_falls_as_rigid_body():
    m = mujoco.MjModel.from_xml_string(cube_xml("full"))
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 200)
    cube = CubeInterface(m, d, "full")
    reading = cube.read()
    assert reading.state.is_solved()
    assert reading.max_error < np.deg2rad(0.5)


def test_set_state_round_trip():
    m, d, cube = make_fixture("full")
    rng = np.random.default_rng(3)
    for _ in range(10):
        state = CubeState.solved().apply_sequence(CubeState.random_moves(20, rng))
        cube.set_state(state)
        mujoco.mj_forward(m, d)
        reading = cube.read()
        assert reading.state == state
        assert reading.max_error < 1e-6


def test_read_is_invariant_to_whole_set_rotation():
    # The core can spin relative to all cubies; read() must still recover the configuration.
    m, d, cube = make_fixture("full")
    state = CubeState.solved().apply_sequence("R U' F2")
    tilt = axis_angle_matrix(np.array([1.0, 2.0, 3.0]), np.deg2rad(30))
    cube.set_state(state, frame=tilt)
    mujoco.mj_forward(m, d)
    reading = cube.read()
    assert reading.state.same_configuration(state)
    assert reading.max_error < 1e-6


def test_single_axis_set_state_rejects_unreachable_states():
    m, d, cube = make_fixture("single_axis")
    cube.set_state(CubeState.solved().apply("U"))
    mujoco.mj_forward(m, d)
    assert cube.read().state.same_configuration(CubeState.solved().apply("U"))
    with pytest.raises(ValueError):
        cube.set_state(CubeState.solved().apply("R"))


@pytest.mark.parametrize("token", ["U", "D'", "R2", "L", "F'", "B"])
def test_full_cube_single_move_from_solved(token):
    _, _, cube = make_fixture("full")
    res = LayerDriver(cube).execute(Move.parse(token))
    assert res.success
    assert res.reading.state.same_configuration(CubeState.solved().apply(token))
    assert res.max_penetration < 1e-3
    assert res.warnings == 0


def test_full_cube_random_sequence_matches_symbolic():
    _, d, cube = make_fixture("full")
    driver = LayerDriver(cube)
    rng = np.random.default_rng(11)
    symbolic = CubeState.solved()
    for move in CubeState.random_moves(15, rng):
        res = driver.execute(move)
        symbolic = symbolic.apply(move)
        assert res.success, (str(move), res)
        assert res.reading.state.same_configuration(symbolic)
        assert res.reading.state.is_valid()
        assert res.max_penetration < 1e-3
    assert np.isfinite(d.qpos).all()


def test_single_axis_sequence():
    _, _, cube = make_fixture("single_axis")
    driver = LayerDriver(cube)
    symbolic = CubeState.solved()
    for token in ["U", "D2", "U'", "D", "U2"]:
        res = driver.execute(Move.parse(token))
        symbolic = symbolic.apply(token)
        assert res.success
        assert res.reading.state.same_configuration(symbolic)


def test_rotation_helpers():
    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        r = axis_angle_matrix(axis, angle)
        np.testing.assert_allclose(log_rotation(r), axis * angle, atol=1e-6)
        assert twist_angle(r, axis) == pytest.approx(angle, abs=1e-6)
        q = mat_to_quat(r)
        assert q[0] >= 0 and np.linalg.norm(q) == pytest.approx(1.0)
