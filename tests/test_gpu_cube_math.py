"""Batched cube readout must agree with the CPU reference (CubeInterface.read, CubeState.canonical)."""

import numpy as np
import pytest
import torch

from manip_rubiks.gpu.cube_math import MULT, CubeMath, indices_to_state, state_to_indices
from manip_rubiks.planning.cube_state import ROTATIONS, CubeState, snap_rotation
from manip_rubiks.simulation.cube_interface import CubeInterface
from manip_rubiks.simulation.rotations import axis_angle_matrix

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def random_rotation(rng, scale):
    axis = rng.standard_normal(3)
    return axis_angle_matrix(axis / np.linalg.norm(axis), scale * rng.uniform())


def physical_cube(rng, state, noise=0.08):
    """Cubie rotations relative to the core: a random frame, the state, and per-cubie jitter."""
    frame = ROTATIONS[rng.integers(24)] @ random_rotation(rng, 0.3)
    return np.stack([frame @ r @ random_rotation(rng, noise) for r in state.rotations])


def test_group_tables():
    for a in range(24):
        for b in range(24):
            np.testing.assert_array_equal(ROTATIONS[MULT[a, b]], ROTATIONS[a] @ ROTATIONS[b])


@pytest.mark.parametrize("device", DEVICES)
def test_readout_matches_cube_interface(device):
    rng = np.random.default_rng(0)
    math = CubeMath(device, torch.float64)
    reader = CubeInterface.__new__(CubeInterface)  # only ``read`` is used, with patched rotations
    batch, expected = [], []
    for _ in range(64):
        state = CubeState.solved().apply_sequence(CubeState.random_moves(6, rng))
        rots = physical_cube(rng, state)
        reader.cubie_rotations = lambda rots=rots: rots
        expected.append(reader.read())
        batch.append(rots)
    got = math.read(torch.as_tensor(np.stack(batch), device=device))
    for k, ref in enumerate(expected):
        assert indices_to_state(got.indices[k].cpu().numpy()) == ref.state
        np.testing.assert_allclose(got.frame[k].cpu().numpy(), ref.frame, atol=1e-9)
        np.testing.assert_allclose(got.errors[k].cpu().numpy(), ref.errors, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_same_configuration_matches_cube_state(device):
    rng = np.random.default_rng(1)
    math = CubeMath(device)
    for _ in range(50):
        a = CubeState.solved().apply_sequence(CubeState.random_moves(4, rng))
        b = a.rotate_whole(ROTATIONS[rng.integers(24)]) if rng.random() < 0.5 else a.apply("R")
        ia = torch.as_tensor(state_to_indices(a), device=device)
        ib = torch.as_tensor(state_to_indices(b), device=device)
        assert bool(math.same_configuration(ib, math.canonical(ia))) == a.same_configuration(b)


def test_snap_matches_cpu():
    rng = np.random.default_rng(2)
    math = CubeMath("cpu", torch.float64)
    rs = np.stack([random_rotation(rng, np.pi) for _ in range(200)])
    got = math.snap(torch.as_tensor(rs)).numpy()
    assert [snap_rotation(r)[0] for r in rs] == list(got)
