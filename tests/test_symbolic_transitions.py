import numpy as np
import pytest

from manip_rubiks.planning.cube_state import (
    FACES,
    ROTATIONS,
    CubeState,
    Move,
    axis_rotation,
    parse_moves,
    snap_rotation,
)


def test_rotation_group_has_24_proper_rotations():
    assert len(ROTATIONS) == 24
    assert len({g.tobytes() for g in ROTATIONS}) == 24
    for g in ROTATIONS:
        assert np.array_equal(g @ g.T, np.eye(3, dtype=int))
        assert round(np.linalg.det(g)) == 1


def test_axis_rotation_right_hand_rule():
    # +90 deg about z maps x to y
    assert np.array_equal(axis_rotation(np.array([0, 0, 1]), 1) @ [1, 0, 0], [0, 1, 0])


def test_snap_rotation_reports_angle():
    theta = np.deg2rad(5.0)
    rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    idx, err = snap_rotation(rz)
    assert np.array_equal(ROTATIONS[idx], np.eye(3, dtype=int))
    assert err == pytest.approx(theta)


@pytest.mark.parametrize("face", FACES)
def test_quarter_turn_has_order_four(face):
    s = CubeState.solved()
    for _ in range(1, 4):
        s = s.apply(Move(face, 1))
        assert not s.is_solved()
    assert s.apply(Move(face, 1)) == CubeState.solved()


@pytest.mark.parametrize("face", FACES)
def test_layer_contains_four_cubies(face):
    assert len(CubeState.solved().layer(face)) == 4


def test_inverse_and_parsing():
    rng = np.random.default_rng(0)
    moves = CubeState.random_moves(30, rng)
    s = CubeState.solved().apply_sequence(moves)
    s = s.apply_sequence([m.inverse() for m in reversed(moves)])
    assert s == CubeState.solved()
    assert [str(m) for m in parse_moves("R U' F2 D")] == ["R", "U'", "F2", "D"]


def test_sexy_move_has_order_six():
    seq = parse_moves("R U R' U'")
    s = CubeState.solved()
    for i in range(6):
        assert (s == CubeState.solved()) == (i == 0)
        s = s.apply_sequence(seq)
    assert s == CubeState.solved()


def test_kociemba_orientation_convention():
    # Reference corner orientations from Kociemba's move tables.
    assert CubeState.solved().apply("U").to_perm_ori()[1] == [0] * 8
    assert CubeState.solved().apply("R").to_perm_ori()[1] == [2, 0, 0, 1, 1, 0, 0, 2]
    assert CubeState.solved().apply("F").to_perm_ori()[1] == [1, 2, 0, 0, 2, 1, 0, 0]
    # Permutations: U cycles URF<-UBR, i.e. slot URF now holds cubie UBR.
    assert CubeState.solved().apply("U").to_perm_ori()[0] == [3, 0, 1, 2, 4, 5, 6, 7]
    assert CubeState.solved().apply("R").to_perm_ori()[0] == [4, 1, 2, 0, 7, 5, 6, 3]


def test_opposite_faces_are_equivalent_up_to_whole_cube_rotation():
    # On a 2x2x2, U and D produce the same relative top/bottom rotation, so
    # they differ only by a whole-cube rotation about z; U and D' do not.
    u = CubeState.solved().apply("U")
    d = CubeState.solved().apply("D")
    assert u != d
    assert u.same_configuration(d)
    assert not u.same_configuration(CubeState.solved().apply("D'"))


def test_random_sequences_stay_valid():
    rng = np.random.default_rng(1)
    s = CubeState.solved()
    for move in CubeState.random_moves(500, rng):
        s = s.apply(move)
        assert s.is_valid()


def test_single_corner_twist_is_detected_as_illegal():
    s = CubeState.solved()
    rot = s.rotations.copy()
    # twist cubie URF about its own diagonal (+x, -y, +z): x -> -y -> z -> x
    twist = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    assert np.array_equal(twist @ np.array([1, -1, 1]), [1, -1, 1])
    rot[0] = twist
    assert not CubeState(rot).is_valid()


def test_swapped_corners_without_twist_is_valid_on_2x2():
    # Corner permutation parity is unconstrained on a 2x2x2 (no edges): a
    # single R turn is a 4-cycle (odd) and is legal.
    s = CubeState.solved().apply("R")
    perm, _ = s.to_perm_ori()
    cycles = 0
    seen = set()
    for i in range(8):
        if i in seen:
            continue
        j = i
        while j not in seen:
            seen.add(j)
            j = perm[j]
        cycles += 1
    assert (8 - cycles) % 2 == 1
    assert s.is_valid()


def test_reachable_state_count_with_fixed_corner():
    # BFS over <U, R, F> (DBL fixed) must reach 3,674,160 states; check the
    # well-known first layers of the distance distribution instead of the
    # full count to keep the test fast.
    moves = [Move(f, k) for f in "URF" for k in (1, 2, 3)]
    frontier = {CubeState.solved()}
    seen = set(frontier)
    sizes = [1]
    for _ in range(4):
        nxt = set()
        for s in frontier:
            for m in moves:
                t = s.apply(m)
                if t not in seen:
                    seen.add(t)
                    nxt.add(t)
        frontier = nxt
        sizes.append(len(nxt))
    assert sizes == [1, 9, 54, 321, 1847]
