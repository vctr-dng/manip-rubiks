"""Symbolic 2x2x2 cube state.

State is stored the same way the physics model is parameterised: every corner
cubie has a rotation (an element of the 24-element cube rotation group)
relative to the cube frame. A cubie's current position is
``rotation @ home_position``. This makes it trivial to compare a simulated
cube (which gives cubie orientations) against the symbolic model.

Cube frame convention: +x = R, -x = L, +y = B, -y = F, +z = U, -z = D.

A face turn ``X`` is a 90 degree clockwise turn when looking at face ``X`` from
outside the cube, i.e. a rotation of -90 degrees about the outward face normal.
``X'`` is counterclockwise and ``X2`` is a half turn.

The Kociemba-style ``(permutation, orientation)`` view is available through
:meth:`CubeState.to_perm_ori` for use by solvers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# Corner names in Kociemba order, and their home positions (signs of x, y, z).
CORNER_NAMES: tuple[str, ...] = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
HOME_SIGNS = np.array(
    [
        (+1, -1, +1),  # URF
        (-1, -1, +1),  # UFL
        (-1, +1, +1),  # ULB
        (+1, +1, +1),  # UBR
        (+1, -1, -1),  # DFR
        (-1, -1, -1),  # DLF
        (-1, +1, -1),  # DBL
        (+1, +1, -1),  # DRB
    ],
    dtype=int,
)

# Outward normals of the six faces.
FACE_NORMALS: dict[str, np.ndarray] = {
    "R": np.array([1, 0, 0]),
    "L": np.array([-1, 0, 0]),
    "B": np.array([0, 1, 0]),
    "F": np.array([0, -1, 0]),
    "U": np.array([0, 0, 1]),
    "D": np.array([0, 0, -1]),
}
FACES: tuple[str, ...] = tuple(FACE_NORMALS)

# For each corner slot, its three facets in clockwise order (seen from outside),
# starting with the U/D facet. Used for the Kociemba orientation convention.
_SLOT_FACETS: tuple[tuple[str, str, str], ...] = (
    ("U", "R", "F"),
    ("U", "F", "L"),
    ("U", "L", "B"),
    ("U", "B", "R"),
    ("D", "F", "R"),
    ("D", "L", "F"),
    ("D", "B", "L"),
    ("D", "R", "B"),
)


def _rotation_group() -> np.ndarray:
    """All 24 proper rotation matrices with integer entries (signed permutations)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for row, (col, s) in enumerate(zip(perm, signs, strict=True)):
                m[row, col] = s
            if round(np.linalg.det(m)) == 1:
                mats.append(m)
    group = np.array(mats)
    # Put the identity first for convenience.
    idx = next(i for i, g in enumerate(group) if np.array_equal(g, np.eye(3, dtype=int)))
    group[[0, idx]] = group[[idx, 0]]
    return group


ROTATIONS: np.ndarray = _rotation_group()  # (24, 3, 3) int


def axis_rotation(axis: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Integer rotation matrix for ``quarter_turns`` * 90 degrees about a unit axis.

    Positive turns follow the right-hand rule (counterclockwise seen from the tip).
    """
    k = quarter_turns % 4
    a = np.asarray(axis, dtype=int)
    cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=int)
    outer = np.outer(a, a)
    cos, sin = ((1, 0), (0, 1), (-1, 0), (0, -1))[k]
    return cos * np.eye(3, dtype=int) + sin * cross + (1 - cos) * outer


def snap_rotation(r: np.ndarray) -> tuple[int, float]:
    """Return (index into ROTATIONS, angular distance in radians) of the closest group element."""
    traces = np.einsum("gij,ij->g", ROTATIONS, np.asarray(r, dtype=float))
    idx = int(np.argmax(traces))
    cos_angle = np.clip((traces[idx] - 1.0) / 2.0, -1.0, 1.0)
    return idx, float(np.arccos(cos_angle))


@dataclass(frozen=True)
class Move:
    face: str
    quarter_turns: int  # 1 = clockwise, 2 = half turn, 3 = counterclockwise

    def __post_init__(self) -> None:
        if self.face not in FACE_NORMALS:
            raise ValueError(f"unknown face {self.face!r}")
        if self.quarter_turns % 4 == 0:
            raise ValueError("a move must turn the face")
        object.__setattr__(self, "quarter_turns", self.quarter_turns % 4)

    @classmethod
    def parse(cls, token: str) -> Move:
        token = token.strip()
        face, suffix = token[0], token[1:]
        turns = {"": 1, "'": 3, "2": 2, "2'": 2}.get(suffix)
        if turns is None:
            raise ValueError(f"cannot parse move {token!r}")
        return cls(face, turns)

    @property
    def normal(self) -> np.ndarray:
        return FACE_NORMALS[self.face]

    @property
    def signed_angle(self) -> float:
        """Rotation angle about the outward face normal, in (-pi, pi]."""
        return {1: -np.pi / 2, 2: np.pi, 3: np.pi / 2}[self.quarter_turns]

    def rotation(self) -> np.ndarray:
        # clockwise from outside == negative rotation about the outward normal
        return axis_rotation(self.normal, -self.quarter_turns)

    def inverse(self) -> Move:
        return Move(self.face, -self.quarter_turns)

    def __str__(self) -> str:
        return self.face + {1: "", 2: "2", 3: "'"}[self.quarter_turns]


def parse_moves(seq: str) -> list[Move]:
    return [Move.parse(t) for t in seq.split()]


def format_moves(moves: list[Move]) -> str:
    return " ".join(str(m) for m in moves)


class CubeState:
    """Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``CORNER_NAMES[i]``)."""

    __slots__ = ("rotations",)

    def __init__(self, rotations: np.ndarray | None = None) -> None:
        if rotations is None:
            rotations = np.broadcast_to(np.eye(3, dtype=int), (8, 3, 3)).copy()
        rotations = np.asarray(rotations, dtype=int)
        if rotations.shape != (8, 3, 3):
            raise ValueError("rotations must have shape (8, 3, 3)")
        self.rotations = rotations

    # ------------------------------------------------------------------ basics
    @classmethod
    def solved(cls) -> CubeState:
        return cls()

    def copy(self) -> CubeState:
        return CubeState(self.rotations.copy())

    def positions(self) -> np.ndarray:
        """Current position signs (8, 3) of every cubie."""
        return np.einsum("nij,nj->ni", self.rotations, HOME_SIGNS)

    def layer(self, face: str) -> np.ndarray:
        """Indices of the cubies currently in ``face``."""
        return np.flatnonzero(self.positions() @ FACE_NORMALS[face] > 0)

    def apply(self, move: Move | str) -> CubeState:
        if isinstance(move, str):
            move = Move.parse(move)
        out = self.rotations.copy()
        rot = move.rotation()
        for i in self.layer(move.face):
            out[i] = rot @ out[i]
        return CubeState(out)

    def apply_sequence(self, moves: list[Move] | str) -> CubeState:
        if isinstance(moves, str):
            moves = parse_moves(moves)
        state = self
        for m in moves:
            state = state.apply(m)
        return state

    def rotate_whole(self, rotation: np.ndarray) -> CubeState:
        """Rotate the whole cube (does not change the puzzle configuration)."""
        return CubeState(np.einsum("ij,njk->nik", np.asarray(rotation, dtype=int), self.rotations))

    def canonical(self) -> CubeState:
        """Whole-cube rotation that brings cubie DBL to its home pose.

        A 2x2x2 cube has no fixed centres, so two states that differ only by a
        whole-cube rotation are the same puzzle configuration. Comparing
        canonical forms compares configurations.
        """
        dbl = CORNER_NAMES.index("DBL")
        return self.rotate_whole(self.rotations[dbl].T)

    def same_configuration(self, other: CubeState) -> bool:
        return np.array_equal(self.canonical().rotations, other.canonical().rotations)

    def is_solved(self) -> bool:
        return self.same_configuration(CubeState.solved())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CubeState) and np.array_equal(self.rotations, other.rotations)

    def __hash__(self) -> int:
        return hash(self.rotations.tobytes())

    def __repr__(self) -> str:
        perm, ori = self.to_perm_ori()
        return f"CubeState(perm={perm}, ori={ori})"

    # ------------------------------------------------------------ validation
    def is_valid(self) -> bool:
        """Every cubie occupies a distinct slot and the total twist is 0 mod 3."""
        if not all(any(np.array_equal(r, g) for g in ROTATIONS) for r in self.rotations):
            return False
        pos = {tuple(p) for p in self.positions()}
        if len(pos) != 8:
            return False
        _, ori = self.to_perm_ori()
        return sum(ori) % 3 == 0

    def to_perm_ori(self) -> tuple[list[int], list[int]]:
        """Kociemba corner representation.

        ``perm[slot]`` is the cubie in ``slot``; ``ori[slot]`` in {0, 1, 2} is the
        index of the slot facet (clockwise from the U/D facet) that shows the
        cubie's U/D sticker.
        """
        pos = self.positions()
        perm, ori = [], []
        for slot, slot_signs in enumerate(HOME_SIGNS):
            matches = np.flatnonzero((pos == slot_signs).all(axis=1))
            if len(matches) != 1:
                raise ValueError("state does not place exactly one cubie per slot")
            cubie = int(matches[0])
            ud_sticker = self.rotations[cubie] @ np.array([0, 0, HOME_SIGNS[cubie, 2]])
            face = next(f for f, n in FACE_NORMALS.items() if np.array_equal(n, ud_sticker))
            perm.append(cubie)
            ori.append(_SLOT_FACETS[slot].index(face))
        return perm, ori

    # -------------------------------------------------------------- scramble
    @staticmethod
    def random_moves(n: int, rng: np.random.Generator, faces: tuple[str, ...] = FACES) -> list[Move]:
        """Random sequence avoiding consecutive moves on the same axis.

        If all allowed faces share one axis, only consecutive moves of the same
        face are avoided.
        """
        axes = {int(np.argmax(np.abs(FACE_NORMALS[f]))) for f in faces}
        moves: list[Move] = []
        last_face = None
        for _ in range(n):
            while True:
                face = faces[rng.integers(len(faces))]
                if last_face is None or len(faces) == 1:
                    break
                same_axis = bool(np.any(FACE_NORMALS[face] * FACE_NORMALS[last_face]))
                if (not same_axis) if len(axes) > 1 else face != last_face:
                    break
            moves.append(Move(face, int(rng.integers(1, 4))))
            last_face = face
        return moves
