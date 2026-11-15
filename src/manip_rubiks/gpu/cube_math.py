"""Batched cube readout in PyTorch, mirroring :meth:`CubeInterface.read` and :meth:`CubeState.same_configuration`.

Symbolic states are represented by ``(..., 8)`` integer tensors of indices
into :data:`~manip_rubiks.planning.cube_state.ROTATIONS`, one per cubie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from manip_rubiks.planning.cube_state import CORNER_NAMES, ROTATIONS, CubeState, snap_rotation

DBL = CORNER_NAMES.index("DBL")


def _group_tables() -> tuple[np.ndarray, np.ndarray]:
    n = len(ROTATIONS)
    mult = np.empty((n, n), dtype=np.int64)
    for a in range(n):
        for b in range(n):
            mult[a, b] = snap_rotation(ROTATIONS[a] @ ROTATIONS[b])[0]
    inv = np.array([snap_rotation(r.T)[0] for r in ROTATIONS], dtype=np.int64)
    return mult, inv


MULT, INV = _group_tables()  # ROTATIONS[MULT[a, b]] = ROTATIONS[a] @ ROTATIONS[b]


def state_to_indices(state: CubeState) -> np.ndarray:
    return np.array([snap_rotation(r)[0] for r in state.rotations], dtype=np.int64)


def indices_to_state(idx: np.ndarray) -> CubeState:
    return CubeState(ROTATIONS[np.asarray(idx)])


@dataclass
class BatchedReading:
    indices: torch.Tensor  # (N, 8) snapped cubie rotations in ``frame``
    frame: torch.Tensor  # (N, 3, 3) symbolic frame relative to the core
    errors: torch.Tensor  # (N, 8) angle between each cubie and its snapped pose [rad]

    @property
    def max_error(self) -> torch.Tensor:
        return self.errors.amax(dim=1)


class CubeMath:
    """Group tables on a device, and the batched readout."""

    def __init__(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> None:
        self.device, self.dtype = torch.device(device), dtype
        self.rotations = torch.as_tensor(ROTATIONS, dtype=dtype, device=self.device)  # (24, 3, 3)
        self.mult = torch.as_tensor(MULT, device=self.device)
        self.inv = torch.as_tensor(INV, device=self.device)

    def snap(self, r: torch.Tensor) -> torch.Tensor:
        """Index of the closest group element for each rotation in ``r`` (..., 3, 3)."""
        traces = torch.einsum("gij,...ij->...g", self.rotations, r)
        return traces.argmax(dim=-1)

    @staticmethod
    def chordal_mean(m: torch.Tensor) -> torch.Tensor:
        """Rotation closest to each ``(N, 3, 3)`` matrix sum (Frobenius)."""
        u, _, vh = torch.linalg.svd(m)
        d = torch.sign(torch.linalg.det(u @ vh))
        u = torch.cat([u[..., :2], u[..., 2:] * d[..., None, None]], dim=-1)
        return u @ vh

    @staticmethod
    def angle(r: torch.Tensor) -> torch.Tensor:
        tr = r.diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))

    def read(self, cubie_rotations: torch.Tensor, iterations: int = 3) -> BatchedReading:
        """Snap ``(N, 8, 3, 3)`` cubie rotations (relative to the core) to symbolic states."""
        rots = cubie_rotations.to(self.dtype)
        frame = torch.eye(3, dtype=self.dtype, device=self.device).expand(rots.shape[0], 3, 3)
        for _ in range(iterations):
            idx = self.snap(frame.transpose(-1, -2)[:, None] @ rots)
            snapped = self.rotations[idx]
            frame = self.chordal_mean((rots @ snapped.transpose(-1, -2)).sum(dim=1))
        idx = self.snap(frame.transpose(-1, -2)[:, None] @ rots)
        posed = frame[:, None] @ self.rotations[idx]
        errors = self.angle(posed.transpose(-1, -2) @ rots)
        return BatchedReading(idx, frame, errors)

    def canonical(self, idx: torch.Tensor) -> torch.Tensor:
        """Whole-cube rotation bringing cubie DBL home (:meth:`CubeState.canonical`)."""
        g = self.inv[idx[..., DBL]]
        return self.mult[g[..., None], idx]

    def same_configuration(self, idx: torch.Tensor, reference_canonical: torch.Tensor) -> torch.Tensor:
        return (self.canonical(idx) == reference_canonical).all(dim=-1)
