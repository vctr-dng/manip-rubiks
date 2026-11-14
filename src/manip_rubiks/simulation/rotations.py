"""Small rotation helpers (numpy only)."""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = skew(a)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def log_rotation(r: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix, robust near pi."""
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-7:
        return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.pi - angle < 1e-4:
        # axis from the symmetric part: R ~ 2aa^T - I
        sym = (r + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(sym)))
        axis = sym[:, i] / np.sqrt(max(sym[i, i], 1e-12))
        return axis / np.linalg.norm(axis) * angle
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w / (2.0 * np.sin(angle)) * angle


def rotation_angle(r: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


def twist_angle(r: np.ndarray, axis: np.ndarray) -> float:
    """Signed twist of rotation ``r`` about unit ``axis`` (swing-twist decomposition), in (-pi, pi]."""
    q = mat_to_quat(r)
    proj = float(np.dot(q[1:], axis))
    return float(2.0 * np.arctan2(proj, q[0])) if (proj or q[0]) else 0.0


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def mat_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to unit quaternion (w, x, y, z) with w >= 0."""
    m = np.asarray(r, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def chordal_mean(rotations: np.ndarray) -> np.ndarray:
    """Rotation closest (Frobenius) to the arithmetic mean of rotation matrices."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def min_jerk(s: float) -> tuple[float, float]:
    """Normalised minimum-jerk profile value and derivative for s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4
