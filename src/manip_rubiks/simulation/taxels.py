"""Distributed fingertip taxels for tactile configuration C (docs/project-draft.md section 9).

Each fingertip pad gets a non-colliding *skin mesh*: an ``ROWS x COLS`` grid of vertices lying on the
palmar surface of the pad capsule, plus a native MuJoCo ``<tactile>`` sensor on that mesh. The pad
capsule stays the collider, so contact dynamics are identical with and without taxels.

MuJoCo's tactile sensor reports three blocks of ``ROWS * COLS`` values, in this order:

``depth``
    How far the vertex lies inside any other geom touching the fingertip body [m]. Because the
    vertices sit on the unloaded pad surface, this is the indentation of the object into the pad.
``slip_along``, ``slip_across``
    Non-negative relative speed between vertex and the touching body along the two tangent
    directions [m/s]. They are magnitudes, so the sign of the slip is lost.

MuJoCo takes the two tangent directions from the *second and third vertex normals* of the mesh. The
builtin ``plate`` mesh stores the surface normal as its third normal, so its second "tangent"
channel would measure motion along the surface normal. That is why the skin is a custom mesh whose
three per-vertex normals are ``(surface normal, along-finger tangent, across-finger tangent)``.

Grid layout: row ``i`` runs along the finger from the base to the tip, column ``j`` runs across the
pad. In the finger's plate frame, ``x`` is the palmar surface normal, ``y`` points to the fingertip
and ``z = x cross y``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from manip_rubiks.simulation.robot_interface import FINGERTIPS

ROWS = 8  # taxels along the finger
COLS = 8  # taxels across the pad
CHANNELS = ("depth", "slip_along", "slip_across")
ARC_HALF_ANGLE = np.deg2rad(60.0)  # the skin wraps +-60 degrees around the pad capsule
SENSOR_PREFIX = "tactile_"


def sensor_name(prefix: str, finger: str) -> str:
    return f"{prefix}{SENSOR_PREFIX}{finger}"


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.asarray(mat, dtype=float).reshape(-1))
    return out


def skin_vertices(radius: float, half_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertices ``(ROWS*COLS, 3)`` and per-vertex frames ``(ROWS*COLS, 3, 3)`` in the plate frame.

    The frame rows are ``(surface normal, along tangent, across tangent)``. The origin is the pad's
    apex on the palmar side, so the vertex ``x`` coordinate is ``<= 0``.
    """
    along = np.linspace(-half_length, half_length, ROWS)
    phi = np.linspace(-ARC_HALF_ANGLE, ARC_HALF_ANGLE, COLS)
    verts, frames = [], []
    for s in along:
        for p in phi:
            normal = np.array([np.cos(p), 0.0, np.sin(p)])
            verts.append([radius * (np.cos(p) - 1.0), s, radius * np.sin(p)])
            frames.append([normal, [0.0, 1.0, 0.0], np.cross(normal, [0.0, 1.0, 0.0])])
    return np.array(verts), np.array(frames)


def _grid_faces() -> list[int]:
    tri: list[int] = []
    for i in range(ROWS - 1):
        for j in range(COLS - 1):
            a, b = i * COLS + j, i * COLS + j + 1
            c, d = (i + 1) * COLS + j, (i + 1) * COLS + j + 1
            tri += [a, b, d, a, d, c]
    return tri


def pad_frame(pad_quat: np.ndarray) -> np.ndarray:
    """Plate frame (columns) in the distal-phalanx frame for a pad capsule with orientation ``pad_quat``.

    The palmar side of every Sharpa Wave fingertip is ``-y`` of the distal frame, and the fingertip
    lies along ``+x``. The plate normal is the part of ``-y`` perpendicular to the capsule axis.
    """
    axis = _quat_to_mat(pad_quat)[:, 2]
    if axis[0] < 0:
        axis = -axis
    normal = np.array([0.0, -1.0, 0.0])
    normal = normal - normal @ axis * axis
    normal /= np.linalg.norm(normal)
    return np.column_stack([normal, axis, np.cross(normal, axis)])


def add_taxel_sensors(spec: mujoco.MjSpec, prefix: str, side: str = "right") -> None:
    """Add a skin mesh, a non-colliding mesh geom and a tactile sensor to every fingertip of ``spec``."""
    for finger in FINGERTIPS:
        pad = spec.geom(f"{prefix}{side}_{finger}_fingertip_collision")
        radius, half_length = float(pad.size[0]), float(pad.size[1])
        frame = pad_frame(pad.quat)
        verts, frames = skin_vertices(radius, half_length)
        name = f"{prefix}taxel_{finger}"
        mesh = spec.add_mesh(name=name)
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # a surface has no volume
        mesh.uservert = verts.reshape(-1).tolist()
        faces = _grid_faces()
        mesh.userface = faces
        mesh.usernormal = frames.reshape(-1).tolist()
        mesh.userfacenormal = [3 * v for v in faces]
        spec.body(f"{prefix}{side}_{finger}_DP").add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            contype=0,
            conaffinity=0,
            group=4,
            pos=np.asarray(pad.pos, dtype=float) + radius * frame[:, 0],
            quat=_mat_to_quat(frame),
        )
        spec.add_sensor(
            name=sensor_name(prefix, finger),
            type=mujoco.mjtSensor.mjSENS_TACTILE,
            objtype=mujoco.mjtObj.mjOBJ_MESH,
            objname=name,
            reftype=mujoco.mjtObj.mjOBJ_GEOM,
            refname=name,
        )
