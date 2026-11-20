"""A controlled rig for tactile sensor experiments (docs/project-draft.md section 10).

The rig is a hand-only scene with the fingertip taxels enabled and a few small box probes. Probes are
free bodies that the rig drives *kinematically*: every physics step it overwrites their pose and
velocity, so the normal indentation, tangential velocity and contact location are known exactly.
They are real bodies, not mocap bodies, because MuJoCo's tactile sensor measures slip against the
other body's velocity and a mocap body has none.

Positions are given in each fingertip's *pad frame*: ``along`` runs towards the fingertip, ``across``
runs over the pad, and ``depth`` is how far the probe's face lies below the unloaded pad surface.
The pad is a capsule, so its surface is lower away from the centre line; the rig follows it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from manip_rubiks.simulation.robot_interface import FINGERTIPS, RobotInterface
from manip_rubiks.simulation.scene_builder import HAND_PREFIX, SceneConfig, build_spec
from manip_rubiks.simulation.tactile_sensors import TactileConfig, TactileSensors
from manip_rubiks.simulation.taxels import COLS, ROWS, pad_frame, skin_vertices

PARK = np.array([0.5, 0.5, 1.0])  # probes wait here, far from the hand
PROBE_MASS = 0.07  # same as the cube [kg], so the contact's effective mass is representative


@dataclass(frozen=True)
class PadFrame:
    """Pad frame of one fingertip in world coordinates at the rig's rest pose."""

    apex: np.ndarray  # surface point at the pad centre, on the palmar side
    along: np.ndarray
    across: np.ndarray
    normal: np.ndarray  # outward palmar surface normal
    radius: float
    half_length: float

    def surface_height(self, across: float) -> float:
        """Height of the unloaded pad surface above the apex at ``across`` (<= 0)."""
        a = min(abs(across), self.radius)
        return -(self.radius - np.sqrt(self.radius**2 - a**2))


class TactileRig:
    """Hand-only scene with kinematic probes and one sensor object per tactile configuration."""

    def __init__(
        self,
        n_probes: int = 1,
        probe_half: tuple[float, float, float] = (0.003, 0.003, 0.005),
        pad_solref: tuple[float, float] | None = None,
        pad_solimp: tuple[float, float, float, float, float] | None = None,
        timestep: float | None = None,
    ) -> None:
        cfg = SceneConfig(variant=None, tactile_taxels=True, pad_solref=pad_solref, pad_solimp=pad_solimp)
        if timestep is not None:
            cfg.timestep = timestep
        spec = build_spec(cfg)
        self.probe_half = probe_half
        for k in range(n_probes):
            # Gravity-compensated: the rig overwrites the pose every step, so the contact force must
            # come from the commanded penetration and not from supporting the probe's weight.
            body = spec.worldbody.add_body(name=f"probe{k}", pos=PARK, gravcomp=1.0)
            body.add_freejoint(name=f"probe{k}_free")
            body.add_geom(
                name=f"probe{k}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(probe_half),
                mass=PROBE_MASS,
                friction=[1.0, 0.005, 0.0001],
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.robot = RobotInterface(self.model, self.data, cfg.side, prefix=HAND_PREFIX)
        self.contact = TactileSensors(self.model, self.data, cfg.side, HAND_PREFIX, TactileConfig(mode="contact"))
        self.force = TactileSensors(self.model, self.data, cfg.side, HAND_PREFIX, TactileConfig(mode="force"))
        self.taxel = TactileSensors(self.model, self.data, cfg.side, HAND_PREFIX, TactileConfig(mode="taxel"))
        self._probe_q, self._probe_v = [], []
        for k in range(n_probes):
            j = self.model.joint(f"probe{k}_free")
            self._probe_q.append(int(j.qposadr[0]))
            self._probe_v.append(int(j.dofadr[0]))
        self._probe_ids = [self.model.geom(f"probe{k}").id for k in range(n_probes)]
        self.frames: dict[str, PadFrame] = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, settle: float = 0.3) -> None:
        """Open hand at rest with every probe parked; recompute the pad frames."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.robot.reset(np.clip(np.zeros(self.robot.n), self.robot.q_min, self.robot.q_max))
        for k in range(len(self._probe_q)):
            self.park(k)
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, round(settle / self.dt))
        self.frames = {f: self._pad_frame(f) for f in FINGERTIPS}
        for ts in (self.contact, self.force, self.taxel):
            ts.reset(0)

    def lock_hand(self, tol: float = 5e-4) -> None:
        """Clamp every hand joint to +-``tol`` rad around the rest pose, so pads have rigid backing.

        Position actuators are compliant and a finger simply flexes away from a probe pressing on its pad,
        which would hide the pad's own compression. Call after :meth:`reset`.
        """
        m, d = self.model, self.data
        for name, adr in zip(self.robot.joint_names, self.robot.qpos_adr, strict=True):
            j = m.joint(name).id
            m.jnt_limited[j] = 1
            m.jnt_range[j] = [d.qpos[adr] - tol, d.qpos[adr] + tol]

    def _pad_frame(self, finger: str) -> PadFrame:
        m, d = self.model, self.data
        pad = m.geom(f"{HAND_PREFIX}right_{finger}_fingertip_collision")
        r_dp = d.xmat[m.body(f"{HAND_PREFIX}right_{finger}_DP").id].reshape(3, 3)
        quat = m.geom_quat[pad.id]
        frame = r_dp @ pad_frame(quat)  # columns: normal, along, across
        radius, half_length = float(pad.size[0]), float(pad.size[1])
        apex = d.geom_xpos[pad.id] + radius * frame[:, 0]
        return PadFrame(apex, frame[:, 1], frame[:, 2], frame[:, 0], radius, half_length)

    def park(self, probe: int) -> None:
        q, v = self._probe_q[probe], self._probe_v[probe]
        self.data.qpos[q : q + 3] = PARK + probe * np.array([0.1, 0.0, 0.0])
        self.data.qpos[q + 3 : q + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[v : v + 6] = 0.0

    def place(
        self,
        probe: int,
        finger: str,
        along: float,
        across: float,
        depth: float,
        v_along: float = 0.0,
        v_across: float = 0.0,
        v_normal: float = 0.0,
    ) -> None:
        """Pose a probe with its face ``depth`` below the pad surface at (along, across), flat on the pad."""
        fr = self.frames[finger]
        hx, hy, hz = self.probe_half
        rot = np.column_stack([fr.along, fr.across, fr.normal])  # probe x -> along, y -> across, z -> normal
        centre = fr.apex + fr.along * along + fr.across * across + fr.normal * (fr.surface_height(across) - depth + hz)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(-1))
        q, v = self._probe_q[probe], self._probe_v[probe]
        self.data.qpos[q : q + 3] = centre
        self.data.qpos[q + 3 : q + 7] = quat
        self.data.qvel[v : v + 3] = fr.along * v_along + fr.across * v_across + fr.normal * v_normal
        self.data.qvel[v + 3 : v + 6] = 0.0

    # ------------------------------------------------------------- stepping
    def run(self, steps: int, driver: Callable[[int], None] | None = None, record: Callable[[int], None] | None = None):
        """Step the physics, calling ``driver(k)`` before and ``record(k)`` after every step."""
        for k in range(steps):
            if driver is not None:
                driver(k)
            mujoco.mj_step(self.model, self.data)
            if record is not None:
                record(k)

    # --------------------------------------------------------------- readings
    def readings(self) -> dict[str, np.ndarray]:
        """Noiseless readings of the three configurations at the current state."""
        return {"A": self.contact.measure(), "B": self.force.measure(), "C": self.taxel.measure()}

    def taxel_si(self) -> np.ndarray:
        """Taxel readings in SI units: depth [m] and slip speeds [m/s], shape (5, ROWS, COLS, 3)."""
        cfg = self.taxel.cfg
        return self.taxel.measure() * np.array([cfg.max_depth, cfg.max_speed, cfg.max_speed])

    def taxel_positions(self, finger: str) -> np.ndarray:
        """(ROWS, COLS, 2) pad coordinates (along, across) of the taxels of ``finger`` [m]."""
        fr = self.frames[finger]
        verts, _ = skin_vertices(fr.radius, fr.half_length)
        return verts[:, 1:3].reshape(ROWS, COLS, 2)
