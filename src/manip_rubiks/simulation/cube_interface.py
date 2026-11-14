"""Ground-truth access to the simulated cube.

Everything here reads privileged simulator state. It is meant for resets,
rewards, evaluation and external test drivers -- never as an input to a
blindfolded policy (docs/project-draft.md section 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from manip_rubiks.planning.cube_state import CORNER_NAMES, FACE_NORMALS, ROTATIONS, CubeState, Move, snap_rotation
from manip_rubiks.simulation.cube_model import Variant
from manip_rubiks.simulation.rotations import (
    axis_angle_matrix,
    chordal_mean,
    mat_to_quat,
    min_jerk,
    rotation_angle,
    twist_angle,
    wrap_angle,
)


@dataclass
class CubeReading:
    """Snapped configuration of the physical cube."""

    state: CubeState  # cubie rotations expressed in ``frame``
    frame: np.ndarray  # rotation of the symbolic cube frame relative to the core
    errors: np.ndarray  # (8,) angle between each cubie and its snapped pose [rad]

    @property
    def max_error(self) -> float:
        return float(self.errors.max())


class CubeInterface:
    """Reads and writes the physical state of a cube built by :mod:`cube_model`."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, variant: Variant, prefix: str = "") -> None:
        self.model, self.data, self.variant, self.prefix = model, data, variant, prefix
        self.core_body = model.body(prefix + "cube_core").id
        if variant == "full":
            self.cubie_bodies = np.array([model.body(f"{prefix}cubie_{n}").id for n in CORNER_NAMES])
            joints = [model.joint(f"{prefix}cubie_{n}_ball") for n in CORNER_NAMES]
            self.ball_qpos = np.array([j.qposadr[0] for j in joints])
            self.ball_dof = np.array([j.dofadr[0] for j in joints])
        elif variant == "single_axis":
            self.layer_body = model.body(prefix + "cube_layer_U").id
            hinge = model.joint(prefix + "cube_U_hinge")
            self.hinge_qpos, self.hinge_dof = int(hinge.qposadr[0]), int(hinge.dofadr[0])
        else:
            raise ValueError(variant)
        free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "cube_free")
        self.free_qpos = int(model.jnt_qposadr[free_id]) if free_id >= 0 else None
        self.free_dof = int(model.jnt_dofadr[free_id]) if free_id >= 0 else None
        hold = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{prefix}hold_{n}") for n in CORNER_NAMES]
        self.hold_eq = np.array(hold) if variant == "full" and min(hold) >= 0 else None
        self.geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, self._cube_body_ids()) & (model.geom_contype > 0))

    # ----------------------------------------------------------------- frames
    def _cube_body_ids(self) -> np.ndarray:
        if self.variant == "full":
            return np.concatenate([[self.core_body], self.cubie_bodies])
        return np.array([self.core_body, self.layer_body])

    def core_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.core_body].copy(), self.data.xmat[self.core_body].reshape(3, 3).copy()

    def cubie_rotations(self) -> np.ndarray:
        """(8, 3, 3) cubie rotations relative to the core body."""
        _, r_core = self.core_pose()
        if self.variant == "full":
            r_world = self.data.xmat[self.cubie_bodies].reshape(8, 3, 3)
            return np.einsum("ji,njk->nik", r_core, r_world)
        out = np.broadcast_to(np.eye(3), (8, 3, 3)).copy()
        angle = self.data.qpos[self.hinge_qpos]
        for i, n in enumerate(CORNER_NAMES):
            if n.startswith("U"):
                out[i] = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), angle)
        return out

    # ---------------------------------------------------------------- reading
    def read(self) -> CubeReading:
        """Snap the physical cube to the nearest symbolic configuration.

        The internal core can rotate freely relative to the set of cubies (a
        whole-cube motion), so the symbolic frame is fitted to the cubies
        rather than taken from the core.
        """
        rots = self.cubie_rotations()
        frame = np.eye(3)
        for _ in range(3):
            snapped = np.array([ROTATIONS[snap_rotation(frame.T @ r)[0]] for r in rots])
            frame = chordal_mean(np.einsum("nij,nkj->nik", rots, snapped.astype(float)))
        snapped = np.array([ROTATIONS[snap_rotation(frame.T @ r)[0]] for r in rots])
        errors = np.array([rotation_angle((frame @ s).T @ r) for s, r in zip(snapped, rots, strict=True)])
        return CubeReading(CubeState(snapped), frame, errors)

    # ---------------------------------------------------------------- writing
    def set_state(self, state: CubeState, frame: np.ndarray | None = None) -> None:
        """Write a symbolic configuration into ``qpos`` (call ``mj_forward`` afterwards)."""
        frame = np.eye(3) if frame is None else frame
        if self.variant == "full":
            for adr, r in zip(self.ball_qpos, state.rotations, strict=True):
                self.data.qpos[adr : adr + 4] = mat_to_quat(frame @ r)
            self.data.qvel[self.ball_dof[0] : self.ball_dof[-1] + 3] = 0.0
            return
        canonical = state.canonical()
        lower = [i for i, n in enumerate(CORNER_NAMES) if n.startswith("D")]
        upper = [i for i, n in enumerate(CORNER_NAMES) if n.startswith("U")]
        if not all(np.array_equal(canonical.rotations[i], np.eye(3)) for i in lower):
            raise ValueError("single_axis cube can only represent U-layer turns from solved")
        r_u = canonical.rotations[upper[0]]
        if not all(np.array_equal(canonical.rotations[i], r_u) for i in upper):
            raise ValueError("U-layer cubies must share one rotation")
        self.data.qpos[self.hinge_qpos] = twist_angle(frame @ r_u, np.array([0.0, 0.0, 1.0]))
        self.data.qvel[self.hinge_dof] = 0.0

    def set_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        if self.free_qpos is None:
            raise RuntimeError("cube root is fixed")
        self.data.qpos[self.free_qpos : self.free_qpos + 3] = pos
        self.data.qpos[self.free_qpos + 3 : self.free_qpos + 7] = quat
        self.data.qvel[self.free_dof : self.free_dof + 6] = 0.0

    def hold(self, index: int, rotation_rel_core: np.ndarray) -> None:
        """Activate the fixture weld holding cubie ``index`` at a rotation relative to the core."""
        eq = int(self.hold_eq[index])
        self.model.eq_data[eq, :6] = 0.0  # anchor and relative position (both frames at the centre)
        self.model.eq_data[eq, 6:10] = mat_to_quat(np.asarray(rotation_rel_core).T)  # core pose in cubie frame
        self.model.eq_data[eq, 10] = 1.0  # torquescale
        self.data.eq_active[eq] = 1

    def release(self, index: int) -> None:
        self.data.eq_active[int(self.hold_eq[index])] = 0

    # ------------------------------------------------------------- diagnostics
    def max_penetration(self, internal_only: bool = False) -> float:
        """Largest penetration depth [m] among active contacts involving the cube."""
        n = self.data.ncon
        if n == 0:
            return 0.0
        geoms = self.data.contact.geom[:n]
        in_cube = np.isin(geoms, self.geom_ids)
        mask = in_cube.all(axis=1) if internal_only else in_cube.any(axis=1)
        if not mask.any():
            return 0.0
        return float(max(0.0, -self.data.contact.dist[:n][mask].min()))

    def cubie_angular_velocities(self) -> np.ndarray:
        """(8, 3) world-frame angular velocity of each cubie relative to the core (full variant)."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.core_body, res, 0)
        w_core = res[:3].copy()
        out = np.zeros((8, 3))
        for k, b in enumerate(self.cubie_bodies):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(b), res, 0)
            out[k] = res[:3] - w_core
        return out


@dataclass
class MoveResult:
    move: Move
    success: bool
    reading: CubeReading
    final_angle_error: float  # |achieved - commanded| layer angle at the end of the hold phase [rad]
    max_penetration: float  # deepest cubie-cubie penetration during the move [m]
    max_stator_deviation: float  # largest rotation of a non-turning cubie [rad]
    max_rotor_swing: float  # largest off-axis rotation of a turning cubie [rad]
    warnings: int
    details: dict = field(default_factory=dict)


@dataclass
class LayerDriverConfig:
    align_duration: float = 0.1  # squaring phase before the turn [s]
    turn_duration: float = 0.35  # per quarter turn [s]
    hold_duration: float = 0.15
    settle_duration: float = 0.15
    # Torques are applied explicitly, so gains must respect the explicit
    # stability limits for cubie inertias of ~3e-6 kg m^2 at dt = 2 ms.
    kp: float = 0.08  # layer angle PD [Nm/rad]
    kd: float = 1.2e-3
    success_tolerance: float = np.deg2rad(3.0)


class LayerDriver:
    """Executes face turns by applying external torques to the cubies.

    This stands in for a perfect manipulator so the cube mechanics can be
    validated independently of the hand: turning cubies receive an axial
    torque from a PD controller tracking a minimum-jerk angle profile, while
    the remaining cubies are held in place by orientation PD torques, like a
    hand gripping the opposite layer. Torques are applied through
    ``xfrc_applied`` so all motion is mediated by the contact model.
    """

    def __init__(self, cube: CubeInterface, config: LayerDriverConfig | None = None) -> None:
        self.cube = cube
        self.cfg = config or LayerDriverConfig()

    def execute(self, move: Move) -> MoveResult:
        if self.cube.variant == "single_axis":
            return self._execute_hinge(move)
        return self._execute_full(move)

    # ----------------------------------------------------------------- full
    def _execute_full(self, move: Move) -> MoveResult:
        cube, m, d, cfg = self.cube, self.cube.model, self.cube.data, self.cfg
        if cube.hold_eq is None:
            raise RuntimeError("full-cube driver needs a cube built with fixture_welds=True")
        start = cube.read()
        rotor = start.state.layer(move.face)
        stator = np.setdiff1d(np.arange(8), rotor)
        targets0 = np.einsum("ij,njk->nik", start.frame, start.state.rotations.astype(float))  # rel. core
        axis_core = start.frame @ FACE_NORMALS[move.face].astype(float)
        total = move.signed_angle
        turn_time = cfg.turn_duration * (2 if move.quarter_turns == 2 else 1)
        dt = m.opt.timestep
        n_align = round(cfg.align_duration / dt)
        n_turn = round(turn_time / dt)
        n_hold = round(cfg.hold_duration / dt)
        n_settle = round(cfg.settle_duration / dt)

        phi_acc = np.zeros(len(rotor))
        max_pen = max_stator = max_swing = 0.0
        warn0 = sum(w.number for w in d.warning)
        bodies = cube.cubie_bodies
        final_err = np.nan

        # 1. Square the cube: weld every cubie onto its snapped pose, as a hand
        #    does before turning. Removes slack left by gravity and the gap.
        for i in range(8):
            cube.hold(i, targets0[i])
        for _ in range(n_align):
            mujoco.mj_step(m, d)
            max_pen = max(max_pen, cube.max_penetration(internal_only=True))
        # 2. Turn: release the rotor, keep the stator welded, drive the rotor
        #    with an axial torque only (alignment comes from the contacts).
        for i in rotor:
            cube.release(i)
        for step in range(n_turn + n_hold + n_settle):
            _, r_core = cube.core_pose()
            axis_world = r_core @ axis_core
            rots = cube.cubie_rotations()
            omegas = cube.cubie_angular_velocities()
            for k, i in enumerate(rotor):
                tw = twist_angle(rots[i] @ targets0[i].T, axis_core)
                phi_acc[k] += wrap_angle(tw - wrap_angle(phi_acc[k]))
            phi = float(phi_acc.mean())
            phidot = float(np.mean(omegas[rotor] @ axis_world))
            d.xfrc_applied[bodies, 3:] = 0.0
            if step < n_turn + n_hold:
                s, sdot = min_jerk(step / n_turn) if step < n_turn else (1.0, 0.0)
                tau = cfg.kp * (total * s - phi) + cfg.kd * (total * sdot / turn_time - phidot)
                d.xfrc_applied[bodies[rotor], 3:] = axis_world * (tau / len(rotor))
            if step == n_turn + n_hold - 1:
                final_err = abs(phi - total)
                # 3. Settle: release everything and let the cube rest passively.
                for i in stator:
                    cube.release(i)
            for k, i in enumerate(rotor):
                expected = axis_angle_matrix(axis_core, phi_acc[k]) @ targets0[i]
                max_swing = max(max_swing, rotation_angle(expected.T @ rots[i]))
            for i in stator:
                max_stator = max(max_stator, rotation_angle(targets0[i].T @ rots[i]))
            mujoco.mj_step(m, d)
            max_pen = max(max_pen, cube.max_penetration(internal_only=True))
        d.xfrc_applied[bodies, 3:] = 0.0
        mujoco.mj_forward(m, d)
        reading = cube.read()
        ok = reading.state.same_configuration(start.state.apply(move)) and reading.max_error < cfg.success_tolerance
        warnings = sum(w.number for w in d.warning) - warn0
        return MoveResult(
            move, bool(ok and warnings == 0), reading, float(final_err), max_pen, max_stator, max_swing, warnings
        )

    # --------------------------------------------------------------- hinge
    def _execute_hinge(self, move: Move) -> MoveResult:
        cube, m, d, cfg = self.cube, self.cube.model, self.cube.data, self.cfg
        if move.face not in ("U", "D"):
            raise ValueError("single_axis cube only supports U and D turns")
        start = cube.read()
        # A D turn (about -z) with the base fixed is equivalent to the U layer
        # turning by the same signed angle about +z.
        total = move.signed_angle
        q0 = float(d.qpos[cube.hinge_qpos])
        turn_time = cfg.turn_duration * (2 if move.quarter_turns == 2 else 1)
        dt = m.opt.timestep
        n_turn, n_hold, n_settle = (round(x / dt) for x in (turn_time, cfg.hold_duration, cfg.settle_duration))
        warn0 = sum(w.number for w in d.warning)
        final_err = np.nan
        for step in range(n_turn + n_hold + n_settle):
            d.qfrc_applied[cube.hinge_dof] = 0.0
            if step < n_turn + n_hold:
                s, sdot = min_jerk(step / n_turn) if step < n_turn else (1.0, 0.0)
                phi = d.qpos[cube.hinge_qpos] - q0
                tau = cfg.kp * (total * s - phi) + cfg.kd * (total * sdot / turn_time - d.qvel[cube.hinge_dof])
                d.qfrc_applied[cube.hinge_dof] = tau
            if step == n_turn + n_hold - 1:
                final_err = abs(d.qpos[cube.hinge_qpos] - q0 - total)
            mujoco.mj_step(m, d)
        d.qfrc_applied[cube.hinge_dof] = 0.0
        mujoco.mj_forward(m, d)
        reading = cube.read()
        ok = reading.state.same_configuration(start.state.apply(move)) and reading.max_error < cfg.success_tolerance
        warnings = sum(w.number for w in d.warning) - warn0
        return MoveResult(
            move,
            bool(ok and warnings == 0),
            reading,
            float(final_err),
            cube.max_penetration(internal_only=True),
            0.0,
            0.0,
            warnings,
        )
