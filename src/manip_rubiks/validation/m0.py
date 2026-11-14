"""Gate M0 -- simulation foundation (docs/project-draft.md section 33).

Acceptance requirements and how they are checked:

1. Robot loads successfully       -> :func:`check_robot_loads`
2. Actuators behave correctly     -> :func:`check_actuators`
3. Cube rotations preserve legal geometry -> :func:`check_cube_geometry`
4. No unacceptable instability during repeated interactions
                                  -> :func:`check_stability` (+ :func:`check_reproducibility`)

Further simulation checks:

* collision filters, contact-free resets, self-collision and tunnelling
                                  -> :func:`manip_rubiks.validation.collisions.check_collisions`

Run ``uv run manip-rubiks-m0`` to execute every check and write a report.
Thresholds are engineering choices, recorded next to the measured values.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.simulation.assets import MENAGERIE_COMMIT
from manip_rubiks.simulation.cube_interface import CubeInterface, LayerDriver
from manip_rubiks.simulation.cube_model import fixture_cube_xml
from manip_rubiks.simulation.robot_interface import JOINTS
from manip_rubiks.simulation.rotations import min_jerk
from manip_rubiks.simulation.scene_builder import HandCubeScene, SceneConfig
from manip_rubiks.validation.collisions import check_collisions
from manip_rubiks.validation.common import CheckResult
from manip_rubiks.validation.common import finite as _finite
from manip_rubiks.validation.common import warnings_count as _warnings


# --------------------------------------------------------------------------- 1
def check_robot_loads() -> CheckResult:
    """The hand+cube scene compiles with the expected structure and no warnings."""
    scene = HandCubeScene()
    m = scene.model
    scene.reset(seed=0)
    hand_bodies = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("hand/")]
    metrics = {
        "nq": m.nq,
        "nv": m.nv,
        "nu": m.nu,
        "hand_joints": scene.robot.n,
        "hand_bodies": len(hand_bodies),
        "cube_bodies": int(np.sum([m.body(i).name.startswith("cube/") for i in range(m.nbody)])),
        "timestep": m.opt.timestep,
        "warnings_after_reset": _warnings(scene.data),
        "hand_mass_kg": float(m.body_subtreemass[m.body("hand/right_hand_C_MC").id]),
    }
    expected = {"nu": 22, "hand_joints": 22, "hand_bodies": 23, "cube_bodies": 9, "warnings_after_reset": 0}
    passed = all(metrics[k] == v for k, v in expected.items()) and _finite(scene.data)
    return CheckResult("robot_loads", passed, metrics, {"expected": expected})


# --------------------------------------------------------------------------- 2
def check_actuators(step: float = 0.3, duration: float = 0.6, tol: float = 0.05, coupling_tol: float = 0.05):
    """Per-actuator behaviour in a hand-only scene.

    Accuracy tests run from an open hand with gravity and contacts disabled so
    they measure the actuator/joint dynamics rather than self-collisions or
    gravity loading:

    * step response: each actuator's target is stepped by up to ``step`` rad
      towards its farther limit while the others hold; checks direction,
      final tracking error and cross-coupling;
    * limits: each joint alone is commanded beyond both ends of its range;
      targets must be clipped and the joint must settle within its range
      (a small transient overshoot into MuJoCo's soft limit is allowed);
    * actuator forces never exceed the configured force range.

    Reported for information (not pass/fail): steady-state sag under gravity
    in the palm-up home pose (P-only position control), and limit violations
    when all joints are driven to their limits at once (self-collision).
    """
    scene = HandCubeScene(SceneConfig(variant=None))
    m, d, robot = scene.model, scene.data, scene.robot
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    q_open = np.clip(np.zeros(robot.n), robot.q_min, robot.q_max)
    n_steps = round(duration / m.opt.timestep)

    def reset_open():
        mujoco.mj_resetData(m, d)
        robot.reset(q_open)
        mujoco.mj_forward(m, d)

    per_joint = {}
    worst_err = worst_coupling = limit_violation = steady_violation = 0.0
    sign_ok = force_ok = True
    for j, name in enumerate(JOINTS):
        reset_open()
        q0 = robot.q()
        up, down = robot.ctrl_max[j] - q0[j], q0[j] - robot.ctrl_min[j]
        delta = min(step, 0.5 * max(up, down)) * (1.0 if up >= down else -1.0)
        target = robot.target()
        target[j] = q0[j] + delta
        robot.set_target(target)
        settle_time = None
        max_force_ratio = 0.0
        for k in range(n_steps):
            mujoco.mj_step(m, d)
            if settle_time is None and abs(d.qpos[robot.qpos_adr[j]] - target[j]) < tol:
                settle_time = (k + 1) * m.opt.timestep
            max_force_ratio = max(max_force_ratio, float(np.max(np.abs(robot.effort()) / robot.force_limit)))
        q1 = robot.q()
        err = float(abs(q1[j] - target[j]))
        coupling = float(np.delete(np.abs(q1 - q0), j).max())
        moved = float((q1[j] - q0[j]) * np.sign(delta))
        sign_ok &= moved > 0.5 * abs(delta)
        force_ok &= max_force_ratio <= 1.0 + 1e-9
        worst_err, worst_coupling = max(worst_err, err), max(worst_coupling, coupling)

        joint_violation = joint_steady = 0.0
        for end in (robot.ctrl_max[j] + 0.5, robot.ctrl_min[j] - 0.5):
            reset_open()
            target = robot.target()
            target[j] = end
            applied = robot.set_target(target)
            force_ok &= bool(robot.ctrl_min[j] <= applied[j] <= robot.ctrl_max[j])
            for _ in range(round(0.8 / m.opt.timestep)):
                mujoco.mj_step(m, d)
                qj = d.qpos[robot.qpos_adr[j]]
                joint_violation = max(joint_violation, float(max(robot.q_min[j] - qj, qj - robot.q_max[j])))
            qj = d.qpos[robot.qpos_adr[j]]
            joint_steady = max(joint_steady, float(max(robot.q_min[j] - qj, qj - robot.q_max[j], 0.0)))
        limit_violation = max(limit_violation, joint_violation)
        steady_violation = max(steady_violation, joint_steady)
        per_joint[name] = {
            "step_rad": round(float(delta), 4),
            "final_error_rad": round(err, 5),
            "settle_time_s": None if settle_time is None else round(settle_time, 3),
            "max_coupling_rad": round(coupling, 5),
            "max_force_ratio": round(max_force_ratio, 3),
            "limit_violation_rad": round(joint_violation, 5),
        }
    warnings = _warnings(d)

    # Informational: gravity sag at home, and all-joints-at-limit self-collision.
    m.opt.gravity[:] = gravity
    m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    scene.reset()
    mujoco.mj_step(m, d, round(1.0 / m.opt.timestep))
    gravity_sag = np.abs(robot.q() - robot.target())
    self_collision_violation = 0.0
    for end in (robot.ctrl_min, robot.ctrl_max):
        scene.reset()
        robot.set_target(end)
        for _ in range(round(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            q = robot.q()
            self_collision_violation = max(
                self_collision_violation, float(np.max(np.maximum(robot.q_min - q, q - robot.q_max)))
            )

    # Delta-action interface: repeated +1 actions saturate at the ctrl limits.
    scene.reset()
    for _ in range(200):
        robot.apply_action(np.ones(robot.n), k_a=0.05)
    action_saturates = bool(np.allclose(robot.target(), robot.ctrl_max))

    metrics = {
        "worst_final_error_rad": round(worst_err, 5),
        "worst_coupling_rad": round(worst_coupling, 5),
        "all_directions_correct": bool(sign_ok),
        "forces_and_targets_within_limits": bool(force_ok),
        "max_transient_limit_violation_rad": round(limit_violation, 5),
        "max_steady_limit_violation_rad": round(steady_violation, 5),
        "delta_action_saturates_at_limits": action_saturates,
        "warnings": warnings,
        "info_gravity_sag_home_max_rad": round(float(gravity_sag.max()), 4),
        "info_gravity_sag_worst_joint": JOINTS[int(gravity_sag.argmax())],
        "info_all_joints_at_limits_violation_rad": round(self_collision_violation, 4),
        "per_joint": per_joint,
    }
    thresholds = {
        "final_error_rad": tol,
        "coupling_rad": coupling_tol,
        "transient_limit_violation_rad": 0.05,
        "steady_limit_violation_rad": 0.01,
        "step_rad": step,
        "duration_s": duration,
    }
    passed = (
        worst_err < tol
        and worst_coupling < coupling_tol
        and sign_ok
        and force_ok
        and limit_violation < 0.05
        and steady_violation < 0.01
        and action_saturates
        and warnings == 0
    )
    notes = [
        "Accuracy tests use an open hand with gravity and contacts off; gravity sag and self-collision "
        "limit violations are reported as info_* values.",
    ]
    return CheckResult("actuators", passed, metrics, thresholds, notes)


# --------------------------------------------------------------------------- 3
def check_cube_geometry(n_full: int = 200, n_single: int = 50, n_states: int = 50, seed: int = 0) -> CheckResult:
    """Externally driven legal move sequences keep the cube legal and consistent.

    * ``full`` cube on a fixture: ``n_full`` random face turns (quarter and
      half turns, all six faces) executed physically by :class:`LayerDriver`.
      After every move the snapped physical configuration must equal the
      symbolic prediction, every cubie must be within the alignment
      tolerance of a legal pose, and cubies must not interpenetrate.
    * ``single_axis`` cube: ``n_single`` random U/D turns.
    * ``set_state``/``read`` round-trip on ``n_states`` random scrambles.
    """
    rng = np.random.default_rng(seed)
    notes = []
    metrics = {}
    passed = True
    for variant, n_moves, faces in (
        ("full", n_full, ("U", "D", "R", "L", "F", "B")),
        ("single_axis", n_single, ("U", "D")),
    ):
        m = mujoco.MjModel.from_xml_string(fixture_cube_xml(variant))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cube = CubeInterface(m, d, variant)
        driver = LayerDriver(cube)
        symbolic = CubeState.solved()
        moves = CubeState.random_moves(n_moves, rng, faces)
        n_ok = n_agree = 0
        first_divergence = None
        snap_err, pen, swing, ang_err = [], [], [], []
        t0 = time.perf_counter()
        for k, move in enumerate(moves):
            res = driver.execute(move)
            symbolic = symbolic.apply(move)
            agree = res.reading.state.same_configuration(symbolic) and res.reading.state.is_valid()
            n_ok += res.success
            n_agree += agree
            if not agree and first_divergence is None:
                first_divergence = k
            snap_err.append(res.reading.max_error)
            pen.append(res.max_penetration)
            swing.append(res.max_rotor_swing)
            ang_err.append(res.final_angle_error)
        wall = time.perf_counter() - t0
        stats = {
            "moves": n_moves,
            "moves_successful": int(n_ok),
            "states_matching_symbolic": int(n_agree),
            "first_divergence": first_divergence,
            "max_snap_error_deg": round(float(np.rad2deg(max(snap_err))), 3),
            "mean_snap_error_deg": round(float(np.rad2deg(np.mean(snap_err))), 3),
            "max_internal_penetration_mm": round(float(max(pen)) * 1e3, 4),
            "max_rotor_swing_deg": round(float(np.rad2deg(max(swing))), 3),
            "max_hold_angle_error_deg": round(float(np.rad2deg(max(ang_err))), 3),
            "warnings": _warnings(d),
            "finite": _finite(d),
            "sim_time_s": round(float(d.time), 2),
            "wall_time_s": round(wall, 2),
        }
        ok = (
            n_ok == n_moves
            and n_agree == n_moves
            and stats["max_internal_penetration_mm"] < 1.0
            and stats["max_snap_error_deg"] < 3.0
            and stats["warnings"] == 0
            and stats["finite"]
        )
        passed &= ok
        metrics[variant] = stats

    # set_state / read round trip
    m = mujoco.MjModel.from_xml_string(fixture_cube_xml("full"))
    d = mujoco.MjData(m)
    cube = CubeInterface(m, d, "full")
    round_trip_ok = 0
    for _ in range(n_states):
        state = CubeState.solved().apply_sequence(CubeState.random_moves(25, rng))
        cube.set_state(state)
        mujoco.mj_forward(m, d)
        reading = cube.read()
        round_trip_ok += reading.state == state and reading.max_error < 1e-6
    metrics["set_state_round_trip"] = f"{round_trip_ok}/{n_states}"
    passed &= round_trip_ok == n_states
    notes.append(
        "Full cube = 8 box cubies on ball joints about the cube centre; legal motion is enforced by contacts, "
        "not by the kinematic tree. Turns are driven by axial torques on the turning layer while the other "
        "layer is held by fixture welds (stand-in for a gripping hand)."
    )
    thresholds = {"max_snap_error_deg": 3.0, "max_internal_penetration_mm": 1.0, "match_rate": 1.0}
    return CheckResult("cube_geometry", bool(passed), metrics, thresholds, notes)


# --------------------------------------------------------------------------- 4
def random_target_trajectory(scene: HandCubeScene, duration: float, rng: np.random.Generator, amplitude: float = 0.3):
    """Smooth random joint targets around the home pose (sum of sinusoids)."""
    robot = scene.robot
    n = robot.n
    k = 3
    freqs = rng.uniform(0.2, 1.2, (k, n))
    phases = rng.uniform(0, 2 * np.pi, (k, n))
    span = (robot.ctrl_max - robot.ctrl_min) * amplitude / k
    dt = scene.model.opt.timestep
    times = np.arange(round(duration / dt)) * dt
    ramp = 0.5  # fade the offset in so the first target is the home pose
    for t in times:
        offset = np.sum(span * (np.sin(2 * np.pi * freqs * t + phases) - np.sin(phases)), axis=0)
        yield scene.home_q + offset * min_jerk(t / ramp)[0]


def _contact_penetrations(
    scene: HandCubeScene, cube_geoms: np.ndarray, hand_geoms: np.ndarray, pad_geoms: np.ndarray
) -> dict:
    """Deepest current penetration [m] per contact category.

    ``hand_cube`` covers rigid hand links; ``pad_cube`` the compliant
    fingertip elastomer pads.
    """
    d = scene.data
    out = {"hand_cube": 0.0, "pad_cube": 0.0, "cube_internal": 0.0, "hand_hand": 0.0, "environment": 0.0}
    n = d.ncon
    if n == 0:
        return out
    geoms = d.contact.geom[:n]
    pen = np.maximum(0.0, -d.contact.dist[:n])
    in_cube = np.isin(geoms, cube_geoms)
    in_hand = np.isin(geoms, hand_geoms)
    in_pad = np.isin(geoms, pad_geoms).any(axis=1)
    touching = in_cube.any(axis=1) & in_hand.any(axis=1)
    masks = {
        "cube_internal": in_cube.all(axis=1),
        "hand_cube": touching & ~in_pad,
        "pad_cube": touching & in_pad,
        "hand_hand": in_hand.all(axis=1),
    }
    masks["environment"] = ~(masks["cube_internal"] | touching | masks["hand_hand"])
    for k, mask in masks.items():
        if mask.any():
            out[k] = float(pen[mask].max())
    return out


def actuator_speed_bounds(scene: HandCubeScene) -> np.ndarray:
    """Fastest speed each joint's actuator could produce on its own [rad/s].

    ``sqrt(2 * tau_max * range / I)``: the actuator's force limit accelerating
    the joint's inertia (diagonal of the mass matrix at the current pose, which
    includes armature and the distal links) across its whole range. A joint
    moving faster than this has gained energy the actuators cannot supply, which
    signals numerical instability rather than a fast but legitimate motion.
    """
    m, d, r = scene.model, scene.data, scene.robot
    mujoco.mj_forward(m, d)
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    inertia = np.diag(full)[r.dof_adr]
    return np.sqrt(2.0 * r.force_limit * (r.q_max - r.q_min) / inertia)


def rollout(scene: HandCubeScene, seed: int, duration: float) -> dict:
    """Random finger motions with the cube in hand; returns stability metrics.

    Metrics are split by whether the cube is held (touching the hand and not
    yet dropped): a falling cube and its floor impact (a task failure, not an
    instability) are reported separately.
    """
    rng = np.random.default_rng(seed)
    scene.reset(seed=seed, pose_noise=0.004, joint_noise=0.05)
    m, d = scene.model, scene.data
    cube_geoms = scene.cube.geom_ids
    hand_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("hand/")])
    pad_geoms = np.array([g for g in hand_geoms if m.geom(g).name.endswith("fingertip_collision")])
    w0 = _warnings(d)
    speed_bounds = actuator_speed_bounds(scene)
    free = scene.cube.free_dof
    finite = True
    in_hand = dict.fromkeys(
        ("max_joint_speed_rad_s", "max_joint_speed_ratio", "max_cube_speed_m_s", "max_cube_angular_speed_rad_s"), 0.0
    )
    pen_in_hand = dict.fromkeys(("hand_cube", "pad_cube", "cube_internal", "hand_hand", "environment"), 0.0)
    pen_any = dict(pen_in_hand)
    max_speed_any = 0.0
    dropped_at = None
    for target in random_target_trajectory(scene, duration, rng):
        scene.robot.set_target(target)
        mujoco.mj_step(m, d)
        if not _finite(d):
            finite = False
            break
        pos, _ = scene.cube.core_pose()
        if dropped_at is None and pos[2] < scene.cfg.wrist_height - 0.05:
            dropped_at = float(d.time)
        # "Held" = not dropped and currently touching the hand, so a cube that
        # has slipped off and is falling no longer counts as in-hand.
        geoms = d.contact.geom[: d.ncon]
        touching = bool(np.any(np.isin(geoms, cube_geoms).any(axis=1) & np.isin(geoms, hand_geoms).any(axis=1)))
        holding = dropped_at is None and touching
        pens = _contact_penetrations(scene, cube_geoms, hand_geoms, pad_geoms)
        speed = float(np.linalg.norm(d.qvel[free : free + 3]))
        max_speed_any = max(max_speed_any, speed)
        for k, v in pens.items():
            pen_any[k] = max(pen_any[k], v)
        if holding:
            for k, v in pens.items():
                pen_in_hand[k] = max(pen_in_hand[k], v)
            qd = np.abs(scene.robot.qd())
            in_hand["max_joint_speed_rad_s"] = max(in_hand["max_joint_speed_rad_s"], float(qd.max()))
            in_hand["max_joint_speed_ratio"] = max(in_hand["max_joint_speed_ratio"], float((qd / speed_bounds).max()))
            in_hand["max_cube_speed_m_s"] = max(in_hand["max_cube_speed_m_s"], speed)
            in_hand["max_cube_angular_speed_rad_s"] = max(
                in_hand["max_cube_angular_speed_rad_s"], float(np.linalg.norm(d.qvel[free + 3 : free + 6]))
            )
    reading = scene.cube.read()
    out = {
        "finite": finite,
        "warnings": _warnings(d) - w0,
        **in_hand,
        **{f"max_pen_in_hand_{k}_mm": v * 1e3 for k, v in pen_in_hand.items()},
        **{f"max_pen_any_{k}_mm": v * 1e3 for k, v in pen_any.items()},
        "max_cube_speed_any_m_s": max_speed_any,
        "final_state_valid": bool(reading.state.is_valid()),
        "final_max_snap_error_deg": float(np.rad2deg(reading.max_error)),
        "cube_in_hand": dropped_at is None,
        "dropped_at_s": dropped_at,
    }
    return out


def check_stability(n_episodes: int = 10, duration: float = 4.0, long_duration: float = 20.0) -> CheckResult:
    """Repeated hand-cube interaction stays numerically and physically stable.

    Runs ``n_episodes`` short rollouts (each with a fresh reset and random,
    smooth finger motions) and one long rollout. Pass criteria:

    * simulation always finite and free of MuJoCo warnings;
    * while the cube is held: joint speeds below what each actuator could
      produce (:func:`actuator_speed_bounds`), bounded cube speeds, rigid hand-link
      penetration into the cube below 5 mm, cubie interpenetration below
      the cube's mechanical gap scale;
    * at all times (including after a drop): cube speed below 1.5x the
      free-fall speed from the wrist height, cube internal penetration small
      enough that the puzzle stays intact, final cube state legal.

    Dropping the cube is a task failure, not an instability, so it is counted
    but does not fail the check.
    """
    scene = HandCubeScene()
    episodes = [rollout(scene, seed, duration) for seed in range(n_episodes)]
    t0 = time.perf_counter()
    long = rollout(scene, 1000, long_duration)
    wall = time.perf_counter() - t0
    runs = [*episodes, long]

    def worst(key):
        return max(e[key] for e in runs)

    free_fall = float(np.sqrt(2 * 9.81 * scene.cfg.wrist_height))
    limits = {
        "max_joint_speed_ratio": 1.0,
        "max_cube_speed_m_s": 1.0,
        "max_cube_angular_speed_rad_s": 60.0,
        "max_pen_in_hand_hand_cube_mm": 5.0,
        "max_pen_in_hand_cube_internal_mm": 1.0,
        "max_cube_speed_any_m_s": round(1.5 * free_fall, 3),
        "max_pen_any_cube_internal_mm": 5.0,
    }
    metrics = {k: round(worst(k), 4) for k in limits}
    metrics.update(
        {
            "info_max_joint_speed_rad_s": round(worst("max_joint_speed_rad_s"), 3),
            "info_max_pen_in_hand_pad_cube_mm": round(worst("max_pen_in_hand_pad_cube_mm"), 3),
            "info_max_pen_in_hand_hand_hand_mm": round(worst("max_pen_in_hand_hand_hand_mm"), 3),
            "info_max_pen_any_environment_mm": round(worst("max_pen_any_environment_mm"), 3),
            "episodes": n_episodes,
            "episode_duration_s": duration,
            "long_rollout_s": long_duration,
            "long_rollout_realtime_factor": round(long_duration / wall, 2),
            "all_finite": all(e["finite"] for e in runs),
            "total_warnings": int(sum(e["warnings"] for e in runs)),
            "all_final_states_valid": all(e["final_state_valid"] for e in runs),
            "episodes_cube_still_in_hand": int(sum(e["cube_in_hand"] for e in episodes)),
            "long_rollout_cube_in_hand": long["cube_in_hand"],
        }
    )
    passed = (
        metrics["all_finite"]
        and metrics["total_warnings"] == 0
        and metrics["all_final_states_valid"]
        and all(metrics[k] < v for k, v in limits.items())
    )
    notes = [
        "Cube drops are counted but are not instability; floor-impact penetration is reported as info.",
        "Joint speed is judged against the actuator-attainable bound, not a fixed rad/s value: a distal "
        "phalanx (4.7 g) released after being pressed past its limit by another finger snaps back at up "
        "to ~60 rad/s under its force-limited actuator and settles within ~30 ms. That is fast but "
        "physically consistent; the bound is 145 rad/s for DIP joints and 30-73 rad/s for the others.",
        "info_max_pen_in_hand_pad_cube_mm and info_max_pen_in_hand_hand_hand_mm reflect Menagerie's very "
        "compliant fingertip pads (elastomer solref 0.06). Pad indentation of several mm is not an "
        "instability but is unrealistically deep; pad compliance must be calibrated in M1 (tactile).",
    ]
    return CheckResult("stability", bool(passed), metrics, limits, notes)


def check_reproducibility(duration: float = 2.0) -> CheckResult:
    """Same seed -> bitwise identical trajectory; reset after a rollout == fresh reset."""
    a, b = HandCubeScene(), HandCubeScene()
    rollout(a, 7, duration)
    rollout(b, 7, duration)
    same_seed = bool(np.array_equal(a.data.qpos, b.data.qpos) and np.array_equal(a.data.qvel, b.data.qvel))
    rollout(b, 8, duration)
    different_seed_differs = not np.array_equal(a.data.qpos, b.data.qpos)
    # reset equivalence: dirty scene vs fresh scene
    fresh = HandCubeScene()
    fresh.reset(seed=3, pose_noise=0.004, joint_noise=0.05)
    a.reset(seed=3, pose_noise=0.004, joint_noise=0.05)
    reset_equal = bool(
        np.array_equal(a.data.qpos, fresh.data.qpos)
        and np.array_equal(a.data.qvel, fresh.data.qvel)
        and np.array_equal(a.data.ctrl, fresh.data.ctrl)
        and a.data.time == fresh.data.time
    )
    metrics = {
        "same_seed_bitwise_identical": same_seed,
        "different_seed_differs": bool(different_seed_differs),
        "reset_after_rollout_equals_fresh_reset": reset_equal,
    }
    return CheckResult("reproducibility", all(metrics.values()), metrics)


def benchmark_throughput(seconds: float = 2.0) -> dict:
    scene = HandCubeScene()
    scene.reset(seed=0)
    n = round(seconds / scene.model.opt.timestep)
    t0 = time.perf_counter()
    mujoco.mj_step(scene.model, scene.data, n)
    wall = time.perf_counter() - t0
    return {"steps_per_second": round(n / wall), "realtime_factor": round(seconds / wall, 2)}


# ------------------------------------------------------------------- report
def run_all(quick: bool = False) -> dict:
    scale = {"n_full": 40, "n_single": 10, "n_states": 10} if quick else {}
    checks = [
        check_robot_loads(),
        check_actuators(),
        check_cube_geometry(**scale),
        check_stability(**({"n_episodes": 3, "duration": 2.0, "long_duration": 5.0} if quick else {})),
        check_collisions(**({"n_seeds": 5} if quick else {})),
        check_reproducibility(),
    ]
    return {
        "gate": "M0 - simulation foundation",
        "passed": all(c.passed for c in checks),
        "quick": quick,
        "environment": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "throughput": benchmark_throughput(),
        "checks": [asdict(c) for c in checks],
    }


def to_markdown(report: dict) -> str:
    lines = [
        f"# Gate M0 report -- {'PASSED' if report['passed'] else 'FAILED'}",
        "",
        f"Environment: MuJoCo {report['environment']['mujoco']}, NumPy {report['environment']['numpy']}, "
        f"Python {report['environment']['python']}, Menagerie `{report['environment']['menagerie_commit'][:12]}`",
        f"Throughput (hand + cube, CPU, 1 thread): {report['throughput']['steps_per_second']} steps/s "
        f"({report['throughput']['realtime_factor']}x real time)",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for c in report["checks"]:
        lines.append(f"| {c['name']} | {'pass' if c['passed'] else '**FAIL**'} |")
    for c in report["checks"]:
        lines += ["", f"## {c['name']} -- {'pass' if c['passed'] else 'FAIL'}", ""]
        for k, v in c["metrics"].items():
            if k == "per_joint":
                lines += [
                    "",
                    "| joint | step | final err | settle [s] | coupling | force ratio |",
                    "|---|---|---|---|---|---|",
                ]
                for j, s in v.items():
                    lines.append(
                        f"| {j} | {s['step_rad']} | {s['final_error_rad']} | {s['settle_time_s']} | "
                        f"{s['max_coupling_rad']} | {s['max_force_ratio']} |"
                    )
            elif isinstance(v, dict):
                lines.append(f"- **{k}**: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
            else:
                lines.append(f"- {k}: {v}")
        if c["thresholds"]:
            lines.append(f"- thresholds: {c['thresholds']}")
        for n in c["notes"]:
            lines.append(f"- note: {n}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gate M0 simulation-foundation checks.")
    parser.add_argument("--quick", action="store_true", help="smaller sample sizes (for CI / smoke tests)")
    parser.add_argument("--out", type=Path, default=Path("reports/m0"))
    args = parser.parse_args(argv)
    report = run_all(quick=args.quick)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "m0_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "m0_report.md").write_text(to_markdown(report))
    for c in report["checks"]:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
    print(f"Gate M0: {'PASSED' if report['passed'] else 'FAILED'}  (report in {args.out})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
