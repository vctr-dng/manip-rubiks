"""Tactile signal sanity checks (contact-derived sensors).

These establish that the sensing pipeline produces physically meaningful
signals. Detailed sensor characterisation (contact location, sliding,
multiple contacts, noise robustness) is separate work.
"""

from __future__ import annotations

import mujoco
import numpy as np

from manip_rubiks.simulation.robot_interface import FINGERTIPS, RobotInterface
from manip_rubiks.simulation.scene_builder import HAND_PREFIX, HandCubeScene, SceneConfig, build_spec
from manip_rubiks.simulation.tactile_sensors import TactileConfig, TactileSensors
from manip_rubiks.validation.common import CheckResult


def _sensors(scene: HandCubeScene, **kw) -> TactileSensors:
    return TactileSensors(scene.model, scene.data, scene.cfg.side, HAND_PREFIX, TactileConfig(**kw))


PROBE_RADIUS = 0.006  # [m]


def probe_fingertips(height: float = 0.01) -> tuple[dict[str, str | None], bool]:
    """Drop a small sphere onto each fingertip pad of the open hand.

    Returns the first fingertip region that registered contact for each
    target finger, and whether every pad reads zero after the probe is removed.
    """
    spec = build_spec(SceneConfig(variant=None))
    probe = spec.worldbody.add_body(name="probe", pos=[0.5, 0.5, 0.1])
    probe.add_freejoint(name="probe_free")
    probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PROBE_RADIUS, 0, 0], mass=0.02)
    m = spec.compile()
    d = mujoco.MjData(m)
    robot = RobotInterface(m, d, "right", prefix=HAND_PREFIX)
    tips = TactileSensors(m, d, "right", HAND_PREFIX, TactileConfig(mode="contact"))
    free = m.joint("probe_free")
    qadr, vadr = int(free.qposadr[0]), int(free.dofadr[0])
    first: dict[str, str | None] = {}
    released = True
    for k, finger in enumerate(FINGERTIPS):
        mujoco.mj_resetData(m, d)
        robot.reset(np.clip(np.zeros(robot.n), robot.q_min, robot.q_max))
        mujoco.mj_forward(m, d)
        site = robot.fingertip_sites[k]
        d.qpos[qadr : qadr + 3] = d.site_xpos[site] + np.array([0.0, 0.0, height + PROBE_RADIUS])  # above the tip
        d.qvel[vadr : vadr + 6] = 0.0
        first[finger] = None
        for _ in range(300):
            mujoco.mj_step(m, d)
            reading = tips.measure()
            if reading.any():
                first[finger] = tips.names[int(np.flatnonzero(reading)[0])]
                break
        d.qpos[qadr : qadr + 3] = [0.5, 0.5, 0.1]
        d.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_step(m, d, 10)
        released &= not tips.measure().any()
    return first, bool(released)


def check_tactile_signals() -> CheckResult:
    # 1. Whole-hand coverage includes every hand collision geom exactly once.
    scene = HandCubeScene()
    m = scene.model
    hand = _sensors(scene, coverage="hand")
    covered = np.concatenate([r.geoms for r in hand.regions])
    in_hand = np.array([m.body(b).name.startswith(HAND_PREFIX) for b in m.geom_bodyid])
    colliding = np.flatnonzero(in_hand & ((m.geom_contype > 0) | (m.geom_conaffinity > 0)))
    coverage_complete = len(covered) == len(np.unique(covered)) and set(covered) == set(colliding)

    # 2. An open hand touching nothing reads zero everywhere.
    bare = HandCubeScene(SceneConfig(variant=None))
    bare.reset()
    bare.robot.reset(np.clip(np.zeros(bare.robot.n), bare.robot.q_min, bare.robot.q_max))
    silent = True
    contact = _sensors(bare, mode="contact", coverage="hand")
    for _ in range(200):
        mujoco.mj_step(bare.model, bare.data)
        silent &= not contact.measure().any()

    # 3. With the cube resting in the hand, hand-hand forces cancel and the net
    #    force on the whole hand equals the cube's weight.
    scene.reset(seed=0)
    scene.step(500)
    weight = scene.cfg.cube_params.mass * float(np.linalg.norm(m.opt.gravity))
    net = hand.region_forces().sum(axis=0)
    balance_error = float(np.linalg.norm(net - np.array([0.0, 0.0, -weight])) / weight)

    # 4. Localisation: a small probe dropped onto each fingertip of the open
    #    hand lights up that pad first, and the reading returns to zero once
    #    the probe is taken away.
    localised, released = probe_fingertips()

    metrics = {
        "hand_regions": len(hand.regions),
        "coverage_complete": bool(coverage_complete),
        "open_hand_silent": bool(silent),
        "resting_force_balance_rel_error": round(balance_error, 4),
        "first_region_touched_per_target": localised,
        "pads_silent_after_release": bool(released),
    }
    limits = {"resting_force_balance_rel_error": 0.02}
    passed = (
        coverage_complete
        and silent
        and balance_error < limits["resting_force_balance_rel_error"]
        and all(f"{f}_tip" == hit for f, hit in localised.items())
        and released
    )
    notes = ["Readings are noiseless (TactileSensors.measure); noise, dropout and latency are unit-tested."]
    return CheckResult("tactile_signals", bool(passed), metrics, limits, notes)
