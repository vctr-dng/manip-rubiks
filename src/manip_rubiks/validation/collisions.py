"""Collision audit for the hand + cube scene.

Checks, in order of increasing dynamics:

1. **Filters** -- every hand collision geom can touch every cubie, the cube
   can touch the floor, stickers are visual only, and the distal phalanges of
   different fingers can collide with each other (self-collision is on).
2. **Reset clearance** -- across noisy resets the cube spawns free of the
   hand and the hand free of itself, so no episode starts with a contact
   impulse.
3. **Settled home pose** -- the cube rests on the hand, not the floor, with
   no rigid hand self-penetration.
4. **Pinch** -- thumb and index flexed into each other at full force: rigid
   links must not pass through each other.
5. **Press** -- the resting cube pushed into the palm with 20 N (a strong
   finger push): no tunnelling into rigid links.
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from manip_rubiks.simulation.robot_interface import FINGERTIPS, JOINTS
from manip_rubiks.simulation.scene_builder import HAND_PREFIX, HandCubeScene, SceneConfig, is_pad
from manip_rubiks.validation.common import CheckResult, finite, warnings_count

PINCH_JOINTS = ("thumb_CMC_FE", "thumb_MCP_FE", "thumb_IP", "index_MCP_FE", "index_PIP", "index_DIP")


class _Geoms:
    """Collision geom groups of a compiled scene."""

    def __init__(self, scene: HandCubeScene) -> None:
        m = self.model = scene.model
        colliding = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        in_hand = np.array([m.body(b).name.startswith(HAND_PREFIX) for b in m.geom_bodyid])
        self.hand = np.flatnonzero(colliding & in_hand)
        self.pads = np.array([g for g in self.hand if is_pad(m.geom(g).name)])
        self.rigid = np.setdiff1d(self.hand, self.pads)
        self.cube = scene.cube.geom_ids if scene.cube is not None else np.array([], int)
        self.excluded = set(m.exclude_signature.tolist())

    def body_pair_excluded(self, a: int, b: int) -> bool:
        m = self.model
        ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
        if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:  # MuJoCo skips parent-child
            return True
        return ((min(ba, bb) << 16) | max(ba, bb)) in self.excluded

    def can_collide(self, a: int, b: int) -> bool:
        m = self.model
        compatible = (m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])
        return bool(compatible) and not self.body_pair_excluded(a, b)

    def self_pairs(self) -> list[tuple[int, int]]:
        """Hand geom pairs that the collision filter allows to touch."""
        return [(a, b) for a, b in itertools.combinations(self.hand, 2) if self.can_collide(a, b)]


def _min_distance(m: mujoco.MjModel, d: mujoco.MjData, pairs, distmax: float = 0.05) -> float:
    fromto = np.zeros(6)
    return min((mujoco.mj_geomDistance(m, d, int(a), int(b), distmax, fromto) for a, b in pairs), default=distmax)


def _max_pen(d: mujoco.MjData, group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Deepest current penetration [m] between a geom of ``group_a`` and one of ``group_b``."""
    n = d.ncon
    if n == 0:
        return 0.0
    g = d.contact.geom[:n]
    mask = (np.isin(g[:, 0], group_a) & np.isin(g[:, 1], group_b)) | (
        np.isin(g[:, 0], group_b) & np.isin(g[:, 1], group_a)
    )
    return float(max(0.0, -d.contact.dist[:n][mask].min())) if mask.any() else 0.0


def audit_filters(scene: HandCubeScene) -> dict:
    m, geoms = scene.model, _Geoms(scene)
    floor = m.geom("floor").id
    hand_cube = [geoms.can_collide(h, c) for h in geoms.hand for c in geoms.cube]
    stickers = [g for g in range(m.ngeom) if m.geom(g).name.startswith(scene.cube.prefix + "sticker_")]
    distal = {f: [g for g in geoms.hand if m.body(m.geom_bodyid[g]).name.endswith(f"_{f}_DP")] for f in FINGERTIPS}
    finger_pairs = [
        geoms.can_collide(a, b)
        for f1, f2 in itertools.combinations(FINGERTIPS, 2)
        for a in distal[f1]
        for b in distal[f2]
    ]
    return {
        "hand_geoms": len(geoms.hand),
        "pad_geoms": len(geoms.pads),
        "hand_cube_pairs_collidable": f"{sum(hand_cube)}/{len(hand_cube)}",
        "cube_floor_collidable": all(geoms.can_collide(c, floor) for c in geoms.cube),
        "stickers_visual_only": bool(stickers)
        and all(m.geom_contype[g] == m.geom_conaffinity[g] == 0 for g in stickers),
        "finger_distal_pairs_collidable": f"{sum(finger_pairs)}/{len(finger_pairs)}",
        "hand_self_pairs_enabled": len(geoms.self_pairs()),
        "_ok": all(hand_cube) and all(finger_pairs),
    }


def reset_clearance(scene: HandCubeScene, n_seeds: int = 20) -> dict:
    geoms = _Geoms(scene)
    hand_cube = [(h, c) for h in geoms.hand for c in geoms.cube]
    self_rigid = [(a, b) for a, b in geoms.self_pairs() if a in geoms.rigid and b in geoms.rigid]
    self_all = geoms.self_pairs()
    worst_cube, worst_rigid, worst_any = np.inf, np.inf, np.inf
    for seed in range(n_seeds):
        scene.reset(seed=seed, pose_noise=0.004, joint_noise=0.05, settle=False)
        m, d = scene.model, scene.data
        worst_cube = min(worst_cube, _min_distance(m, d, hand_cube))
        worst_rigid = min(worst_rigid, _min_distance(m, d, self_rigid))
        worst_any = min(worst_any, _min_distance(m, d, self_all))
    return {
        "seeds": n_seeds,
        "min_hand_cube_clearance_mm": round(worst_cube * 1e3, 3),
        "min_rigid_self_clearance_mm": round(worst_rigid * 1e3, 3),
        "info_min_self_clearance_incl_pads_mm": round(worst_any * 1e3, 3),
    }


def settled_home(scene: HandCubeScene, duration: float = 1.0) -> dict:
    geoms = _Geoms(scene)
    scene.reset(seed=0)
    scene.step(round(duration / scene.model.opt.timestep))
    m, d = scene.model, scene.data
    floor = np.array([m.geom("floor").id])
    n = d.ncon
    g = d.contact.geom[:n]
    touching_hand = bool(np.any(np.isin(g, geoms.cube).any(axis=1) & np.isin(g, geoms.hand).any(axis=1)))
    return {
        "cube_touches_hand": touching_hand,
        "cube_touches_floor": _max_pen(d, geoms.cube, floor) > 0
        or bool(np.any(np.isin(g, geoms.cube).any(axis=1) & np.isin(g, floor).any(axis=1))),
        "rigid_self_penetration_mm": round(_max_pen(d, geoms.rigid, geoms.rigid) * 1e3, 3),
        "info_pad_self_penetration_mm": round(_max_pen(d, geoms.pads, geoms.hand) * 1e3, 3),
        "warnings": warnings_count(d),
    }


def pinch(duration: float = 2.0) -> dict:
    scene = HandCubeScene(SceneConfig(variant=None))
    geoms = _Geoms(scene)
    m, d, r = scene.model, scene.data, scene.robot
    scene.reset()
    q_open = np.clip(np.zeros(r.n), r.q_min, r.q_max)
    r.reset(q_open)
    mujoco.mj_step(m, d, 100)
    target = q_open.copy()
    for j in PINCH_JOINTS:
        target[JOINTS.index(j)] = r.ctrl_max[JOINTS.index(j)]
    r.set_target(target)
    rigid = pad = 0.0
    contact_seen = False
    thumb = [g for g in geoms.hand if "_thumb_" in m.body(m.geom_bodyid[g]).name]
    index = [g for g in geoms.hand if "_index_" in m.body(m.geom_bodyid[g]).name]
    for _ in range(round(duration / m.opt.timestep)):
        mujoco.mj_step(m, d)
        rigid = max(rigid, _max_pen(d, geoms.rigid, geoms.rigid))
        pad = max(pad, _max_pen(d, geoms.pads, geoms.hand))
        contact_seen |= _max_pen(d, np.array(thumb), np.array(index)) > 0
    return {
        "thumb_index_contact_detected": bool(contact_seen),
        "max_rigid_self_penetration_mm": round(rigid * 1e3, 3),
        "info_max_pad_penetration_mm": round(pad * 1e3, 3),
        "finite": finite(d),
        "warnings": warnings_count(d),
    }


def press(scene: HandCubeScene, force: float = 20.0, duration: float = 1.0) -> dict:
    geoms = _Geoms(scene)
    scene.reset(seed=0)
    m, d = scene.model, scene.data
    d.xfrc_applied[scene.cube.core_body, :3] = scene.r_mount @ np.array([-force, 0.0, 0.0])  # into the palm
    rigid = 0.0
    for _ in range(round(duration / m.opt.timestep)):
        mujoco.mj_step(m, d)
        rigid = max(rigid, _max_pen(d, geoms.cube, geoms.rigid))
    d.xfrc_applied[scene.cube.core_body] = 0.0
    pos, _ = scene.cube.core_pose()
    rest, _ = scene.cube_rest_pose()
    return {
        "force_n": force,
        "max_rigid_hand_cube_penetration_mm": round(rigid * 1e3, 3),
        "cube_displacement_mm": round(float(np.linalg.norm(pos - rest)) * 1e3, 2),
        "finite": finite(d),
        "warnings": warnings_count(d),
    }


def check_collisions(n_seeds: int = 20) -> CheckResult:
    scene = HandCubeScene()
    filters = audit_filters(scene)
    clearance = reset_clearance(scene, n_seeds)
    home = settled_home(scene)
    pinch_res = pinch()
    press_res = press(scene)
    limits = {
        "min_hand_cube_clearance_mm": 0.0,  # strictly positive: no contact at spawn
        "min_rigid_self_clearance_mm": 0.0,
        "home_rigid_self_penetration_mm": 0.5,
        "pinch_max_rigid_self_penetration_mm": 5.0,
        "press_max_rigid_hand_cube_penetration_mm": 2.0,
    }
    ok = filters.pop("_ok")
    passed = (
        ok
        and filters["cube_floor_collidable"]
        and filters["stickers_visual_only"]
        and clearance["min_hand_cube_clearance_mm"] > limits["min_hand_cube_clearance_mm"]
        and clearance["min_rigid_self_clearance_mm"] > limits["min_rigid_self_clearance_mm"]
        and home["cube_touches_hand"]
        and not home["cube_touches_floor"]
        and home["rigid_self_penetration_mm"] < limits["home_rigid_self_penetration_mm"]
        and home["warnings"] == 0
        and pinch_res["thumb_index_contact_detected"]
        and pinch_res["max_rigid_self_penetration_mm"] < limits["pinch_max_rigid_self_penetration_mm"]
        and pinch_res["finite"]
        and pinch_res["warnings"] == 0
        and press_res["max_rigid_hand_cube_penetration_mm"] < limits["press_max_rigid_hand_cube_penetration_mm"]
        and press_res["finite"]
        and press_res["warnings"] == 0
    )
    notes = [
        "Rigid = palm and phalanx geoms; pads = compliant fingertip elastomer geoms, reported as info "
        "until their compliance is calibrated together with the tactile sensor.",
        "Self pairs exclude parent-child bodies (skipped by MuJoCo) and the Menagerie <exclude> list.",
    ]
    return CheckResult(
        "collisions",
        bool(passed),
        {
            "filters": filters,
            "reset_clearance": clearance,
            "settled_home": home,
            "pinch": pinch_res,
            "press": press_res,
        },
        limits,
        notes,
    )
