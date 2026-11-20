"""Gate M1 -- tactile foundation (docs/project-draft.md sections 9-10, 33).

Sensor experiments on the :class:`~manip_rubiks.validation.tactile_rig.TactileRig`:

``static_contact`` (T1)
    Increasing indentation of every fingertip: the readings must grow monotonically, stay silent on the
    other fingers, and return to zero after release.
``sliding`` (T2)
    Tangential sweeps along and across the pad at several speeds, in both directions.
``contact_location`` (T3)
    A grid of press positions on every pad; how well each configuration localises the contact.
``multiple_contacts`` (T4)
    Two probes at several separations; do the taxels see two contacts?
``robustness`` (T5)
    Noise, dropout and latency sweeps on detection and localisation.
``repeatability``
    Identical presses must give identical readings.
``pad_calibration``
    Pad penetration under known loads for the default pad and stiffer candidates.

Run ``manip-rubiks-m1`` to write ``reports/m1/m1_report.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from manip_rubiks.simulation.assets import MENAGERIE_COMMIT
from manip_rubiks.simulation.robot_interface import FINGERTIPS
from manip_rubiks.simulation.scene_builder import HAND_PREFIX
from manip_rubiks.simulation.tactile_sensors import TactileSensors
from manip_rubiks.simulation.taxels import COLS, ROWS
from manip_rubiks.validation.common import CheckResult
from manip_rubiks.validation.tactile_rig import TactileRig

SETTLE_STEPS = 250  # physics steps (0.5 s) for a press to settle
CONTROL_STEPS = 20  # physics steps per control step (control_dt 0.04 s)
MM = 1e3


def _r(x, n: int = 4):
    """Round numpy scalars and arrays into plain JSON-friendly Python values."""
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), n).tolist()
    return round(float(x), n)


def count_blobs(mask: np.ndarray) -> int:
    """Number of 4-connected components of a boolean grid."""
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and not seen[i, j]:
                n += 1
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        stack += [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
    return n


def centroid(depth: np.ndarray, positions: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Depth-weighted centroid of a taxel map in pad coordinates (along, across) [m]."""
    w = np.clip(depth - floor * depth.max(), 0.0, None)
    if w.sum() <= 0:
        return np.full(2, np.nan)
    return (w[..., None] * positions).sum(axis=(0, 1)) / w.sum()


def _press(rig: TactileRig, finger: str, along: float, across: float, depth: float, steps: int = SETTLE_STEPS) -> None:
    rig.reset()
    rig.run(steps, lambda k: rig.place(0, finger, along, across, depth))


# --------------------------------------------------------------------------- T1
def check_static_contact(depths=(0.0005, 0.001, 0.002, 0.003), fingers=FINGERTIPS) -> CheckResult:
    rig = TactileRig()
    per_finger, ok = {}, True
    for finger in fingers:
        k = FINGERTIPS.index(finger)
        others = [i for i in range(len(FINGERTIPS)) if i != k]
        force, taxel_sum, taxel_max, crosstalk = [], [], [], 0.0
        for depth in depths:
            _press(rig, finger, 0.0, 0.0, depth)
            r, si = rig.readings(), rig.taxel_si()
            force.append(float(np.linalg.norm(r["B"][k])))
            taxel_sum.append(float(si[k, ..., 0].sum()))
            taxel_max.append(float(si[k, ..., 0].max()))
            crosstalk = max(crosstalk, float(np.abs(r["B"][others]).max()), float(np.abs(si[others]).max()))
            contact_deepest = float(r["A"][k])
        # Release: the probe leaves and every reading must return to zero.
        rig.run(150, lambda i, f=finger: rig.place(0, f, 0.0, 0.0, -0.02))
        r, si = rig.readings(), rig.taxel_si()
        released = bool(not r["A"].any() and np.abs(r["B"]).max() < 1e-3 and si.max() < 1e-9)
        monotonic = {
            "force": bool(np.all(np.diff(force) >= -1e-6)),
            "taxel_depth": bool(np.all(np.diff(taxel_sum) >= -1e-9)),
        }
        rises = bool(force[-1] > 2 * max(force[0], 1e-6) and taxel_sum[-1] > 2 * max(taxel_sum[0], 1e-9))
        finger_ok = all(monotonic.values()) and rises and crosstalk < 1e-9 and released and contact_deepest == 1.0
        ok &= finger_ok
        per_finger[finger] = {
            "pass": finger_ok,
            "force_N": _r(np.array(force)),
            "taxel_depth_sum_mm": _r(np.array(taxel_sum) * MM, 3),
            "taxel_depth_max_mm": _r(np.array(taxel_max) * MM, 3),
            "monotonic": monotonic,
            "crosstalk_on_other_fingers": crosstalk,
            "released_to_zero": released,
            "binary_contact_at_deepest": contact_deepest,
        }
    return CheckResult(
        "T1_static_contact",
        bool(ok),
        {"commanded_depth_mm": [d * MM for d in depths], "per_finger": per_finger},
        {"monotonic": "non-decreasing in commanded depth", "crosstalk": "< 1e-9", "release": "all readings zero"},
        [
            "Depth is the commanded penetration of a flat probe below the unloaded pad surface; the fingers are "
            "position controlled, so the realised indentation is smaller than the command.",
            "The binary contact flag needs 0.05 N, so it stays off for the lightest touches while taxel depth "
            "already responds.",
        ],
    )


# --------------------------------------------------------------------------- T2
def _slide(rig: TactileRig, finger: str, axis: str, speed: float, depth: float = 0.0015, window: float = 0.002):
    """Press at the start, slide at ``speed`` and return the mean readings while within ``window`` of the centre."""
    k = FINGERTIPS.index(finger)
    rig.reset()
    start = -0.005 if speed > 0 else 0.005
    n_press, n_slide = SETTLE_STEPS, int(0.012 / abs(speed) / rig.dt) + 1
    samples = []

    def pos(i: int) -> float:
        return start + speed * max(i - n_press, 0) * rig.dt

    def drive(i: int) -> None:
        v = 0.0 if i < n_press else speed
        if axis == "along":
            rig.place(0, finger, pos(i), 0.0, depth, v_along=v)
        else:
            rig.place(0, finger, 0.0, pos(i), depth, v_across=v)

    def record(i: int) -> None:
        if i >= n_press and abs(pos(i)) <= window:
            si, f = rig.taxel_si()[k], rig.force.measure()[k]
            a = rig.contact.measure()[k]
            samples.append([si[..., 1].max(), si[..., 2].max(), si[..., 0].max(), *f, a])

    rig.run(n_press + n_slide, drive, record)
    return np.mean(samples, axis=0)  # slip_along, slip_across, depth, Bx, By, Bz, A


def check_sliding(speeds=(0.01, 0.03, 0.06, 0.1), fingers=FINGERTIPS) -> CheckResult:
    rig = TactileRig(probe_half=(0.0025, 0.0025, 0.005))
    rel_tol, cross_tol = 0.10, 0.05
    sweep, ok = {}, True
    for axis in ("along", "across"):
        own, other = (0, 1) if axis == "along" else (1, 0)
        rows, tang = [], []
        for v in speeds:
            plus = _slide(rig, "index", axis, +v)
            minus = _slide(rig, "index", axis, -v)
            err = abs(plus[own] - v) / v
            cross = plus[other] / v
            blind = abs(plus[own] - minus[own]) / v
            # Config B: the tangential friction force changes sign with the direction of motion.
            bt = (plus[5], minus[5]) if axis == "along" else (plus[4], minus[4])
            tang.append(abs(bt[0]))
            sign_flips = bool(np.sign(bt[0]) == -np.sign(bt[1]) and bt[0] != 0)
            rows.append(
                {
                    "speed_m_s": v,
                    "slip_measured_m_s": _r(plus[own], 5),
                    "rel_error": _r(err),
                    "cross_channel_ratio": _r(cross, 5),
                    "sign_blind_diff_ratio": _r(blind, 5),
                    "B_tangential_force_N": _r(abs(bt[0]), 4),
                    "B_sign_flips_with_direction": sign_flips,
                }
            )
            ok &= bool(err <= rel_tol and cross <= cross_tol and sign_flips)
        mono = bool(np.all(np.diff(tang) > 0))
        ok &= mono
        sweep[axis] = {"rows": rows, "B_tangential_force_increases_with_speed": mono}
    # Static press: the slip channels stay at zero.
    _press(rig, "index", 0.0, 0.0, 0.0015)
    static_slip = float(rig.taxel_si()[FINGERTIPS.index("index"), ..., 1:].max())
    ok &= static_slip < 2e-3
    # Same check on every fingertip at a single speed (along the finger).
    per_finger = {}
    for finger in fingers:
        m = _slide(rig, finger, "along", +0.03)
        err = abs(m[0] - 0.03) / 0.03
        per_finger[finger] = {"slip_along_m_s": _r(m[0], 5), "rel_error": _r(err), "cross_ratio": _r(m[1] / 0.03, 5)}
        ok &= bool(err <= rel_tol and m[1] / 0.03 <= cross_tol)
    _press(rig, "index", 0.0, 0.0, 0.0015)
    a_static = float(rig.contact.measure()[FINGERTIPS.index("index")])
    a_sliding = float(_slide(rig, "index", "along", +0.03)[6])
    return CheckResult(
        "T2_sliding",
        bool(ok),
        {
            "index_sweep": sweep,
            "static_slip_max_m_s": static_slip,
            "all_fingertips_along_0.03_m_s": per_finger,
            "signed_slip_channels": False,
            "binary_contact_static_vs_sliding": [a_static, a_sliding],
            "binary_contact_sees_sliding": bool(a_static != a_sliding),
        },
        {
            "slip_rel_error": f"<= {rel_tol}",
            "cross_channel_ratio": f"<= {cross_tol}",
            "static_slip_m_s": "< 2e-3",
        },
        [
            "Configuration C reports slip speed magnitudes along (channel 1) and across (channel 2) the pad: both "
            "channels are non-negative, so the direction of slip is not observable from C alone (sign_blind_diff_ratio "
            "~ 0). Configuration B's tangential force is signed.",
            "Configuration A is a single bit per fingertip and does not change between a static press and a slide.",
        ],
    )


# --------------------------------------------------------------------------- T3
def _grid(n: int, extent: float = 0.004) -> list[float]:
    return list(np.linspace(-extent, extent, n))


def check_contact_location(grid: int = 5, fingers=FINGERTIPS) -> CheckResult:
    rig = TactileRig(probe_half=(0.002, 0.002, 0.005))
    mean_tol, corr_tol = 1.0e-3, 0.98
    per_finger, ok = {}, True
    for finger in fingers:
        k = FINGERTIPS.index(finger)
        pos = rig.taxel_positions(finger)
        cmd, est, force = [], [], []
        for a in _grid(grid):
            for c in _grid(grid):
                _press(rig, finger, a, c, 0.002)
                cmd.append((a, c))
                est.append(centroid(rig.taxel_si()[k][..., 0], pos))
                force.append(rig.force.measure()[k])
        cmd, est, force = np.array(cmd), np.array(est), np.array(force)
        err = np.linalg.norm(cmd - est, axis=1)
        corr = [float(np.corrcoef(cmd[:, i], est[:, i])[0, 1]) for i in range(2)]
        x = np.c_[force, np.ones(len(force))]
        r2 = []
        for i in range(2):
            coef, *_ = np.linalg.lstsq(x, cmd[:, i], rcond=None)
            resid = cmd[:, i] - x @ coef
            r2.append(float(1 - (resid**2).sum() / ((cmd[:, i] - cmd[:, i].mean()) ** 2).sum()))
        finger_ok = bool(np.nanmean(err) <= mean_tol and min(corr) >= corr_tol)
        ok &= finger_ok
        per_finger[finger] = {
            "pass": finger_ok,
            "C_mean_error_mm": _r(np.nanmean(err) * MM, 3),
            "C_max_error_mm": _r(np.nanmax(err) * MM, 3),
            "C_corr_along_across": _r(np.array(corr), 4),
            "B_linear_fit_R2_along_across": _r(np.array(r2), 3),
            "A_distinct_values": 1,
        }
    pitch = {}
    for f in fingers:
        extent = np.ptp(rig.taxel_positions(f), axis=(0, 1))  # (along, across)
        pitch[f] = _r(extent / np.array([ROWS - 1, COLS - 1]) * MM, 2)
    return CheckResult(
        "T3_contact_location",
        bool(ok),
        {
            "grid_points": grid * grid,
            "positions_mm": [round(v * MM, 2) for v in _grid(grid)],
            "taxel_pitch_mm_along_across": pitch,
            "per_finger": per_finger,
        },
        {"C_mean_error": f"<= {mean_tol * MM} mm", "C_corr": f">= {corr_tol}"},
        [
            "C localises the contact as the depth-weighted centroid of the taxel map. A is one bit per fingertip, "
            "so it carries no location. B is a net force vector, but in this noiseless set-up a linear map of it "
            "recovers the across-pad position almost perfectly and the along-pad position only partly "
            "(B_linear_fit_R2): the pad is a curved capsule, so the force direction depends on where it is pressed "
            "across it. That is not a reliable location sensor on a real contact patch or under noise, and B cannot "
            "separate two contacts (T4) while C can.",
        ],
    )


# --------------------------------------------------------------------------- T4
def check_multiple_contacts(separations=(0.002, 0.003, 0.004, 0.006, 0.008), finger="index") -> CheckResult:
    k = FINGERTIPS.index(finger)
    rig = TactileRig(n_probes=2, probe_half=(0.0015, 0.0015, 0.005))
    blobs, ok = {}, True
    rig.reset()
    rig.run(SETTLE_STEPS, lambda i: rig.place(0, finger, 0.0, 0.0, 0.002))
    single = count_blobs(rig.taxel_si()[k][..., 0] > 0.5 * rig.taxel_si()[k][..., 0].max())
    for sep in separations:
        rig.reset()

        def drive(i: int, sep: float = sep) -> None:
            rig.place(0, finger, -sep / 2, 0.0, 0.002)
            rig.place(1, finger, sep / 2, 0.0, 0.002)

        rig.run(SETTLE_STEPS, drive)
        d = rig.taxel_si()[k][..., 0]
        blobs[f"{sep * MM:.0f}mm"] = count_blobs(d > 0.5 * d.max())
    resolved = [s for s in separations if blobs[f"{s * MM:.0f}mm"] == 2]
    needed = [s for s in separations if s >= 0.006]
    ok = bool(single == 1 and all(s in resolved for s in needed))
    return CheckResult(
        "T4_multiple_contacts",
        ok,
        {
            "finger": finger,
            "probe_width_mm": 3.0,
            "blobs_single_probe": single,
            "blobs_by_centre_separation": blobs,
            "resolved_from_mm": _r(min(resolved) * MM, 1) if resolved else None,
            "taxel_pitch_along_mm": _r(np.ptp(rig.taxel_positions(finger)[..., 0]) / (ROWS - 1) * MM, 2),
        },
        {"single_probe_blobs": 1, "two_blobs_from_separation": "6 mm"},
        [
            "Two probes 3 mm wide on one pad. A blob is a 4-connected region of taxels above half the peak depth. "
            "A and B return a single value per fingertip and cannot tell one contact from two.",
        ],
    )


# --------------------------------------------------------------------------- T5
def _score(maps: np.ndarray) -> tuple[float, np.ndarray]:
    med = np.median(maps, axis=0)
    return float(med.max()), med


def check_robustness(quick: bool = False, finger: str = "index") -> CheckResult:
    k = FINGERTIPS.index(finger)
    rig = TactileRig(probe_half=(0.002, 0.002, 0.005))
    pos = rig.taxel_positions(finger)
    window, thr_det = 5, 0.02  # median over 5 reads; 0.02 of full scale is 0.3 mm of indentation
    settings = [(s, 0.0, 0) for s in ((0.0, 0.005, 0.02) if quick else (0.0, 0.002, 0.005, 0.01, 0.02))]
    settings += [(0.0, p, 0) for p in ((0.1, 0.3) if quick else (0.1, 0.3, 0.5))]
    nominal = (0.005, 0.1, 2)
    settings += [nominal]
    sensors = {}
    for s in settings:
        cfg = replace(rig.taxel.cfg, noise_std=s[0], dropout=s[1], latency=s[2])
        sensors[s] = TactileSensors(rig.model, rig.data, "right", HAND_PREFIX, cfg)
    for i, ts in enumerate(sensors.values()):
        ts.reset(i)
    n = 3 if quick else 5
    samples = [(a, c) for a in _grid(n) for c in _grid(n)] + [None] * (n * n)
    stats = {s: {"tp": 0, "fp": 0, "fn": 0, "err": []} for s in settings}
    for sample in samples:
        rig.reset()
        if sample is None:
            rig.run(SETTLE_STEPS, lambda i: rig.place(0, finger, 0.0, 0.0, -0.02))
        else:
            rig.run(SETTLE_STEPS, lambda i, s=sample: rig.place(0, finger, s[0], s[1], 0.002))
        for s, ts in sensors.items():
            reads = np.stack([ts.read()[k, ..., 0] for _ in range(window + s[2])])[-window:]
            score, med = _score(reads)
            detected = score > thr_det
            st = stats[s]
            if sample is None:
                st["fp"] += int(detected)
            elif detected:
                st["tp"] += 1
                est = centroid(med, pos, floor=0.3)
                st["err"].append(float(np.linalg.norm(est - np.array(sample))))
            else:
                st["fn"] += 1
    table = {}
    for s, st in stats.items():
        precision = st["tp"] / max(st["tp"] + st["fp"], 1)
        recall = st["tp"] / max(st["tp"] + st["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        table[f"noise={s[0]},dropout={s[1]},latency={s[2]}"] = {
            "precision": _r(precision, 3),
            "recall": _r(recall, 3),
            "F1": _r(f1, 3),
            "localisation_error_mm": _r(np.mean(st["err"]) * MM, 3) if st["err"] else None,
        }
    nom = table[f"noise={nominal[0]},dropout={nominal[1]},latency={nominal[2]}"]
    clean = table["noise=0.0,dropout=0.0,latency=0"]
    # Latency: the reading lags the physical event by exactly ``latency`` reads.
    delays = {}
    for latency in (0, 1, 2, 4):
        ts = TactileSensors(rig.model, rig.data, "right", HAND_PREFIX, replace(rig.taxel.cfg, latency=latency))
        rig.reset()
        ts.reset(0)
        for _ in range(8):  # fill the delay line with the empty (no contact) reading
            rig.run(CONTROL_STEPS, lambda i: rig.place(0, finger, 0.0, 0.0, -0.02))
            ts.read()
        delay = None
        for step in range(12):
            rig.run(CONTROL_STEPS, lambda i: rig.place(0, finger, 0.0, 0.0, 0.002))
            if ts.read()[k, ..., 0].max() > thr_det and delay is None:
                delay = step
        delays[latency] = delay
    base = delays[0]
    latency_ok = all(d is not None and base is not None and d - base == latency for latency, d in delays.items())
    ok = bool(
        nom["F1"] >= 0.9
        and nom["localisation_error_mm"] is not None
        and nom["localisation_error_mm"] <= 2.0
        and clean["F1"] >= 0.99
        and latency_ok
    )
    return CheckResult(
        "T5_robustness",
        ok,
        {
            "finger": finger,
            "samples": len(samples),
            "estimator": f"median of {window} reads, detection threshold {thr_det} of full scale (0.3 mm)",
            "sweep": table,
            "nominal_corruption": {"noise_std": nominal[0], "dropout": nominal[1], "latency": nominal[2]},
            "detection_delay_in_reads_by_latency": delays,
            "latency_adds_exactly_its_value": latency_ok,
        },
        {"nominal_F1": ">= 0.9", "nominal_localisation_error": "<= 2 mm", "clean_F1": ">= 0.99"},
        [
            "noise_std is in normalised taxel units (1.0 = max_depth 15 mm / max_speed 0.15 m/s), so 0.005 is "
            "0.075 mm of depth noise. Dropout zeroes a whole fingertip for one read.",
            "Detection delay is counted in control steps (one read per 20 physics steps) after the probe appears.",
        ],
    )


# ------------------------------------------------------------------ repeatability
def check_repeatability(repeats: int = 5) -> CheckResult:
    rig = TactileRig()
    runs = []
    for _ in range(repeats):
        _press(rig, "index", 0.001, -0.001, 0.002)
        runs.append(np.concatenate([rig.taxel.measure().ravel(), rig.force.measure().ravel()]))
    spread = float(np.abs(np.array(runs) - runs[0]).max())
    return CheckResult(
        "repeatability",
        bool(spread < 1e-9),
        {"repeats": repeats, "max_deviation_between_repeats": spread},
        {"max_deviation": "< 1e-9"},
        ["Identical press, repeated after a full reset: the physical readings are deterministic."],
    )


# ------------------------------------------------------------------ pad calibration
CANDIDATES: dict[str, tuple[tuple[float, float] | None, tuple[float, ...] | None]] = {
    "default (solref 0.06 0.9)": (None, None),
    "solref 0.04 1.0": ((0.04, 1.0), None),
    "solref 0.02 1.0": ((0.02, 1.0), None),
    "solref 0.01 1.0": ((0.01, 1.0), None),
    "solref 0.005 1.0": ((0.005, 1.0), None),
    "solref 0.01 1.0, solimp 0.95 0.99 0.001 0.5 2": ((0.01, 1.0), (0.95, 0.99, 0.001, 0.5, 2.0)),
}


TARGET_FORCE = 10.0  # [N]
TARGET_PENETRATION = 2e-3  # [m]; an assumption, see the notes of ``check_pad_calibration``


def pad_curve(solref, solimp, forces, finger: str = "index", steps: int = 1500) -> dict[float, float | None]:
    """Steady pad penetration [m] under known normal loads; ``None`` where the probe punches through."""
    rig = TactileRig(probe_half=(0.008, 0.008, 0.004), pad_solref=solref, pad_solimp=solimp)
    rig.lock_hand()  # rigid backing: the pad's own compression, not the finger's flex
    m, d = rig.model, rig.data
    pad = m.geom(f"{HAND_PREFIX}right_{finger}_fingertip_collision").id
    probe, body = m.geom("probe0").id, m.body("probe0").id
    f6 = np.zeros(6)
    curve: dict[float, float | None] = {}
    for force in forces:
        rig.reset()
        fr = rig.frames[finger]
        rig.place(0, finger, 0.0, 0.0, -0.0005)
        d.xfrc_applied[body, :3] = -fr.normal * force
        pens, normals = [], []
        for i in range(steps):
            mujoco.mj_step(m, d)
            if i >= steps * 2 // 3:
                pen, normal = 0.0, 0.0
                for c in range(d.ncon):
                    ct = d.contact[c]
                    if {ct.geom1, ct.geom2} == {pad, probe}:
                        mujoco.mj_contactForce(m, d, c, f6)
                        normal += f6[0]
                        pen = max(pen, -ct.dist)
                pens.append(pen)
                normals.append(normal)
        curve[force] = float(np.mean(pens)) if np.mean(normals) > 0.9 * force else None
    return curve


def check_pad_calibration(quick: bool = False) -> CheckResult:
    forces = (2.0, 10.0) if quick else (1.0, 2.0, 5.0, 10.0, 20.0)
    all_names = list(CANDIDATES)
    names = [all_names[0], all_names[2], all_names[3]] if quick else all_names
    table, recommended, best = {}, None, 0.0
    for name in names:
        solref, solimp = CANDIDATES[name]
        curve = pad_curve(solref, solimp, forces)
        table[name] = {f"{f:g} N": (None if v is None else _r(v * MM, 3)) for f, v in curve.items()}
        pen = curve.get(TARGET_FORCE)
        if pen is not None and pen <= TARGET_PENETRATION and pen > best:
            recommended, best = name, pen  # the softest candidate (deepest indentation) that meets the target
    default = table[names[0]]
    lost_at = [k for k, v in default.items() if v is None]
    mono = all(np.all(np.diff([v for v in row.values() if v is not None]) > 0) for row in table.values())
    return CheckResult(
        "pad_calibration",
        bool(mono and recommended is not None),
        {
            "penetration_mm_by_load": table,
            "default_pad_cannot_carry_load_from": lost_at[0] if lost_at else None,
            "target": f"<= {TARGET_PENETRATION * MM:g} mm at {TARGET_FORCE:g} N",
            "recommended_pad": recommended,
            "default_pad_unchanged": True,
        },
        {"penetration_increases_with_load": True, "a_candidate_meets_the_target": True},
        [
            "None means the pad could not carry the load: the steady contact force stayed below 90% of the applied "
            "load because the probe sank through the pad. The load is a "
            "70 g flat probe pressed on the index pad with the hand clamped, so the pad is rigidly backed.",
            "MuJoCo's soft contacts scale with the effective mass of the pair, so these penetrations belong to "
            "a contact between the finger and a light object such as the cube; they are not a material property.",
            f"ASSUMPTION: the target of {TARGET_PENETRATION * MM:g} mm at {TARGET_FORCE:g} N is a typical figure for "
            "a silicone fingertip pad. It is not taken from the Sharpa Wave datasheet and should be confirmed.",
            "The recommendation is not applied: SceneConfig keeps Menagerie's pad so trained grasp and rotation "
            "policies stay valid. Set SceneConfig.pad_solref / pad_solimp to use a candidate.",
        ],
    )


# ------------------------------------------------------------------------- report
def run_all(quick: bool = False) -> dict:
    q = quick
    fingers = ("index", "thumb") if q else FINGERTIPS
    checks = [
        check_static_contact(depths=(0.001, 0.003) if q else (0.0005, 0.001, 0.002, 0.003), fingers=fingers),
        check_sliding(speeds=(0.03, 0.1) if q else (0.01, 0.03, 0.06, 0.1), fingers=fingers),
        check_contact_location(grid=3 if q else 5, fingers=fingers),
        check_multiple_contacts(separations=(0.003, 0.006, 0.008) if q else (0.002, 0.003, 0.004, 0.006, 0.008)),
        check_robustness(quick=q),
        check_repeatability(3 if q else 5),
        check_pad_calibration(quick=q),
    ]
    return {
        "gate": "M1 - tactile foundation",
        "passed": all(c.passed for c in checks),
        "quick": quick,
        "environment": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "checks": [asdict(c) for c in checks],
    }


def _fmt(v) -> str:
    return json.dumps(v) if isinstance(v, (dict, list)) else str(v)


def to_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        f"# Gate M1 report -- {'PASSED' if report['passed'] else 'FAILED'}",
        "",
        f"Environment: MuJoCo {env['mujoco']}, NumPy {env['numpy']}, Python {env['python']}, "
        f"Menagerie `{env['menagerie_commit'][:12]}`" + (" (quick run)" if report["quick"] else ""),
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for c in report["checks"]:
        lines.append(f"| {c['name']} | {'pass' if c['passed'] else '**FAIL**'} |")
    for c in report["checks"]:
        lines += ["", f"## {c['name']} -- {'pass' if c['passed'] else 'FAIL'}", ""]
        for k, v in c["metrics"].items():
            if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                lines += ["", f"**{k}**", ""]
                cols = list(next(iter(v.values())).keys())
                lines += ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
                for name, row in v.items():
                    lines.append(f"| {name} | " + " | ".join(_fmt(row.get(col)) for col in cols) + " |")
                lines.append("")
            else:
                lines.append(f"- {k}: {_fmt(v)}")
        if c["thresholds"]:
            lines.append(f"- thresholds: {_fmt(c['thresholds'])}")
        for n in c["notes"]:
            lines.append(f"- note: {n}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gate M1 tactile-foundation experiments.")
    parser.add_argument("--quick", action="store_true", help="smaller sweeps (for CI / smoke tests)")
    parser.add_argument("--out", type=Path, default=Path("reports/m1"))
    args = parser.parse_args(argv)
    report = run_all(quick=args.quick)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "m1_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "m1_report.md").write_text(to_markdown(report))
    for c in report["checks"]:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
    print(f"Gate M1: {'PASSED' if report['passed'] else 'FAILED'}  (report in {args.out})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
