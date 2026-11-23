"""Grasp-stabilisation benchmark: sustained blindfolded holds under wrist reorientation and pushes.

Every policy runs the same held-out seeds in each condition:

=================  ==========================================================
``palm_down``      one reorientation to palm-down, held; no pushes
``sideways``       one reorientation to a 90 degree tilt in a random direction; no pushes
``reorient``       random reorientation sequence up to palm-down; no pushes
``reorient_push``  random reorientations plus random pushes (the training distribution)
=================  ==========================================================

Four diagnostic conditions share the ``reorient_push`` seeds and change only the pushes. They are
not part of the default run; select them with ``--condition``:

=============================  ======================================================
``reorient_push_0.5x``         pushes at half the force and torque
``reorient_push_1.5x``         pushes at one and a half times the force and torque
``reorient_push_force``        force of each push only (no torque)
``reorient_push_torque``       torque of each push only (no force)
=============================  ======================================================

An episode succeeds when it reaches the time limit with the cube in the hand,
the puzzle configuration unchanged, and every cubie within the alignment
tolerance (``GraspEnv`` ``is_success``). Policies receive only the
non-privileged observation; ground truth is used for scoring only.

Example::

    uv run manip-rubiks-eval-grasp --policy passive --policy scripted \\
        --policy learned=runs/grasp/ppo_asym_s0 --episodes 100
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manip_rubiks.evaluation.core import (
    EpisodeTrace,
    base_parser,
    build_policy,
    build_report,
    cached,
    make_jobs,
    policy_label,
    run_jobs,
    summary_table,
    wilson_interval,
    write_json_report,
)

CONDITIONS: tuple[str, ...] = ("palm_down", "sideways", "reorient", "reorient_push")
PUSH_DIAGNOSTICS: dict[str, dict[str, Any]] = {
    "reorient_push_0.5x": {"push_scale": 0.5},
    "reorient_push_1.5x": {"push_scale": 1.5},
    "reorient_push_force": {"push_kind": "force"},
    "reorient_push_torque": {"push_kind": "torque"},
}
ALL_CONDITIONS: tuple[str, ...] = (*CONDITIONS, *PUSH_DIAGNOSTICS)
TARGET_SUCCESS = 0.95


@dataclass
class EpisodeResult:
    policy: str
    condition: str
    seed: int
    success: bool
    dropped: bool
    layer_turned: bool
    unstable: bool
    steps: int
    final_misalignment: float  # [rad]
    max_misalignment: float  # [rad]
    final_displacement: float  # [m]
    mean_effort: float  # mean |actuator force| / force limit
    episode_return: float
    # What the episode looked like when it ended (for diagnosing failures):
    steps_since_push: int  # control steps since a push was last active; 0 = pushing now, -1 = never pushed
    last_push_force: float  # force magnitude of the last push [N]
    last_push_torque: float  # torque magnitude of the last push [N m]
    wrist_moving: bool  # the wrist was mid-reorientation in the last step
    final_tilt: float  # wrist tilt away from palm-up in the last step [rad]


def condition_options(condition: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if condition == "palm_down":
        return {"tilt": np.pi, "azimuth": float(rng.uniform(0, 2 * np.pi)), "pushes": False}
    if condition == "sideways":
        return {"tilt": np.pi / 2, "azimuth": float(rng.uniform(0, 2 * np.pi)), "pushes": False}
    if condition == "reorient":
        return {"pushes": False}
    if condition == "reorient_push":
        return {}
    if condition in PUSH_DIAGNOSTICS:
        return dict(PUSH_DIAGNOSTICS[condition])
    raise ValueError(f"unknown condition {condition!r}")


def run_episode(env, policy, label: str, condition: str, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed, options=condition_options(condition, seed))
    policy.reset()
    trace = EpisodeTrace()
    last_push_step, last_force, last_torque = -1, 0.0, 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        trace.update(env, reward, info)
        if info["pushing"]:
            wrench = env.data.xfrc_applied[env.cube.core_body]
            last_push_step = int(info["steps"])
            last_force, last_torque = float(np.linalg.norm(wrench[:3])), float(np.linalg.norm(wrench[3:]))
        if terminated or truncated:
            break
    return EpisodeResult(
        policy=label,
        condition=condition,
        seed=seed,
        success=bool(truncated and info["is_success"]),
        dropped=bool(info["dropped"]),
        layer_turned=not info["state_preserved"],
        unstable=bool(info["unstable"]),
        steps=int(info["steps"]),
        final_misalignment=float(info["misalignment"]),
        max_misalignment=float(trace.max_misalignment),
        final_displacement=float(info["displacement"]),
        mean_effort=trace.mean_effort,
        episode_return=float(trace.total_reward),
        steps_since_push=int(info["steps"]) - last_push_step if last_push_step >= 0 else -1,
        last_push_force=last_force,
        last_push_torque=last_torque,
        wrist_moving=bool(env.wrist_moving(float(info["steps"] - 1) * env.cfg.control_dt)),
        final_tilt=float(info["tilt"]),
    )


def _worker(job: tuple[str, str, list[int]]) -> list[EpisodeResult]:
    from manip_rubiks.environments import GraspEnv

    spec, condition, seeds = job
    env = cached("env", GraspEnv)
    policy = cached(spec, lambda: build_policy(spec))
    return [run_episode(env, policy, policy_label(spec), condition, s) for s in seeds]


def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    k = sum(r.success for r in results)
    mis = np.degrees([r.final_misalignment for r in results])
    return {
        "episodes": n,
        "success_rate": k / n,
        "success_ci95": wilson_interval(k, n),
        "drop_rate": float(np.mean([r.dropped for r in results])),
        "layer_turn_rate": float(np.mean([r.layer_turned for r in results])),
        "unstable_rate": float(np.mean([r.unstable for r in results])),
        "misalignment_median_deg": float(np.median(mis)),
        "misalignment_p95_deg": float(np.percentile(mis, 95)),
        "displacement_median_mm": float(np.median([r.final_displacement for r in results]) * 1000),
        "mean_effort": float(np.mean([r.mean_effort for r in results])),
        "mean_return": float(np.mean([r.episode_return for r in results])),
    }


def run_benchmark(
    specs: list[str], episodes: int, workers: int, conditions: tuple[str, ...] = CONDITIONS, chunk: int = 10
) -> list[EpisodeResult]:
    return run_jobs(_worker, make_jobs(specs, conditions, episodes, chunk), workers)


def write_report(results: list[EpisodeResult], specs: list[str], out: Path, elapsed: float) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    labels = [policy_label(s) for s in specs]
    conditions = [c for c in ALL_CONDITIONS if any(r.condition == c for r in results)]
    table = summary_table(results, labels, conditions, summarise)
    report = build_report(specs, results, elapsed, table, {"target_success_rate": TARGET_SUCCESS})
    write_json_report(out / "grasp_report.json", report)

    lines = [
        "# Grasp stabilisation benchmark",
        "",
        "Sustained blindfolded hold of the 2x2x2 cube with the fixed-wrist Sharpa Wave while gravity is",
        "rotated to emulate wrist reorientation. Success: the episode reaches its 10 s time limit with the",
        "cube in the hand, the puzzle configuration unchanged and every cubie within 10 degrees of",
        f"alignment. Target: {TARGET_SUCCESS:.0%} success. Generated by `manip-rubiks-eval-grasp`",
        f"({len(results)} episodes, {elapsed:.0f} s).",
        "",
        "| Policy | Condition | Success (95% CI) | Dropped | Layer turned | Misalign. median / p95 | Effort |",
        "|---|---|---|---|---|---|---|",
    ]
    for lab in labels:
        for c in conditions:
            s = table[lab][c]
            lo, hi = s["success_ci95"]
            lines.append(
                f"| {lab} | {c} | {s['success_rate']:.0%} ({lo:.0%}-{hi:.0%}) | {s['drop_rate']:.0%} | "
                f"{s['layer_turn_rate']:.0%} | {s['misalignment_median_deg']:.1f} / "
                f"{s['misalignment_p95_deg']:.1f} deg | {s['mean_effort']:.2f} |"
            )
    lines += [
        "",
        "Conditions: `palm_down` and `sideways` reorient once (to 180 and 90 degrees) and hold without",
        "pushes; `reorient` follows random reorientation sequences up to palm-down; `reorient_push` adds",
        "random pushes on the cube (up to 1.5 N and 0.02 N m), as in training. The `reorient_push_*`",
        "diagnostic conditions replay the same seeds with the pushes scaled or reduced to force or torque.",
        "Effort is the mean actuator force as a fraction of its limit.",
        "",
    ]
    (out / "grasp_report.md").write_text("\n".join(lines))
    return report


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__, ALL_CONDITIONS, Path("reports/grasp"), "passive, scripted or label=run_dir")
    args = p.parse_args(argv)
    specs = args.policies or ["passive", "scripted"]
    conditions = tuple(args.conditions or CONDITIONS)
    t0 = time.time()
    results = run_benchmark(specs, args.episodes, args.workers, conditions)
    report = write_report(results, specs, args.out, time.time() - t0)
    for lab, rows in report["summary"].items():
        for c, s in rows.items():
            print(
                f"{lab:>10s} {c:>14s} success {s['success_rate']:.0%}  drop {s['drop_rate']:.0%}  "
                f"turned {s['layer_turn_rate']:.0%}  misalign {s['misalignment_median_deg']:.1f} deg"
            )


if __name__ == "__main__":
    main()
