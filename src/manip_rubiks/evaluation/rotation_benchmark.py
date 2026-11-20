"""Quarter-turn reorientation benchmark on the CPU reference environment.

Every policy runs the same held-out seeds in each condition:

==============  =============================================================
``turn_<c>``    a single quarter-turn command ``c`` (``+x``, ``-x``, ``+y``,
                ``-y``, ``+z``, ``-z``); the episode ends when the goal is
                reached (success) or the attempt fails
``chain``       the training task: random commands about ``--axes`` for the
                full 16 s episode
==============  =============================================================

A goal counts as reached once the cube stays within the goal tolerance
(0.4 rad) for 5 control steps. An attempt fails if the cube is dropped, a
layer turns, the simulation becomes unstable, or the goal is not reached
within 6 s. In the chain condition, a goal still pending at the time limit is
not counted. The teacher policies receive the privileged ``teacher``
observation they were trained with; everything else is non-privileged.

Example::

    uv run manip-rubiks-eval-rotate --policy passive --policy gpu=runs/rotation_gpu/gpu_s0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manip_rubiks.environments.rotation_env import COMMAND_NAMES
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

TURN_CONDITIONS: tuple[str, ...] = tuple(f"turn_{c}" for c in COMMAND_NAMES)
CONDITIONS: tuple[str, ...] = (*TURN_CONDITIONS, "chain")
TARGET_GOAL_SUCCESS = 0.90
TARGET_DROP_RATE = 0.05


@dataclass
class EpisodeResult:
    policy: str
    condition: str
    seed: int
    goals_reached: int
    goals_failed: int  # 0 or 1: the attempt that ended the episode
    outcome: str  # success, timed_out, dropped, layer_turned, unstable, pending (time limit mid-goal)
    steps: int
    first_goal_s: float  # time to reach the first goal [s], nan if never reached
    max_misalignment: float  # [rad]
    final_displacement: float  # [m]
    mean_effort: float  # mean |actuator force| / force limit
    episode_return: float


def _outcome(info: dict[str, Any], terminated: bool) -> str:
    if not terminated:
        return "pending"
    if info["unstable"]:
        return "unstable"
    if info["dropped"]:
        return "dropped"
    if not info["state_preserved"]:
        return "layer_turned"
    return "timed_out"


def run_episode(env, policy, label: str, condition: str, seed: int) -> EpisodeResult:
    single = condition != "chain"
    options = {"commands": [COMMAND_NAMES.index(condition.removeprefix("turn_"))]} if single else {}
    obs, info = env.reset(seed=seed, options=options)
    policy.reset()
    trace = EpisodeTrace()
    first_goal = float("nan")
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        trace.update(env, reward, info)
        if info["goal_reached"] and np.isnan(first_goal):
            first_goal = info["steps"] * env.cfg.control_dt
        if terminated or truncated or (single and info["goals_reached"] > 0):
            break
    reached = int(info["goals_reached"])
    outcome = "success" if single and reached else _outcome(info, terminated)
    if not single and outcome == "pending":
        outcome = "success"  # survived the whole chain
    return EpisodeResult(
        policy=label,
        condition=condition,
        seed=seed,
        goals_reached=reached,
        goals_failed=int(terminated),
        outcome=outcome,
        steps=int(info["steps"]),
        first_goal_s=first_goal,
        max_misalignment=float(trace.max_misalignment),
        final_displacement=float(info["displacement"]),
        mean_effort=trace.mean_effort,
        episode_return=float(trace.total_reward),
    )


def _worker(job: tuple[str, str, str, list[int]]) -> list[EpisodeResult]:
    from manip_rubiks.environments import RotationConfig, RotationEnv

    spec, axes, condition, seeds = job
    env = cached(f"env:{axes}", lambda: RotationEnv(RotationConfig(goal_axes=axes)))
    policy = cached(spec, lambda: build_policy(spec))
    return [run_episode(env, policy, policy_label(spec), condition, s) for s in seeds]


def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    reached = sum(r.goals_reached for r in results)
    attempts = reached + sum(r.goals_failed for r in results)
    first = [r.first_goal_s for r in results if not np.isnan(r.first_goal_s)]
    return {
        "episodes": len(results),
        "goal_attempts": attempts,
        "goal_success_rate": reached / attempts if attempts else float("nan"),
        "goal_success_ci95": wilson_interval(reached, attempts),
        "goals_per_episode": reached / len(results),
        "drop_rate": float(np.mean([r.outcome == "dropped" for r in results])),
        "timeout_rate": float(np.mean([r.outcome == "timed_out" for r in results])),
        "layer_turn_rate": float(np.mean([r.outcome == "layer_turned" for r in results])),
        "unstable_rate": float(np.mean([r.outcome == "unstable" for r in results])),
        "first_goal_median_s": float(np.median(first)) if first else float("nan"),
        "mean_effort": float(np.mean([r.mean_effort for r in results])),
        "mean_return": float(np.mean([r.episode_return for r in results])),
    }


def run_benchmark(
    specs: list[str], episodes: int, workers: int, conditions: tuple[str, ...], axes: str, chunk: int = 10
) -> list[EpisodeResult]:
    return run_jobs(_worker, make_jobs(specs, conditions, episodes, chunk, extra=(axes,)), workers)


def _pct(x: float) -> str:
    return "-" if np.isnan(x) else f"{x:.0%}"


def write_report(
    results: list[EpisodeResult], specs: list[str], out: Path, elapsed: float, axes: str
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    labels = [policy_label(s) for s in specs]
    conditions = [c for c in CONDITIONS if any(r.condition == c for r in results)]
    table = summary_table(results, labels, conditions, summarise)
    header = {
        "chain_axes": axes,
        "target_goal_success_rate": TARGET_GOAL_SUCCESS,
        "target_drop_rate": TARGET_DROP_RATE,
    }
    report = build_report(specs, results, elapsed, table, header)
    write_json_report(out / "rotation_report.json", report)

    lines = [
        "# Quarter-turn reorientation benchmark",
        "",
        "In-hand whole-cube reorientation with the fixed-wrist Sharpa Wave, palm up, locked cube layers.",
        "A goal is the current orientation turned by 90 degrees about a hand axis (x: palm normal, y: across",
        "the palm, z: along the fingers). It is reached once the cube stays within 0.4 rad for 0.2 s; an",
        "attempt fails on a drop, a layer turn, instability or after 6 s. Target: "
        f"{TARGET_GOAL_SUCCESS:.0%} goal success with under {TARGET_DROP_RATE:.0%} drops. Generated by",
        f"`manip-rubiks-eval-rotate` ({len(results)} episodes, {elapsed:.0f} s; chain axes `{axes}`).",
        "",
        "| Policy | Condition | Goal success (95% CI) | Goals / episode | Dropped | Timed out | Time to goal |",
        "|---|---|---|---|---|---|---|",
    ]
    for lab in labels:
        for c in conditions:
            s = table[lab][c]
            lo, hi = s["goal_success_ci95"]
            t = s["first_goal_median_s"]
            lines.append(
                f"| {lab} | {c} | {_pct(s['goal_success_rate'])} ({_pct(lo)}-{_pct(hi)}) | "
                f"{s['goals_per_episode']:.2f} | {s['drop_rate']:.0%} | {s['timeout_rate']:.0%} | "
                f"{'-' if np.isnan(t) else f'{t:.1f} s'} |"
            )
    lines += [
        "",
        "`turn_<c>` runs a single command per episode; `chain` runs random commands for 16 s, as in",
        "training. Time to goal is the median time to reach the first goal.",
        "",
    ]
    (out / "rotation_report.md").write_text("\n".join(lines))
    return report


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__, CONDITIONS, Path("reports/rotation"), "passive or label=run_dir")
    p.add_argument("--axes", default="xyz", help="hand axes of the chain condition's commands")
    args = p.parse_args(argv)
    specs = args.policies or ["passive"]
    conditions = tuple(args.conditions or CONDITIONS)
    t0 = time.time()
    results = run_benchmark(specs, args.episodes, args.workers, conditions, args.axes)
    report = write_report(results, specs, args.out, time.time() - t0, args.axes)
    for lab, rows in report["summary"].items():
        for c, s in rows.items():
            print(
                f"{lab:>10s} {c:>8s} goal success {_pct(s['goal_success_rate'])}  "
                f"goals/ep {s['goals_per_episode']:.2f}  drop {s['drop_rate']:.0%}  timeout {s['timeout_rate']:.0%}"
            )


if __name__ == "__main__":
    main()
