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

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manip_rubiks.environments.rotation_env import COMMAND_NAMES
from manip_rubiks.evaluation.grasp_benchmark import build_policy, policy_label, wilson_interval

TURN_CONDITIONS: tuple[str, ...] = tuple(f"turn_{c}" for c in COMMAND_NAMES)
CONDITIONS: tuple[str, ...] = (*TURN_CONDITIONS, "chain")
EVAL_SEED_OFFSET = 100_000  # disjoint from training (0..) and in-training evaluation (10_000..) seeds
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
    total, efforts, max_mis, first_goal = 0.0, [], 0.0, float("nan")
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        efforts.append(float(np.mean(np.abs(env.robot.effort()) / env.robot.force_limit)))
        max_mis = max(max_mis, info["misalignment"])
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
        max_misalignment=float(max_mis),
        final_displacement=float(info["displacement"]),
        mean_effort=float(np.mean(efforts)),
        episode_return=float(total),
    )


_CACHE: dict[str, Any] = {}


def _worker(job: tuple[str, str, str, list[int]]) -> list[EpisodeResult]:
    from manip_rubiks.environments import RotationConfig, RotationEnv

    spec, axes, condition, seeds = job
    key = f"env:{axes}"
    if key not in _CACHE:
        _CACHE[key] = RotationEnv(RotationConfig(goal_axes=axes))
    if spec not in _CACHE:
        _CACHE[spec] = build_policy(spec)
    return [run_episode(_CACHE[key], _CACHE[spec], policy_label(spec), condition, s) for s in seeds]


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
    seeds = [EVAL_SEED_OFFSET + i for i in range(episodes)]
    jobs = [(s, axes, c, seeds[i : i + chunk]) for s in specs for c in conditions for i in range(0, episodes, chunk)]
    if workers <= 1:
        return [r for job in jobs for r in _worker(job)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        return [r for batch in pool.map(_worker, jobs) for r in batch]


def _pct(x: float) -> str:
    return "-" if np.isnan(x) else f"{x:.0%}"


def write_report(
    results: list[EpisodeResult], specs: list[str], out: Path, elapsed: float, axes: str
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    labels = [policy_label(s) for s in specs]
    conditions = [c for c in CONDITIONS if any(r.condition == c for r in results)]
    table = {
        lab: {c: summarise([r for r in results if r.policy == lab and r.condition == c]) for c in conditions}
        for lab in labels
    }
    report = {
        "policies": dict(zip(labels, specs, strict=True)),
        "chain_axes": axes,
        "target_goal_success_rate": TARGET_GOAL_SUCCESS,
        "target_drop_rate": TARGET_DROP_RATE,
        "seeds": f"{min(r.seed for r in results)}..{max(r.seed for r in results)}",
        "wall_clock_s": elapsed,
        "summary": table,
        "episodes": [{k: round(v, 5) if isinstance(v, float) else v for k, v in asdict(r).items()} for r in results],
    }
    text = json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=1)
    rows = ",\n  ".join(json.dumps(e, separators=(",", ":")) for e in report["episodes"])
    (out / "rotation_report.json").write_text(f'{text[:-2]},\n "episodes": [\n  {rows}\n ]\n}}\n')

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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", dest="policies", help="passive or label=run_dir")
    p.add_argument("--episodes", type=int, default=100, help="episodes per policy and condition")
    p.add_argument("--condition", action="append", dest="conditions", choices=CONDITIONS)
    p.add_argument("--axes", default="xyz", help="hand axes of the chain condition's commands")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    p.add_argument("--out", type=Path, default=Path("reports/rotation"))
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
