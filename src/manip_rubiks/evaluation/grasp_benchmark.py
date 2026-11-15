"""Grasp-stabilisation benchmark: sustained blindfolded holds under wrist reorientation and pushes.

Every policy runs the same held-out seeds in each condition:

=================  ==========================================================
``palm_down``      one reorientation to palm-down, held; no pushes
``sideways``       one reorientation to a 90 degree tilt in a random direction; no pushes
``reorient``       random reorientation sequence up to palm-down; no pushes
``reorient_push``  random reorientations plus random pushes (the training distribution)
=================  ==========================================================

An episode succeeds when it reaches the time limit with the cube in the hand,
the puzzle configuration unchanged, and every cubie within the alignment
tolerance (``GraspEnv`` ``is_success``). Policies receive only the
non-privileged observation; ground truth is used for scoring only.

Example::

    uv run manip-rubiks-eval-grasp --policy passive --policy scripted \\
        --policy learned=runs/grasp/ppo_asym_s0 --episodes 100
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

CONDITIONS: tuple[str, ...] = ("palm_down", "sideways", "reorient", "reorient_push")
EVAL_SEED_OFFSET = 100_000  # disjoint from training (0..) and in-training evaluation (10_000..) seeds
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
    raise ValueError(f"unknown condition {condition!r}")


def build_policy(spec: str):
    """``passive``, ``scripted`` or ``<label>=<run dir, model.zip or policy.pt>``.

    Run directories of GPU training (``policy.pt``) and of Stable-Baselines3
    training (``model.zip``) are both recognised.
    """
    from manip_rubiks.policies import PassivePolicy, ScriptedGraspPolicy

    if spec == "passive":
        return PassivePolicy()
    if spec == "scripted":
        return ScriptedGraspPolicy()
    _, _, path = spec.partition("=")
    p = Path(path)
    if p.suffix == ".pt" or (p / "policy.pt").exists() or (p / "best" / "policy.pt").exists():
        from manip_rubiks.policies.torch_policy import TorchGraspPolicy

        return TorchGraspPolicy(p) if p.suffix == ".pt" else TorchGraspPolicy.from_run(p)
    from manip_rubiks.policies.learned import LearnedPolicy

    return LearnedPolicy(p) if p.suffix == ".zip" else LearnedPolicy.from_run(p)


def policy_label(spec: str) -> str:
    return spec.partition("=")[0]


_CACHE: dict[str, Any] = {}


def run_episode(env, policy, label: str, condition: str, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed, options=condition_options(condition, seed))
    policy.reset()
    total, efforts, max_mis = 0.0, [], 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        efforts.append(float(np.mean(np.abs(env.robot.effort()) / env.robot.force_limit)))
        max_mis = max(max_mis, info["misalignment"])
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
        max_misalignment=float(max_mis),
        final_displacement=float(info["displacement"]),
        mean_effort=float(np.mean(efforts)),
        episode_return=float(total),
    )


def _worker(job: tuple[str, str, list[int]]) -> list[EpisodeResult]:
    from manip_rubiks.environments import GraspEnv

    spec, condition, seeds = job
    if "env" not in _CACHE:
        _CACHE["env"] = GraspEnv()
    if spec not in _CACHE:
        _CACHE[spec] = build_policy(spec)
    return [run_episode(_CACHE["env"], _CACHE[spec], policy_label(spec), condition, s) for s in seeds]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


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
    seeds = [EVAL_SEED_OFFSET + i for i in range(episodes)]
    jobs = [(s, c, seeds[i : i + chunk]) for s in specs for c in conditions for i in range(0, episodes, chunk)]
    if workers <= 1:
        return [r for job in jobs for r in _worker(job)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        return [r for batch in pool.map(_worker, jobs) for r in batch]


def write_report(results: list[EpisodeResult], specs: list[str], out: Path, elapsed: float) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    labels = [policy_label(s) for s in specs]
    conditions = [c for c in CONDITIONS if any(r.condition == c for r in results)]
    table = {
        lab: {c: summarise([r for r in results if r.policy == lab and r.condition == c]) for c in conditions}
        for lab in labels
    }
    report = {
        "policies": dict(zip(labels, specs, strict=True)),
        "target_success_rate": TARGET_SUCCESS,
        "seeds": f"{min(r.seed for r in results)}..{max(r.seed for r in results)}",
        "wall_clock_s": elapsed,
        "summary": table,
        # One compact row per episode, floats rounded, so reports stay small enough to version.
        "episodes": [{k: round(v, 5) if isinstance(v, float) else v for k, v in asdict(r).items()} for r in results],
    }
    text = json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=1)
    rows = ",\n  ".join(json.dumps(e, separators=(",", ":")) for e in report["episodes"])
    (out / "grasp_report.json").write_text(f'{text[:-2]},\n "episodes": [\n  {rows}\n ]\n}}\n')

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
        "random pushes on the cube (up to 1.5 N and 0.02 N m), as in training. Effort is the mean",
        "actuator force as a fraction of its limit.",
        "",
    ]
    (out / "grasp_report.md").write_text("\n".join(lines))
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", dest="policies", help="passive, scripted or label=run_dir")
    p.add_argument("--episodes", type=int, default=100, help="episodes per policy and condition")
    p.add_argument("--condition", action="append", dest="conditions", choices=CONDITIONS)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    p.add_argument("--out", type=Path, default=Path("reports/grasp"))
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
