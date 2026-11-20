"""Machinery shared by the held-out CPU benchmarks.

A benchmark (:mod:`~manip_rubiks.evaluation.grasp_benchmark`,
:mod:`~manip_rubiks.evaluation.rotation_benchmark`, and those of later tasks)
contributes what is specific to its task: the conditions, how one episode is
run and scored, how results are summarised and how the Markdown report reads.
This module provides the rest:

* policy specs (``passive``, ``scripted`` or ``label=run_dir``) and their loading,
* the held-out evaluation seeds, split into chunks of episodes and spread over
  worker processes, each caching its environment and policies,
* Wilson confidence intervals and the per-episode effort / misalignment trace,
* the JSON report layout (summary table plus one compact row per episode) and
  the command-line arguments every benchmark shares.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

EVAL_SEED_OFFSET = 100_000  # disjoint from training (0..) and in-training evaluation (10_000..) seeds

_CACHE: dict[str, Any] = {}


# ------------------------------------------------------------------ policies
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


def cached(key: str, factory: Callable[[], Any]) -> Any:
    """Per-process cache: workers build each environment and policy once."""
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


# ---------------------------------------------------------------- statistics
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


class EpisodeTrace:
    """Quantities every benchmark accumulates over an episode's steps."""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.efforts: list[float] = []
        self.max_misalignment = 0.0  # [rad]

    def update(self, env, reward: float, info: dict[str, Any]) -> None:
        self.total_reward += reward
        self.efforts.append(float(np.mean(np.abs(env.robot.effort()) / env.robot.force_limit)))
        self.max_misalignment = max(self.max_misalignment, info["misalignment"])

    @property
    def mean_effort(self) -> float:
        return float(np.mean(self.efforts))


# ----------------------------------------------------------------- execution
def make_jobs(
    specs: Sequence[str],
    conditions: Sequence[str],
    episodes: int,
    chunk: int = 10,
    extra: tuple = (),
) -> list[tuple]:
    """``(spec, *extra, condition, seeds)`` for every policy, condition and chunk of held-out seeds."""
    seeds = [EVAL_SEED_OFFSET + i for i in range(episodes)]
    return [(s, *extra, c, seeds[i : i + chunk]) for s in specs for c in conditions for i in range(0, episodes, chunk)]


def run_jobs(worker: Callable[[tuple], list], jobs: list[tuple], workers: int) -> list:
    """Run the jobs (in this process for ``workers <= 1``) and concatenate their results in job order."""
    if workers <= 1:
        return [r for job in jobs for r in worker(job)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        return [r for batch in pool.map(worker, jobs) for r in batch]


# ------------------------------------------------------------------- reports
def summary_table(
    results: list, labels: Sequence[str], conditions: Sequence[str], summarise: Callable[[list], dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{policy label: {condition: summary}}``."""
    return {
        lab: {c: summarise([r for r in results if r.policy == lab and r.condition == c]) for c in conditions}
        for lab in labels
    }


def build_report(
    specs: Sequence[str],
    results: list,
    elapsed: float,
    table: dict[str, dict[str, dict[str, Any]]],
    header: dict[str, Any],
) -> dict[str, Any]:
    """The JSON report: ``header`` (task targets and settings) sits between the policies and the seeds."""
    return {
        "policies": dict(zip((policy_label(s) for s in specs), specs, strict=True)),
        **header,
        "seeds": f"{min(r.seed for r in results)}..{max(r.seed for r in results)}",
        "wall_clock_s": elapsed,
        "summary": table,
        # One compact row per episode, floats rounded, so reports stay small enough to version.
        "episodes": [{k: round(v, 5) if isinstance(v, float) else v for k, v in asdict(r).items()} for r in results],
    }


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    """Indented summary followed by one line per episode."""
    text = json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=1)
    rows = ",\n  ".join(json.dumps(e, separators=(",", ":")) for e in report["episodes"])
    path.write_text(f'{text[:-2]},\n "episodes": [\n  {rows}\n ]\n}}\n')


def base_parser(doc: str | None, conditions: Sequence[str], out: Path, policy_help: str) -> argparse.ArgumentParser:
    """Arguments every benchmark shares (``--policy`` and ``--condition`` defaults are the caller's)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", dest="policies", help=policy_help)
    p.add_argument("--episodes", type=int, default=100, help="episodes per policy and condition")
    p.add_argument("--condition", action="append", dest="conditions", choices=conditions)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    p.add_argument("--out", type=Path, default=out)
    return p
