"""Machinery shared by the benchmarks: jobs, intervals, traces and the JSON report layout."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from manip_rubiks.evaluation.core import (
    EVAL_SEED_OFFSET,
    EpisodeTrace,
    base_parser,
    build_report,
    cached,
    make_jobs,
    policy_label,
    run_jobs,
    summary_table,
    wilson_interval,
    write_json_report,
)


@dataclass
class Row:
    policy: str
    condition: str
    seed: int
    value: float


def test_wilson_interval():
    lo, hi = wilson_interval(95, 100)
    assert lo < 0.95 < hi and lo > 0.88 and hi < 0.98
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(np.isnan(wilson_interval(0, 0)))


def test_jobs_cover_every_policy_condition_and_held_out_seed_once():
    jobs = make_jobs(["a", "b=run"], ["c1", "c2"], episodes=25, chunk=10)
    assert len(jobs) == 2 * 2 * 3  # 25 episodes -> chunks of 10, 10, 5
    assert jobs[0] == ("a", "c1", [EVAL_SEED_OFFSET + i for i in range(10)])
    assert jobs[2][2] == [EVAL_SEED_OFFSET + i for i in range(20, 25)]
    assert {j[0] for j in jobs} == {"a", "b=run"} and {j[1] for j in jobs} == {"c1", "c2"}
    assert make_jobs(["a"], ["c"], 3, extra=("xy",))[0] == ("a", "xy", "c", [EVAL_SEED_OFFSET + i for i in range(3)])


def test_run_jobs_keeps_job_order_in_process():
    jobs = make_jobs(["p"], ["c"], episodes=4, chunk=2)
    assert run_jobs(lambda job: [job[2][0], job[2][1]], jobs, workers=1) == [EVAL_SEED_OFFSET + i for i in range(4)]


def test_cached_builds_once():
    calls = []
    assert cached("core-test-key", lambda: calls.append(1) or "v") == "v"
    assert cached("core-test-key", lambda: calls.append(1) or "other") == "v"
    assert len(calls) == 1


def test_episode_trace_accumulates_reward_effort_and_misalignment():
    robot = SimpleNamespace(effort=lambda: np.array([1.0, -3.0]), force_limit=np.array([2.0, 4.0]))
    env = SimpleNamespace(robot=robot)
    trace = EpisodeTrace()
    trace.update(env, 1.5, {"misalignment": 0.2})
    trace.update(env, -0.5, {"misalignment": 0.1})
    assert trace.total_reward == pytest.approx(1.0)
    assert trace.max_misalignment == pytest.approx(0.2)
    assert trace.mean_effort == pytest.approx(0.625)


def test_report_layout(tmp_path):
    results = [Row("a", "c1", 100_000, 1.23456789), Row("a", "c1", 100_001, 2.0), Row("b", "c1", 100_000, 3.0)]
    table = summary_table(
        results, ["a", "b"], ["c1"], lambda rows: {"n": len(rows), "mean": float(np.mean([r.value for r in rows]))}
    )
    assert table == {"a": {"c1": {"n": 2, "mean": pytest.approx(1.617284)}}, "b": {"c1": {"n": 1, "mean": 3.0}}}
    report = build_report(["a", "b=run"], results, 1.5, table, {"target": 0.9})
    assert list(report) == ["policies", "target", "seeds", "wall_clock_s", "summary", "episodes"]
    assert report["policies"] == {"a": "a", "b": "b=run"} and report["seeds"] == "100000..100001"
    assert report["episodes"][0]["value"] == 1.23457  # rounded for compact reports
    write_json_report(tmp_path / "r.json", report)
    text = (tmp_path / "r.json").read_text()
    assert json.loads(text)["episodes"] == report["episodes"]
    assert text.count('{"policy"') == 3 and all(
        line.startswith("  {") for line in text.splitlines() if '"policy"' in line
    )


def test_policy_label_and_parser_defaults(tmp_path):
    assert policy_label("gpu=runs/x") == "gpu" and policy_label("passive") == "passive"
    args = base_parser("doc", ("c1", "c2"), tmp_path, "help").parse_args(["--condition", "c2"])
    assert args.conditions == ["c2"] and args.policies is None and args.episodes == 100 and args.out == tmp_path
