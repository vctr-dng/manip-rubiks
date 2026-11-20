"""Grasp benchmark: seeding, scoring and report generation."""

import json

import numpy as np
import pytest

from manip_rubiks.evaluation.grasp_benchmark import (
    CONDITIONS,
    condition_options,
    run_benchmark,
    write_report,
)


def test_condition_options_are_seeded():
    assert condition_options("palm_down", 3) == condition_options("palm_down", 3)
    assert condition_options("palm_down", 3)["tilt"] == pytest.approx(np.pi)
    assert condition_options("sideways", 3)["azimuth"] != condition_options("sideways", 4)["azimuth"]
    assert condition_options("reorient_push", 0) == {}
    with pytest.raises(ValueError):
        condition_options("nope", 0)


def test_benchmark_scores_baselines_and_writes_report(tmp_path):
    results = run_benchmark(["passive", "scripted"], episodes=1, workers=1, conditions=("palm_down",))
    by_policy = {r.policy: r for r in results}
    assert by_policy["passive"].dropped and not by_policy["passive"].success
    assert not by_policy["scripted"].dropped and not by_policy["scripted"].layer_turned
    report = write_report(results, ["passive", "scripted"], tmp_path, elapsed=1.0)
    assert set(report["summary"]) == {"passive", "scripted"}
    assert json.loads((tmp_path / "grasp_report.json").read_text())["summary"]["passive"]["palm_down"]["episodes"] == 1
    assert "| scripted | palm_down |" in (tmp_path / "grasp_report.md").read_text()
    assert set(CONDITIONS) >= {"palm_down"}
