"""Grasp benchmark: seeding, scoring and report generation."""

import json

import numpy as np
import pytest

from manip_rubiks.evaluation.grasp_benchmark import (
    ALL_CONDITIONS,
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


def test_push_diagnostic_conditions_only_change_the_pushes():
    assert set(ALL_CONDITIONS) - set(CONDITIONS) == {
        "reorient_push_0.5x",
        "reorient_push_1.5x",
        "reorient_push_force",
        "reorient_push_torque",
    }
    assert condition_options("reorient_push_1.5x", 7) == {"push_scale": 1.5}
    assert condition_options("reorient_push_torque", 7) == {"push_kind": "torque"}
    options = condition_options("reorient_push_force", 7)
    options["push_kind"] = "both"  # callers get a copy, not the shared table
    assert condition_options("reorient_push_force", 7) == {"push_kind": "force"}


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
    # The failure trace: a passive hand is never pushed here and ends with the cube gone.
    assert by_policy["passive"].steps_since_push == -1
    assert by_policy["passive"].last_push_force == 0.0
    assert by_policy["scripted"].final_tilt == pytest.approx(np.pi, abs=0.05)


def test_trace_records_the_last_push():
    results = run_benchmark(["passive"], episodes=2, workers=1, conditions=("reorient_push_1.5x",))
    assert all(r.condition == "reorient_push_1.5x" for r in results)
    pushed = [r for r in results if r.steps_since_push >= 0]
    for r in pushed:
        assert 0 < r.last_push_force <= 1.5 * 1.5 + 1e-6
        assert r.last_push_torque <= 1.5 * 0.02 + 1e-9
