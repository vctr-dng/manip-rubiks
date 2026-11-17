"""Reorientation benchmark: episode scoring and report generation."""

import json

import numpy as np

from manip_rubiks.environments import RotationConfig, RotationEnv
from manip_rubiks.evaluation.rotation_benchmark import main, run_episode
from manip_rubiks.policies import PassivePolicy


def test_passive_hand_times_out_on_a_single_turn():
    env = RotationEnv(RotationConfig(goal_timeout=10))
    r = run_episode(env, PassivePolicy(), "passive", "turn_+x", 100_000)
    assert r.outcome == "timed_out" and r.goals_reached == 0 and r.goals_failed == 1
    assert np.isnan(r.first_goal_s) and r.steps == 10
    env.close()


def test_report_is_written(tmp_path):
    main(["--policy", "passive", "--episodes", "2", "--condition", "turn_-z", "--workers", "1", f"--out={tmp_path}"])
    report = json.loads((tmp_path / "rotation_report.json").read_text())
    s = report["summary"]["passive"]["turn_-z"]
    assert s["episodes"] == 2 and s["goal_success_rate"] == 0.0
    assert "| passive | turn_-z |" in (tmp_path / "rotation_report.md").read_text()
