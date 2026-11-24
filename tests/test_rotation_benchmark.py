"""Reorientation benchmark: episode scoring and report generation."""

import json

import numpy as np

from manip_rubiks.environments import RotationConfig, RotationEnv
from manip_rubiks.evaluation.rotation_benchmark import CHAIN_COMMANDS, command_plan, main, run_episode
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


ALWAYS_REACHED = {
    "goal_tolerance": 4.0,
    "easy_tolerance": 4.0,
}  # wider than any angle: a goal is reached after the hold


def test_command_plan_is_seeded_and_covers_all_axes():
    assert command_plan("chain", 3) is None
    assert command_plan("turn_-z", 3) == [5]
    assert command_plan("turn_+z_composed", 3) == [4]
    plan = command_plan("chain_composed", 3)
    assert plan == command_plan("chain_composed", 3) and len(plan) == CHAIN_COMMANDS
    assert plan != command_plan("chain_composed", 4)
    assert {c // 2 for c in sum((command_plan("chain_composed", s) for s in range(20)), [])} == {0, 1, 2}


def test_composed_z_turn_counts_one_command_for_three_goals():
    env = RotationEnv(RotationConfig(**ALWAYS_REACHED))
    r = run_episode(env, PassivePolicy(), "passive", "turn_+z_composed", 100_000)
    assert r.outcome == "success" and r.goals_reached == 3
    assert (r.commands_reached, r.commands_failed) == (1, 0)
    plain = run_episode(env, PassivePolicy(), "passive", "turn_+x", 100_000)
    assert plain.outcome == "success" and (plain.goals_reached, plain.commands_reached) == (1, 1)
    env.close()


def test_composed_chain_counts_commands_not_goals():
    env = RotationEnv(RotationConfig(**ALWAYS_REACHED))
    r = run_episode(env, PassivePolicy(), "passive", "chain_composed", 100_000)
    plan = command_plan("chain_composed", 100_000)
    assert r.commands_reached == CHAIN_COMMANDS  # the whole queue completes, then the environment's own chain
    assert r.goals_reached >= CHAIN_COMMANDS + 2 * sum(c >= 4 for c in plan)
    env.close()


def test_passive_hand_fails_a_composed_z_turn_on_its_first_sub_goal():
    env = RotationEnv(RotationConfig(goal_timeout=10))
    r = run_episode(env, PassivePolicy(), "passive", "turn_-z_composed", 100_000)
    assert r.outcome == "timed_out" and r.goals_reached == 0
    assert (r.commands_reached, r.commands_failed) == (0, 1)
    env.close()


def test_composed_report_has_command_success(tmp_path):
    args = ["--policy", "passive", "--episodes", "2", "--condition", "turn_+z_composed", "--workers", "1"]
    main([*args, "--z-variant", "2", f"--out={tmp_path}"])
    report = json.loads((tmp_path / "rotation_report.json").read_text())
    s = report["summary"]["passive"]["turn_+z_composed"]
    assert report["z_variant"] == 2 and s["command_attempts"] == 2 and s["command_success_rate"] == 0.0
    assert "Command success" in (tmp_path / "rotation_report.md").read_text()
