"""Tactile-signal and episode-logic validation checks."""

from manip_rubiks.validation.episodes import check_episode_logic
from manip_rubiks.validation.tactile import check_tactile_signals


def test_tactile_signals_check():
    result = check_tactile_signals()
    assert result.passed, result.metrics


def test_episode_logic_check():
    result = check_episode_logic(rollout_steps=10)
    assert result.passed, result.metrics
