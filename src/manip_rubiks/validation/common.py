"""Shared helpers for the simulation validation checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def warnings_count(d: mujoco.MjData) -> int:
    return int(sum(w.number for w in d.warning))


def finite(d: mujoco.MjData) -> bool:
    return bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all())
