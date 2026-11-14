"""Scripted grasp baselines.

Both policies use only the non-privileged observation, so they are valid
blindfolded controllers and a reference point for learned grasps.
"""

from __future__ import annotations

import numpy as np

from manip_rubiks.simulation.robot_interface import JOINTS
from manip_rubiks.simulation.scene_builder import HOME_POSE

_FINGER_FLEXION = {"MCP_FE": 1.4, "PIP": 1.4, "DIP": 0.9}

# Flexion targets beyond the cube surface. Position actuators saturate at their
# force limits, so the fingers wrap the cube and press with bounded force.
POWER_GRASP: dict[str, float] = {
    **HOME_POSE,
    "thumb_CMC_FE": 1.7,
    "thumb_CMC_AA": -0.1,
    "thumb_MCP_FE": 0.9,
    "thumb_IP": 1.0,
    **{f"{f}_{j}": v for f in ("index", "middle", "ring", "pinky") for j, v in _FINGER_FLEXION.items()},
}


class PassivePolicy:
    """Zero action: the hand keeps its home cage."""

    def __init__(self, n_joints: int = len(JOINTS)) -> None:
        self.n = n_joints

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(self.n, dtype=np.float32)


class ScriptedGraspPolicy:
    """Drive the joint targets to a fixed grasp pose as fast as the action bound allows.

    ``k_a`` must match the environment's largest target change per step so the
    normalised action reaches the pose without overshoot.
    """

    def __init__(self, pose: dict[str, float] | None = None, k_a: float = 0.05) -> None:
        pose = pose or POWER_GRASP
        self.q_grasp = np.array([pose[j] for j in JOINTS])
        self.k_a = k_a

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.clip((self.q_grasp - obs["q_target"]) / self.k_a, -1.0, 1.0).astype(np.float32)
