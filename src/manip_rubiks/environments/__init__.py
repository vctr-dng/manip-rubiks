"""Gymnasium environments."""

import gymnasium as gym

from manip_rubiks.environments.grasp_env import GRASP_OBS_KEYS, GraspConfig, GraspEnv, GraspRewardWeights
from manip_rubiks.environments.hand_cube_env import OBS_KEYS, EnvConfig, HandCubeEnv
from manip_rubiks.environments.rotation_env import (
    COMMAND_NAMES,
    ROTATION_OBS_KEYS,
    ROTATION_STUDENT_OBS_KEYS,
    TEACHER_KEYS,
    RotationConfig,
    RotationEnv,
    RotationRewardWeights,
)

ENV_ID = "ManipRubiks/HandCube-v0"
GRASP_ENV_ID = "ManipRubiks/Grasp-v0"
ROTATION_ENV_ID = "ManipRubiks/Rotate-v0"

for _id, _entry in (
    (ENV_ID, "manip_rubiks.environments.hand_cube_env:HandCubeEnv"),
    (GRASP_ENV_ID, "manip_rubiks.environments.grasp_env:GraspEnv"),
    (ROTATION_ENV_ID, "manip_rubiks.environments.rotation_env:RotationEnv"),
):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_entry)

__all__ = [
    "COMMAND_NAMES",
    "ENV_ID",
    "GRASP_ENV_ID",
    "GRASP_OBS_KEYS",
    "OBS_KEYS",
    "ROTATION_ENV_ID",
    "ROTATION_OBS_KEYS",
    "ROTATION_STUDENT_OBS_KEYS",
    "TEACHER_KEYS",
    "EnvConfig",
    "GraspConfig",
    "GraspEnv",
    "GraspRewardWeights",
    "HandCubeEnv",
    "RotationConfig",
    "RotationEnv",
    "RotationRewardWeights",
]
