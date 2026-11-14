"""Gymnasium environments."""

import gymnasium as gym

from manip_rubiks.environments.grasp_env import GRASP_OBS_KEYS, GraspConfig, GraspEnv, GraspRewardWeights
from manip_rubiks.environments.hand_cube_env import OBS_KEYS, EnvConfig, HandCubeEnv

ENV_ID = "ManipRubiks/HandCube-v0"
GRASP_ENV_ID = "ManipRubiks/Grasp-v0"

for _id, _entry in (
    (ENV_ID, "manip_rubiks.environments.hand_cube_env:HandCubeEnv"),
    (GRASP_ENV_ID, "manip_rubiks.environments.grasp_env:GraspEnv"),
):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_entry)

__all__ = [
    "ENV_ID",
    "GRASP_ENV_ID",
    "GRASP_OBS_KEYS",
    "OBS_KEYS",
    "EnvConfig",
    "GraspConfig",
    "GraspEnv",
    "GraspRewardWeights",
    "HandCubeEnv",
]
