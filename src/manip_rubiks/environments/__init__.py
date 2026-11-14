"""Gymnasium environments."""

import gymnasium as gym

from manip_rubiks.environments.hand_cube_env import OBS_KEYS, EnvConfig, HandCubeEnv

ENV_ID = "ManipRubiks/HandCube-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="manip_rubiks.environments.hand_cube_env:HandCubeEnv")

__all__ = ["ENV_ID", "OBS_KEYS", "EnvConfig", "HandCubeEnv"]
