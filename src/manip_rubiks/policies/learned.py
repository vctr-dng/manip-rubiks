"""Blindfolded deployment of a trained PPO policy.

Only the actor runs. Observations are normalised with the statistics saved
during training. The critic's privileged entry, if the model has one, is
filled with zeros: the actor never reads it, and nothing privileged is ever
passed in.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from manip_rubiks.training.asymmetric import PRIVILEGED_KEY


class LearnedPolicy:
    """Callable ``obs -> action`` around a saved PPO model and its VecNormalize statistics."""

    def __init__(self, model_path: str | Path, vecnormalize_path: str | Path | None = None) -> None:
        model_path = Path(model_path)
        self.model = PPO.load(model_path, device="cpu")
        if vecnormalize_path is None:
            candidate = model_path.with_name("vecnormalize.pkl")
            vecnormalize_path = candidate if candidate.exists() else None
        self.normalizer = None
        if vecnormalize_path is not None:
            with open(vecnormalize_path, "rb") as f:
                self.normalizer = pickle.load(f)  # VecNormalize without its venv
            self.normalizer.training = False
        space = self.model.observation_space
        self.privileged_dim = int(space[PRIVILEGED_KEY].shape[0]) if PRIVILEGED_KEY in space.spaces else 0

    @classmethod
    def from_run(cls, run_dir: str | Path, best: bool = True) -> LearnedPolicy:
        run_dir = Path(run_dir)
        sub = run_dir / "best" if best and (run_dir / "best" / "best_model.zip").exists() else run_dir
        model = sub / ("best_model.zip" if sub.name == "best" else "model.zip")
        return cls(model, sub / "vecnormalize.pkl")

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if PRIVILEGED_KEY in obs:
            raise ValueError("a deployed policy must not receive privileged state")
        x = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
        if self.privileged_dim:
            x[PRIVILEGED_KEY] = np.zeros(self.privileged_dim, dtype=np.float32)
        if self.normalizer is not None:
            x = self.normalizer.normalize_obs(x)
        action, _ = self.model.predict(x, deterministic=True)
        return np.asarray(action, dtype=np.float32)
