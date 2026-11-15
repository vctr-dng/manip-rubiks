"""Blindfolded deployment of a policy trained on the GPU (:mod:`manip_rubiks.training.train_grasp_gpu`).

Only the actor and its observation normaliser are loaded; the critic and its
privileged inputs are discarded. The policy runs on the CPU against the
reference :class:`~manip_rubiks.environments.grasp_env.GraspEnv`, flattening
its observation in the layout recorded in the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from manip_rubiks.training.asymmetric import PRIVILEGED_KEY
from manip_rubiks.training.ppo_gpu import GaussianActor


class TorchGraspPolicy:
    """Callable ``obs -> action`` around a GPU-trained checkpoint (``policy.pt``)."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.layout = [(k, tuple(s)) for k, s in ckpt["obs_layout"]]
        self.actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"], tuple(ckpt["actor_sizes"]), ckpt["log_std_init"])
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval().to(device)
        self.device = device
        self.timesteps = int(ckpt.get("timesteps", 0))

    @classmethod
    def from_run(cls, run_dir: str | Path, best: bool = True) -> TorchGraspPolicy:
        run_dir = Path(run_dir)
        path = run_dir / "best" / "policy.pt"
        return cls(path if best and path.exists() else run_dir / "policy.pt")

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if PRIVILEGED_KEY in obs:
            raise ValueError("a deployed policy must not receive privileged state")
        flat = []
        for key, shape in self.layout:
            x = np.asarray(obs[key], dtype=np.float32)
            if x.shape != shape:
                raise ValueError(f"observation {key!r} has shape {x.shape}, the policy expects {shape}")
            flat.append(x.reshape(-1))
        x = torch.as_tensor(np.concatenate(flat), device=self.device)[None]
        action = self.actor.act(x)[0].cpu().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)
