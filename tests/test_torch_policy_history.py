"""The deployed policy stacks its last observation frames like the training wrapper."""

import numpy as np
import torch

from manip_rubiks.policies.torch_policy import TorchGraspPolicy
from manip_rubiks.training.ppo_gpu import GaussianActor

LAYOUT = [("a", (2,)), ("b", (1,))]


def make_policy(tmp_path, history):
    dim = 3 * history
    actor = GaussianActor(dim, 2, (8,), 0.0)
    path = tmp_path / "policy.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "obs_layout": LAYOUT,
            "obs_dim": dim,
            "history": history,
            "act_dim": 2,
            "actor_sizes": (8,),
            "log_std_init": 0.0,
        },
        path,
    )
    return TorchGraspPolicy(path), actor


def obs(i):
    return {"a": np.array([i, i + 0.5], dtype=np.float32), "b": np.array([-i], dtype=np.float32)}


def test_policy_feeds_newest_first_and_refills_after_reset(tmp_path):
    policy, actor = make_policy(tmp_path, history=3)
    seen = []
    policy.actor.act = lambda x: seen.append(x[0].numpy().copy()) or torch.zeros(1, 2)
    policy.reset()
    for i in (1, 2, 3, 4):
        policy(obs(i))
    frame = lambda i: np.array([i, i + 0.5, -i], dtype=np.float32)  # noqa: E731
    np.testing.assert_allclose(seen[0], np.concatenate([frame(1)] * 3))
    np.testing.assert_allclose(seen[3], np.concatenate([frame(4), frame(3), frame(2)]))
    policy.reset()
    policy(obs(9))
    np.testing.assert_allclose(seen[4], np.concatenate([frame(9)] * 3))


def test_checkpoints_without_history_use_one_frame(tmp_path):
    policy, _ = make_policy(tmp_path, history=1)
    assert policy.history == 1
    assert policy(obs(1)).shape == (2,)
