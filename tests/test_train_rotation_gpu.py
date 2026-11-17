"""End-to-end check of the GPU reorientation training CLI (slow)."""

import csv

import mlflow
import numpy as np
import pytest
import torch

from manip_rubiks.environments.rotation_env import RotationEnv


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_rotation_training_cli_logs_goal_metrics_and_deploys_on_the_cpu(tmp_path, monkeypatch):
    from manip_rubiks.policies.torch_policy import TorchGraspPolicy
    from manip_rubiks.training.train_rotation_gpu import main as train_main

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlruns = tmp_path / "mlruns"
    train_main(
        [
            "--name=t",
            f"--out={tmp_path}",
            f"--mlflow-uri=file://{mlruns}",
            "--experiment=test",
            "--timesteps=2048",
            "--num-envs=64",
            "--n-steps=16",
            "--eval-every=1024",
            "--eval-envs=8",
            "--bank-size=64",
            "--video-episodes=4",
            "--axes=x",
        ]
    )
    run = tmp_path / "t"
    for f in ("policy.pt", "best/policy.pt", "config.json", "summary.json"):
        assert (run / f).exists(), f
    policy = TorchGraspPolicy.from_run(run)
    assert [k for k, _ in policy.layout][-2:] == ["command", "teacher"]
    env = RotationEnv()
    obs, _ = env.reset(seed=0)
    a = policy(obs)
    assert a.shape == (22,) and np.all(np.abs(a) <= 1.0)
    mlflow.set_tracking_uri(f"file://{mlruns}")
    (logged,) = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert logged.info.status == "FINISHED" and logged.data.tags["task"] == "rotation"
    for k in ("eval/goals_per_episode", "eval/goal_success_rate", "eval/timed_out_rate", "eval/success_rate"):
        assert k in logged.data.metrics, k
    rows = list(csv.DictReader((run / "eval" / "000002048" / "episodes.csv").open()))
    assert len(rows) == 8 and {"goals_reached", "theta", "outcome"} <= set(rows[0])
    assert all(
        r["outcome"] in {"success", "timed_out", "dropped", "unstable", "layer_turned", "misaligned"} for r in rows
    )
    mlflow.set_tracking_uri(None)
