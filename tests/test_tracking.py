"""MLflow tracking: SB3 logger output, parameter flattening, server check."""

import mlflow
import numpy as np
import pytest
from stable_baselines3.common.logger import Logger

from manip_rubiks.training import tracking
from manip_rubiks.training.tracking import (
    MLflowOutputFormat,
    check_server,
    docker_env_file,
    flatten_params,
    uri_from_env_file,
)


@pytest.fixture
def file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # the client lacks sqlalchemy; tests need no server
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test")
    yield
    mlflow.set_tracking_uri(None)


def test_logger_dumps_become_mlflow_metrics(file_store, tmp_path):
    logger = Logger(str(tmp_path), [MLflowOutputFormat()])
    with mlflow.start_run() as run:
        logger.record("eval/mean_reward", 12.5)
        logger.record("eval/success_rate", np.float32(0.75))
        logger.record("episode/dropped", True)
        logger.record("time/fps", 1200)
        logger.record("train/note", "text is skipped")
        logger.record("train/hidden", 1.0, exclude="mlflow")
        logger.record("train/bad", float("nan"))
        logger.dump(step=4096)
        logger.record("eval/mean_reward", 20.0)
        logger.dump(step=8192)
    client = mlflow.MlflowClient()
    metrics = client.get_run(run.info.run_id).data.metrics
    assert metrics == {"eval/mean_reward": 20.0, "eval/success_rate": 0.75, "episode/dropped": 1.0, "time/fps": 1200.0}
    history = client.get_metric_history(run.info.run_id, "eval/mean_reward")
    assert [(m.step, m.value) for m in history] == [(4096, 12.5), (8192, 20.0)]


def test_flatten_params():
    flat = flatten_params(
        {"args": {"lr": 3e-4, "name": "x"}, "cfg": {"env": {"noise": (0.1, 0.2)}, "w": np.array([1, 2])}}
    )
    assert flat == {"args.lr": 3e-4, "args.name": "x", "cfg.env.noise": "[0.1, 0.2]", "cfg.w": "[1, 2]"}


def test_unreachable_server_fails_fast():
    with pytest.raises(RuntimeError, match="docker/up.sh"):
        check_server("http://127.0.0.1:9", timeout=0.5)
    check_server("file:///tmp/anything")  # non-HTTP stores are not checked


def _env(tmp_path, text):
    path = tmp_path / "docker.env"
    path.write_text(text)
    return path


def test_uri_from_docker_env_file(tmp_path):
    bind = "MLFLOW_BIND_ADDRESS=10.1.2.3\nMLFLOW_PORT=5050\n"
    assert uri_from_env_file(_env(tmp_path, bind)) == "http://10.1.2.3:5050"
    assert uri_from_env_file(_env(tmp_path, "MLFLOW_BIND_ADDRESS=0.0.0.0\n")) == "http://127.0.0.1:5000"
    explicit = "MLFLOW_TRACKING_URI=http://tracker:1\nMLFLOW_BIND_ADDRESS=10.1.2.3\n"
    assert uri_from_env_file(_env(tmp_path, explicit)) == "http://tracker:1"
    assert uri_from_env_file(_env(tmp_path, "MLFLOW_BIND_ADDRESS=\nMLFLOW_PORT=5000\n")) is None


def test_docker_env_file_found_from_a_worktree(tmp_path):
    main, wt = tmp_path / "main", tmp_path / "wt"
    (main / "docker").mkdir(parents=True)
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / "docker" / ".env").write_text("MLFLOW_BIND_ADDRESS=10.1.2.3\n")
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert docker_env_file(wt) == main / "docker" / ".env"
    assert docker_env_file(main) == main / "docker" / ".env"
    assert docker_env_file(tmp_path / "elsewhere") is None


def test_env_file_takes_precedence_over_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-shell:1")
    monkeypatch.setattr(tracking, "docker_env_file", lambda: _env(tmp_path, "MLFLOW_BIND_ADDRESS=10.1.2.3\n"))
    assert tracking.tracking_uri() == "http://10.1.2.3:5000"
    monkeypatch.setattr(tracking, "docker_env_file", lambda: None)
    assert tracking.tracking_uri() == "http://from-shell:1"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert tracking.tracking_uri() == tracking.DEFAULT_TRACKING_URI
