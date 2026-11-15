"""Periodic training artifacts: episode outcomes, statistics, files, reproducibility, upload schedule."""

import csv
import json

import mlflow
import numpy as np
import pytest

from manip_rubiks.environments import GraspEnv
from manip_rubiks.training.artifacts import (
    ArtifactLogger,
    DistributionStats,
    normalizer_stats,
    outcome,
    pick_video_episodes,
    render_gif,
    write_csv,
    write_json,
    write_repro,
)


def test_outcome_labels():
    assert outcome(True, False, False, False) == "success"
    assert outcome(False, True, True, True) == "unstable"
    assert outcome(False, True, True, False) == "dropped"
    assert outcome(False, False, True, False) == "layer_turned"
    assert outcome(False, False, False, False) == "misaligned"


def test_distribution_stats():
    stats = DistributionStats(act_dim=2, tactile_dim=3, bins=4)
    stats.add(np.array([[-1.0, 0.1], [1.0, 0.3]]), np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 0.0]]))
    stats.add(np.zeros((0, 2)), np.zeros((0, 3)))  # empty batches are ignored
    s = stats.summary()
    assert s["steps"] == 2
    np.testing.assert_allclose(s["action"]["mean"], [0.0, 0.2])
    np.testing.assert_allclose(s["action"]["std"], [1.0, 0.1])
    np.testing.assert_allclose(s["action"]["saturation_fraction"], [1.0, 0.0])
    np.testing.assert_array_equal(s["action"]["histogram"]["counts"], [[1, 0, 0, 1], [0, 0, 2, 0]])
    np.testing.assert_allclose(s["tactile"]["active_fraction"], [0.0, 1.0, 0.5])
    np.testing.assert_allclose(s["tactile"]["mean"], [0.0, 2.0, 1.0])


def test_normalizer_stats():
    out = normalizer_stats({"q": (np.zeros((2, 2)), np.full((2, 2), 4.0))})
    np.testing.assert_allclose(out["q"]["std"], [2.0] * 4)


def test_json_and_csv_files(tmp_path):
    write_json(tmp_path / "a" / "d.json", {"x": np.arange(3), "y": np.float32(0.5)})
    assert json.loads((tmp_path / "a" / "d.json").read_text()) == {"x": [0, 1, 2], "y": 0.5}
    write_csv(tmp_path / "e.csv", [{"a": 1, "b": True}, {"a": 2.5, "c": "x"}])
    rows = list(csv.DictReader((tmp_path / "e.csv").open()))
    assert rows == [{"a": "1", "b": "1", "c": ""}, {"a": "2.5", "b": "", "c": "x"}]


def test_pick_video_episodes():
    assert pick_video_episodes(["dropped", "success", "success", "misaligned"]) == [1, 0]
    assert pick_video_episodes(["success", "success"]) == [0]
    assert pick_video_episodes([]) == []


def test_repro_snapshot_records_the_commit(tmp_path):
    tags = write_repro(tmp_path)
    info = json.loads((tmp_path / "repro.json").read_text())
    assert info["git"]["commit"] == tags["git.commit"] and len(tags["git.commit"]) == 40
    assert tags["git.dirty"] in ("true", "false")
    assert (tmp_path / "git.diff").exists() == info["git"]["dirty"]
    assert (tmp_path / "uv.lock").exists()
    assert "mujoco" in info["packages"] and info["menagerie_commit"]


def test_repro_snapshot_outside_git(tmp_path):
    tags = write_repro(tmp_path / "out", source=tmp_path)
    assert tags == {"git.commit": "unknown"}
    assert json.loads((tmp_path / "out" / "repro.json").read_text())["git"] is None


def test_schedule():
    log = ArtifactLogger("unused", total_timesteps=1000, milestones=4, video_every=3)
    assert [log.next_eval()[1] for _ in range(7)] == [True, False, False, True, False, False, True]
    assert log.milestones_reached(200) == []
    assert log.milestones_reached(600) == [250, 500]
    assert log.milestones_reached(600) == []
    assert log.milestones_reached(5000) == [750, 1000]
    assert ArtifactLogger("unused", 1000, milestones=0, video_every=0).next_eval() == (0, False)


@pytest.fixture
def file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test")
    yield
    mlflow.set_tracking_uri(None)


def test_logger_uploads_periodic_and_final_files(file_store, tmp_path):
    run_dir = tmp_path / "run"
    log = ArtifactLogger(run_dir, total_timesteps=100, milestones=2)
    (run_dir / "latest").mkdir(parents=True)
    (run_dir / "latest" / "model.zip").write_text("m")
    (run_dir / "best").mkdir()
    (run_dir / "best" / "model.zip").write_text("b")
    (run_dir / "config.json").write_text("{}")
    write_csv(log.eval_dir(60) / "episodes.csv", [{"a": 1}])
    with mlflow.start_run() as run:
        log.start()
        log.checkpoint(run_dir / "latest", 60, improved=True, best_dir=run_dir / "best")
        log.checkpoint(run_dir / "latest", 70, improved=False, best_dir=run_dir / "best")
        log.log_eval(60)
        log.finalize()
    client = mlflow.MlflowClient()

    def ls(path=None):
        return {a.path for a in client.list_artifacts(run.info.run_id, path)}

    assert ls() == {"repro", "checkpoints", "eval", "run"}
    assert ls("checkpoints") == {"checkpoints/latest", "checkpoints/best", "checkpoints/milestones"}
    assert ls("checkpoints/milestones") == {"checkpoints/milestones/000000060"}
    assert ls("eval/000000060") == {"eval/000000060/episodes.csv"}
    assert ls("run") == {"run/best", "run/config.json"}  # periodic dirs are not uploaded twice
    assert client.get_run(run.info.run_id).data.tags["git.commit"] != "unknown"


def test_render_gif(tmp_path):
    from PIL import Image

    env = GraspEnv(render_mode="rgb_array")
    env.reset(seed=0)
    qpos = []
    for _ in range(4):  # distinct frames (Pillow merges identical consecutive ones)
        env.step(np.ones(22))
        qpos.append(env.data.qpos.copy())
    path = render_gif(env, qpos, tmp_path / "v.gif", stride=2, width=160)
    with Image.open(path) as gif:
        assert gif.size == (160, 120) and gif.n_frames == 2
    env.close()
