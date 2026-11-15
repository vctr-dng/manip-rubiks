"""Artifacts uploaded to MLflow while a grasp policy trains.

Both trainers write the same layout, locally under the run directory and in
the MLflow run:

=====================================  =====================================================
``repro/``                             once at start: ``repro.json`` (git commit, branch,
                                       dirty flag, package versions, Menagerie commit),
                                       ``git.diff`` when the tree is dirty, ``uv.lock``
``checkpoints/latest/``                after every evaluation, overwritten: everything
                                       needed to deploy or resume the policy
``checkpoints/best/``                  whenever the evaluation score improves
``checkpoints/milestones/<steps>/``    the latest checkpoint at the first evaluation past
                                       each ``1/--milestones`` of the budget, kept
``eval/<steps>/episodes.csv``          one row per evaluation episode (outcome, return,
                                       per-term reward sums, end-of-episode metrics)
``eval/<steps>/distributions.json``    observation normaliser, policy standard deviation,
                                       action and tactile statistics
``eval/<steps>/videos/*.gif``          every ``--video-every`` evaluations: one successful
                                       and one failed full-difficulty episode
=====================================  =====================================================

Upload failures are reported and never interrupt training.
"""

from __future__ import annotations

import contextlib
import csv
import json
import shutil
import subprocess
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from manip_rubiks.simulation.assets import MENAGERIE_COMMIT

PACKAGES = ("mujoco", "mujoco-warp", "warp-lang", "torch", "stable-baselines3", "gymnasium", "numpy", "mlflow-skinny")
OUTCOMES = ("success", "unstable", "dropped", "layer_turned", "misaligned")


def outcome(success: bool, dropped: bool, layer_turned: bool, unstable: bool) -> str:
    """Single label for how an episode ended (``misaligned``: kept the cube, but cubies off-grid)."""
    if success:
        return "success"
    if unstable:
        return "unstable"
    if dropped:
        return "dropped"
    if layer_turned:
        return "layer_turned"
    return "misaligned"


# ---------------------------------------------------------------- files
def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))
    return path


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write dict rows; columns are the union of the keys, in first-seen order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})
    return path


def _csv_value(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return v


# ---------------------------------------------------------- statistics
class DistributionStats:
    """Accumulate action and tactile statistics over evaluation steps."""

    def __init__(self, act_dim: int, tactile_dim: int, bins: int = 20, saturation: float = 0.99) -> None:
        self.edges = np.linspace(-1.0, 1.0, bins + 1)
        self.saturation = saturation
        self.n = 0
        self.a_sum, self.a_sq = np.zeros(act_dim), np.zeros(act_dim)
        self.a_sat = np.zeros(act_dim)
        self.a_hist = np.zeros((act_dim, bins), dtype=np.int64)
        self.t_sum, self.t_sq = np.zeros(tactile_dim), np.zeros(tactile_dim)
        self.t_active = np.zeros(tactile_dim)

    def add(self, actions: np.ndarray, tactile: np.ndarray) -> None:
        """``actions``: (batch, act_dim), ``tactile``: (batch, tactile_dim) raw (unnormalised) readings."""
        if not len(actions):
            return
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(len(actions), -1), -1.0, 1.0)
        t = np.asarray(tactile, dtype=np.float64).reshape(len(tactile), -1)
        self.n += len(a)
        self.a_sum += a.sum(0)
        self.a_sq += (a**2).sum(0)
        self.a_sat += (np.abs(a) >= self.saturation).sum(0)
        idx = np.clip(np.searchsorted(self.edges, a, side="right") - 1, 0, len(self.edges) - 2)
        for j in range(a.shape[1]):
            self.a_hist[j] += np.bincount(idx[:, j], minlength=self.a_hist.shape[1])
        self.t_sum += t.sum(0)
        self.t_sq += (t**2).sum(0)
        self.t_active += (np.abs(t) > 1e-6).sum(0)

    def summary(self) -> dict[str, Any]:
        n = max(self.n, 1)
        a_mean, t_mean = self.a_sum / n, self.t_sum / n
        return {
            "steps": self.n,
            "action": {
                "mean": a_mean,
                "std": np.sqrt(np.maximum(self.a_sq / n - a_mean**2, 0.0)),
                "saturation_fraction": self.a_sat / n,
                "histogram": {"edges": self.edges, "counts": self.a_hist},
            },
            "tactile": {
                "mean": t_mean,
                "std": np.sqrt(np.maximum(self.t_sq / n - t_mean**2, 0.0)),
                "active_fraction": self.t_active / n,
            },
        }


def normalizer_stats(stats: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """``{key: (mean, var)}`` -> ``{key: {"mean": ..., "std": ...}}`` (flattened)."""
    return {
        k: {"mean": np.asarray(m, dtype=np.float64).ravel(), "std": np.sqrt(np.asarray(v, dtype=np.float64).ravel())}
        for k, (m, v) in stats.items()
    }


# -------------------------------------------------------------- videos
def pick_video_episodes(outcomes: Sequence[str]) -> list[int]:
    """Indices of the first success and the first failure, when present."""
    picked = []
    for wanted in (True, False):
        i = next((i for i, o in enumerate(outcomes) if (o == "success") == wanted), None)
        if i is not None:
            picked.append(i)
    return picked


def render_gif(env, qpos: np.ndarray, path: Path, stride: int = 2, width: int = 320) -> Path:
    """Replay joint positions in a ``render_mode="rgb_array"`` environment and save an animated GIF.

    Only ``env.data.qpos`` is overwritten (then ``mj_forward``), so ``env``
    should be dedicated to rendering.
    """
    import mujoco
    from PIL import Image

    frames = []
    for q in np.asarray(qpos)[::stride]:
        env.data.qpos[:] = q
        mujoco.mj_forward(env.model, env.data)
        img = Image.fromarray(env.render())
        frames.append(img.resize((width, round(img.height * width / img.width)), Image.Resampling.BILINEAR))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_ms = round(1000 * stride * env.cfg.control_dt)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True)
    return path


# ------------------------------------------------------ reproducibility
def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def write_repro(dest: Path, source: Path | None = None) -> dict[str, str]:
    """Snapshot code and environment versions into ``dest``; return MLflow tags.

    ``source`` is any path inside the repository (default: this package).
    """
    source = source or Path(__file__).resolve().parent
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "command": sys.argv,
        "menagerie_commit": MENAGERIE_COMMIT,
        "packages": {},
    }
    for pkg in PACKAGES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            info["packages"][pkg] = metadata.version(pkg)
    tags = {"git.commit": "unknown"}
    try:
        root = Path(_run(["git", "rev-parse", "--show-toplevel"], source).strip())
        status = _run(["git", "status", "--porcelain"], root)
        info["git"] = {
            "commit": _run(["git", "rev-parse", "HEAD"], root).strip(),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root).strip(),
            "dirty": bool(status.strip()),
            "status": status.splitlines(),
        }
        dest.mkdir(parents=True, exist_ok=True)
        if info["git"]["dirty"]:
            (dest / "git.diff").write_text(_run(["git", "diff", "HEAD", "--binary"], root))
        if (root / "uv.lock").exists():
            shutil.copy2(root / "uv.lock", dest / "uv.lock")
        tags = {
            "git.commit": info["git"]["commit"],
            "git.branch": info["git"]["branch"],
            "git.dirty": str(info["git"]["dirty"]).lower(),
        }
    except (OSError, subprocess.CalledProcessError):
        info["git"] = None
    write_json(dest / "repro.json", info)
    return tags


# ------------------------------------------------------------- logging
class ArtifactLogger:
    """Upload run files to the active MLflow run and decide when periodic artifacts are due."""

    PERIODIC_DIRS = ("repro", "latest", "eval")  # uploaded during training, skipped by :meth:`finalize`

    def __init__(self, run_dir: Path, total_timesteps: int, milestones: int = 10, video_every: int = 4) -> None:
        self.run_dir = Path(run_dir)
        self.milestone_steps = (
            [round(total_timesteps * (i + 1) / milestones) for i in range(milestones)] if milestones > 0 else []
        )
        self.video_every = video_every
        self.evals = 0

    def log(self, path: Path, artifact_path: str) -> None:
        try:
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path)
            elif path.exists():
                mlflow.log_artifact(str(path), artifact_path)
        except Exception as e:  # noqa: BLE001 - a tracking hiccup must not kill a long run
            warnings.warn(f"MLflow upload of {path} failed: {e}", stacklevel=2)

    def start(self) -> None:
        """Snapshot code versions (``repro/``) and tag the run with the git commit."""
        tags = write_repro(self.run_dir / "repro")
        try:
            mlflow.set_tags(tags)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"MLflow tagging failed: {e}", stacklevel=2)
        self.log(self.run_dir / "repro", "repro")

    def next_eval(self) -> tuple[int, bool]:
        """Count an evaluation; return its index and whether it records videos."""
        index = self.evals
        self.evals += 1
        return index, self.video_every > 0 and index % self.video_every == 0

    def milestones_reached(self, timesteps: int) -> list[int]:
        """Milestones crossed since the last call (each is reported once)."""
        reached = [m for m in self.milestone_steps if m <= timesteps]
        self.milestone_steps = [m for m in self.milestone_steps if m > timesteps]
        return reached

    def checkpoint(self, latest_dir: Path, timesteps: int, improved: bool, best_dir: Path | None = None) -> None:
        """Upload the latest checkpoint; keep a copy when a milestone was crossed since the last call."""
        self.log(latest_dir, "checkpoints/latest")
        if self.milestones_reached(timesteps):  # named by the checkpoint's actual timesteps
            self.log(latest_dir, f"checkpoints/milestones/{timesteps:09d}")
        if improved and best_dir is not None:
            self.log(best_dir, "checkpoints/best")

    def eval_dir(self, timesteps: int) -> Path:
        return self.run_dir / "eval" / f"{timesteps:09d}"

    def log_eval(self, timesteps: int) -> None:
        self.log(self.eval_dir(timesteps), f"eval/{timesteps:09d}")

    def finalize(self, extra_skip: Iterable[str] = ()) -> None:
        """Upload the run directory under ``run/``, except what was already uploaded periodically."""
        skip = {*self.PERIODIC_DIRS, *extra_skip}
        if not self.run_dir.exists():
            return
        for item in sorted(self.run_dir.iterdir()):
            if item.name in skip:
                continue
            self.log(item, f"run/{item.name}" if item.is_dir() else "run")
