"""Experiment tracking with MLflow.

The MLflow server runs in Docker (``docker/up.sh``), configured by
``docker/.env``. :func:`tracking_uri` reads that file with python-dotenv
(from the main checkout when running in a git worktree, since ``.env`` is not
versioned) and takes ``MLFLOW_TRACKING_URI`` from it, or builds the URI from
``MLFLOW_BIND_ADDRESS`` and ``MLFLOW_PORT``. Without the file, the
``MLFLOW_TRACKING_URI`` environment variable applies, then
http://127.0.0.1:5000.

:class:`MLflowOutputFormat` plugs into the Stable-Baselines3 logger, so
everything SB3 and the training callbacks record (``rollout/``, ``train/``,
``eval/``, ``curriculum/``, ``episode/``) becomes an MLflow metric of the
active run, stepped by environment timesteps.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from dotenv import dotenv_values
from stable_baselines3.common.logger import KVWriter

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _main_checkout(root: Path) -> Path | None:
    """Main checkout of a git worktree (whose ``.git`` is a file pointing into ``<main>/.git/worktrees``)."""
    git = root / ".git"
    if not git.is_file():
        return None
    line = git.read_text().strip()
    if not line.startswith("gitdir:"):
        return None
    gitdir = Path(line.removeprefix("gitdir:").strip())
    gitdir = gitdir if gitdir.is_absolute() else (root / gitdir).resolve()
    return gitdir.parent.parent.parent if gitdir.parent.name == "worktrees" else None


def docker_env_file(root: Path = REPO_ROOT) -> Path | None:
    """``docker/.env`` of this checkout, or of the main checkout when running from a worktree."""
    main = _main_checkout(root)
    for base in (root, main):
        if base is not None and (base / "docker" / ".env").is_file():
            return base / "docker" / ".env"
    return None


def uri_from_env_file(path: Path) -> str | None:
    """Tracking URI described by a Docker ``.env`` file.

    ``MLFLOW_TRACKING_URI`` wins if the file sets it; otherwise the URI is
    built from the server's ``MLFLOW_BIND_ADDRESS`` and ``MLFLOW_PORT``.
    """
    values = {k: v for k, v in dotenv_values(path).items() if v}
    if "MLFLOW_TRACKING_URI" in values:
        return values["MLFLOW_TRACKING_URI"]
    host = values.get("MLFLOW_BIND_ADDRESS")
    if host is None:
        return None
    if host in ("0.0.0.0", "::"):  # listening on every interface: reach it locally
        host = "127.0.0.1"
    return f"http://{host}:{values.get('MLFLOW_PORT', '5000')}"


def tracking_uri() -> str:
    """``docker/.env`` (where the server is configured), then ``$MLFLOW_TRACKING_URI``, then the local default."""
    path = docker_env_file()
    uri = uri_from_env_file(path) if path is not None else None
    return uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def check_server(uri: str, timeout: float = 3.0) -> None:
    """Fail fast with a useful message when an HTTP tracking server is down.

    MLflow's own client retries for minutes before giving up.
    """
    if not uri.startswith(("http://", "https://")):
        return
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"MLflow server unreachable at {uri} ({e}). Start it with docker/up.sh or check docker/.env."
        ) from e


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys; sequences become strings."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{name}."))
        elif isinstance(value, np.ndarray):
            flat[name] = str(value.tolist())
        elif isinstance(value, (list, tuple)):
            flat[name] = str(list(value))
        else:
            flat[name] = value
    return flat


class MLflowOutputFormat(KVWriter):
    """Write the scalar values of each SB3 logger dump to the active MLflow run."""

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        metrics = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key)
            if excluded is not None and "mlflow" in excluded:
                continue
            if isinstance(value, (bool, np.bool_)):
                value = float(value)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                metrics[key] = float(value)
        if metrics:
            mlflow.log_metrics(metrics, step=int(step))

    def close(self) -> None:
        pass
