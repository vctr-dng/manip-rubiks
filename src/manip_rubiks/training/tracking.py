"""Experiment tracking with MLflow.

The MLflow server runs in Docker (``docker/up.sh``) and listens on
http://127.0.0.1:5000 by default. ``MLFLOW_TRACKING_URI`` overrides that.

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
from typing import Any

import mlflow
import numpy as np
from stable_baselines3.common.logger import KVWriter

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


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
            f"MLflow server unreachable at {uri} ({e}). Start it with docker/up.sh or set MLFLOW_TRACKING_URI."
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
