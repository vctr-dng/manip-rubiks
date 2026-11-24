"""Worker pools run single-threaded and leave the caller's thread settings alone."""

import os
from pathlib import Path

import pytest

from manip_rubiks.parallel import THREAD_VARS, map_resilient, spawn_pool


def _thread_settings(_: int) -> tuple[str | None, ...]:
    return tuple(os.environ.get(k) for k in THREAD_VARS)


def test_workers_get_one_thread_and_the_environment_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with spawn_pool(2) as pool:
        seen = list(pool.map(_thread_settings, range(2)))
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert seen == [("1", "1", "1")] * 2
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "MKL_NUM_THREADS" not in os.environ


def _square(x: int) -> int:
    return x * x


def _crash_once(job: tuple[str, int]) -> int:
    """Kill the worker process the first time job 3 runs (a stand-in for a native crash)."""
    marker, x = job
    if x == 3 and not Path(marker).exists():
        Path(marker).touch()
        os._exit(1)
    return x * x


def _fail(x: int) -> int:
    raise KeyError(x)


def test_map_resilient_keeps_order():
    assert map_resilient(_square, list(range(7)), workers=3) == [x * x for x in range(7)]


def test_map_resilient_reruns_the_jobs_a_crashed_pool_left_unfinished(tmp_path, capfd):
    jobs = [(str(tmp_path / "crashed"), x) for x in range(8)]
    assert map_resilient(_crash_once, jobs, workers=2) == [x * x for x in range(8)]
    assert "worker pool broke" in capfd.readouterr().err


def test_map_resilient_does_not_retry_errors_raised_by_the_job():
    with pytest.raises(KeyError):
        map_resilient(_fail, [1, 2], workers=2)
