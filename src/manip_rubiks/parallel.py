"""Process pools for CPU-bound simulation work."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any

THREAD_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextmanager
def spawn_pool(workers: int) -> Iterator[ProcessPoolExecutor]:
    """A pool of spawned workers that each run one BLAS/OpenMP thread.

    The workers step MuJoCo and evaluate a small policy, so extra threads buy nothing. Left at
    their default, every worker starts one OpenMP thread per core: sixteen workers on a 20-core
    machine then crash the pool within seconds (``BrokenProcessPool``). Spawned workers inherit
    the environment at start-up, so the limit is set while the pool lives and restored afterwards;
    the calling process keeps its own thread settings.
    """
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            yield pool
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def map_resilient(
    fn: Callable[[Any], Any], jobs: Sequence[Any], workers: int, max_pool_failures: int = 20
) -> list[Any]:
    """``[fn(job) for job in jobs]`` on a pool, surviving worker crashes; results are in job order.

    On this project's workstation a worker process occasionally dies with a native crash (a
    segmentation fault while importing torch or stepping MuJoCo), and one dead worker breaks the
    whole ``ProcessPoolExecutor``. The jobs are deterministic and independent, so the results
    already collected are kept and only the unfinished jobs run again in a fresh pool. An
    exception raised by ``fn`` itself is not retried, and ``BrokenProcessPool`` is raised after
    ``max_pool_failures`` broken pools in a row without progress.
    """
    results: dict[int, Any] = {}
    failures = 0
    while len(results) < len(jobs):
        pending = [i for i in range(len(jobs)) if i not in results]
        with spawn_pool(max(1, min(workers, len(pending)))) as pool:
            futures = {pool.submit(fn, jobs[i]): i for i in pending}
            wait(futures)
        done_before = len(results)
        for fut, i in futures.items():
            error = fut.exception()
            if error is None:
                results[i] = fut.result()
            elif not isinstance(error, BrokenProcessPool):
                raise error
        if len(results) == len(jobs):
            break
        failures = failures + 1 if len(results) == done_before else 0
        if failures >= max_pool_failures:
            raise BrokenProcessPool(f"{max_pool_failures} worker pools in a row broke without finishing a job")
        print(
            f"worker pool broke; {len(jobs) - len(results)}/{len(jobs)} jobs unfinished, running them again",
            file=sys.stderr,
            flush=True,
        )
    return [results[i] for i in range(len(jobs))]
