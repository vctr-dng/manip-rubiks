"""Train the blindfolded grasp-stabilisation policy with PPO on the GPU.

Thousands of grasp worlds run in parallel on MuJoCo Warp
(:class:`~manip_rubiks.gpu.grasp_env.GraspEnvGPU`) and PPO runs on the same
device (:mod:`manip_rubiks.training.gpu_loop`). As on the CPU, the actor sees
proprioception, tactile readings and the wrist's gravity direction, the
critic also sees the privileged cube state, and tilt range and pushes ramp up
from ``--difficulty-start`` to full difficulty over the first
``--curriculum-fraction`` of training.

Evaluation runs one deterministic episode in each of ``--eval-envs`` worlds at
full difficulty, starting from held-out reset states (CPU seeds 10000..). The
checkpoint with the best success rate (then mean return) is kept in ``best/``.
Benchmark it on the CPU reference with ``manip-rubiks-eval-grasp``. After every
evaluation the latest checkpoint, a per-episode table, a distribution snapshot
and, periodically, videos replayed from the GPU worlds are uploaded as well
(see :mod:`manip_rubiks.training.artifacts`).

Example::

    docker/up.sh                                       # MLflow server on http://127.0.0.1:5000
    uv run manip-rubiks-train-grasp-gpu --name gpu_s0
    uv run manip-rubiks-eval-grasp --policy gpu=runs/grasp_gpu/gpu_s0
"""

from __future__ import annotations

import argparse

from manip_rubiks.environments.grasp_env import GraspConfig, GraspEnv
from manip_rubiks.gpu.grasp_env import GraspEnvGPU
from manip_rubiks.training import gpu_loop


def _config(args: argparse.Namespace, difficulty: float) -> GraspConfig:
    return GraspConfig(difficulty=difficulty)


TASK = gpu_loop.Task(
    name="grasp",
    env_cls=GraspEnvGPU,
    cpu_env_cls=GraspEnv,
    make_config=_config,
    end_fields=("misalignment", "displacement", "rotation"),
    episode_metrics=("misalignment", "displacement", "rotation", "dropped", "state_preserved"),
    description=__doc__,
)


def main(argv: list[str] | None = None) -> None:
    gpu_loop.main(TASK, argv)


if __name__ == "__main__":
    main()
