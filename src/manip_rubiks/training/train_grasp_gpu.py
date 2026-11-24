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
from dataclasses import replace

from manip_rubiks.environments.grasp_env import GraspConfig, GraspEnv, GraspRewardWeights
from manip_rubiks.gpu.grasp_env import GraspEnvGPU
from manip_rubiks.gpu.history import HistoryEnvGPU
from manip_rubiks.training import gpu_loop


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--push-scale", type=float, default=1.0, help="train with pushes this many times as strong")
    p.add_argument("--critic-wrench", action="store_true", help="the critic also sees the active push wrench")
    p.add_argument("--history", type=int, default=1, help="stack the actor's last k observation frames")
    p.add_argument("--align-barrier", type=float, default=0.0, help="weight of the layer-misalignment barrier")


def _config(args: argparse.Namespace, difficulty: float) -> GraspConfig:
    return GraspConfig(
        difficulty=difficulty,
        push_scale=args.push_scale,
        critic_wrench=args.critic_wrench,
        reward=GraspRewardWeights(align_barrier=args.align_barrier),
    )


def _eval_config(cfg: GraspConfig) -> GraspConfig:
    """Evaluation always uses the nominal pushes and reward, so runs stay comparable."""
    return replace(cfg, push_scale=1.0, reward=GraspRewardWeights())


def _wrap_env(env, args: argparse.Namespace):
    return HistoryEnvGPU(env, args.history) if args.history > 1 else env


TASK = gpu_loop.Task(
    name="grasp",
    env_cls=GraspEnvGPU,
    cpu_env_cls=GraspEnv,
    make_config=_config,
    eval_config=_eval_config,
    wrap_env=_wrap_env,
    add_args=_add_args,
    end_fields=("misalignment", "displacement", "rotation"),
    episode_metrics=("misalignment", "displacement", "rotation", "dropped", "state_preserved"),
    description=__doc__,
)


def main(argv: list[str] | None = None) -> None:
    gpu_loop.main(TASK, argv)


if __name__ == "__main__":
    main()
