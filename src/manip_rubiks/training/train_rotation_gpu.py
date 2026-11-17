"""Train the in-hand quarter-turn reorientation teacher with PPO on the GPU.

Thousands of reorientation worlds run in parallel on MuJoCo Warp
(:class:`~manip_rubiks.gpu.rotation_env.RotationEnvGPU`) and PPO runs on the
same device (:mod:`manip_rubiks.training.gpu_loop`). The actor sees
proprioception, tactile readings, the active quarter-turn command and the
privileged ``teacher`` observation (goal error, cube position and angular
velocity). The critic also sees the full privileged cube state. The goal
tolerance tightens from ``RotationConfig.easy_tolerance`` to
``goal_tolerance`` over the first ``--curriculum-fraction`` of training.

Evaluation runs one deterministic 16 s episode in each of ``--eval-envs``
worlds and reports goals reached per episode, the per-goal success rate
(goals reached / goals reached or failed; a goal still pending at the time
limit is not counted) overall and per command, and how episodes ended. The
checkpoint with the most goals per episode (then mean return) is kept in
``best/``. Benchmark it on the CPU reference with ``manip-rubiks-eval-rotate``.

Example::

    uv run manip-rubiks-train-rotate-gpu --name gpu_s0 --mlflow-uri http://127.0.0.1:5000
    uv run manip-rubiks-eval-rotate --policy gpu=runs/rotation_gpu/gpu_s0
"""

from __future__ import annotations

import argparse

import torch

from manip_rubiks.environments.rotation_env import COMMAND_NAMES, N_COMMANDS, RotationConfig, RotationEnv
from manip_rubiks.gpu.hand_cube_env import StepInfo
from manip_rubiks.gpu.rotation_env import RotationEnvGPU
from manip_rubiks.training import gpu_loop


def _config(args: argparse.Namespace, difficulty: float) -> RotationConfig:
    return RotationConfig(goal_axes=args.axes, difficulty=difficulty)


def _flags(term: torch.Tensor, trunc: torch.Tensor, info: StepInfo) -> gpu_loop.Flags:
    return {**gpu_loop.grasp_like_flags(term, trunc, info), "timed_out": info.timed_out}


def _eval_step(info: StepInfo, active: torch.Tensor, flags: gpu_loop.Flags, state: dict) -> None:
    """Count reached and failed goals per command over the evaluation episodes."""
    if not state:
        dev = active.device
        state.update(
            reached=torch.zeros(N_COMMANDS, device=dev),
            failed=torch.zeros(N_COMMANDS, device=dev),
            episodes=int(active.numel()),
        )
    reached = info.goal_reached & active
    failed = flags["terminated"] & active
    state["reached"].index_add_(0, info.command[reached], torch.ones(int(reached.sum()), device=active.device))
    state["failed"].index_add_(0, info.command[failed], torch.ones(int(failed.sum()), device=active.device))


def _eval_summary(state: dict) -> dict[str, float]:
    reached, failed = state["reached"], state["failed"]
    attempts = reached + failed
    out = {
        "eval/goals_per_episode": float(reached.sum()) / state["episodes"],
        "eval/goal_success_rate": float(reached.sum() / attempts.sum().clamp(min=1)),
    }
    for c, name in enumerate(COMMAND_NAMES):
        if attempts[c] > 0:
            sign = "pos" if name[0] == "+" else "neg"  # MLflow metric names cannot contain "+"
            out[f"eval/goal_success/{sign}_{name[1]}"] = float(reached[c] / attempts[c])
    return out


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--axes", default="xyz", help="hand axes of the quarter-turn commands (x: palm normal)")


TASK = gpu_loop.Task(
    name="rotation",
    env_cls=RotationEnvGPU,
    cpu_env_cls=RotationEnv,
    make_config=_config,
    end_fields=("misalignment", "displacement", "theta", "goals_reached"),
    episode_metrics=("goals_reached", "timed_out", "dropped", "misalignment", "displacement"),
    failure_order=("unstable", "dropped", "layer_turned", "timed_out"),
    flags=_flags,
    score=lambda m: (m["eval/goals_per_episode"], m["eval/mean_return"]),
    eval_step=_eval_step,
    eval_summary=_eval_summary,
    add_args=_add_args,
    # Exploration starts at std 1: at the grasp default (std 0.37) undirected finger motion barely
    # rotates the cube, and no goal is ever reached.
    defaults={"timesteps": 60_000_000, "difficulty_start": 0.0, "curriculum_fraction": 0.3, "log_std_init": 0.0},
    description=__doc__,
)


def main(argv: list[str] | None = None) -> None:
    gpu_loop.main(TASK, argv)


if __name__ == "__main__":
    main()
