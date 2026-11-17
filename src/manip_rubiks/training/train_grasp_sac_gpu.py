"""Train the blindfolded grasp-stabilisation policy with SAC on the GPU.

Same environment, observations, curriculum, evaluation and MLflow artifacts as
:mod:`manip_rubiks.training.train_grasp_gpu`; only the learning algorithm
changes (:mod:`manip_rubiks.training.sac_gpu`). Every batched environment
step adds ``--num-envs`` transitions to the replay buffer and is followed by
``--updates-per-step`` gradient steps on batches of ``--batch-size``.

Example::

    docker/up.sh                                       # MLflow server on http://127.0.0.1:5000
    uv run manip-rubiks-train-grasp-sac --name sac_s0
    uv run manip-rubiks-eval-grasp --policy sac=runs/grasp_sac/sac_s0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch

from manip_rubiks.environments.grasp_env import GraspConfig, GraspEnv
from manip_rubiks.gpu.grasp_env import EVAL_BANK_SEEDS, TRAIN_BANK_SEEDS, GraspEnvGPU
from manip_rubiks.gpu.reset_bank import load_reset_bank
from manip_rubiks.training.artifacts import ArtifactLogger, normalizer_stats, write_csv, write_json
from manip_rubiks.training.sac_gpu import SAC, SACConfig
from manip_rubiks.training.tracking import check_server, flatten_params, tracking_uri
from manip_rubiks.training.train_grasp_gpu import EPISODE_METRICS, EvalResult, _obs_slice, evaluate, record_videos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=time.strftime("sac_%Y%m%d_%H%M%S"))
    p.add_argument("--out", type=Path, default=Path("runs/grasp_sac"))
    p.add_argument("--timesteps", type=int, default=8_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty-start", type=float, default=0.2)
    p.add_argument("--curriculum-fraction", type=float, default=0.4)
    d = SACConfig()
    p.add_argument("--buffer-size", type=int, default=d.buffer_size, help="replay transitions")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--updates-per-step", type=int, default=d.updates_per_step)
    p.add_argument("--learning-starts", type=int, default=d.learning_starts, help="batched env steps")
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--n-step", type=int, default=d.n_step)
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--actor-lr", type=float, default=d.actor_lr)
    p.add_argument("--critic-lr", type=float, default=d.critic_lr)
    p.add_argument("--alpha-lr", type=float, default=d.alpha_lr)
    p.add_argument("--init-alpha", type=float, default=d.init_alpha)
    p.add_argument("--target-entropy-scale", type=float, default=d.target_entropy_scale)
    p.add_argument("--log-std-init", type=float, default=d.log_std_init)
    p.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=d.layer_norm)
    p.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    p.add_argument(
        "--freeze-normalizer-after",
        type=int,
        default=d.freeze_normalizer_after,
        help="timesteps after which observation statistics stop updating (0: never)",
    )
    p.add_argument("--log-every", type=int, default=10, help="batched env steps between metric logs")
    p.add_argument("--eval-every", type=int, default=500_000, help="timesteps between evaluations")
    p.add_argument("--eval-envs", type=int, default=256, help="worlds (= episodes) per evaluation")
    p.add_argument("--video-every", type=int, default=4, help="evaluations between eval videos (0: no videos)")
    p.add_argument("--video-episodes", type=int, default=16, help="eval worlds recorded for the videos")
    p.add_argument("--milestones", type=int, default=10, help="checkpoints kept over the run (0: none)")
    p.add_argument("--bank-size", type=int, default=len(TRAIN_BANK_SEEDS), help="training reset states")
    p.add_argument("--mlflow-uri", default=None, help="tracking URI (default: $MLFLOW_TRACKING_URI or local server)")
    p.add_argument("--experiment", default="grasp", help="MLflow experiment name")
    return p.parse_args(argv)


def sac_config(args: argparse.Namespace) -> SACConfig:
    return SACConfig(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        n_step=args.n_step,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        init_alpha=args.init_alpha,
        target_entropy_scale=args.target_entropy_scale,
        log_std_init=args.log_std_init,
        layer_norm=args.layer_norm,
        max_grad_norm=args.max_grad_norm,
        freeze_normalizer_after=args.freeze_normalizer_after,
    )


def distribution_snapshot(ev: EvalResult, sac: SAC, env: GraspEnvGPU, timesteps: int) -> dict:
    mean, var = sac.actor.obs_rms.mean.cpu().numpy(), sac.actor.obs_rms.var.cpu().numpy()
    stats = {k: (mean[_obs_slice(env, k)], var[_obs_slice(env, k)]) for k, _ in env.layout}
    stats["privileged"] = (sac.privileged_rms.mean.cpu().numpy(), sac.privileged_rms.var.cpu().numpy())
    return {
        **ev.distributions,
        "obs_normalizer": normalizer_stats(stats),
        "alpha": float(sac.alpha),
        "timesteps": timesteps,
    }


def save_checkpoint(path: Path, sac: SAC, env: GraspEnvGPU, args: argparse.Namespace, timesteps: int, **extra) -> None:
    """Networks, normalisers, temperature and optimizers (not the replay buffer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **sac.state_dict(),
            **extra,
            "algorithm": "sac",
            "obs_layout": env.layout,
            "obs_dim": env.obs_dim,
            "privileged_dim": env.privileged_dim,
            "act_dim": env.n_act,
            "actor_sizes": sac.cfg.actor_sizes,
            "log_std_init": sac.cfg.log_std_init,
            "sac_config": asdict(sac.cfg),
            "timesteps": timesteps,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    uri = args.mlflow_uri or tracking_uri()
    check_server(uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.experiment)
    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    config["grasp_config"] = asdict(GraspConfig())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=float))
    artifacts = ArtifactLogger(run_dir, args.timesteps, args.milestones, args.video_every)
    with mlflow.start_run(run_name=args.name):
        mlflow.log_params(flatten_params(config))
        mlflow.set_tags({"task": "grasp", "algorithm": "SAC", "backend": "mujoco_warp", "run_dir": str(run_dir)})
        artifacts.start()
        try:
            _train(args, run_dir, artifacts)
        finally:  # keep whatever was saved, also when training is interrupted
            artifacts.finalize()


def _train(args: argparse.Namespace, run_dir: Path, artifacts: ArtifactLogger) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = GraspConfig(difficulty=args.difficulty_start)
    env = GraspEnvGPU(
        args.num_envs, cfg, device=args.device, seed=args.seed, bank=load_reset_bank(cfg, range(args.bank_size))
    )
    eval_env = GraspEnvGPU(
        args.eval_envs,
        GraspConfig(),
        device=args.device,
        seed=args.seed + 1,
        bank=load_reset_bank(cfg, EVAL_BANK_SEEDS),
    )
    sac = SAC(args.num_envs, env.obs_dim, env.privileged_dim, env.n_act, sac_config(args), device)
    iterations = max(1, args.timesteps // args.num_envs)
    ramp = max(1, int(args.curriculum_fraction * args.timesteps))
    best = (-1.0, -np.inf)
    render_env: GraspEnv | None = None
    next_eval, timesteps = args.eval_every, 0
    actor_obs, critic_obs = env.reset(seed=args.seed)
    episodes: dict[str, list[torch.Tensor]] = defaultdict(list)
    losses: dict[str, list[float]] = defaultdict(list)
    t_env = t_update = 0.0
    window_steps = 0
    t_start = t_window = time.time()
    for it in range(iterations):
        difficulty = min(1.0, args.difficulty_start + (1 - args.difficulty_start) * timesteps / ramp)
        env.set_difficulty(difficulty)
        t0 = time.time()
        action = sac.act(actor_obs, critic_obs)
        next_actor_obs, next_critic_obs, reward, term, trunc, info, terminal = env.step(action)
        sac.record(critic_obs, action, reward, term, trunc, terminal)
        actor_obs, critic_obs = next_actor_obs, next_critic_obs
        done = term | trunc
        if bool(done.any()):
            episodes["return"].append(info.episode_return[done])
            episodes["length"].append(info.episode_length[done].float())
            episodes["success"].append((trunc & info.is_success)[done].float())
            for k in EPISODE_METRICS:
                episodes[k].append(getattr(info, k)[done].float())
        t1 = time.time()
        for k, v in sac.update().items():
            losses[k].append(v)
        t_update += time.time() - t1
        t_env += t1 - t0
        timesteps += args.num_envs
        window_steps += args.num_envs
        last = it == iterations - 1
        evaluate_now = timesteps >= next_eval or last
        if (it + 1) % args.log_every and not evaluate_now:
            continue

        window = time.time() - t_window
        metrics = {f"train/{k}": float(np.mean(v)) for k, v in losses.items()}
        metrics.update(
            {
                "curriculum/difficulty": difficulty,
                "time/fps": window_steps / max(window, 1e-9),
                "time/update_fraction": t_update / max(t_env + t_update, 1e-9),
                "time/elapsed_s": time.time() - t_start,
                "replay/size": float(len(sac.buffer)),
            }
        )
        if episodes:
            cat = {k: torch.cat(v) for k, v in episodes.items()}
            metrics["rollout/ep_rew_mean"] = float(cat["return"].mean())
            metrics["rollout/ep_len_mean"] = float(cat["length"].mean())
            metrics["rollout/success_rate"] = float(cat["success"].mean())
            metrics.update({f"episode/{k}": float(cat[k].mean()) for k in EPISODE_METRICS})
        episodes.clear()
        losses.clear()
        t_env = t_update = 0.0
        window_steps = 0
        if evaluate_now:
            next_eval += args.eval_every
            _, videos = artifacts.next_eval()
            ev = evaluate(eval_env, sac.actor, record=args.video_episodes if videos else 0)
            metrics.update(ev.metrics)
            score = (ev.metrics["eval/success_rate"], ev.metrics["eval/mean_return"])
            improved = score > best
            if improved:
                best = score
                save_checkpoint(run_dir / "best" / "policy.pt", sac, env, args, timesteps)
                metrics["eval/best_success_rate"] = score[0]
            out = artifacts.eval_dir(timesteps)
            write_csv(out / "episodes.csv", ev.episodes)
            write_json(out / "distributions.json", distribution_snapshot(ev, sac, env, timesteps))
            resume = {
                "best_score": best,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "env_rng": env.gen.get_state(),
            }
            save_checkpoint(run_dir / "latest" / "policy.pt", sac, env, args, timesteps, **resume)
            artifacts.checkpoint(run_dir / "latest", timesteps, improved, run_dir / "best")
            if videos:
                try:
                    render_env = render_env or GraspEnv(GraspConfig(), render_mode="rgb_array")
                    record_videos(ev, render_env, out / "videos")
                except Exception as e:  # noqa: BLE001 - e.g. no OpenGL context; training goes on without videos
                    print(f"video recording failed ({e}); set MUJOCO_GL=egl on a headless machine", flush=True)
                    artifacts.video_every = 0
            artifacts.log_eval(timesteps)
            print(
                f"[{timesteps / 1e6:6.2f}M] difficulty {difficulty:.2f}  fps {metrics['time/fps']:6.0f}  "
                f"alpha {metrics.get('train/alpha', float('nan')):.4f}  "
                f"eval success {score[0]:.2f}  return {score[1]:7.1f}",
                flush=True,
            )
        mlflow.log_metrics(metrics, step=timesteps)
        t_window = time.time()
    save_checkpoint(run_dir / "policy.pt", sac, env, args, timesteps)
    wall = time.time() - t_start
    summary = {
        "timesteps": timesteps,
        "wall_clock_s": wall,
        "steps_per_s": timesteps / wall,
        "best_eval_success_rate": best[0],
        "best_eval_mean_return": best[1],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    mlflow.log_metrics({f"summary/{k}": float(v) for k, v in summary.items() if np.isfinite(v)}, step=timesteps)
    env.close()
    eval_env.close()
    if render_env is not None:
        render_env.close()


if __name__ == "__main__":
    main()
