"""Train the blindfolded grasp-stabilisation policy with PPO on the GPU.

Thousands of grasp worlds run in parallel on MuJoCo Warp
(:class:`~manip_rubiks.gpu.grasp_env.GraspEnvGPU`) and PPO runs on the same
device (:mod:`manip_rubiks.training.ppo_gpu`). As on the CPU, the actor sees
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
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch

from manip_rubiks.environments.grasp_env import GraspConfig, GraspEnv
from manip_rubiks.gpu.grasp_env import EVAL_BANK_SEEDS, TRAIN_BANK_SEEDS, GraspEnvGPU
from manip_rubiks.gpu.reset_bank import load_reset_bank
from manip_rubiks.training.artifacts import (
    ArtifactLogger,
    DistributionStats,
    normalizer_stats,
    outcome,
    pick_video_episodes,
    render_gif,
    write_csv,
    write_json,
)
from manip_rubiks.training.ppo_gpu import PPO, GaussianActor, PPOConfig
from manip_rubiks.training.tracking import check_server, flatten_params, tracking_uri

EPISODE_METRICS = ("misalignment", "displacement", "rotation", "dropped", "state_preserved")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=time.strftime("gpu_%Y%m%d_%H%M%S"))
    p.add_argument("--out", type=Path, default=Path("runs/grasp_gpu"))
    p.add_argument("--timesteps", type=int, default=24_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty-start", type=float, default=0.2)
    p.add_argument("--curriculum-fraction", type=float, default=0.4)
    p.add_argument("--n-steps", type=int, default=32, help="rollout length per world")
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--eval-every", type=int, default=1_000_000, help="timesteps between evaluations")
    p.add_argument("--eval-envs", type=int, default=256, help="worlds (= episodes) per evaluation")
    p.add_argument("--video-every", type=int, default=1, help="evaluations between eval videos (0: no videos)")
    p.add_argument("--video-episodes", type=int, default=16, help="eval worlds recorded for the videos")
    p.add_argument("--milestones", type=int, default=10, help="checkpoints kept over the run (0: none)")
    p.add_argument("--bank-size", type=int, default=len(TRAIN_BANK_SEEDS), help="training reset states")
    p.add_argument(
        "--mlflow-uri",
        default=None,
        help="tracking URI (default: from docker/.env, else $MLFLOW_TRACKING_URI, else local server)",
    )
    p.add_argument("--experiment", default="grasp", help="MLflow experiment name")
    return p.parse_args(argv)


@dataclass
class EvalResult:
    metrics: dict[str, float]
    episodes: list[dict] = field(default_factory=list)  # one row per world
    distributions: dict = field(default_factory=dict)
    qpos: list[np.ndarray] = field(default_factory=list)  # joint positions of the first ``record`` worlds


@torch.no_grad()
def evaluate(env: GraspEnvGPU, actor: GaussianActor, seed: int = 0, record: int = 0) -> EvalResult:
    """One deterministic full-difficulty episode per world.

    Also collects one table row per world (CPU seed of its initial state,
    outcome, per-term reward sums), action and tactile statistics, and the
    joint positions of the first ``record`` worlds for video replay.
    """
    obs, _ = env.reset(seed=seed)
    n = env.num_envs
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    ends = ("misalignment", "displacement", "rotation")
    keys = ("success", "return", "length", "dropped", "layer_turned", "unstable", *ends)
    result = {k: torch.zeros(n, device=env.device) for k in keys}
    terms: dict[str, torch.Tensor] = {}
    tactile = _obs_slice(env, "tactile")
    dist = DistributionStats(env.n_act, tactile.stop - tactile.start)
    record = min(record, n)
    qpos: list[np.ndarray] = []
    end = np.full(record, env.max_steps)
    for t in range(env.max_steps):
        if record:
            qpos.append(env.qpos[:record].cpu().numpy())
        action = actor.act(obs)
        active = ~done
        dist.add(action[active].cpu().numpy(), obs[active][:, tactile].cpu().numpy())
        obs, _, _, term, trunc, info, _ = env.step(action)
        for k, v in info.reward_terms.items():
            acc = terms.setdefault(k, torch.zeros(n, device=env.device))
            acc[active] += v[active]
        new = (term | trunc) & ~done
        result["success"][new] = (trunc & info.is_success)[new].float()
        result["return"][new] = info.episode_return[new]
        result["length"][new] = info.episode_length[new].float()
        result["dropped"][new] = info.dropped[new].float()
        result["layer_turned"][new] = (~info.state_preserved)[new].float()
        result["unstable"][new] = info.unstable[new].float()
        for k in ends:
            result[k][new] = getattr(info, k)[new]
        if record:
            end[new[:record].cpu().numpy()] = t + 1
        done |= new
        if bool(done.all()):
            break
    metrics = {f"eval/{k}_rate": float(result[k].mean()) for k in ("success", "dropped", "layer_turned")}
    metrics["eval/mean_return"] = float(result["return"].mean())
    r = {k: v.cpu().numpy() for k, v in result.items()}
    term_sums = {k: v.cpu().numpy() for k, v in terms.items()}
    seeds = env.bank.seeds
    rows = []
    for i in range(n):
        flags = (bool(r[k][i]) for k in ("success", "dropped", "layer_turned", "unstable"))
        rows.append(
            {
                "world": i,
                "reset_seed": int(seeds[i % len(seeds)]),
                "outcome": outcome(*flags),
                "return": float(r["return"][i]),
                "length": int(r["length"][i]),
                **{k: float(r[k][i]) for k in ends},
                **{f"reward/{k}": float(v[i]) for k, v in term_sums.items()},
            }
        )
    trajectories = [np.stack([q[i] for q in qpos[: end[i]]]) for i in range(record)]
    return EvalResult(metrics, rows, dist.summary(), trajectories)


def _obs_slice(env: GraspEnvGPU, key: str) -> slice:
    start = 0
    for k, shape in env.layout:
        size = int(np.prod(shape))
        if k == key:
            return slice(start, start + size)
        start += size
    raise KeyError(key)


def distribution_snapshot(ev: EvalResult, ppo: PPO, env: GraspEnvGPU, timesteps: int) -> dict:
    mean, var = ppo.actor.obs_rms.mean.cpu().numpy(), ppo.actor.obs_rms.var.cpu().numpy()
    stats = {k: (mean[_obs_slice(env, k)], var[_obs_slice(env, k)]) for k, _ in env.layout}
    priv = ppo.critic.privileged_rms
    stats["privileged"] = (priv.mean.cpu().numpy(), priv.var.cpu().numpy())
    return {
        **ev.distributions,
        "obs_normalizer": normalizer_stats(stats),
        "policy_std": ppo.actor.log_std.detach().exp().cpu().numpy(),
        "return_normalizer_std": float(ppo.return_rms.var.sqrt()),
        "timesteps": timesteps,
    }


def record_videos(ev: EvalResult, render_env: GraspEnv, out: Path) -> None:
    """Replay one successful and one failed recorded world on the CPU model."""
    labels = [row["outcome"] for row in ev.episodes[: len(ev.qpos)]]
    for i in pick_video_episodes(labels):
        render_gif(render_env, ev.qpos[i], out / f"{labels[i]}_world{i}.gif")


def save_checkpoint(path: Path, ppo: PPO, env: GraspEnvGPU, args: argparse.Namespace, timesteps: int, **extra) -> None:
    """Actor, critic, normalisers and optimizer, plus what is needed to rebuild or resume them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **ppo.state_dict(),
            **extra,
            "obs_layout": env.layout,
            "obs_dim": env.obs_dim,
            "privileged_dim": env.privileged_dim,
            "act_dim": env.n_act,
            "actor_sizes": ppo.cfg.actor_sizes,
            "log_std_init": ppo.cfg.log_std_init,
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
        mlflow.set_tags({"task": "grasp", "algorithm": "PPO", "backend": "mujoco_warp", "run_dir": str(run_dir)})
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
    ppo_cfg = PPOConfig(
        n_steps=args.n_steps,
        n_minibatches=args.n_minibatches,
        n_epochs=args.n_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        log_std_init=args.log_std_init,
    )
    ppo = PPO(args.num_envs, env.obs_dim, env.privileged_dim, env.n_act, ppo_cfg, device)
    batch = args.num_envs * args.n_steps
    iterations = max(1, args.timesteps // batch)
    ramp = max(1, int(args.curriculum_fraction * args.timesteps))
    best = (-1.0, -np.inf)
    render_env: GraspEnv | None = None
    next_eval, timesteps = args.eval_every, 0
    actor_obs, critic_obs = env.reset(seed=args.seed)
    t_start = time.time()
    for it in range(iterations):
        difficulty = min(1.0, args.difficulty_start + (1 - args.difficulty_start) * timesteps / ramp)
        env.set_difficulty(difficulty)
        episodes: dict[str, list[torch.Tensor]] = defaultdict(list)
        t0 = time.time()
        for step in range(args.n_steps):
            action = ppo.act(actor_obs, critic_obs, step)
            actor_obs, critic_obs, reward, term, trunc, info, terminal = env.step(action)
            ppo.record(step, reward, term, trunc, terminal)
            done = term | trunc
            if bool(done.any()):
                episodes["return"].append(info.episode_return[done])
                episodes["length"].append(info.episode_length[done].float())
                episodes["success"].append((trunc & info.is_success)[done].float())
                for k in EPISODE_METRICS:
                    episodes[k].append(getattr(info, k)[done].float())
        t_collect = time.time() - t0
        losses = ppo.update(critic_obs)
        timesteps += batch
        elapsed = time.time() - t0
        metrics = {f"train/{k}": v for k, v in losses.items() if np.isfinite(v)}
        metrics.update(
            {
                "curriculum/difficulty": difficulty,
                "time/fps": batch / elapsed,
                "time/collect_fraction": t_collect / elapsed,
                "time/elapsed_s": time.time() - t_start,
            }
        )
        if episodes:
            cat = {k: torch.cat(v) for k, v in episodes.items()}
            metrics["rollout/ep_rew_mean"] = float(cat["return"].mean())
            metrics["rollout/ep_len_mean"] = float(cat["length"].mean())
            metrics["rollout/success_rate"] = float(cat["success"].mean())
            metrics.update({f"episode/{k}": float(cat[k].mean()) for k in EPISODE_METRICS})
        if timesteps >= next_eval or it == iterations - 1:
            next_eval += args.eval_every
            _, videos = artifacts.next_eval()
            ev = evaluate(eval_env, ppo.actor, record=args.video_episodes if videos else 0)
            metrics.update(ev.metrics)
            score = (ev.metrics["eval/success_rate"], ev.metrics["eval/mean_return"])
            improved = score > best
            if improved:
                best = score
                save_checkpoint(run_dir / "best" / "policy.pt", ppo, env, args, timesteps)
                metrics["eval/best_success_rate"] = score[0]
            out = artifacts.eval_dir(timesteps)
            write_csv(out / "episodes.csv", ev.episodes)
            write_json(out / "distributions.json", distribution_snapshot(ev, ppo, env, timesteps))
            resume = {
                "best_score": best,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "env_rng": env.gen.get_state(),
            }
            save_checkpoint(run_dir / "latest" / "policy.pt", ppo, env, args, timesteps, **resume)
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
                f"[{timesteps / 1e6:6.2f}M] difficulty {difficulty:.2f}  fps {batch / elapsed:6.0f}  "
                f"eval success {score[0]:.2f}  return {score[1]:7.1f}",
                flush=True,
            )
        mlflow.log_metrics(metrics, step=timesteps)
    save_checkpoint(run_dir / "policy.pt", ppo, env, args, timesteps)
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
