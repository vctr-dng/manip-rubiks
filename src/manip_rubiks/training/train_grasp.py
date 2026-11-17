"""Train a blindfolded grasp-stabilisation policy with PPO.

The actor sees proprioception, tactile readings and the wrist's gravity
direction. With ``--asymmetric`` (the default) the critic also sees the
privileged cube state (draft section 25). Tilt range and pushes ramp up from
``--difficulty-start`` to full difficulty over the first
``--curriculum-fraction`` of training, while evaluation always runs at full
difficulty.

Example::

    docker/up.sh                                  # MLflow server on http://127.0.0.1:5000
    uv run manip-rubiks-train-grasp --timesteps 12_000_000 --name ppo_asym

Metrics, parameters and the trained model are logged to MLflow (experiment
``--experiment``, run ``--name``). Checkpoints are also kept in ``runs/grasp``.
After every evaluation the latest checkpoint, a per-episode table, a
distribution snapshot and, periodically, videos are uploaded as well (see
:mod:`manip_rubiks.training.artifacts`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from torch import nn

from manip_rubiks.environments import GRASP_OBS_KEYS, GraspConfig, GraspEnv
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
from manip_rubiks.training.asymmetric import PRIVILEGED_KEY, AsymmetricActorCriticPolicy, PrivilegedObservation
from manip_rubiks.training.tracking import MLflowOutputFormat, check_server, flatten_params, tracking_uri

EPISODE_METRICS = ("misalignment", "displacement", "rotation", "dropped", "state_preserved")
VIDEO_SEED = 20_000  # video episodes use seeds VIDEO_SEED.., the same at every evaluation


def make_env(difficulty: float, asymmetric: bool):
    def _init():
        env = GraspEnv(GraspConfig(difficulty=difficulty))
        return PrivilegedObservation(env) if asymmetric else env

    return _init


def make_vec_env(n: int, difficulty: float, asymmetric: bool, seed: int) -> VecMonitor:
    venv = SubprocVecEnv([make_env(difficulty, asymmetric) for _ in range(n)], start_method="forkserver")
    venv.seed(seed)
    return VecMonitor(venv)


class DifficultyCurriculum(BaseCallback):
    """Linear difficulty ramp, applied at the start of every rollout."""

    def __init__(self, start: float, ramp_steps: int) -> None:
        super().__init__()
        self.start, self.ramp_steps = start, max(ramp_steps, 1)

    def difficulty(self) -> float:
        return float(min(1.0, self.start + (1.0 - self.start) * self.num_timesteps / self.ramp_steps))

    def _on_rollout_start(self) -> None:
        d = self.difficulty()
        self.training_env.env_method("set_difficulty", d)
        self.logger.record("curriculum/difficulty", d)

    def _on_step(self) -> bool:
        return True


class EpisodeMetrics(BaseCallback):
    """Log privileged end-of-episode metrics (evaluation only, never fed to the actor)."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                for k in EPISODE_METRICS:
                    self.logger.record_mean(f"episode/{k}", float(info[k]))
        return True


class SaveNormalizerOnBest(BaseCallback):
    """Store the VecNormalize statistics next to each new best model."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        self.model.get_vec_normalize_env().save(str(self.path))
        return True


class GraspEvalCallback(EvalCallback):
    """:class:`EvalCallback` that also writes the periodic training artifacts.

    While SB3 evaluates, every step's action and raw tactile reading and every
    finished episode's metrics are collected. After each evaluation it writes
    ``eval/<steps>/episodes.csv`` and ``distributions.json``, saves and uploads
    the latest checkpoint, uploads ``best/`` when it improved, and every
    ``video_every`` evaluations renders one successful and one failed episode.
    """

    def __init__(self, eval_env, *, run_dir: Path, artifacts: ArtifactLogger, video_episodes: int = 8, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.run_dir, self.artifacts, self.video_episodes = run_dir, artifacts, video_episodes
        self._video_env: GraspEnv | None = None
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.rows: list[dict] = []
        self.dist: DistributionStats | None = None
        self.terms: dict[int, dict[str, float]] = {}

    def _on_step(self) -> bool:
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if due:
            self._reset_buffers()
            best_before = self.best_mean_reward
        keep_going = super()._on_step()
        if due:
            self._after_eval(improved=self.best_mean_reward > best_before)
        return keep_going

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        super()._log_success_callback(locals_, globals_)
        i, info = locals_["i"], locals_["info"]
        tactile = self.eval_env.get_original_obs()["tactile"]
        if self.dist is None:
            self.dist = DistributionStats(locals_["actions"].shape[-1], int(np.prod(tactile.shape[1:])))
        self.dist.add(locals_["actions"][i : i + 1], tactile[i : i + 1])
        terms = self.terms.setdefault(i, {})
        for k, v in info.get("reward_terms", {}).items():
            terms[k] = terms.get(k, 0.0) + v
        if not locals_["done"]:
            return
        ep = info.get("episode", {})
        self.rows.append(
            {
                "env": i,
                "episode": int(locals_["episode_counts"][i]),
                "outcome": outcome(info["is_success"], info["dropped"], not info["state_preserved"], info["unstable"]),
                "return": float(ep.get("r", locals_["current_rewards"][i])),
                "length": int(ep.get("l", locals_["current_lengths"][i])),
                **{k: info[k] for k in ("is_success", "dropped", "unstable", "state_preserved")},
                **{k: info[k] for k in ("misalignment", "displacement", "rotation")},
                **{f"reward/{k}": v for k, v in terms.items()},
            }
        )
        self.terms.pop(i)

    def _after_eval(self, improved: bool) -> None:
        steps = int(self.num_timesteps)
        _, videos = self.artifacts.next_eval()
        out = self.artifacts.eval_dir(steps)
        write_csv(out / "episodes.csv", self.rows)
        snapshot = self.dist.summary() if self.dist is not None else {}
        rms = self.eval_env.obs_rms  # synced from the training normaliser before each evaluation
        snapshot["obs_normalizer"] = normalizer_stats({k: (r.mean, r.var) for k, r in rms.items()})
        snapshot["policy_std"] = th.exp(self.model.policy.log_std.detach()).cpu().numpy()
        snapshot["timesteps"] = steps
        write_json(out / "distributions.json", snapshot)

        latest = self.run_dir / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        self.model.save(latest / "model.zip")  # includes the optimizer state
        self.model.get_vec_normalize_env().save(str(latest / "vecnormalize.pkl"))
        write_json(latest / "state.json", {"timesteps": steps, "best_mean_reward": float(self.best_mean_reward)})
        self.artifacts.checkpoint(latest, steps, improved, self.run_dir / "best")
        if videos:
            self._record_videos(out / "videos")
        self.artifacts.log_eval(steps)

    def _record_videos(self, out: Path) -> None:
        try:
            if self._video_env is None:
                self._video_env = GraspEnv(GraspConfig(difficulty=1.0), render_mode="rgb_array")
            env = self._video_env
            space = self.model.observation_space
            privileged = space[PRIVILEGED_KEY].shape if PRIVILEGED_KEY in space.spaces else None
            episodes: list[tuple[int, str, np.ndarray]] = []
            for j in range(self.video_episodes):
                obs, _ = env.reset(seed=VIDEO_SEED + j)
                qpos, done = [env.data.qpos.copy()], False
                while not done:
                    x = {k: v[None] for k, v in obs.items()}
                    if privileged is not None:  # the actor never reads it
                        x[PRIVILEGED_KEY] = np.zeros((1, *privileged), np.float32)
                    action, _ = self.model.predict(self.eval_env.normalize_obs(x), deterministic=True)
                    obs, _, term, trunc, info = env.step(action[0])
                    qpos.append(env.data.qpos.copy())
                    done = term or trunc
                label = outcome(info["is_success"], info["dropped"], not info["state_preserved"], info["unstable"])
                episodes.append((VIDEO_SEED + j, label, np.array(qpos)))
                if len(pick_video_episodes([e[1] for e in episodes])) == 2:
                    break
            for i in pick_video_episodes([e[1] for e in episodes]):
                seed, label, qpos = episodes[i]
                render_gif(env, qpos, out / f"{label}_seed{seed}.gif")
        except Exception as e:  # noqa: BLE001 - e.g. no OpenGL context; training goes on without videos
            print(f"video recording failed ({e}); set MUJOCO_GL=egl on a headless machine", file=sys.stderr)
            self.artifacts.video_every = 0

    def close_video_env(self) -> None:
        if self._video_env is not None:
            self._video_env.close()
            self._video_env = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=time.strftime("ppo_%Y%m%d_%H%M%S"))
    p.add_argument("--out", type=Path, default=Path("runs/grasp"))
    p.add_argument("--timesteps", type=int, default=12_000_000)
    p.add_argument("--n-envs", type=int, default=24)
    p.add_argument("--n-eval-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--asymmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--difficulty-start", type=float, default=0.2)
    p.add_argument("--curriculum-fraction", type=float, default=0.4)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per environment")
    p.add_argument("--batch-size", type=int, default=2560)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--eval-every", type=int, default=500_000, help="timesteps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--video-every", type=int, default=4, help="evaluations between eval videos (0: no videos)")
    p.add_argument("--video-episodes", type=int, default=8, help="episodes tried to find a success and a failure")
    p.add_argument("--milestones", type=int, default=10, help="checkpoints kept over the run (0: none)")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument(
        "--mlflow-uri",
        default=None,
        help="tracking URI (default: from docker/.env, else $MLFLOW_TRACKING_URI, else local server)",
    )
    p.add_argument("--experiment", default="grasp", help="MLflow experiment name")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    th.set_num_threads(args.torch_threads)
    uri = args.mlflow_uri or tracking_uri()
    check_server(uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.experiment)
    run_dir = args.out / args.name
    (run_dir / "best").mkdir(parents=True, exist_ok=True)
    config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    config["grasp_config"] = asdict(GraspConfig())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=float))
    artifacts = ArtifactLogger(run_dir, args.timesteps, args.milestones, args.video_every)
    with mlflow.start_run(run_name=args.name):
        mlflow.log_params(flatten_params(config))
        mlflow.set_tags({"task": "grasp", "algorithm": "PPO", "run_dir": str(run_dir)})
        artifacts.start()
        try:
            _train(args, run_dir, artifacts)
        finally:  # keep whatever was saved, also when training is interrupted
            artifacts.finalize()


def _train(args: argparse.Namespace, run_dir: Path, artifacts: ArtifactLogger) -> None:
    venv = VecNormalize(make_vec_env(args.n_envs, args.difficulty_start, args.asymmetric, args.seed), gamma=args.gamma)
    eval_venv = VecNormalize(
        make_vec_env(args.n_eval_envs, 1.0, args.asymmetric, args.seed + 10_000), training=False, norm_reward=False
    )
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [512, 512]},
        "activation_fn": nn.ELU,
        "log_std_init": args.log_std_init,
        "actor_keys": GRASP_OBS_KEYS,
        "critic_keys": (PRIVILEGED_KEY,) if args.asymmetric else (),
    }
    model = PPO(
        AsymmetricActorCriticPolicy,
        venv,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    model.set_logger(Logger(str(run_dir), [HumanOutputFormat(sys.stdout), MLflowOutputFormat()]))
    eval_cb = GraspEvalCallback(
        eval_venv,
        run_dir=run_dir,
        artifacts=artifacts,
        video_episodes=args.video_episodes,
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(args.eval_every // args.n_envs, 1),
        best_model_save_path=str(run_dir / "best"),
        callback_on_new_best=SaveNormalizerOnBest(run_dir / "best" / "vecnormalize.pkl"),
        deterministic=True,
    )
    callbacks = CallbackList(
        [
            DifficultyCurriculum(args.difficulty_start, int(args.curriculum_fraction * args.timesteps)),
            EpisodeMetrics(),
            eval_cb,
        ]
    )
    t0 = time.time()
    try:
        model.learn(args.timesteps, callback=callbacks)
    finally:
        model.save(run_dir / "model.zip")
        venv.save(str(run_dir / "vecnormalize.pkl"))
        elapsed = time.time() - t0
        summary = {
            "timesteps": int(model.num_timesteps),
            "wall_clock_s": elapsed,
            "steps_per_s": model.num_timesteps / max(elapsed, 1e-9),
            "best_eval_mean_reward": float(getattr(eval_cb, "best_mean_reward", np.nan)),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_metrics(
            {f"summary/{k}": float(v) for k, v in summary.items() if np.isfinite(v)}, step=int(model.num_timesteps)
        )
        venv.close()
        eval_venv.close()
        eval_cb.close_video_env()


if __name__ == "__main__":
    main()
