"""Episode logic of the Gymnasium environment: seeding, resets, termination, leakage."""

from __future__ import annotations

import time
import warnings

import numpy as np
from gymnasium.utils.env_checker import check_env

from manip_rubiks.environments import OBS_KEYS, EnvConfig, HandCubeEnv
from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.validation.common import CheckResult

NOISY = {"pose_noise": 0.004, "joint_noise": 0.05}


def _obs_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _max_obs_diff(a: dict, b: dict) -> float:
    return max(float(np.abs(a[k] - b[k]).max()) for k in OBS_KEYS)


def check_episode_logic(rollout_steps: int = 40) -> CheckResult:
    env = HandCubeEnv(EnvConfig(scramble_moves=4, **NOISY))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(HandCubeEnv(EnvConfig(scramble_moves=4, **NOISY)), skip_render_check=True)
        checker = "pass"
    except Exception as exc:  # report the checker's complaint rather than crash the suite
        checker = f"{type(exc).__name__}: {exc}"

    # Seeding.
    o1, i1 = env.reset(seed=11)
    o2, i2 = env.reset(seed=11)
    o3, i3 = env.reset(seed=12)
    same_seed = _obs_equal(o1, o2) and np.array_equal(i1["privileged"], i2["privileged"])
    same_scramble = i1["initial_state"] == i2["initial_state"]
    different_seed = not np.array_equal(o1["q"], o3["q"]) and i1["initial_state"] != i3["initial_state"]

    # Reset after a rollout equals a reset of a fresh environment, and both
    # keep stepping identically.
    rng = np.random.default_rng(0)
    for _ in range(rollout_steps):
        env.step(rng.uniform(-1, 1, env.robot.n))
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=4)
    reset_ms = (time.perf_counter() - t0) * 1e3
    fresh = HandCubeEnv(EnvConfig(scramble_moves=4, **NOISY))
    obs_fresh, _ = fresh.reset(seed=4)
    d, df = env.data, fresh.data
    reset_equal = (
        _obs_equal(obs, obs_fresh)
        and np.array_equal(d.qpos, df.qpos)
        and np.array_equal(d.qvel, df.qvel)
        and np.array_equal(d.ctrl, df.ctrl)
        and d.time == df.time
    )
    t0 = time.perf_counter()
    for k in range(10):
        a = np.full(env.robot.n, 0.3 * (-1) ** k)
        reset_equal &= _obs_equal(env.step(a)[0], fresh.step(a)[0])
    steps_per_s = 10 / (time.perf_counter() - t0) * 2  # two environments stepped

    # Leakage: a scramble is physically invisible, so observations must match.
    scrambled = CubeState.solved().apply_sequence("R U' F2 R2 U")
    oa, ia = env.reset(seed=1, options={"state": CubeState.solved()})
    ob, ib = fresh.reset(seed=1, options={"state": scrambled})
    leak = _max_obs_diff(oa, ob)
    for _ in range(10):
        action = rng.uniform(-1, 1, env.robot.n)
        oa, *_ = env.step(action)
        ob, *_ = fresh.step(action)
        leak = max(leak, _max_obs_diff(oa, ob))
    privileged_differs = not np.allclose(ia["privileged"][13:], ib["privileged"][13:])

    # Termination and truncation.
    short = HandCubeEnv(EnvConfig(max_episode_steps=6))
    short.reset(seed=0)
    flags = [short.step(np.zeros(short.robot.n))[2:4] for _ in range(6)]
    truncation_ok = flags == [(False, False)] * 5 + [(False, True)]
    short.reset(seed=0)
    pos, _ = short.cube.core_pose()
    short.cube.set_pose(pos + np.array([0.0, 0.2, 0.0]), short.data.xquat[short.cube.core_body].copy())
    *_, terminated, truncated, info = short.step(np.zeros(short.robot.n))
    drop_ok = bool(info["dropped"] and terminated and not truncated)

    metrics = {
        "gymnasium_env_checker": checker,
        "same_seed_identical_start": bool(same_seed),
        "same_seed_identical_scramble": bool(same_scramble),
        "different_seed_differs": bool(different_seed),
        "reset_after_rollout_equals_fresh_env": bool(reset_equal),
        "max_obs_diff_solved_vs_scrambled": float(f"{leak:.3g}"),
        "privileged_state_sees_scramble": bool(privileged_differs),
        "truncates_at_horizon": bool(truncation_ok),
        "drop_terminates": drop_ok,
        "info_reset_ms": round(reset_ms, 1),
        "info_control_steps_per_s": round(steps_per_s),
    }
    limits = {"max_obs_diff_solved_vs_scrambled": 1e-4}
    passed = (
        checker == "pass"
        and same_seed
        and same_scramble
        and different_seed
        and reset_equal
        and leak < limits["max_obs_diff_solved_vs_scrambled"]
        and privileged_differs
        and truncation_ok
        and drop_ok
    )
    notes = [
        "Observation leakage: cubies are physically identical, so solved and scrambled cubes must give the "
        "same observations under the same actions; only float rounding (different cubie order) may differ.",
    ]
    return CheckResult("episode_logic", bool(passed), metrics, limits, notes)
