"""The GPU reorientation environment must reproduce the CPU reference (RotationEnv)."""

import mujoco
import numpy as np
import pytest
import torch

from manip_rubiks.environments.rotation_env import RotationConfig, RotationEnv
from manip_rubiks.simulation.rotations import mat_to_quat

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

N = 4
STATE_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart")


@pytest.fixture(scope="module")
def setup():
    from manip_rubiks.gpu.reset_bank import build_reset_bank
    from manip_rubiks.gpu.rotation_env import RotationEnvGPU

    cfg = RotationConfig()
    bank = build_reset_bank(cfg, range(N), workers=1, env_cls=RotationEnv)
    env = RotationEnvGPU(N, cfg, bank=bank, seed=0)
    yield cfg, bank, env
    env.close()


def sync(env) -> None:
    import warp as wp

    torch.cuda.synchronize()
    with wp.ScopedDevice(env.wp_device):
        wp.capture_launch(env._forward_graph)
    wp.synchronize_device()


def copy_state(env, world: int, data) -> None:
    for f in STATE_FIELDS:
        getattr(env, f)[world] = torch.as_tensor(getattr(data, f), dtype=torch.float32, device=env.device)
    sync(env)


def match_goal(env, world: int, cpu: RotationEnv) -> None:
    """Give GPU world ``world`` the CPU environment's goal and command."""
    env.goal[world] = torch.as_tensor(cpu.goal, dtype=torch.float32, device=env.device)
    env.command[world] = cpu.command
    cq = env._cube_quantities()
    env._cq = cq
    env.r_hand = env._hand_rotation(cq)
    env.theta_prev[world] = env.math.angle(env.r_hand[world].T @ env.goal[world])


def test_bank_reproduces_cpu_resets_with_locked_layers(setup):
    cfg, bank, _ = setup
    cpu = RotationEnv(cfg)
    for i, seed in enumerate(bank.seeds):
        cpu.reset(seed=int(seed))
        np.testing.assert_array_equal(bank.qpos[i], cpu.data.qpos)
    assert cpu.model.neq == 8 and cpu.data.eq_active.all()


def test_reset_observations_match_cpu(setup):
    from manip_rubiks.gpu.rotation_env import flatten_rotation_observation

    cfg, bank, env = setup
    actor, critic = env.reset(seed=0)
    assert actor.shape == (N, env.obs_dim) and critic.shape == (N, env.obs_dim + env.privileged_dim)
    theta = env.math.angle(env.r_hand.transpose(-1, -2) @ env.goal)
    np.testing.assert_allclose(theta.cpu().numpy(), np.pi / 2, atol=0.1)  # first goal: a quarter-turn away
    assert torch.isin(env.command, env.allowed).all()
    cpu = RotationEnv(cfg)
    for i in range(N):
        obs, info = cpu.reset(seed=int(bank.seeds[i]))
        match_goal(env, i, cpu)
        got = env._actor_obs()[i].cpu().numpy()
        want = flatten_rotation_observation(obs, env.obs_keys)
        np.testing.assert_allclose(got[:-3], want[:-3], atol=2e-4)
        np.testing.assert_allclose(got[-3:], want[-3:], atol=1e-3)  # settling cube angular velocity, float32 noise
        np.testing.assert_allclose(env._privileged(env._cq)[i].cpu().numpy(), info["privileged"], atol=1e-3)


def test_short_rollout_tracks_cpu(setup):
    cfg, bank, env = setup
    env.reset(seed=0)
    cpus = [RotationEnv(cfg) for _ in range(N)]
    for i, e in enumerate(cpus):
        e.reset(seed=int(bank.seeds[i]))
        match_goal(env, i, e)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(-0.3, 0.3, (N, 22)).astype(np.float32)
        _, _, reward, term, trunc, info, _ = env.step(torch.as_tensor(a, device=env.device))
        ref = [e.step(a[i]) for i, e in enumerate(cpus)]
        np.testing.assert_allclose(info.theta.cpu().numpy(), [r[4]["theta"] for r in ref], atol=2e-3)
        np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in ref], atol=0.05)
        assert not term.any() and not trunc.any()


def test_reaching_a_goal_issues_the_next(setup):
    cfg, bank, env = setup
    env.reset(seed=0)
    cpu = RotationEnv(cfg)
    cpu.reset(seed=int(bank.seeds[1]), options={"commands": [0]})  # spin about the palm normal
    pos, _ = cpu.cube.core_pose()
    cpu.cube.set_pose(pos, mat_to_quat(cpu.r_mount @ cpu.goal @ cpu.cube_reading.frame.T))
    mujoco.mj_forward(cpu.model, cpu.data)
    copy_state(env, 1, cpu.data)
    old_goal = torch.as_tensor(cpu.goal, dtype=torch.float32, device=env.device)
    env.goal[1] = old_goal
    env.command[1] = 0
    env.hold[1] = 0
    env.goal_steps[1] = 0
    for _ in range(cfg.hold_steps):
        _, _, reward, term, _, info, _ = env.step(torch.zeros(N, 22, device=env.device))
        assert not term[1]
    assert info.goal_reached[1] and info.goals_reached[1] == 1
    assert info.reward_terms["goal"][1] == cfg.reward.goal
    assert env.math.angle(old_goal.T @ env.goal[1]).item() == pytest.approx(np.pi / 2, abs=1e-4)
    assert env.goal_steps[1] == 0 and env.hold[1] == 0


def test_goal_timeout_terminates_and_resets():
    from manip_rubiks.gpu.reset_bank import build_reset_bank
    from manip_rubiks.gpu.rotation_env import RotationEnvGPU

    cfg = RotationConfig(goal_timeout=3)
    env = RotationEnvGPU(2, cfg, bank=build_reset_bank(cfg, range(2), workers=1, env_cls=RotationEnv), seed=0)
    env.reset(seed=0)
    for _ in range(3):
        _, _, reward, term, trunc, info, _ = env.step(torch.zeros(2, 22, device=env.device))
    assert term.all() and info.timed_out.all() and not info.is_success.any()
    assert (info.reward_terms["timeout"] == -cfg.reward.timeout).all()
    assert (info.reward_terms["failure"] == 0).all()
    assert (env.steps == 0).all() and (env.goal_steps == 0).all()
    env.close()
