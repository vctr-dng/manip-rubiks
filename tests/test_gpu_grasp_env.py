"""The GPU grasp environment must reproduce the CPU reference (GraspEnv)."""

import mujoco
import numpy as np
import pytest
import torch

from manip_rubiks.environments.grasp_env import GraspConfig, GraspEnv
from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.policies import ScriptedGraspPolicy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

N = 4
STATE_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart")


@pytest.fixture(scope="module")
def setup():
    from manip_rubiks.gpu.grasp_env import GraspEnvGPU
    from manip_rubiks.gpu.reset_bank import build_reset_bank

    cfg = GraspConfig(difficulty=0.0)  # palm up, no pushes: deterministic and comparable
    bank = build_reset_bank(cfg, range(N), workers=1)
    env = GraspEnvGPU(N, cfg, bank=bank, seed=0)
    yield cfg, bank, env
    env.close()


def copy_state(env, world, data):
    for f in STATE_FIELDS:
        getattr(env, f)[world] = torch.as_tensor(getattr(data, f), dtype=torch.float32, device=env.device)
    env.xfrc[world] = 0.0
    torch.cuda.synchronize()
    import warp as wp

    with wp.ScopedDevice(env.wp_device):
        wp.capture_launch(env._forward_graph)
    wp.synchronize_device()


def test_bank_reproduces_cpu_resets(setup):
    cfg, bank, _ = setup
    cpu = GraspEnv(cfg)
    for i, seed in enumerate(bank.seeds):
        cpu.reset(seed=int(seed))
        np.testing.assert_array_equal(bank.qpos[i], cpu.data.qpos)
        np.testing.assert_array_equal(bank.ctrl[i], cpu.data.ctrl)


def test_reset_observations_match_cpu(setup):
    from manip_rubiks.gpu.grasp_env import flatten_observation

    cfg, bank, env = setup
    actor, critic = env.reset(seed=0)
    assert actor.shape == (N, env.obs_dim) and critic.shape == (N, env.obs_dim + env.privileged_dim)
    cpu = GraspEnv(cfg)
    for i in range(N):
        obs, info = cpu.reset(seed=int(bank.seeds[i]))
        np.testing.assert_allclose(actor[i].cpu().numpy(), flatten_observation(obs), atol=2e-4)
        np.testing.assert_allclose(critic[i, env.obs_dim :].cpu().numpy(), info["privileged"], atol=1e-3)


def test_sensing_matches_cpu_in_a_grasp(setup):
    """Copy CPU grasp states (with fingertip contacts) into the GPU worlds and compare readouts exactly."""
    cfg, _, env = setup
    env.reset(seed=0)
    cpu = GraspEnv(cfg)
    obs, _ = cpu.reset(seed=3)
    policy = ScriptedGraspPolicy()
    for _ in range(40):
        obs, *_ = cpu.step(policy(obs))
    # After mj_step, contact forces belong to the state before the last integration; recompute
    # them at the current state, which is what the GPU forward pass below does.
    mujoco.mj_forward(cpu.model, cpu.data)
    assert np.abs(cpu.tactile.measure()).sum() > 0.1  # fingertips are loaded
    copy_state(env, 1, cpu.data)
    tactile = env._measure_tactile()[1].cpu().numpy()
    ref_tactile = cpu.tactile.measure()
    # Fingertip contacts are identical, but MuJoCo Warp generates a couple of extra contacts
    # elsewhere (55 vs 53 here), which redistributes friction by a few percent.
    np.testing.assert_allclose(tactile, ref_tactile, atol=0.05 * np.abs(ref_tactile).max())
    cq = env._cube_quantities()
    priv = env._privileged(cq)[1].cpu().numpy()
    ref = cpu.privileged_state()
    np.testing.assert_allclose(priv[:10], ref[:10], atol=1e-4)  # position, quaternion, linear velocity
    np.testing.assert_allclose(priv[13:], ref[13:], atol=1e-4)  # cubie rotations
    reading = cpu.cube.read()
    assert env.math.same_configuration(cq["reading"].indices[1], env.math.canonical(cq["reading"].indices[1]))
    np.testing.assert_allclose(cq["reading"].max_error[1].item(), reading.max_error, atol=1e-4)


def test_short_rollout_tracks_cpu(setup):
    cfg, bank, env = setup
    env.reset(seed=0)
    cpus = [GraspEnv(cfg) for _ in range(N)]
    for i, e in enumerate(cpus):
        e.reset(seed=int(bank.seeds[i]))
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(-1, 1, (N, 22)).astype(np.float32)
        _, _, reward, term, trunc, info, _ = env.step(torch.as_tensor(a, device=env.device))
        ref = [e.step(a[i]) for i, e in enumerate(cpus)]
        np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in ref], atol=0.02)
        np.testing.assert_allclose(info.misalignment.cpu().numpy(), [r[4]["misalignment"] for r in ref], atol=5e-3)
        assert not term.any() and not trunc.any()


def test_gravity_schedule_matches_cpu(setup):
    from manip_rubiks.gpu.grasp_env import GraspEnvGPU  # noqa: F401

    cfg, _, env = setup
    cpu = GraspEnv(GraspConfig())
    cpu.reset(seed=7)
    moves = cpu.schedule[: env.n_moves]
    env.sched_valid[:] = False
    for j, m in enumerate(moves):
        env.sched_t0[:, j], env.sched_t1[:, j] = m.t_start, m.t_end
        env.sched_from[:, j] = torch.as_tensor(m.d_from, dtype=torch.float32, device=env.device)
        env.sched_to[:, j] = torch.as_tensor(m.d_to, dtype=torch.float32, device=env.device)
        env.sched_valid[:, j] = True
    for t in np.arange(0.0, 10.0, 0.13):
        got = env.gravity_direction_at(torch.full((N,), t, device=env.device))[0].cpu().numpy()
        np.testing.assert_allclose(got, cpu.gravity_direction_at(t), atol=1e-5)


def test_layer_turn_terminates_and_world_resets(setup):
    from manip_rubiks.gpu.grasp_env import flatten_observation

    cfg, bank, env = setup
    env.reset(seed=0)
    cpu = GraspEnv(cfg)
    cpu.reset(seed=int(bank.seeds[2]))
    cpu.cube.set_state(CubeState.solved().apply("R"))
    mujoco.mj_forward(cpu.model, cpu.data)
    copy_state(env, 2, cpu.data)
    actor, _, reward, term, trunc, info, _ = env.step(torch.zeros(N, 22, device=env.device))
    assert term.tolist() == [False, False, True, False] and not trunc.any()
    assert not info.state_preserved[2] and not info.is_success[2]
    assert reward[2] < -cfg.reward.failure / 2
    assert env.steps[2] == 0 and env.steps[0] == 1  # world 2 was reset automatically
    assert env.math.same_configuration(env._cube_quantities()["reading"].indices[2], env.ref_canonical[2])
    assert actor[2, -25:-3].abs().sum() == 0  # prev_action cleared
    assert np.isfinite(flatten_observation(cpu.reset(seed=0)[0])).all()


def test_truncation_after_time_limit(setup):
    cfg, _, env = setup
    env.reset(seed=0)
    for k in range(env.max_steps):
        _, _, _, term, trunc, info, _ = env.step(torch.zeros(N, 22, device=env.device))
        if k < env.max_steps - 1:
            assert not trunc.any()
    assert trunc.all() and not term.any()
    assert info.is_success.all() and (info.episode_length == env.max_steps).all()
    assert (env.steps == 0).all()


def test_push_scale_critic_wrench_and_barrier_on_the_gpu():
    from manip_rubiks.environments.grasp_env import GraspRewardWeights
    from manip_rubiks.gpu.grasp_env import GraspEnvGPU
    from manip_rubiks.gpu.reset_bank import build_reset_bank

    cfg = GraspConfig(
        difficulty=1.0,
        push_rate=50.0,
        push_scale=2.0,
        critic_wrench=True,
        reward=GraspRewardWeights(align_barrier=3.0),
    )
    bank = build_reset_bank(cfg, range(N), workers=1)
    env = GraspEnvGPU(N, cfg, bank=bank, seed=0)
    cpu = GraspEnv(cfg)
    cpu.reset(seed=0)
    assert env.privileged_dim == cpu.privileged_state().size
    actor, critic = env.reset(seed=0)
    assert critic.shape == (N, env.obs_dim + env.privileged_dim)
    norm = torch.tensor([cfg.push_force] * 3 + [cfg.push_torque] * 3, device=env.device)
    peak = 0.0
    for _ in range(15):
        actor, critic, _, _, _, info, _ = env.step(torch.zeros(N, env.n_act, device=env.device))
        torch.testing.assert_close(critic[:, -6:], env.xfrc[:, env.core] / norm)
        peak = max(peak, float(critic[:, -6:-3].norm(dim=1).max()))
    assert 0.3 * 2.0 <= peak <= 2.0 + 1e-5, "pushes are applied, at twice the nominal strength at most"
    assert (info.reward_terms["barrier"] <= 0).all()
    env.close()
