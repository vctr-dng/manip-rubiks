"""Configuration C (native tactile taxels) on the GPU must reproduce the CPU reference."""

import mujoco
import numpy as np
import pytest
import torch

from manip_rubiks.environments.grasp_env import GraspConfig, GraspEnv
from manip_rubiks.environments.hand_cube_env import EnvConfig
from manip_rubiks.policies import ScriptedGraspPolicy
from manip_rubiks.simulation.tactile_sensors import TactileConfig
from manip_rubiks.simulation.taxels import COLS, ROWS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

N = 4


@pytest.fixture(scope="module")
def setup():
    from manip_rubiks.gpu.grasp_env import GraspEnvGPU
    from manip_rubiks.gpu.reset_bank import build_reset_bank

    env_cfg = EnvConfig(pose_noise=0.003, joint_noise=0.05, tactile=TactileConfig(mode="taxel"))
    cfg = GraspConfig(env=env_cfg, difficulty=0.0)
    bank = build_reset_bank(cfg, range(N), workers=1)
    env = GraspEnvGPU(N, cfg, bank=bank, seed=0)
    yield cfg, bank, env
    env.close()


def test_observation_contains_the_taxel_block(setup):
    cfg, _, env = setup
    assert env.tactile_shape == (5, ROWS, COLS, 3)
    actor, _ = env.reset(seed=0)
    assert actor.shape == (N, env.obs_dim)
    assert torch.isfinite(actor).all()
    sizes = dict(env.layout)
    assert int(np.prod(sizes["tactile"])) == 960


def test_taxels_match_cpu_in_a_grasp(setup):
    """Copy a CPU grasp state (fingertips loaded) into a GPU world and compare the taxel readout."""
    import warp as wp

    cfg, _, env = setup
    env.reset(seed=0)
    cpu = GraspEnv(cfg)
    obs, _ = cpu.reset(seed=3)
    policy = ScriptedGraspPolicy()
    for _ in range(40):
        obs, *_ = cpu.step(policy(obs))
    mujoco.mj_forward(cpu.model, cpu.data)
    ref = cpu.tactile.measure()
    assert ref[..., 0].max() > 0.0  # at least one taxel is indented

    for f in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        getattr(env, f)[1] = torch.as_tensor(getattr(cpu.data, f), dtype=torch.float32, device=env.device)
    env.xfrc[1] = 0.0
    torch.cuda.synchronize()
    with wp.ScopedDevice(env.wp_device):
        wp.capture_launch(env._forward_graph)
    wp.synchronize_device()

    gpu = env._measure_tactile()[1].cpu().numpy()
    assert gpu.shape == ref.shape
    # Warp runs in float32 and may generate a few different contacts elsewhere in the hand;
    # the skin readings themselves agree to ~1e-5 here.
    np.testing.assert_allclose(gpu, ref, atol=0.01)


def test_taxel_corruption_runs_on_the_gpu(setup, monkeypatch):
    _, _, env = setup
    env.reset(seed=0)
    monkeypatch.setattr(env, "tactile_cfg", TactileConfig(mode="taxel", noise_std=0.05, dropout=0.5))
    reading = env._corrupt(env._measure_tactile())
    assert reading.shape == (N, 5, ROWS, COLS, 3)
    assert reading.min() >= 0.0 and reading.max() <= 1.0
