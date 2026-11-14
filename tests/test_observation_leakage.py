"""The actor observation must not carry privileged cube state (docs/project-draft.md section 3)."""

import numpy as np
import pytest

from manip_rubiks.environments import OBS_KEYS, EnvConfig, HandCubeEnv
from manip_rubiks.planning.cube_state import CubeState


@pytest.fixture(scope="module")
def pair():
    cfg = EnvConfig(pose_noise=0.003, joint_noise=0.05)
    return HandCubeEnv(cfg), HandCubeEnv(cfg)


def test_observation_contains_only_robot_and_tactile_signals(pair):
    env, _ = pair
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(OBS_KEYS)
    assert set(env.observation_space.spaces) == set(OBS_KEYS)
    n = env.robot.n
    sizes = {k: v.size for k, v in obs.items()}
    assert sizes == {
        "q": n,
        "qd": n,
        "effort": n,
        "q_target": n,
        "tactile": int(np.prod(env.tactile.shape)),
        "prev_action": n,
    }


def test_observations_do_not_depend_on_the_scramble(pair):
    """Cubies are physically identical, so a scramble is invisible without vision.

    If any observation channel differed between two cubes that differ only in
    their puzzle state, it would leak the symbolic configuration.
    """
    a, b = pair
    scrambled = CubeState.solved().apply_sequence("R U' F2 R2 U")
    oa, ia = a.reset(seed=1)
    ob, ib = b.reset(seed=1, options={"state": scrambled})
    assert not ia["cube_state"].same_configuration(ib["cube_state"])
    rng = np.random.default_rng(0)
    for _ in range(10):
        for k in OBS_KEYS:
            # Cubie order differs in qpos, so rounding differs; over longer horizons contact
            # events (stick-slip) amplify it chaotically, hence the short horizon.
            np.testing.assert_allclose(oa[k], ob[k], atol=1e-4, err_msg=k)
        action = rng.uniform(-1, 1, a.robot.n)
        oa, *_ = a.step(action)
        ob, *_ = b.step(action)
    # The privileged channel does see the difference.
    assert not np.allclose(ia["privileged"][13:], ib["privileged"][13:])


def test_privileged_state_layout(pair):
    env, _ = pair
    _, info = env.reset(seed=0)
    p = info["privileged"]
    assert p.shape == (3 + 4 + 6 + 72,)
    np.testing.assert_allclose(np.linalg.norm(p[3:7]), 1.0)
    rots = p[13:].reshape(8, 3, 3)
    np.testing.assert_allclose(np.einsum("nij,nkj->nik", rots, rots), np.broadcast_to(np.eye(3), (8, 3, 3)), atol=1e-9)
