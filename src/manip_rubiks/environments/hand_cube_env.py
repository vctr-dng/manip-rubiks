"""Gymnasium environment for the fixed-wrist hand holding a 2x2x2 cube.

This is the base environment that task environments (grasp, rotation, face
turns, ...) build on by overriding :meth:`HandCubeEnv.compute_reward` and
:meth:`HandCubeEnv.task_terminated`. It owns the episode logic: seeded resets,
cube scrambling, control-rate stepping, drop and instability termination, and
time-limit truncation.

Observation (docs/project-draft.md section 4, ``o_t``) -- non-privileged only:

==============  ================================================================
``q``           joint positions (22)
``qd``          joint velocities (22)
``effort``      actuator forces, the analogue of joint-torque feedback (22)
``q_target``    current joint-position targets, i.e. the policy's own command (22)
``tactile``     :class:`~manip_rubiks.simulation.tactile_sensors.TactileSensors` reading
``prev_action`` previous normalised action (22)
==============  ================================================================

Nothing about the cube is observed directly. Ground truth is returned in
``info`` only (``info["privileged"]`` and ``info["cube_state"]``), for
rewards, asymmetric critics and evaluation (draft sections 3 and 25).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from manip_rubiks.planning.cube_state import FACES, CubeState, parse_moves
from manip_rubiks.simulation.scene_builder import HAND_PREFIX, HandCubeScene, SceneConfig
from manip_rubiks.simulation.tactile_sensors import TactileConfig, TactileSensors

OBS_KEYS: tuple[str, ...] = ("q", "qd", "effort", "q_target", "tactile", "prev_action")
JOINT_MARGIN = 0.5  # observation bound beyond the joint range [rad]
MAX_JOINT_SPEED = 100.0  # observation bound on joint speed [rad/s]; observed peaks are ~12 rad/s


@dataclass
class EnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    control_dt: float = 0.04  # policy period [s] (25 Hz)
    k_a: float = 0.05  # largest joint-target change per control step [rad]
    max_episode_steps: int = 250  # truncation horizon (10 s at 25 Hz)
    scramble_moves: int = 0  # random legal moves applied to the cube at reset
    scramble_faces: tuple[str, ...] | None = None  # None: all faces the cube variant supports
    pose_noise: float = 0.0  # cube placement noise at reset [m]
    joint_noise: float = 0.0  # initial joint noise at reset [rad]
    drop_distance: float = 0.05  # cube centre this far from its rest position counts as dropped [m]


class HandCubeEnv(gym.Env):
    """Fixed-wrist Sharpa Wave with a cube resting in the palm.

    Actions are normalised joint-target deltas ``a in [-1, 1]^22`` (draft
    section 12). ``reset`` accepts ``options={"state": CubeState}`` or
    ``options={"moves": "R U' ..."}`` to choose the initial puzzle state;
    otherwise ``EnvConfig.scramble_moves`` random moves are applied.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        self.cfg = config or EnvConfig()
        if self.cfg.scene.variant is None:
            raise ValueError("HandCubeEnv needs a cube; use a scene variant other than None")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.scene = HandCubeScene(self.cfg.scene)
        self.model, self.data = self.scene.model, self.scene.data
        self.robot, self.cube = self.scene.robot, self.scene.cube
        self.tactile = TactileSensors(self.model, self.data, self.cfg.scene.side, HAND_PREFIX, self.cfg.tactile)
        self.n_substeps = max(1, round(self.cfg.control_dt / self.model.opt.timestep))
        self.metadata = {**self.metadata, "render_fps": round(1.0 / (self.n_substeps * self.model.opt.timestep))}
        self.rest_pos, _ = self.scene.cube_rest_pose()
        if self.cfg.scene.lock_layers and self.cfg.scramble_moves:
            raise ValueError("locked layers hold the solved cube; use scramble_moves=0")
        self.scramble_faces = self.cfg.scramble_faces or (
            ("U", "D") if self.cfg.scene.variant == "single_axis" else FACES
        )

        n = self.robot.n
        r = self.robot

        def box(low, high) -> spaces.Box:
            return spaces.Box(np.asarray(low, np.float32), np.asarray(high, np.float32), dtype=np.float32)

        if self.cfg.tactile.mode == "contact":
            tactile = spaces.Box(0.0, 1.0, self.tactile.shape, np.float32)
        else:
            f = self.cfg.tactile.max_force
            tactile = spaces.Box(-f, f, self.tactile.shape, np.float32)
        self.observation_space = spaces.Dict(
            {
                # Soft joint limits can be exceeded slightly under contact.
                "q": box(r.q_min - JOINT_MARGIN, r.q_max + JOINT_MARGIN),
                "qd": box(np.full(n, -MAX_JOINT_SPEED), np.full(n, MAX_JOINT_SPEED)),
                "effort": box(-r.force_limit, r.force_limit),
                "q_target": box(r.ctrl_min, r.ctrl_max),
                "tactile": tactile,
                "prev_action": spaces.Box(-1.0, 1.0, (n,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)

        self._prev_action = np.zeros(n)
        self._steps = 0
        self._tactile_reading = np.zeros(self.tactile.shape)
        self.initial_state = CubeState.solved()
        self.cube_reading = None  # snapped ground-truth reading from the last ``_info`` call
        self._renderer: mujoco.Renderer | None = None

    # ------------------------------------------------------------- episode
    def _initial_cube_state(self, options: dict[str, Any]) -> CubeState:
        if "state" in options:
            return options["state"]
        if "moves" in options:
            moves = options["moves"]
            return CubeState.solved().apply_sequence(parse_moves(moves) if isinstance(moves, str) else moves)
        moves = CubeState.random_moves(self.cfg.scramble_moves, self.np_random, self.scramble_faces)
        return CubeState.solved().apply_sequence(moves)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.initial_state = self._initial_cube_state(options)
        if self.cfg.scene.lock_layers and not self.initial_state.same_configuration(CubeState.solved()):
            raise ValueError("locked layers hold the solved cube; it cannot start scrambled")
        scene_seed, tactile_seed = (int(s) for s in self.np_random.integers(2**31, size=2))
        self.scene.reset(
            seed=scene_seed,
            state=self.initial_state,
            pose_noise=self.cfg.pose_noise,
            joint_noise=self.cfg.joint_noise,
        )
        self.tactile.reset(tactile_seed)
        self._prev_action = np.zeros(self.robot.n)
        self._steps = 0
        self._tactile_reading = self.tactile.read()
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.robot.apply_action(action, self.cfg.k_a)
        mujoco.mj_step(self.model, self.data, self.n_substeps)
        self._prev_action = action
        self._steps += 1
        self._tactile_reading = self.tactile.read()

        info = self._info()
        reward = float(self.compute_reward(action, info))
        terminated = bool(info["dropped"] or info["unstable"] or self.task_terminated(info))
        truncated = bool(not terminated and self._steps >= self.cfg.max_episode_steps)
        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------- task hooks
    def compute_reward(self, action: np.ndarray, info: dict[str, Any]) -> float:
        """Task reward; the base environment has no task."""
        return 0.0

    def task_terminated(self, info: dict[str, Any]) -> bool:
        """Task-specific success or failure termination."""
        return False

    # ----------------------------------------------------- observations
    def _observation(self) -> dict[str, np.ndarray]:
        r = self.robot
        obs = {
            "q": r.q(),
            "qd": r.qd(),
            "effort": r.effort(),
            "q_target": r.target(),
            "tactile": self._tactile_reading,
            "prev_action": self._prev_action,
        }
        return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}

    def privileged_state(self) -> np.ndarray:
        """Flat ground-truth cube state for critics and rewards (never for the actor).

        Layout: cube centre relative to its rest position (3), core quaternion
        (4), core linear and angular velocity (6), cubie rotations relative to
        the core, row-major (8 x 9).
        """
        pos, _ = self.cube.core_pose()
        quat = self.data.xquat[self.cube.core_body]
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.cube.core_body, vel, 0)
        return np.concatenate([pos - self.rest_pos, quat, vel[3:], vel[:3], self.cube.cubie_rotations().ravel()])

    def _info(self) -> dict[str, Any]:
        pos, _ = self.cube.core_pose()
        unstable = self.scene.warnings() > 0 or not (
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        self.cube_reading = self.cube.read()  # privileged; cached for task rewards
        return {
            "steps": self._steps,
            "sim_time": float(self.data.time),
            "dropped": bool(np.linalg.norm(pos - self.rest_pos) > self.cfg.drop_distance),
            "unstable": bool(unstable),
            "cube_state": self.cube_reading.state,
            "initial_state": self.initial_state,
            "privileged": self.privileged_state(),
        }

    # ----------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
