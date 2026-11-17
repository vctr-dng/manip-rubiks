"""In-hand whole-cube reorientation by chained quarter-turns (docs/project-draft.md section 24).

The hand holds the cube palm-up with a fixed wrist. Each goal is the current
goal orientation rotated by 90 degrees about one hand axis. The six commands
are ``+x, -x, +y, -y, +z, -z`` in the hand frame: x is the palm normal (a
spin on the palm), y points across the palm (towards the thumb for the right
hand) and z along the fingers. A goal counts as reached once the cube stays
within ``goal_tolerance`` of it for ``hold_steps`` control steps, and then the
next command is issued at once. An episode fails if a goal is not reached
within ``goal_timeout`` steps, or if the cube is dropped or a layer turns.

The cube's layers are locked by default (``SceneConfig.lock_layers``), so it
moves as one rigid body and the skill can be learned without accidental face
turns. Orientations are compared with the stickers taken into account: the
24 orientations are all different goals.

Observation: the base observation plus ``command``, a one-hot of the active
command. ``command`` is not privileged; a blindfolded robot knows what it was
asked to do. With ``teacher=True`` (the default) the actor also gets
``teacher``, privileged ground truth for a teacher policy (draft section 25):
the rotation from the current orientation to the goal in the hand frame (first
two columns, 6), the cube position relative to its rest position (3) and the
cube angular velocity (3), all in the hand frame. A tactile student replaces
it once a state estimator exists.

Reward (privileged ground truth is allowed here, draft sections 13 and 25):

    r =   w_progress * (theta_{t-1} - theta_t) / 0.1 rad
        + w_goal     on reaching a goal
        - w_disp     * displacement / 1 cm
        - w_align    * layer misalignment / 0.1 rad
        - w_effort   * mean((actuator force / force limit)^2)
        - w_rate     * mean((a_t - a_{t-1})^2)
        - w_fail     on termination (dropped, unstable, layer turned, goal timeout)

where ``theta`` is the angle between the cube and goal orientations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from manip_rubiks.environments.hand_cube_env import OBS_KEYS, EnvConfig, HandCubeEnv
from manip_rubiks.planning.cube_state import ROTATIONS, snap_rotation
from manip_rubiks.simulation.rotations import axis_angle_matrix, rotation_angle
from manip_rubiks.simulation.scene_builder import SceneConfig

N_COMMANDS = 6
COMMAND_NAMES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")
TEACHER_KEYS: tuple[str, ...] = ("teacher",)
TEACHER_DIM = 12
# Teacher observation bounds: rotation entries, position [m], angular velocity [rad/s].
TEACHER_BOUND = np.array([1.0] * 6 + [0.2] * 3 + [50.0] * 3, dtype=np.float32)
ROTATION_STUDENT_OBS_KEYS: tuple[str, ...] = (*OBS_KEYS, "command")
ROTATION_OBS_KEYS: tuple[str, ...] = (*ROTATION_STUDENT_OBS_KEYS, *TEACHER_KEYS)


def command_rotation(command: int) -> np.ndarray:
    """Hand-frame rotation of a quarter-turn command (0..5 = +x, -x, +y, -y, +z, -z)."""
    axis = np.zeros(3)
    axis[command // 2] = 1.0
    return np.rint(axis_angle_matrix(axis, np.pi / 2 if command % 2 == 0 else -np.pi / 2))


def allowed_commands(axes: str) -> np.ndarray:
    """Command indices about the hand axes named in ``axes`` (e.g. ``"xz"``)."""
    out = [2 * "xyz".index(a) + s for a in axes for s in (0, 1)]
    if not out:
        raise ValueError("at least one goal axis is needed")
    return np.array(sorted(set(out)))


def _rotation_env_config() -> EnvConfig:
    return EnvConfig(scene=SceneConfig(lock_layers=True), pose_noise=0.003, joint_noise=0.05, max_episode_steps=400)


@dataclass
class RotationRewardWeights:
    progress: float = 1.0
    goal: float = 10.0
    displacement: float = 0.05
    alignment: float = 0.2
    effort: float = 0.1
    action_rate: float = 0.05
    failure: float = 50.0


@dataclass
class RotationConfig:
    env: EnvConfig = field(default_factory=_rotation_env_config)
    goal_axes: str = "xyz"  # hand axes the quarter-turn commands rotate about
    goal_tolerance: float = 0.4  # largest angle to the goal that counts as reached [rad]
    easy_tolerance: float = 0.6  # tolerance at difficulty 0 (curriculum)
    hold_steps: int = 5  # consecutive steps within tolerance to reach a goal
    goal_timeout: int = 150  # steps allowed per goal (6 s at 25 Hz)
    difficulty: float = 1.0  # in [0, 1]; tightens the goal tolerance (curriculum)
    teacher: bool = True  # add the privileged ``teacher`` observation
    align_tolerance: float = np.deg2rad(10.0)  # largest cubie misalignment for an intact cube [rad]
    reward: RotationRewardWeights = field(default_factory=RotationRewardWeights)

    def tolerance(self, difficulty: float) -> float:
        return self.easy_tolerance + (self.goal_tolerance - self.easy_tolerance) * float(difficulty)


class RotationEnv(HandCubeEnv):
    """Reorient the cube in the hand through a chain of quarter-turn goals.

    ``reset(options={"commands": [c0, c1, ...]})`` issues those commands first
    (indices into :data:`COMMAND_NAMES`), then random ones. The base
    environment's cube-state options still apply, but locked layers only
    accept the solved cube.
    """

    def __init__(self, config: RotationConfig | None = None, render_mode: str | None = None) -> None:
        self.rot_cfg = config or RotationConfig()
        super().__init__(self.rot_cfg.env, render_mode)
        self.r_mount = self.scene.r_mount  # hand frame -> world
        _, r_rest = self.scene.cube_rest_pose()
        self.r_rest_hand = self.r_mount.T @ r_rest
        self.commands = allowed_commands(self.rot_cfg.goal_axes)
        self.difficulty = float(self.rot_cfg.difficulty)
        extra = {"command": spaces.Box(0.0, 1.0, (N_COMMANDS,), np.float32)}
        if self.rot_cfg.teacher:
            extra["teacher"] = spaces.Box(-TEACHER_BOUND, TEACHER_BOUND, dtype=np.float32)
        self.obs_keys = ROTATION_OBS_KEYS if self.rot_cfg.teacher else ROTATION_STUDENT_OBS_KEYS
        self.observation_space = spaces.Dict({**self.observation_space.spaces, **extra})
        self.goal = np.eye(3)
        self.command = 0
        self.goals_reached = 0
        self.goal_steps = 0
        self.hold = 0
        self.theta = 0.0
        self._theta_prev = 0.0
        self._queued: list[int] = []
        self._last_action = np.zeros(self.robot.n)

    # -------------------------------------------------------- curriculum
    def set_difficulty(self, difficulty: float) -> None:
        """Tighten the goal tolerance; takes effect at the next reset."""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    @property
    def tolerance(self) -> float:
        return self.rot_cfg.tolerance(self.difficulty)

    # -------------------------------------------------------------- goals
    def cube_rotation_hand(self) -> np.ndarray:
        """Whole-cube orientation in the hand frame, fitted to the cubies."""
        if self.cube_reading is None:  # before the first ``_info`` call
            self.cube_reading = self.cube.read()
        _, r_core = self.cube.core_pose()
        return self.r_mount.T @ r_core @ self.cube_reading.frame

    def snapped_orientation(self, r_hand: np.ndarray) -> np.ndarray:
        """Nearest of the 24 cube orientations aligned with the rest pose."""
        k, _ = snap_rotation(self.r_rest_hand.T @ r_hand)
        return self.r_rest_hand @ ROTATIONS[k]

    def _next_command(self) -> int:
        if self._queued:
            return int(self._queued.pop(0))
        return int(self.np_random.choice(self.commands))

    def _issue_goal(self) -> None:
        self.command = self._next_command()
        self.goal = command_rotation(self.command) @ self.goal
        self.goal_steps = 0
        self.hold = 0
        self.theta = rotation_angle(self.cube_rotation_hand().T @ self.goal)
        self._theta_prev = self.theta

    # ----------------------------------------------------------- episode
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        super().reset(seed=seed, options=options)
        self._queued = [int(c) for c in options.get("commands", [])]
        self.goal = self.snapped_orientation(self.cube_rotation_hand())
        self.goals_reached = 0
        self._last_action = np.zeros(self.robot.n)
        self._issue_goal()
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        self._last_action = self._prev_action.copy()
        return super().step(action)

    # ------------------------------------------------------ task hooks
    def _info(self) -> dict[str, Any]:
        info = super()._info()
        r_hand = self.cube_rotation_hand()
        self.theta = rotation_angle(r_hand.T @ self.goal)
        if self._steps > 0:
            self.goal_steps += 1
            self.hold = self.hold + 1 if self.theta < self.tolerance else 0
        preserved = self.cube_reading.state.same_configuration(self.initial_state)
        misalignment = self.cube_reading.max_error
        timed_out = self.goal_steps >= self.rot_cfg.goal_timeout and self.hold < self.rot_cfg.hold_steps
        info.update(
            displacement=float(np.linalg.norm(info["privileged"][:3])),
            misalignment=misalignment,
            state_preserved=bool(preserved),
            theta=self.theta,
            command=self.command,
            goal_steps=self.goal_steps,
            goals_reached=self.goals_reached,
            timed_out=bool(timed_out),
            is_success=bool(
                not info["dropped"]
                and not info["unstable"]
                and preserved
                and not timed_out
                and misalignment < self.rot_cfg.align_tolerance
            ),
        )
        return info

    def compute_reward(self, action: np.ndarray, info: dict[str, Any]) -> float:
        """Total reward; the signed per-term contributions are stored in ``info["reward_terms"]``.

        Also advances the goal chain: when the current goal is reached, the
        next one is issued and ``info["goals_reached"]`` is updated.
        """
        w = self.rot_cfg.reward
        effort = self.robot.effort() / self.robot.force_limit
        failed = info["dropped"] or info["unstable"] or not info["state_preserved"] or info["timed_out"]
        reached = self.hold >= self.rot_cfg.hold_steps and not failed
        terms = {
            "progress": w.progress * (self._theta_prev - self.theta) / 0.1,
            "goal": w.goal if reached else 0.0,
            "displacement": -w.displacement * info["displacement"] / 0.01,
            "alignment": -w.alignment * info["misalignment"] / 0.1,
            "effort": -w.effort * float(np.mean(effort**2)),
            "action_rate": -w.action_rate * float(np.mean((action - self._last_action) ** 2)),
            "failure": -w.failure if failed else 0.0,
        }
        self._theta_prev = self.theta
        if reached:
            self.goals_reached += 1
            self._issue_goal()
            info.update(goals_reached=self.goals_reached, command=self.command, theta=self.theta, goal_steps=0)
        info["goal_reached"] = bool(reached)
        info["reward_terms"] = {k: float(v) for k, v in terms.items()}
        return float(sum(terms.values()))

    def task_terminated(self, info: dict[str, Any]) -> bool:
        return not info["state_preserved"] or info["timed_out"]

    def teacher_state(self) -> np.ndarray:
        """Privileged teacher input: goal error (6), position (3), angular velocity (3), hand frame."""
        err = self.goal @ self.cube_rotation_hand().T
        p = self.privileged_state()
        pos = self.r_mount.T @ p[:3]
        ang = self.r_mount.T @ p[10:13]
        return np.clip(np.concatenate([err[:, :2].ravel(), pos, ang]), -TEACHER_BOUND, TEACHER_BOUND)

    def _observation(self) -> dict[str, np.ndarray]:
        obs = super()._observation()
        onehot = np.zeros(N_COMMANDS, dtype=np.float32)
        onehot[self.command] = 1.0
        obs["command"] = onehot
        if self.rot_cfg.teacher:
            obs["teacher"] = self.teacher_state().astype(np.float32)
        return obs
