"""Grasp stabilisation: keep hold of the cube while the wrist reorients (docs/project-draft.md section 24).

Palm-up, the cube simply rests in the hand. To make holding it a real task,
the environment emulates an arm reorienting the fixed wrist by rotating the
gravity vector around the hand. A static wrist rotation and a rotated gravity
vector are equivalent for the hand-cube dynamics. Arm accelerations are not
modelled. On top of that, random pushes (force and torque) hit the cube.

Each episode starts palm-up with the hand in its home cage. After a short
random delay the "wrist" follows a sequence of minimum-jerk reorientations to
random tilts up to ``max_tilt`` (pi = palm down), each followed by a dwell.
The policy has to close a grasp before the cube slides off and keep it
through the whole sequence.

Observation: the base observation plus ``gravity``, the unit gravity
direction in the hand frame. A real robot knows its wrist orientation from
the arm's joint encoders, so this is proprioception, not privileged state.

Reward (privileged ground truth is allowed here, draft sections 13 and 25):

    r = w_alive
        - w_disp   * displacement / 1 cm
        - w_rot    * whole-cube rotation / 0.5 rad
        - w_align  * layer misalignment / 0.1 rad
        - w_vel    * (|v| / 0.1 m/s + |w| / 2 rad/s)
        - w_effort * mean((actuator force / force limit)^2)
        - w_rate   * mean((a_t - a_{t-1})^2)
        - w_fail   on termination (dropped, unstable, or an unintended layer turn)

``info["is_success"]`` reports whether the hand currently holds the cube
intact: not dropped, the puzzle configuration unchanged, and every cubie
within ``align_tolerance`` of its snapped pose. Read at the end of a
truncated episode, it is the sustained-grasp success criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from manip_rubiks.environments.hand_cube_env import OBS_KEYS, EnvConfig, HandCubeEnv
from manip_rubiks.simulation.rotations import axis_angle_matrix, min_jerk, rotation_angle

GRASP_OBS_KEYS: tuple[str, ...] = (*OBS_KEYS, "gravity")
STANDARD_GRAVITY = 9.81


@dataclass
class GraspRewardWeights:
    alive: float = 1.0
    displacement: float = 0.1
    rotation: float = 0.1
    alignment: float = 0.2
    velocity: float = 0.05
    effort: float = 0.2
    action_rate: float = 0.05
    failure: float = 50.0


@dataclass
class GraspConfig:
    env: EnvConfig = field(default_factory=lambda: EnvConfig(pose_noise=0.003, joint_noise=0.05))
    difficulty: float = 1.0  # in [0, 1]; scales the tilt range and the push magnitudes (curriculum)
    max_tilt: float = np.pi  # largest wrist tilt away from palm-up [rad]
    start_delay: tuple[float, float] = (0.2, 0.6)  # palm-up time before the first reorientation [s]
    move_duration: tuple[float, float] = (0.8, 2.0)  # duration of one reorientation [s]
    dwell: tuple[float, float] = (0.4, 1.5)  # hold time between reorientations [s]
    push_rate: float = 0.3  # expected pushes per second
    push_force: float = 1.5  # largest push force on the cube [N] (cube weight: ~0.7 N)
    push_torque: float = 0.02  # largest push torque on the cube [N m]
    push_duration: tuple[float, float] = (0.1, 0.3)  # [s]
    align_tolerance: float = np.deg2rad(10.0)  # largest cubie misalignment for a successful hold [rad]
    reward: GraspRewardWeights = field(default_factory=GraspRewardWeights)


@dataclass(frozen=True)
class WristMove:
    t_start: float
    t_end: float
    d_from: np.ndarray  # unit gravity direction in the hand frame at the start
    d_to: np.ndarray


def slerp_direction(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Great-circle interpolation between unit vectors ``a`` and ``b``."""
    angle = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if angle < 1e-9:
        return b.copy()
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:  # antipodal: any perpendicular axis works
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    return axis_angle_matrix(axis, s * angle) @ a


class GraspEnv(HandCubeEnv):
    """Hold the cube through random wrist reorientations and pushes.

    ``reset(options={"tilt": t, "azimuth": p})`` replaces the random schedule
    with a single reorientation to tilt ``t`` in the in-palm direction ``p``
    (0 = towards the fingertips, pi/2 = towards the thumb side), held until
    the end of the episode. ``options={"pushes": False}`` disables pushes for
    that episode. The base environment's cube-state options still apply.
    """

    def __init__(self, config: GraspConfig | None = None, render_mode: str | None = None) -> None:
        self.grasp_cfg = config or GraspConfig()
        super().__init__(self.grasp_cfg.env, render_mode)
        self.g_nominal = self.model.opt.gravity.copy()
        self.r_mount = self.scene.r_mount  # hand frame -> world
        self.d_nominal = self.r_mount.T @ self.g_nominal / np.linalg.norm(self.g_nominal)
        self.difficulty = float(self.grasp_cfg.difficulty)
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "gravity": spaces.Box(-1.0, 1.0, (3,), np.float32)}
        )
        self.schedule: list[WristMove] = []
        self.gravity_dir = self.d_nominal.copy()
        self._push_until = -1.0
        self._pushes_enabled = True
        self._last_action = np.zeros(self.robot.n)
        self._r_cube0 = np.eye(3)

    # -------------------------------------------------------- curriculum
    def set_difficulty(self, difficulty: float) -> None:
        """Scale tilt range and pushes; takes effect at the next reset."""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ------------------------------------------------------ wrist motion
    def tilted_direction(self, tilt: float, azimuth: float) -> np.ndarray:
        """Gravity direction after tilting the wrist by ``tilt`` towards in-palm direction ``azimuth``.

        Tilting towards a direction ``u`` in the palm plane means the hand
        rotates so that gravity gains a component along ``u``, i.e. the cube
        tends to slide that way. Hand frame: +x palm normal, +y across the
        palm (thumb side for the right hand), +z along the fingers.
        """
        u = np.array([0.0, np.sin(azimuth), np.cos(azimuth)])
        axis = np.cross(self.d_nominal, u)
        if np.linalg.norm(axis) < 1e-9:
            return self.d_nominal.copy()
        return axis_angle_matrix(axis, tilt) @ self.d_nominal

    def _sample_schedule(self, options: dict[str, Any]) -> list[WristMove]:
        cfg, rng = self.grasp_cfg, self.np_random
        horizon = self.cfg.max_episode_steps * self.cfg.control_dt
        t = float(rng.uniform(*cfg.start_delay))
        d = self.d_nominal.copy()
        if "tilt" in options:
            target = self.tilted_direction(float(options["tilt"]), float(options.get("azimuth", 0.0)))
            return [WristMove(t, t + float(np.mean(cfg.move_duration)), d, target)]
        moves = []
        while t < horizon:
            tilt = rng.uniform(0.0, cfg.max_tilt * self.difficulty)
            target = self.tilted_direction(tilt, rng.uniform(0.0, 2 * np.pi))
            t_end = t + float(rng.uniform(*cfg.move_duration))
            moves.append(WristMove(t, t_end, d, target))
            d = target
            t = t_end + float(rng.uniform(*cfg.dwell))
        return moves

    def gravity_direction_at(self, t: float) -> np.ndarray:
        d = self.d_nominal
        for m in self.schedule:
            if t < m.t_start:
                break
            if t < m.t_end:
                s, _ = min_jerk((t - m.t_start) / (m.t_end - m.t_start))
                return slerp_direction(m.d_from, m.d_to, s)
            d = m.d_to
        return d

    def tilt(self) -> float:
        return float(np.arccos(np.clip(self.gravity_dir @ self.d_nominal, -1.0, 1.0)))

    def _set_gravity(self, d_hand: np.ndarray) -> None:
        self.gravity_dir = d_hand / np.linalg.norm(d_hand)
        self.model.opt.gravity[:] = STANDARD_GRAVITY * (self.r_mount @ self.gravity_dir)

    # ------------------------------------------------------------ pushes
    def _update_push(self, t: float) -> None:
        cfg, rng = self.grasp_cfg, self.np_random
        body = self.cube.core_body
        if t < self._push_until:
            return
        self.data.xfrc_applied[body] = 0.0
        scale = cfg.push_force * self.difficulty
        if not self._pushes_enabled or scale <= 0 or rng.random() >= cfg.push_rate * self.cfg.control_dt:
            return

        def random_unit() -> np.ndarray:
            v = rng.standard_normal(3)
            return v / np.linalg.norm(v)

        self.data.xfrc_applied[body, :3] = random_unit() * scale * rng.uniform(0.3, 1.0)
        self.data.xfrc_applied[body, 3:] = random_unit() * cfg.push_torque * self.difficulty * rng.uniform(0.3, 1.0)
        self._push_until = t + float(rng.uniform(*cfg.push_duration))

    # ----------------------------------------------------------- episode
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        self._set_gravity(self.d_nominal)  # the scene settles palm-up
        super().reset(seed=seed, options=options)
        self.schedule = self._sample_schedule(options)
        self._pushes_enabled = bool(options.get("pushes", True))
        self._push_until = -1.0
        self._last_action = np.zeros(self.robot.n)
        self._r_cube0 = self._cube_rotation()
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        t = self._steps * self.cfg.control_dt  # start of the control period about to be simulated
        self._set_gravity(self.gravity_direction_at(t))
        self._update_push(t)
        self._last_action = self._prev_action.copy()
        return super().step(action)

    # ------------------------------------------------------ task hooks
    def _cube_rotation(self) -> np.ndarray:
        """Whole-cube orientation, fitted to the cubies (the core may spin inside them)."""
        _, r_core = self.cube.core_pose()
        return r_core @ self.cube_reading.frame

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        v = info["privileged"][7:13]
        preserved = self.cube_reading.state.same_configuration(self.initial_state)
        misalignment = self.cube_reading.max_error
        info.update(
            displacement=float(np.linalg.norm(info["privileged"][:3])),
            rotation=rotation_angle(self._r_cube0.T @ self._cube_rotation()),
            misalignment=misalignment,
            state_preserved=bool(preserved),
            linear_speed=float(np.linalg.norm(v[:3])),
            angular_speed=float(np.linalg.norm(v[3:])),
            tilt=self.tilt(),
            pushing=bool(np.any(self.data.xfrc_applied[self.cube.core_body])),
            is_success=bool(
                not info["dropped"]
                and not info["unstable"]
                and preserved
                and misalignment < self.grasp_cfg.align_tolerance
            ),
        )
        return info

    def compute_reward(self, action: np.ndarray, info: dict[str, Any]) -> float:
        w = self.grasp_cfg.reward
        effort = self.robot.effort() / self.robot.force_limit
        r = (
            w.alive
            - w.displacement * info["displacement"] / 0.01
            - w.rotation * info["rotation"] / 0.5
            - w.alignment * info["misalignment"] / 0.1
            - w.velocity * (info["linear_speed"] / 0.1 + info["angular_speed"] / 2.0)
            - w.effort * float(np.mean(effort**2))
            - w.action_rate * float(np.mean((action - self._last_action) ** 2))
        )
        if info["dropped"] or info["unstable"] or not info["state_preserved"]:
            r -= w.failure
        return r

    def task_terminated(self, info: dict[str, Any]) -> bool:
        return not info["state_preserved"]

    def _observation(self) -> dict[str, np.ndarray]:
        obs = super()._observation()
        obs["gravity"] = self.gravity_dir.astype(np.float32)
        return obs

    def close(self) -> None:
        self.model.opt.gravity[:] = self.g_nominal
        super().close()
