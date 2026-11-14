"""Hand + cube scene with a fixed wrist (docs/project-draft.md sections 6 and 12).

The Sharpa Wave is mounted palm-up with the fingers pointing along world +x,
and the cube rests on the palm. In the hand frame the fingers extend along +z
and flex towards +x (the palm normal), so the mount rotation maps hand +x to
world +z and hand +z to world +x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from manip_rubiks.planning.cube_state import CubeState
from manip_rubiks.simulation.assets import sharpa_wave_xml
from manip_rubiks.simulation.cube_interface import CubeInterface
from manip_rubiks.simulation.cube_model import CubeParams, Variant, cube_spec
from manip_rubiks.simulation.robot_interface import JOINTS, RobotInterface
from manip_rubiks.simulation.rotations import axis_angle_matrix, mat_to_quat

HAND_PREFIX = "hand/"
CUBE_PREFIX = "cube/"

# A light cage: fingers flexed into a wall in front of the cube, thumb opposed.
HOME_POSE: dict[str, float] = {
    "thumb_CMC_FE": 1.2,
    "thumb_CMC_AA": 0.0,
    "thumb_MCP_FE": 0.3,
    "thumb_MCP_AA": 0.0,
    "thumb_IP": 0.3,
    "index_MCP_FE": 0.9,
    "index_MCP_AA": 0.0,
    "index_PIP": 0.6,
    "index_DIP": 0.3,
    "middle_MCP_FE": 0.9,
    "middle_MCP_AA": 0.0,
    "middle_PIP": 0.6,
    "middle_DIP": 0.3,
    "ring_MCP_FE": 0.9,
    "ring_MCP_AA": 0.0,
    "ring_PIP": 0.6,
    "ring_DIP": 0.3,
    "pinky_CMC": 0.1,
    "pinky_MCP_FE": 0.9,
    "pinky_MCP_AA": 0.0,
    "pinky_PIP": 0.6,
    "pinky_DIP": 0.3,
}

_PARENT_XML = """
<mujoco model="hand_cube_scene">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-25"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <camera name="overview" pos="0.25 -0.2 0.5" xyaxes="0.725 0.689 0 -0.31 0.33 0.89"/>
  </worldbody>
</mujoco>
"""


@dataclass
class SceneConfig:
    variant: Variant | None = "full"  # None builds a hand-only scene
    cube_params: CubeParams = field(default_factory=CubeParams)
    side: str = "right"
    wrist_height: float = 0.3  # world z of the wrist mount [m]
    tilt: float = np.deg2rad(10.0)  # raise the fingertips by this angle [rad]
    timestep: float = 0.002
    cube_offset: float = 0.06  # cube centre along the fingers from the wrist [m]
    drop_height: float = 0.003  # clearance between palm and cube at reset [m]
    home_pose: dict[str, float] = field(default_factory=lambda: dict(HOME_POSE))
    settle_time: float = 0.3  # passive settling after reset [s]


def mount_rotation(tilt: float) -> np.ndarray:
    palm_up = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # columns: hand x, y, z in world
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -tilt) @ palm_up


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    parent = mujoco.MjSpec.from_string(_PARENT_XML.format(timestep=cfg.timestep))
    hand = mujoco.MjSpec.from_file(str(sharpa_wave_xml(cfg.side)))
    r_mount = mount_rotation(cfg.tilt)
    frame = parent.worldbody.add_frame(name="wrist_mount", pos=[0.0, 0.0, cfg.wrist_height], quat=mat_to_quat(r_mount))
    parent.attach(hand, frame=frame, prefix=HAND_PREFIX)
    if cfg.variant is not None:
        cube = cube_spec(cfg.variant, cfg.cube_params, fixed=False)
        parent.attach(cube, frame=parent.worldbody.add_frame(name="cube_mount"), prefix=CUBE_PREFIX)
    return parent


class HandCubeScene:
    """Compiled scene with robot and cube interfaces and deterministic resets."""

    def __init__(self, cfg: SceneConfig | None = None) -> None:
        self.cfg = cfg or SceneConfig()
        self.spec = build_spec(self.cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.robot = RobotInterface(self.model, self.data, self.cfg.side, prefix=HAND_PREFIX)
        self.cube = (
            CubeInterface(self.model, self.data, self.cfg.variant, prefix=CUBE_PREFIX) if self.cfg.variant else None
        )
        self.home_q = np.array([self.cfg.home_pose[j] for j in JOINTS])
        self.r_mount = mount_rotation(self.cfg.tilt)
        self.wrist_pos = np.array([0.0, 0.0, self.cfg.wrist_height])

    def cube_rest_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Nominal cube pose above the palm, aligned with the hand."""
        palm_top = 0.0215 + self.cfg.drop_height + self.cfg.cube_params.size / 2
        offset_hand = np.array([palm_top, 0.0, self.cfg.cube_offset])  # along palm normal, along fingers
        pos = self.wrist_pos + self.r_mount @ offset_hand
        r_cube = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -self.cfg.tilt)  # cube z = palm normal
        return pos, r_cube

    def reset(
        self,
        seed: int | None = None,
        state: CubeState | None = None,
        pose_noise: float = 0.0,
        joint_noise: float = 0.0,
        settle: bool = True,
    ) -> None:
        """Deterministic reset: hand to home, cube onto the palm, then settle passively."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        q = self.home_q + joint_noise * rng.uniform(-1, 1, self.robot.n)
        self.robot.reset(q)
        if self.cube is None:
            mujoco.mj_forward(self.model, self.data)
            return
        pos, r_cube = self.cube_rest_pose()
        pos = pos + pose_noise * rng.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.0])
        yaw = pose_noise * 10.0 * rng.uniform(-1, 1)  # ~0.1 rad per cm of position noise
        r_cube = r_cube @ axis_angle_matrix(np.array([0.0, 0.0, 1.0]), yaw)
        self.cube.set_pose(pos, mat_to_quat(r_cube))
        self.cube.set_state(state or CubeState.solved())
        mujoco.mj_forward(self.model, self.data)
        if settle:
            mujoco.mj_step(self.model, self.data, round(self.cfg.settle_time / self.model.opt.timestep))

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, n)

    def warnings(self) -> int:
        return int(sum(w.number for w in self.data.warning))
