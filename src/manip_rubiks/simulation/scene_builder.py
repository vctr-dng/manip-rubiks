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
from manip_rubiks.simulation.taxels import add_taxel_sensors

HAND_PREFIX = "hand/"
CUBE_PREFIX = "cube/"
MAX_RESET_ATTEMPTS = 100

# A light cage: fingers flexed into a wall in front of the cube, thumb opposed.
# Thumb and pinky are abducted slightly so the hand starts free of self-contact.
HOME_POSE: dict[str, float] = {
    "thumb_CMC_FE": 1.2,
    "thumb_CMC_AA": -0.15,
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
    "pinky_MCP_AA": 0.1,
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
    # Cube centre across the palm (hand y, negative = away from the thumb) [m].
    # At 0 the cube would spawn 15 mm inside the thumb metacarpal.
    cube_lateral: float = -0.02
    drop_height: float = 0.003  # clearance between palm and cube at reset [m]
    home_pose: dict[str, float] = field(default_factory=lambda: dict(HOME_POSE))
    settle_time: float = 0.3  # passive settling after reset [s]
    # Contact time constant and damping ratio for the rigid hand links (palm,
    # phalanges). Menagerie ships them at 10-20 ms, soft enough for a strong
    # pinch to push two links ~15 mm into each other. ``None`` keeps
    # Menagerie's values. The compliant fingertip pads are never changed.
    link_solref: tuple[float, float] | None = (0.005, 1.0)
    # Weld every cubie to the core so the cube moves as one rigid body (full
    # variant only). The welds hold the solved configuration, so scrambles are
    # not supported with locked layers.
    lock_layers: bool = False
    # Add a skin mesh and a native tactile sensor to every fingertip (configuration C,
    # ``simulation/taxels.py``). The skin does not collide, so dynamics are unchanged.
    tactile_taxels: bool = False
    # Override the fingertip pads' contact softness. ``None`` keeps Menagerie's elastomer
    # values (solref 0.06 0.9, MuJoCo's default solimp), which the trained policies assume.
    pad_solref: tuple[float, float] | None = None
    pad_solimp: tuple[float, float, float, float, float] | None = None


def is_pad(geom_name: str) -> bool:
    """Fingertip elastomer pads (compliant contact, the tactile surface)."""
    return geom_name.endswith("fingertip_collision")


def mount_rotation(tilt: float) -> np.ndarray:
    palm_up = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # columns: hand x, y, z in world
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -tilt) @ palm_up


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    parent = mujoco.MjSpec.from_string(_PARENT_XML.format(timestep=cfg.timestep))
    hand = mujoco.MjSpec.from_file(str(sharpa_wave_xml(cfg.side)))
    if cfg.link_solref is not None:
        for g in hand.geoms:
            if (g.contype or g.conaffinity) and not is_pad(g.name):
                g.solref = list(cfg.link_solref)
    for g in hand.geoms:
        if is_pad(g.name):
            if cfg.pad_solref is not None:
                g.solref = list(cfg.pad_solref)
            if cfg.pad_solimp is not None:
                g.solimp = list(cfg.pad_solimp)
    r_mount = mount_rotation(cfg.tilt)
    frame = parent.worldbody.add_frame(name="wrist_mount", pos=[0.0, 0.0, cfg.wrist_height], quat=mat_to_quat(r_mount))
    parent.attach(hand, frame=frame, prefix=HAND_PREFIX)
    if cfg.tactile_taxels:
        add_taxel_sensors(parent, HAND_PREFIX, cfg.side)
    if cfg.variant is not None:
        cube = cube_spec(cfg.variant, cfg.cube_params, fixed=False, lock_layers=cfg.lock_layers)
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
        self._hand_geom = np.array([self.model.body(b).name.startswith(HAND_PREFIX) for b in self.model.geom_bodyid])
        self.reset_attempts = 0

    def cube_rest_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Nominal cube pose above the palm, aligned with the hand."""
        palm_top = 0.0215 + self.cfg.drop_height + self.cfg.cube_params.size / 2
        offset_hand = np.array([palm_top, self.cfg.cube_lateral, self.cfg.cube_offset])  # normal, across, along
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
        """Deterministic reset: hand to home, cube onto the palm, then settle passively.

        Noisy joint and cube poses are rejection-sampled until nothing touches
        the hand, so no episode starts with a contact impulse. After
        ``MAX_RESET_ATTEMPTS`` failures the nominal (collision-free) pose is
        used. ``self.reset_attempts`` records how many samples were drawn.
        """
        rng = np.random.default_rng(seed)
        noisy = joint_noise > 0 or (pose_noise > 0 and self.cube is not None)
        self.reset_attempts = 0
        for _ in range(MAX_RESET_ATTEMPTS if noisy else 1):
            self.reset_attempts += 1
            self._place(rng, state, pose_noise, joint_noise)
            if not noisy or not self._hand_in_contact():
                break
        else:
            self._place(rng, state, 0.0, 0.0)
        if self.cube is not None and settle:
            mujoco.mj_step(self.model, self.data, round(self.cfg.settle_time / self.model.opt.timestep))

    def _place(self, rng: np.random.Generator, state: CubeState | None, pose_noise: float, joint_noise: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset(self.home_q + joint_noise * rng.uniform(-1, 1, self.robot.n))
        if self.cube is not None:
            pos, r_cube = self.cube_rest_pose()
            pos = pos + pose_noise * rng.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.0])
            yaw = pose_noise * 10.0 * rng.uniform(-1, 1)  # ~0.1 rad per cm of position noise
            r_cube = r_cube @ axis_angle_matrix(np.array([0.0, 0.0, 1.0]), yaw)
            self.cube.set_pose(pos, mat_to_quat(r_cube))
            self.cube.set_state(state or CubeState.solved())
        mujoco.mj_forward(self.model, self.data)

    def _hand_in_contact(self) -> bool:
        """Any penetrating contact involving a hand geom (with itself or the cube)."""
        n = self.data.ncon
        if n == 0:
            return False
        geoms = self.data.contact.geom[:n]
        involves_hand = self._hand_geom[geoms].any(axis=1)
        return bool(np.any(involves_hand & (self.data.contact.dist[:n] < 0)))

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, n)

    def warnings(self) -> int:
        return int(sum(w.number for w in self.data.warning))
