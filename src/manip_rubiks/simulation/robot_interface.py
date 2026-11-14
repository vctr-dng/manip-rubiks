"""Joint-level interface to the Sharpa Wave hand (docs/project-draft.md section 31)."""

from __future__ import annotations

import mujoco
import numpy as np

# Joint order follows the Menagerie actuator order.
FINGER_JOINTS: dict[str, tuple[str, ...]] = {
    "thumb": ("thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP"),
    "index": ("index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP"),
    "middle": ("middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP"),
    "ring": ("ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP"),
    "pinky": ("pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP"),
}
JOINTS: tuple[str, ...] = tuple(j for js in FINGER_JOINTS.values() for j in js)
FINGERTIPS: tuple[str, ...] = tuple(FINGER_JOINTS)


class RobotInterface:
    """Read joint state and command position targets for the hand.

    Actions follow section 12 of the project draft: a normalised delta
    ``a in [-1, 1]^22`` updates the joint-position target as
    ``q_target <- clip(q_target + k_a * a, q_min, q_max)``.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str = "right", prefix: str = "") -> None:
        self.model, self.data = model, data
        names = [f"{prefix}{side}_{j}" for j in JOINTS]
        self.joint_names = tuple(names)
        joints = [model.joint(n) for n in names]
        self.qpos_adr = np.array([j.qposadr[0] for j in joints])
        self.dof_adr = np.array([j.dofadr[0] for j in joints])
        self.actuator_ids = np.array([model.actuator(n).id for n in names])
        for a, j in zip(self.actuator_ids, joints, strict=True):
            if model.actuator_trnid[a, 0] != j.id:
                raise ValueError(f"actuator {model.actuator(a).name} does not drive joint {j.name}")
        self.q_min = model.jnt_range[[j.id for j in joints], 0].copy()
        self.q_max = model.jnt_range[[j.id for j in joints], 1].copy()
        self.ctrl_min = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_max = model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.force_limit = model.actuator_forcerange[self.actuator_ids, 1].copy()
        self.fingertip_sites = np.array([model.site(f"{prefix}{side}_{f}_fingertip").id for f in FINGERTIPS])

    @property
    def n(self) -> int:
        return len(self.joint_names)

    # ---------------------------------------------------------------- state
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def qd(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    def effort(self) -> np.ndarray:
        """Actuator forces (the simulated analogue of joint-torque feedback)."""
        return self.data.actuator_force[self.actuator_ids].copy()

    def target(self) -> np.ndarray:
        return self.data.ctrl[self.actuator_ids].copy()

    def fingertip_positions(self) -> np.ndarray:
        return self.data.site_xpos[self.fingertip_sites].copy()

    # -------------------------------------------------------------- command
    def set_target(self, q_target: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(q_target, dtype=float), self.ctrl_min, self.ctrl_max)
        self.data.ctrl[self.actuator_ids] = q
        return q

    def apply_action(self, action: np.ndarray, k_a: float = 0.05) -> np.ndarray:
        """Delta joint-target action from section 12 (``k_a`` in rad per control step)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.set_target(self.target() + k_a * a)

    def reset(self, q: np.ndarray) -> None:
        q = np.clip(np.asarray(q, dtype=float), self.q_min, self.q_max)
        self.data.qpos[self.qpos_adr] = q
        self.data.qvel[self.dof_adr] = 0.0
        self.set_target(q)
