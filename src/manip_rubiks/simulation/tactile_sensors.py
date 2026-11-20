"""Simulated tactile sensing derived from MuJoCo contacts (docs/project-draft.md sections 8-9).

Two of the draft's sensor configurations are implemented here:

``contact`` (configuration A)
    One binary value per sensing region: is anything pressing on it?

``force`` (configuration B)
    The net contact force applied *to* each region, expressed in the
    region's local frame (fingertip site frame for fingertips, body frame
    otherwise), shape ``(n_regions, 3)``.

``taxel`` (configuration C)
    MuJoCo's native mesh-based tactile sensor on every fingertip: an
    ``ROWS x COLS`` grid with three channels per taxel (indentation depth and
    two non-negative tangential slip speeds), shape ``(5, 8, 8, 3)``. The scene
    must be built with ``SceneConfig(tactile_taxels=True)``; see
    ``simulation/taxels.py`` for the geometry and channel definitions. Values
    are divided by ``max_depth`` / ``max_speed`` and clipped to ``[0, 1]``.

A *region* is a group of hand collision geoms. By default the five
fingertip pads are the regions. ``coverage="hand"`` adds the palm and every
phalanx, following Yin et al. (2023), who found that coarse whole-hand
contact coverage is more useful than fingertip resolution alone.

Readings are computed from the simulator's contact list, so they include
contacts with anything (cube, floor, other fingers), just as a physical
skin would. Noise, dropout and latency are applied per reading, with their
own seeded random generator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from manip_rubiks.simulation.robot_interface import FINGERTIPS
from manip_rubiks.simulation.taxels import COLS, ROWS, sensor_name

Mode = Literal["contact", "force", "taxel"]
Coverage = Literal["fingertips", "hand"]


@dataclass(frozen=True)
class TactileConfig:
    mode: Mode = "force"
    coverage: Coverage = "fingertips"
    contact_threshold: float = 0.05  # normal force for a binary contact [N]
    noise_std: float = 0.0  # additive Gaussian noise: newtons (force) or normalised units (taxel)
    noise_rel: float = 0.0  # multiplicative noise (fraction of the reading)
    dropout: float = 0.0  # probability that a region (fingertip) reads zero on a given step
    latency: int = 0  # delay in readings (number of ``read`` calls)
    max_force: float = 50.0  # saturation of each force component [N]
    # Taxel normalisation. In scripted grasps the soft default pads indent 10-15 mm (99th percentile
    # 14.8 mm) and slip speeds reach 0.14 m/s (99th percentile), so these map the usual range to [0, 1].
    max_depth: float = 0.015  # taxel indentation that maps to 1.0 [m]
    max_speed: float = 0.15  # taxel slip speed that maps to 1.0 [m/s]


@dataclass(frozen=True)
class Region:
    name: str
    geoms: np.ndarray  # collision geom ids belonging to this region
    body: int  # body whose frame is used when no site is given
    site: int  # fingertip site id, or -1


class TactileSensors:
    """Per-region contact or force readings for the Sharpa Wave hand.

    Call :meth:`reset` at the start of an episode and :meth:`read` once per
    control step. :meth:`measure` returns the noiseless, undelayed value.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: str = "right",
        prefix: str = "",
        config: TactileConfig | None = None,
    ) -> None:
        self.model, self.data = model, data
        self.cfg = config or TactileConfig()
        if self.cfg.mode == "taxel" and self.cfg.coverage != "fingertips":
            raise ValueError("taxel mode is only defined for fingertip coverage")
        self.regions = self._build_regions(prefix, side)
        self._taxel_adr = self._find_taxel_sensors(prefix, side) if self.cfg.mode == "taxel" else ()
        self._geom_region = np.full(model.ngeom, -1, dtype=int)
        for k, r in enumerate(self.regions):
            self._geom_region[r.geoms] = k
        self._rng = np.random.default_rng()
        self._buffer: deque[np.ndarray] = deque()

    # ----------------------------------------------------------------- setup
    def _build_regions(self, prefix: str, side: str) -> tuple[Region, ...]:
        m = self.model
        colliding = (m.geom_contype > 0) | (m.geom_conaffinity > 0)

        def body_geoms(body: int, only: str | None = None) -> np.ndarray:
            ids = np.flatnonzero((m.geom_bodyid == body) & colliding)
            if only is not None:
                ids = ids[[only in m.geom(i).name for i in ids]]
            return ids

        regions = []
        for f in FINGERTIPS:
            dp = m.body(f"{prefix}{side}_{f}_DP").id
            site = m.site(f"{prefix}{side}_{f}_fingertip").id
            pad = body_geoms(dp, only="fingertip")
            if pad.size == 0:
                raise ValueError(f"no fingertip pad geom on {f}")
            regions.append(Region(f"{f}_tip", pad, dp, site))
        if self.cfg.coverage == "hand":
            # Every other hand body with collision geoms becomes one region, so
            # the regions together cover the whole hand surface.
            pads = np.concatenate([r.geoms for r in regions])
            body_prefix = f"{prefix}{side}_"
            for b in range(m.nbody):
                name = m.body(b).name
                if not name.startswith(body_prefix):
                    continue
                geoms = np.setdiff1d(body_geoms(b), pads)
                if geoms.size:
                    label = name.removeprefix(body_prefix)
                    regions.append(Region("palm" if label == "hand_C_MC" else label, geoms, b, -1))
        return tuple(regions)

    def _find_taxel_sensors(self, prefix: str, side: str) -> tuple[int, ...]:
        m = self.model
        adr = []
        for f in FINGERTIPS:
            name = sensor_name(prefix, f)
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name) < 0:
                raise ValueError(f"taxel mode needs sensor '{name}': set SceneConfig(tactile_taxels=True)")
            adr.append(int(m.sensor_adr[m.sensor(name).id]))
        return tuple(adr)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.regions)

    @property
    def shape(self) -> tuple[int, ...]:
        n = len(self.regions)
        if self.cfg.mode == "taxel":
            return (n, ROWS, COLS, 3)
        return (n,) if self.cfg.mode == "contact" else (n, 3)

    # ------------------------------------------------------------- readings
    def region_forces(self) -> np.ndarray:
        """(n_regions, 3) net contact force on each region, in world coordinates."""
        m, d = self.model, self.data
        out = np.zeros((len(self.regions), 3))
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            r1, r2 = self._geom_region[c.geom1], self._geom_region[c.geom2]
            if r1 < 0 and r2 < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            # Contact frame rows: normal (from geom1 towards geom2) and two tangents.
            f_world = c.frame.reshape(3, 3).T @ f6[:3]  # force exerted on geom2 by geom1
            if r2 >= 0:
                out[r2] += f_world
            if r1 >= 0:
                out[r1] -= f_world
        return out

    def _region_rotations(self) -> np.ndarray:
        d = self.data
        rots = np.empty((len(self.regions), 3, 3))
        for k, r in enumerate(self.regions):
            mat = d.site_xmat[r.site] if r.site >= 0 else d.xmat[r.body]
            rots[k] = mat.reshape(3, 3)
        return rots

    def taxels(self) -> np.ndarray:
        """Normalised native tactile readings, shape ``(n_tips, ROWS, COLS, 3)`` (depth, along, across)."""
        d = self.data
        raw = np.stack([d.sensordata[a : a + 3 * ROWS * COLS].reshape(3, ROWS, COLS) for a in self._taxel_adr])
        scale = np.array([self.cfg.max_depth, self.cfg.max_speed, self.cfg.max_speed])
        return np.clip(raw.transpose(0, 2, 3, 1) / scale, 0.0, 1.0)

    def measure(self) -> np.ndarray:
        """Noiseless, undelayed reading for the current simulator state."""
        if self.cfg.mode == "taxel":
            return self.taxels()
        forces = self.region_forces()
        if self.cfg.mode == "contact":
            return (np.linalg.norm(forces, axis=1) > self.cfg.contact_threshold).astype(np.float64)
        local = np.einsum("nji,nj->ni", self._region_rotations(), forces)  # R^T f
        return np.clip(local, -self.cfg.max_force, self.cfg.max_force)

    def _corrupt(self, x: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self._rng
        x = x.copy()
        if cfg.mode != "contact":
            if cfg.noise_rel > 0:
                x *= 1.0 + cfg.noise_rel * rng.standard_normal(x.shape)
            if cfg.noise_std > 0:
                x += cfg.noise_std * rng.standard_normal(x.shape)
            lo, hi = (0.0, 1.0) if cfg.mode == "taxel" else (-cfg.max_force, cfg.max_force)
            x = np.clip(x, lo, hi)
        if cfg.dropout > 0:
            x[rng.random(len(self.regions)) < cfg.dropout] = 0.0
        return x

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._buffer.clear()

    def read(self) -> np.ndarray:
        """Corrupted, delayed reading; call once per control step."""
        self._buffer.append(self._corrupt(self.measure()))
        while len(self._buffer) > self.cfg.latency + 1:
            self._buffer.popleft()
        return self._buffer[0].copy()
