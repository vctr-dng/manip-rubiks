"""MJCF generation for articulated 2x2x2 cube models.

Two variants are provided (see docs/project-draft.md section 7):

``single_axis`` (Model A)
    The four D cubies form one rigid base body; the four U cubies form a
    child body attached with a hinge about the cube z axis. Only U/D turns are
    possible. Cheap and easy to debug.

``full`` (Model C)
    A core body with eight cubie bodies, each attached to the core with a
    *ball joint located at the cube centre*. The ball joint fixes the radial
    distance of every cubie, and face-to-face box contacts between neighbouring
    cubies block every motion except rigid rotations of whole layers, exactly
    like the interlocking pieces of a real cube. The kinematic tree therefore
    never has to change when different faces turn.

The cube frame follows :mod:`manip_rubiks.planning.cube_state`:
+x = R, -x = L, +y = B, -y = F, +z = U, -z = D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from manip_rubiks.planning.cube_state import CORNER_NAMES, HOME_SIGNS

Variant = Literal["single_axis", "full"]

STICKER_RGBA = {
    "U": "1 1 1 1",
    "D": "1 0.85 0 1",
    "F": "0 0.62 0.27 1",
    "B": "0 0.27 0.68 1",
    "R": "0.78 0.05 0.12 1",
    "L": "1 0.35 0 1",
}
_AXIS_FACES = (("L", "R"), ("F", "B"), ("D", "U"))  # (negative, positive) face per axis


@dataclass(frozen=True)
class CubeParams:
    size: float = 0.05  # cube edge length [m]
    gap: float = 0.0004  # clearance between neighbouring cubies [m]
    mass: float = 0.07  # total mass [kg]
    core_mass: float = 0.004  # mass of the internal core [kg]
    # Cubie-cubie contact. MuJoCo combines friction with max(), so fingertip
    # contacts keep the (higher) fingertip friction.
    friction: float = 0.1
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float] = (0.95, 0.99, 0.0005)
    # Joint properties (ball joints for "full", hinge for "single_axis").
    damping: float = 2e-4
    armature: float = 2e-6
    frictionloss: float = 2e-4
    sticker_fraction: float = 0.84  # sticker side / cubie side
    sticker_thickness: float = 0.0004

    @property
    def cubie(self) -> float:
        return self.size / 2

    @property
    def half_extent(self) -> float:
        return (self.cubie - self.gap) / 2

    @property
    def cubie_mass(self) -> float:
        return (self.mass - self.core_mass) / 8


def cubie_center(index: int, params: CubeParams) -> np.ndarray:
    return HOME_SIGNS[index] * params.cubie / 2


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _stickers(index: int, params: CubeParams) -> str:
    center = cubie_center(index, params)
    h = params.half_extent
    s = h * params.sticker_fraction
    t = params.sticker_thickness / 2
    out = []
    for axis in range(3):
        sign = HOME_SIGNS[index, axis]
        face = _AXIS_FACES[axis][int(sign > 0)]
        pos = center.copy()
        pos[axis] += sign * (h + t)
        size = np.array([s, s, s])
        size[axis] = t
        out.append(
            f'<geom class="sticker" name="sticker_{CORNER_NAMES[index]}_{face}" '
            f'pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{STICKER_RGBA[face]}"/>'
        )
    return "\n".join(out)


def _cubie_geoms(index: int, params: CubeParams) -> str:
    name = CORNER_NAMES[index]
    return f'<geom class="cubie" name="cubie_{name}" pos="{_fmt(cubie_center(index, params))}"/>\n' + _stickers(
        index, params
    )


def cube_xml(
    variant: Variant = "full",
    params: CubeParams | None = None,
    fixed: bool = False,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixture_welds: bool = False,
) -> str:
    """Return a standalone MJCF document for the cube.

    Args:
        variant: ``"single_axis"`` or ``"full"``.
        params: geometry and contact parameters.
        fixed: if True the root body is welded to the world (test fixture),
            otherwise it gets a free joint named ``cube_free``.
        pos: position of the cube centre.
        fixture_welds: add one inactive weld constraint per cubie
            (``hold_<name>``, cubie to core). The external layer driver in
            :mod:`cube_interface` activates them to hold the non-turning layer
            rigidly; they are not meant for hand scenes.
    """
    p = params or CubeParams()
    h = p.half_extent
    free = "" if fixed else '<freejoint name="cube_free"/>'
    defaults = f"""
  <default>
    <default class="cube">
      <default class="cubie">
        <geom type="box" size="{_fmt([h, h, h])}" mass="{p.cubie_mass:.6g}" friction="{p.friction} 0.005 0.0001"
              condim="3" solref="{_fmt(p.solref)}" solimp="{_fmt(p.solimp)}" rgba="0.08 0.08 0.08 1" group="2"/>
        <joint damping="{p.damping}" armature="{p.armature}" frictionloss="{p.frictionloss}"/>
      </default>
      <default class="sticker">
        <geom type="box" contype="0" conaffinity="0" mass="0" group="1"/>
      </default>
    </default>
  </default>"""

    if variant == "full":
        cubies = []
        for i, name in enumerate(CORNER_NAMES):
            cubies.append(
                f"""
      <body name="cubie_{name}" childclass="cubie">
        <joint name="cubie_{name}_ball" type="ball"/>
        {_cubie_geoms(i, p)}
      </body>"""
            )
        body = f"""
    <body name="cube_core" pos="{_fmt(pos)}" childclass="cube">
      {free}
      <geom name="cube_core" type="sphere" size="{p.cubie * 0.3:.6g}" mass="{p.core_mass}"
            contype="0" conaffinity="0" group="3"/>
      <site name="cube_center" size="0.002" group="4"/>
      {"".join(cubies)}
    </body>"""
    elif variant == "single_axis":
        lower = [i for i, n in enumerate(CORNER_NAMES) if n.startswith("D")]
        upper = [i for i, n in enumerate(CORNER_NAMES) if n.startswith("U")]
        body = f"""
    <body name="cube_core" pos="{_fmt(pos)}" childclass="cube">
      {free}
      <site name="cube_center" size="0.002" group="4"/>
      {"".join(_cubie_geoms(i, p) for i in lower)}
      <body name="cube_layer_U">
        <joint name="cube_U_hinge" class="cubie" type="hinge" axis="0 0 1"/>
        {"".join(_cubie_geoms(i, p) for i in upper)}
      </body>
    </body>"""
    else:
        raise ValueError(f"unknown cube variant {variant!r}")

    equality = ""
    if fixture_welds and variant == "full":
        welds = "".join(
            f'\n    <weld name="hold_{n}" body1="cubie_{n}" body2="cube_core" active="false" solref="0.004 1"/>'
            for n in CORNER_NAMES
        )
        equality = f"<equality>{welds}\n  </equality>"

    return f"""<mujoco model="cube2x2_{variant}">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  {defaults}
  <worldbody>
    {body}
  </worldbody>
  {equality}
</mujoco>
"""


def cube_spec(variant: Variant = "full", params: CubeParams | None = None, fixed: bool = False) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(cube_xml(variant, params, fixed))


def fixture_cube_xml(variant: Variant = "full", params: CubeParams | None = None) -> str:
    """Cube welded to the world with driver welds, floor and light (validation fixture)."""
    xml = cube_xml(variant, params, fixed=True, pos=(0.0, 0.0, 0.1), fixture_welds=True)
    extra = """
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.3 0.35 0.4 1"/>
  </worldbody>
</mujoco>"""
    return xml.replace("</mujoco>\n", "").rstrip() + extra
