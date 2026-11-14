# manip-rubiks

Blindfolded Rubik's cube solving through tactile robotic manipulation, simulated in MuJoCo with a
Sharpa Wave hand. See [`docs/project-draft.md`](docs/project-draft.md) for the research plan.

## Status

| Gate | Status | Evidence |
|---|---|---|
| **M0 — Simulation foundation** | passed | [`reports/m0/m0_report.md`](reports/m0/m0_report.md) |
| M1 — Tactile foundation | not started | |

## Setup

```bash
uv sync
uv run pytest             # fast tests (~5 s)
uv run pytest -m slow     # gate M0 in quick mode (~1 min)
uv run manip-rubiks-m0    # full gate M0 validation, writes reports/m0/
```

The Sharpa Wave model is fetched on first use from MuJoCo Menagerie at a pinned commit into
`~/.cache/manip_rubiks/` (override with `MANIP_RUBIKS_CACHE`). Its meshes are too large to vendor.

## Layout

```text
src/manip_rubiks/
  planning/cube_state.py         symbolic 2x2x2 state (per-cubie rotations), moves, Kociemba view
  simulation/cube_model.py       MJCF generator: single_axis (Model A) and full (Model C) cubes
  simulation/cube_interface.py   ground-truth cube I/O, snapping, external layer driver
  simulation/robot_interface.py  joint state, position targets, delta actions (draft section 12)
  simulation/scene_builder.py    palm-up fixed-wrist hand + cube scene, deterministic resets
  simulation/assets.py           pinned Menagerie fetch
  validation/m0.py               gate M0 checks and report
tests/                           pytest suite
```

## Simulation design (M0)

**Cube frame.** +x = R, -x = L, +y = B, -y = F, +z = U, -z = D. `X` is a clockwise quarter turn seen
from outside face X. On a 2x2x2 there are no fixed centres, so states are compared modulo whole-cube
rotation (`CubeState.same_configuration`).

**Full 2x2x2 cube (Model C).** A core body plus eight 25 mm box cubies, each on a *ball joint at the
cube centre*. The ball joint fixes every cubie's radius, and face-to-face contacts between neighbouring
cubies (0.4 mm gap) block every motion except rigid rotations of whole layers, like the interlocking
pieces of a real cube. The kinematic tree never changes when different faces turn. It has 24 internal
DoF, and 3 of them are a harmless null mode: the core can spin relative to all cubies at once.
`CubeInterface.read()` therefore fits the symbolic frame to the cubies rather than to the core.

**Single-axis cube (Model A).** A rigid D layer with the U layer on a hinge. It is for single-skill
experiments.

**Hand scene.** The Sharpa Wave right hand is mounted palm-up with a fixed wrist, fingers along +x
and tilted 10°. The 50 mm, 70 g cube rests on the palm, caged by the fingers in the `HOME_POSE`.

**Privileged state.** `CubeInterface` reads simulator ground truth and is only for resets, rewards
and evaluation. It must never feed a blindfolded policy (draft section 3).

## Known limitations / follow-ups

- **Fingertip pad compliance.** Menagerie's elastomer pads (`solref 0.06`) are very soft: the pads
  sink up to ~6 mm into the cube and ~13 mm into each other under full actuator force. This is not
  an instability, but it should be calibrated in M1 together with the tactile sensor.
- **Self-collision at joint limits.** Driving all joints to their limits at once lets the thumb push
  `index_DIP` ~0.12 rad past its soft limit.
- **Gravity sag.** P-only position actuators sag up to 0.04 rad under gravity (worst: `pinky_CMC`).
- **Cube geometry.** Cubies are sharp-edged boxes with no detents or corner cutting. After a
  passive settle, cubies rest ~2° off-grid within the gap play. The external driver squares the cube
  before each turn, standing in for a gripping hand.
- **No Model B yet.** The restricted multi-axis cube (Model B) is not implemented separately. Use the
  full cube with a restricted move set (`CubeState.random_moves(..., faces=...)`).
