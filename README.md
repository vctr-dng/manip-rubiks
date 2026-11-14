# manip-rubiks

Blindfolded Rubik's cube solving through tactile robotic manipulation, simulated in MuJoCo with a
Sharpa Wave hand. See [`docs/project-draft.md`](docs/project-draft.md) for the research plan.

## Setup

```bash
uv sync
uv run pytest             # fast tests (~15 s)
uv run pytest -m slow     # gate M0 in quick mode (~1 min)
uv run manip-rubiks-m0    # full gate M0 validation, writes reports/m0/
```

```python
import gymnasium as gym
import manip_rubiks.environments  # registers ManipRubiks/HandCube-v0
from manip_rubiks.environments import EnvConfig

env = gym.make("ManipRubiks/HandCube-v0", config=EnvConfig(scramble_moves=5))
obs, info = env.reset(seed=0)  # obs: q, qd, effort, q_target, tactile, prev_action
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Offscreen rendering (`render_mode="rgb_array"`) on a headless Linux machine needs `MUJOCO_GL=egl`.

The Sharpa Wave model is fetched on first use from MuJoCo Menagerie at a pinned commit into
`~/.cache/manip_rubiks/` (override with `MANIP_RUBIKS_CACHE`). Its meshes are too large to vendor.

## Layout

```text
src/manip_rubiks/
  planning/cube_state.py         symbolic 2x2x2 state (per-cubie rotations), moves, Kociemba view
  simulation/cube_model.py       MJCF generator: single_axis (Model A) and full (Model C) cubes
  simulation/cube_interface.py   ground-truth cube I/O, snapping, external layer driver
  simulation/robot_interface.py  joint state, position targets, delta actions (draft section 12)
  simulation/scene_builder.py    palm-up fixed-wrist hand + cube scene, contact-free deterministic resets
  simulation/tactile_sensors.py  contact-derived tactile readings (binary contact / 3D force per region)
  simulation/assets.py           pinned Menagerie fetch
  environments/hand_cube_env.py  Gymnasium base environment (episode logic, observation/privileged split)
  validation/m0.py               validation suite entry point and report
  validation/collisions.py       collision filters, reset clearance, pinch and press audits
  validation/tactile.py          tactile signal sanity checks
  validation/episodes.py         environment seeding, reset, termination and leakage checks
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
The cube spawns 20 mm off-centre, away from the thumb; at the centre it would start 15 mm inside the
thumb metacarpal. Rigid hand links (palm, phalanges) use a 5 ms contact time constant instead of
Menagerie's 10–20 ms, which let a strong pinch push two links ~15 mm into each other
(`SceneConfig.link_solref`). The compliant fingertip pads are left unchanged.

**Resets.** `HandCubeScene.reset` is deterministic per seed. Noisy joint and cube poses are
rejection-sampled until nothing touches the hand, so no episode starts with a contact impulse.

**Tactile sensing.** `TactileSensors` aggregates MuJoCo contact forces per region: the five fingertip
pads, or the whole hand (`coverage="hand"`, 22 regions). Readings are a binary contact per region
(draft configuration A) or the net force in the region's local frame (configuration B), with seeded
noise, dropout and latency. Distributed taxels (configuration C) are not implemented yet.

**Environment.** `HandCubeEnv` runs at 25 Hz (20 physics steps per action) with normalised joint-target
deltas. It terminates when the cube leaves the hand (more than 50 mm from rest) or the simulation
becomes unstable, and truncates after 250 steps. Task environments override `compute_reward` and
`task_terminated`. Observations hold only proprioception, tactile readings and the previous action.
Cubies are physically identical, so a scramble cannot be seen in them; a test enforces this.

**Privileged state.** `CubeInterface` reads simulator ground truth and is only for resets, rewards
and evaluation. It must never feed a blindfolded policy (draft section 3). The environment exposes it
only in `info["privileged"]` (for asymmetric critics) and `info["cube_state"]`.

## Known limitations / follow-ups

- **Fingertip pad compliance.** Menagerie's elastomer pads (`solref 0.06`) are very soft: the pads
  sink up to ~5 mm into the cube and ~10 mm into other links under full actuator force. This is not
  an instability, but it should be calibrated in M1 together with the tactile sensor.
- **Release snaps.** A distal phalanx pressed past its limit by another finger springs back at up to
  ~60 rad/s when released, then settles within ~30 ms. The actuators have no velocity limit. The
  stability check compares joint speeds with what each actuator could produce on its own, not with a
  fixed rad/s value.
- **Self-collision at joint limits.** Driving all joints to their limits at once lets the thumb push
  `index_DIP` ~0.06 rad past its soft limit.
- **Gravity sag.** P-only position actuators sag up to 0.01 rad under gravity (worst: `pinky_CMC`).
- **Cube geometry.** Cubies are sharp-edged boxes with no detents or corner cutting. After a
  passive settle, cubies rest ~2° off-grid within the gap play. The external driver squares the cube
  before each turn, standing in for a gripping hand.
- **No Model B yet.** The restricted multi-axis cube (Model B) is not implemented separately. Use the
  full cube with a restricted move set (`CubeState.random_moves(..., faces=...)`).
