# manip-rubiks

Blindfolded Rubik's cube solving through tactile robotic manipulation, simulated in MuJoCo with a
Sharpa Wave hand. See [`docs/project-draft.md`](docs/project-draft.md) for the research plan.

## Setup

```bash
uv sync
uv run pytest             # fast tests (~15 s)
uv run pytest -m slow     # gate M0 in quick mode (~1 min)
uv run manip-rubiks-m0    # full gate M0 validation, writes reports/m0/

docker/up.sh                                            # MLflow server on http://127.0.0.1:5000
uv run manip-rubiks-train-grasp --name ppo_asym_s0      # PPO grasp policy, writes runs/grasp/ (~2.3 h)
uv run manip-rubiks-eval-grasp --policy passive --policy scripted \
    --policy learned=runs/grasp/ppo_asym_s0             # writes reports/grasp/
```

```python
import gymnasium as gym
import manip_rubiks.environments  # registers ManipRubiks/HandCube-v0
from manip_rubiks.environments import EnvConfig

env = gym.make("ManipRubiks/HandCube-v0", config=EnvConfig(scramble_moves=5))
obs, info = env.reset(seed=0)  # obs: q, qd, effort, q_target, tactile, prev_action
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Training logs metrics, parameters and the final and best checkpoints to MLflow (experiment `grasp`,
run `--name`). The tracking server is `$MLFLOW_TRACKING_URI` if set, otherwise the local Docker server
started by `docker/up.sh`; `--mlflow-uri` and `--experiment` override both. Training stops at once if
the server is unreachable.

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
  environments/grasp_env.py      grasp stabilisation under wrist reorientation and pushes
  policies/grasp_policy.py       passive and scripted power-grasp baselines
  policies/learned.py            blindfolded deployment of a trained PPO actor
  training/asymmetric.py         PPO actor-critic whose critic alone sees privileged cube state
  training/train_grasp.py        PPO training with a difficulty curriculum
  training/tracking.py           MLflow tracking: SB3 logger output, server check
  evaluation/grasp_benchmark.py  held-out grasp benchmark and report
  validation/m0.py               validation suite entry point and report
  validation/collisions.py       collision filters, reset clearance, pinch and press audits
  validation/tactile.py          tactile signal sanity checks
  validation/episodes.py         environment seeding, reset, termination and leakage checks
tests/                           pytest suite
docker/                          MLflow tracking server (Docker Compose, started by up.sh)
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

## Grasp stabilisation

**Task.** Palm-up, the cube simply rests in the hand. `GraspEnv` (`ManipRubiks/Grasp-v0`) emulates an
arm reorienting the fixed wrist by rotating gravity around the hand. For the hand-cube dynamics this
is equivalent to a static wrist rotation; arm accelerations are not modelled. Each 10 s episode
starts palm-up. After 0.2–0.6 s the wrist follows minimum-jerk reorientations to random tilts up to
palm-down, with dwells in between, while random pushes (up to 1.5 N and 0.02 N m, ~3 per episode)
hit the cube. The actor observes the base observation plus `gravity`, the gravity direction in the
hand frame: a real robot knows its wrist orientation from the arm's encoders.

**Success** means the episode reaches its time limit with the cube in the hand, the puzzle
configuration unchanged and every cubie within 10° of alignment. Twisting a layer while holding the
cube counts as failure, since it corrupts the memorised puzzle state. A layer turned past 45°
changes the snapped configuration and ends the episode.

**Reward** (privileged ground truth allowed): alive bonus minus penalties on cube displacement,
whole-cube rotation, layer misalignment, cube velocity, actuator effort and action rate, plus a large
penalty on drops or unintended layer turns.

**Training.** PPO (Stable-Baselines3) with an asymmetric actor-critic (draft section 25): the
critic also reads the simulator's cube state, and the actor's MLP is fed only the non-privileged
slice of the features, which a test enforces. Tilt range and push strength ramp linearly from 20 %
to full difficulty over the first 40 % of training. PPO was chosen over SAC because many parallel
environments are available. Throughput is physics-bound: a grasping control step costs ~4.5 ms of
`mj_step`, and synchronous stepping waits for the slowest worker on the 8P+8E-core CPU, giving
~1.2–1.5k steps/s with 24 workers. The reference run (`ppo_asym_s0`, 12M steps, 2.9 h) plateaued
once full difficulty was reached at ~5M steps. The best checkpoint (7M steps, selected on
in-training evaluation reward) is evaluated below.

**Results** ([`reports/grasp/grasp_report.md`](reports/grasp/grasp_report.md), 100 held-out
episodes per cell, 95 % Wilson intervals):

| Condition | Passive cage | Scripted power grasp | PPO (blindfolded) |
|---|---|---|---|
| Palm-down hold | 0 % (100 % dropped) | 31 % | **98 %** (93–99 %) |
| 90° tilt, random direction | 30 % | 35 % | **100 %** (96–100 %) |
| Random reorientations | 3 % | 29 % | **100 %** (96–100 %) |
| Random reorientations + pushes | 0 % | 13 % (67 % layer turned) | **77 %** (68–84 %) |

The scripted power grasp never drops the cube, but its squeeze shears the R and L layers by
~13° (median) and pushes then turn a layer outright. The learned grasp keeps layers within 2.8°
(median) using ~40 % of the scripted grasp's actuator effort. Under pushes it still fails in
23 % of episodes: 18 % unintended layer turns and 5 % drops. The 95 % target is met for wrist
reorientation but not yet for external pushes.

## Known limitations / follow-ups

- **Push robustness.** The grasp policy turns a layer in 18 % of pushed episodes. Possible next
  steps: longer training at full difficulty, a recurrent actor, stronger alignment shaping, and
  multiple seeds for the comparison. Trained weights (~6 MB) are not versioned: they exceed the
  repository's large-file limit, so `runs/` must be regenerated with `manip-rubiks-train-grasp`.

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
