# manip-rubiks

Blindfolded Rubik's cube solving through tactile robotic manipulation, simulated in MuJoCo with a
Sharpa Wave hand. See [`docs/project-draft.md`](docs/project-draft.md) for the research plan and
[`docs/milestones.md`](docs/milestones.md) for the milestones, their stages and what is done.

## Setup

```bash
uv sync
uv run pytest             # fast tests (~15 s)
uv run pytest -m slow     # gate M0 in quick mode (~1 min)
uv run manip-rubiks-m0    # full gate M0 validation, writes reports/m0/

docker/up.sh                                            # MLflow server on http://127.0.0.1:5000
uv run manip-rubiks-train-grasp-gpu --name gpu_s0       # PPO grasp policy on the GPU, writes runs/grasp_gpu/
uv run manip-rubiks-train-grasp-sac --name sac_s0       # SAC alternative on the GPU, writes runs/grasp_sac/
uv run manip-rubiks-train-grasp --name ppo_asym_s0      # CPU reference (Stable-Baselines3), writes runs/grasp/
uv run manip-rubiks-eval-grasp --policy passive --policy scripted \
    --policy gpu=runs/grasp_gpu/gpu_s0 \
    --policy learned=runs/grasp/ppo_asym_s0             # CPU benchmark, writes reports/grasp/
uv run manip-rubiks-train-rotate-gpu --name gpu_s0 --axes xy   # reorientation teacher, writes runs/rotation_gpu/
uv run manip-rubiks-eval-rotate --policy passive \
    --policy gpu=runs/rotation_gpu/gpu_s0 --axes xy     # CPU benchmark, writes reports/rotation/
```

GPU training needs an NVIDIA GPU with a CUDA 12 or newer driver. The first run compiles MuJoCo Warp
kernels (~1 min, then cached by Warp) and builds the reset bank (~20 s, then cached).

```python
import gymnasium as gym
import manip_rubiks.environments  # registers ManipRubiks/HandCube-v0
from manip_rubiks.environments import EnvConfig

env = gym.make("ManipRubiks/HandCube-v0", config=EnvConfig(scramble_moves=5))
obs, info = env.reset(seed=0)  # obs: q, qd, effort, q_target, tactile, prev_action
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Training logs metrics, parameters and the final and best checkpoints to MLflow (experiment `grasp`,
run `--name`). The tracking server is read from `docker/.env` (`MLFLOW_TRACKING_URI` if set there,
otherwise `http://$MLFLOW_BIND_ADDRESS:$MLFLOW_PORT`), the file that configures the Docker server
started by `docker/up.sh`. From a git worktree the main checkout's `docker/.env` is used. Without that
file, `$MLFLOW_TRACKING_URI` applies, then `http://127.0.0.1:5000`. `--mlflow-uri` and `--experiment`
override everything. Training stops at once if the server is unreachable.

During training, each run also uploads (both trainers, same layout):

| MLflow artifact path | When | Content |
|---|---|---|
| `repro/` | at start | `repro.json` (git commit, branch, dirty flag, package versions, Menagerie commit), `git.diff` if the tree is dirty, `uv.lock`; the run is tagged `git.commit` / `git.dirty` |
| `checkpoints/latest/` | every evaluation, overwritten | model, normalisers and optimizer state (GPU: plus RNG states and best score) |
| `checkpoints/best/` | when the evaluation score improves | best checkpoint |
| `checkpoints/milestones/<steps>/` | first evaluation past each 1/`--milestones` of the budget (default 10) | kept copies of the latest checkpoint |
| `eval/<steps>/episodes.csv` | every evaluation | one row per episode: outcome (`success`, `dropped`, `layer_turned`, `unstable`, `misaligned`), return, length, end-of-episode metrics, summed contribution of each reward term (GPU: CPU seed of the initial state) |
| `eval/<steps>/distributions.json` | every evaluation | observation-normaliser mean/std per key, policy std per action, action mean/std/saturation/histogram, tactile mean/std/active fraction per taxel |
| `eval/<steps>/videos/*.gif` | every `--video-every` evaluations (default 1, i.e. every evaluation; 0 disables) | one successful and one failed full-difficulty episode (GPU: replayed from the recorded worlds) |
| `run/` | at the end, also when interrupted | config, summary, final and best checkpoints |

Wrist reorientation is emulated by rotating gravity, so in the videos the hand stays still while the
cube is pulled in changing directions.

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
  environments/rotation_env.py   in-hand quarter-turn reorientation (privileged teacher observation)
  policies/grasp_policy.py       passive and scripted power-grasp baselines
  policies/learned.py            blindfolded deployment of a trained PPO actor
  policies/torch_policy.py       blindfolded CPU deployment of a GPU-trained actor
  gpu/hand_cube_env.py           batched hand-cube base environment on MuJoCo Warp (thousands of worlds)
  gpu/grasp_env.py               batched grasp environment
  gpu/rotation_env.py            batched reorientation environment
  gpu/cube_math.py               batched cube-state readout in PyTorch
  gpu/reset_bank.py              cached initial states produced by the CPU reset
  training/asymmetric.py         PPO actor-critic whose critic alone sees privileged cube state
  training/train_grasp.py        PPO training with a difficulty curriculum (CPU, Stable-Baselines3)
  training/ppo_gpu.py            asymmetric PPO on GPU rollouts
  training/gpu_loop.py           task-agnostic GPU PPO loop: curriculum, evaluation, checkpoints, MLflow
  training/train_grasp_gpu.py    GPU grasp training
  training/train_rotation_gpu.py GPU reorientation training
  training/sac_gpu.py            asymmetric SAC with an n-step GPU replay buffer
  training/train_grasp_sac_gpu.py  GPU SAC training with the same curriculum and evaluation
  training/tracking.py           MLflow tracking: SB3 logger output, server check
  training/artifacts.py          periodic MLflow artifacts: checkpoints, eval tables, snapshots, videos
  evaluation/core.py             shared benchmark machinery: policy specs, seeds, workers, report layout
  evaluation/grasp_benchmark.py  held-out grasp benchmark and report
  evaluation/rotation_benchmark.py  held-out reorientation benchmark and report
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

**Training.** PPO with an asymmetric actor-critic (draft section 25): the critic also reads the
simulator's cube state, and the actor is fed only the non-privileged observation, which tests
enforce. Tilt range and push strength ramp linearly from 20 % to full difficulty over the first
40 % of training. PPO was chosen over SAC because many parallel environments are available.

- *GPU* (`manip-rubiks-train-grasp-gpu`, the default): 1024 worlds on MuJoCo Warp, PPO in PyTorch on
  the same device, 32-step rollouts (32k samples per update). The reference run (`gpu_s0`, 24M steps,
  87 min at ~4.6k steps/s including evaluations) reached 80 % evaluation success at full difficulty
  after 7M steps (~25 min) and then plateaued at 80–89 %. The best checkpoint (19M steps, selected on
  evaluation success over 256 held-out episodes) is evaluated below.
- *CPU reference* (`manip-rubiks-train-grasp`, Stable-Baselines3): 24 worker processes. Throughput is
  physics-bound: a grasping control step costs ~4.5 ms of `mj_step`, and synchronous stepping waits
  for the slowest worker on the 8P+8E-core CPU, giving ~1.2–1.5k steps/s. The reference run
  (`ppo_asym_s0`, 12M steps, 2.9 h) plateaued once full difficulty was reached at ~5M steps; its
  best checkpoint (7M steps, selected on in-training evaluation reward) is evaluated below.

**Results** on the CPU reference simulation ([`reports/grasp/grasp_report.md`](reports/grasp/grasp_report.md),
100 held-out episodes per cell, 95 % Wilson intervals). The GPU-trained policy runs unchanged on the
float64 CPU simulator here, so its row also checks that the two simulations agree:

| Condition | Passive cage | Scripted power grasp | PPO, CPU (SB3) | PPO, GPU |
|---|---|---|---|---|
| Palm-down hold | 0 % (100 % dropped) | 31 % | 98 % (93–99 %) | **100 %** (96–100 %) |
| 90° tilt, random direction | 30 % | 35 % | **100 %** (96–100 %) | **100 %** (96–100 %) |
| Random reorientations | 3 % | 29 % | **100 %** (96–100 %) | **100 %** (96–100 %) |
| Random reorientations + pushes | 0 % | 13 % (67 % layer turned) | 77 % (68–84 %) | **86 %** (78–91 %) |

The scripted power grasp never drops the cube, but its squeeze shears the R and L layers by
~13° (median) and pushes then turn a layer outright. Both learned grasps keep layers within
2.6–2.8° (median). The GPU policy uses about half the actuator effort of the CPU one (0.06 vs 0.11
of the force limit) and has a tighter misalignment tail without pushes (p95 3.2–3.4° vs 3.9–6.1°).
Under pushes it fails in 14 % of episodes (12 % unintended layer turns, 2 % drops) against 23 %
for the CPU policy. The intervals overlap and each policy comes from a single seed trained for a
different number of steps, so this is not evidence that one trainer is better. The 95 % target is
met for wrist reorientation but not yet for external pushes.

**SAC** (`manip-rubiks-train-grasp-sac`, `training/sac_gpu.py`) is the off-policy alternative on the
same GPU worlds: tanh-squashed Gaussian actor on the non-privileged observation, twin LayerNorm
critics that also read the privileged cube state, automatic temperature, and a per-world replay
buffer with 3-step returns that bootstrap truncated episodes from their final state. Each batched
environment step (1024 transitions) is followed by `--updates-per-step` gradient steps on batches
of 4096. Two settings matter (6M-step runs, curriculum ramping over 9.6M steps as for `gpu_s0`):

- *Sample reuse.* With 4 updates per step (16 reuses per transition) every run peaked at ~0.3M
  steps, then degraded while Q-values first rose and then collapsed. One update per step (4 reuses)
  trains stably and is the default.
- *Reward scale.* Unscaled rewards (+1 per step, −50 on failure) give large TD errors; scaling by
  0.1 alone did not stop the degradation but is kept with one update per step (the default).

The trainer's default `--curriculum-fraction` is 0.4 of `--timesteps`, as for PPO; the runs below
used `--timesteps 6000000 --curriculum-fraction 1.6` to match `gpu_s0`'s absolute schedule.

Against PPO on the identical 6M-step schedule (`ppo_ref6m_s0`, `ppo_ref6m_s1`), SAC is less reliable
across seeds. CPU benchmark, 100 episodes per cell, success rate:

| Condition | PPO s0 | PPO s1 | SAC s1 | SAC s2 | SAC s3 (earlier) | SAC 12M s0 |
|---|---|---|---|---|---|---|
| Palm-down hold | 99 % | 100 % | 94 % | 90 % | 97 % | 100 % |
| 90° tilt, random direction | 99 % | 100 % | 99 % | 64 % | 73 % | 100 % |
| Random reorientations | 100 % | 100 % | 98 % | 60 % | 83 % | 100 % |
| Random reorientations + pushes | 76 % | 82 % | 74 % | 38 % | 51 % | 80 % |

- *Seed spread.* On pushes, three 6M SAC seeds span 38–74 % against 76–82 % for two PPO seeds. One
  SAC seed (`sac_s1`) matches PPO; `sac_s2` drops the cube on tilts and reorientations (11–20 %).
- *Longer run.* At 12M steps (`--timesteps 12000000 --curriculum-fraction 0.8`) a SAC seed reaches
  100 % on the first three conditions and 80 % with pushes, on par with the best 6M PPO. PPO was not
  run at 12M, and this is a single seed.
- *Normalizer freeze.* `--freeze-normalizer-after 3000000` (6M steps) gave 52 % with pushes against
  74 % for the default on a neighbouring seed; one seed each, so only suggestive.
- *Training evaluation misleads.* In-training success (full-difficulty worlds) ended at 64 % for the
  12M run while the benchmark scores 80–100 %; compare policies on the benchmark.

Reports: `reports/grasp_sac/` (passive, scripted, `sac_s1`, the normalizer-freeze run and the 12M run) and `reports/grasp_ppo/` (`sac_s2` and PPO).
PPO remains the default trainer: it is faster per step and more consistent across seeds.

## In-hand reorientation

**Task.** `RotationEnv` (`ManipRubiks/Rotate-v0`): palm-up, fixed wrist, the hand turns the whole cube
by chained quarter-turns. Each goal is the current goal orientation rotated by 90° about a hand axis
(x: palm normal, y: across the palm, z: along the fingers). A goal is reached once the cube stays
within 0.4 rad of it for 0.2 s (5 steps), and the next command follows at once. An attempt fails on
a drop, a layer turn, instability, or after 6 s. Episodes last 16 s.

**Locked layers.** `SceneConfig(lock_layers=True)` welds every cubie to the core, so the cube moves
as one rigid body while keeping the articulated body tree and every state readout. With MuJoCo's
default weld impedance a power grasp deflects the welds by ~4.5°; the locked welds use
`solimp="0.999 0.9999 0.001"` and hold within ~0.1°. The welds hold the solved configuration, so
locked cubes cannot be scrambled. Face turns will need the unlocked cube.

**Teacher observation.** The actor sees the base observation, a one-hot of the active command, and
`teacher`: privileged ground truth (the rotation to the goal, cube position and angular velocity, in
the hand frame), as in draft section 25. A tactile student is meant to replace it once a state
estimator exists. `RotationConfig(teacher=False)` removes it.

**Reward.** Progress towards the goal (1 per 0.1 rad), +10 per goal, small penalties on cube
displacement, layer misalignment, effort and action rate, −50 for a drop, instability or layer turn,
and −10 for a timeout. A timeout costs less than a drop, so dropping is never a cheaper way to end a
hopeless attempt.

**Exploration.** The first version learned nothing in 8M steps per axis: no goal was ever reached.
Undirected finger motion at the grasp exploration noise (std 0.37) rotates the cube by only
0.1–0.37 rad, but the easiest goal needed ~1 rad. The goal tolerance now tightens from 1.3 rad
(~0.27 rad of rotation reaches a goal) to 0.4 rad over the first 30 % of training, and exploration
starts at std 1.

**Training.** 1024 GPU worlds, asymmetric PPO via the shared loop (`training/gpu_loop.py`). After
8M steps per axis, x and y reached 2.4 and 2.0 goals per episode and were still improving; z
(rotation about the finger axis) reached none. The reference run (`gpu_s0`, axes x and y, 60M
steps, 2.5 h at 6.8k steps/s while sharing the GPU) was still improving at the end (7.7 goals per
evaluation episode).

**Results** on the CPU reference simulation
([`reports/rotation/rotation_report.md`](reports/rotation/rotation_report.md), 100 held-out episodes
per condition, 95 % Wilson intervals):

| Condition | Passive | PPO teacher (GPU) |
|---|---|---|
| Single turn +x / −x | 0 % / 0 % | **100 %** (96–100 %) / **97 %** (92–99 %) |
| Single turn +y / −y | 0 % / 0 % | **98 %** (93–99 %) / **100 %** (96–100 %) |
| Single turn +z / −z | 0 % / 0 % | 0 % / 0 % (not trained) |
| 16 s chain (x, y), per goal | 0 % | **97 %** (96–98 %), 7.8 goals per episode |

A quarter-turn takes 0.9–1.5 s (median). Single turns drop the cube in 0–3 % of episodes. In the
chain, 12 % of episodes end in a drop and 8 % in a timeout over ~8 turns each, i.e. about 1.5 % drops
per turn. The 90 % per-turn target is met for x and y. Rotations about the finger axis remain an
open problem: they may need a different starting grasp or help from the arm.

## GPU simulation

`GraspEnvGPU` (`gpu/grasp_env.py`) runs thousands of grasp worlds in one MuJoCo Warp simulation.
Tactile readings, the cube readout, rewards, the wrist schedule and pushes are computed with PyTorch
tensors that share memory with Warp, and the 20 physics substeps of a control step replay as one
captured CUDA graph. Wrist reorientation uses one gravity vector per world. The CPU `GraspEnv`
remains the reference implementation (draft section 34, risk 4):

- **Initial states** come from a reset bank (`gpu/reset_bank.py`): each entry is the simulator state
  right after `GraspEnv.reset(seed=s)`, so they follow the CPU distribution exactly. Training uses
  seeds 0–8191, evaluation 10000–11023, and the CPU benchmark 100000 onwards. Banks are cached in
  `~/.cache/manip_rubiks/reset_banks/`.
- **Agreement**, checked by `tests/test_gpu_grasp_env.py`: observations after a reset match the CPU to
  2e-4, the gravity schedule to 1e-5, and the first control steps track the CPU rewards to 0.02.
  Termination, truncation and automatic resets behave the same way.
- **Known differences.** Physics runs in float32. MuJoCo Warp generates slightly different contact
  sets than CPU MuJoCo (55 vs 53 contacts in a grasp, with identical fingertip contacts), which
  shifts friction forces and tactile readings by up to ~3 %. Random numbers come from a seeded torch
  generator, so episodes follow the same distributions but not the same sequences. Instability is
  detected from non-finite or runaway velocities, since MuJoCo Warp has no per-world solver warnings.
- **Throughput** on an RTX A4000 (grasp scene, ~45 contacts per world): 3.0k env-steps/s with
  256 worlds, 5.8k with 1024 and 6.6k with 4096. Each world is expensive (Newton solver with elliptic
  cones, up to 100 iterations), so throughput saturates around 2k worlds. 1024 worlds keep
  the PPO batch small at almost full throughput.

## Known limitations / follow-ups

- **Reorientation.** The rotation policy is a privileged teacher, trained on a locked cube, palm-up,
  without pushes, from a single seed. Rotations about the finger axis (z) are not learned.
  Next for z: a different starting grasp (e.g. a thumb-index pinch, which frees the palm for a spin
  about the finger axis), help from the arm, a longer high-noise exploration phase for that axis
  alone, and seeding from the x/y behaviour.

- **Push robustness.** The grasp policies turn a layer in 12–18 % of pushed episodes. Possible next
  steps: a recurrent actor, stronger alignment shaping, and multiple seeds for the comparison. Trained
  weights (~6 MB) are not versioned: they exceed the repository's large-file limit. They are logged
  as MLflow artifacts, and `runs/` can be regenerated with `manip-rubiks-train-grasp-gpu`.

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
