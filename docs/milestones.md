# Milestones and stages

This merges the milestones (M0–M6, `project-draft.md` section 33) with the curriculum stages (0–9,
section 24). Each milestone lists its acceptance gate, its stages, and what is already done. There is
no timeline.

Legend: ✅ done · 🟡 partial · ⬜ not started

## Progress at a glance

| Milestone | Goal | Stages | Status |
|---|---|---|---|
| M0 | Simulation foundation | 0 | ✅ |
| M1 | Tactile foundation | 0 (sensors) | 🟡 |
| — | Manipulation prerequisites (grasp, rigid rotation) | 1, 2 | ✅ x/y axes · 🟡 z axis and pushes |
| M2 | Single-turn policy | 3 | ⬜ |
| M3 | Blindfolded estimator | 4, 5 | ⬜ |
| M4 | Multi-turn execution | 6 | ⬜ |
| M5 | Restricted solving | 7, 8 | ⬜ (symbolic layer done) |
| M6 | World-model comparison | 9 | ⬜ |
| Extensions | Inspection, risk-aware planning, arm, 3×3×3 | — | ⬜ |

Infrastructure already in place and shared by all later work: MuJoCo Warp GPU environments (1024 worlds,
~5.8k steps/s) checked against the CPU reference; asymmetric PPO and SAC; MLflow tracking; benchmarks
with 95 % Wilson intervals; tests that guard against leaking privileged cube state to actors.

---

## M0 — Simulation foundation ✅

**Gate:** robot loads, actuators behave, cube rotations stay legal, no instability over repeated
interaction. [`reports/m0/m0_report.md`](../reports/m0/m0_report.md): PASSED, 8 checks (robot_loads,
actuators, cube_geometry, stability, collisions, tactile_signals, episode_logic, reproducibility).

**Stage 0 — Pure simulation validation** ✅
- ✅ Sharpa Wave (22 actuators) on a fixed wrist, with a cube resting in the palm.
- ✅ Model A (single-axis cube) and Model C (full 2×2×2, ball-jointed cubies). Random move sequences
  match the symbolic state (`tests/test_cube_kinematics.py`).
- ✅ Symbolic layer: `CubeState`, `Move`, legality checks, `to_perm_ori` (`planning/cube_state.py`).
- ✅ Collision audit, contact-free deterministic resets, reproducibility.
- ✅ Gymnasium `HandCubeEnv` (25 Hz, normalised joint-target deltas) and a test that actor observations
  exclude privileged state.
- 🟡 Model B (restricted multi-axis cube) is not a separate model. The plan is to use Model C with a
  restricted move set.
- 🟡 Cubies are sharp boxes with no detents, and the external driver squares the cube before each turn.
  This matters when the hand has to do the turning (M2).
- ⬜ Franka arm, which the draft defers until after single-turn success.

## M1 — Tactile foundation 🟡

**Gate:** detect contacts and changes in contact location, capture sliding information, produce
consistent observations, and document a sensor interface.

- ✅ Configuration A (binary contact) and B (net force per region in the local frame), on five fingertips
  or on the whole hand (22 regions). Seeded noise, dropout and latency (`simulation/tactile_sensors.py`).
- ✅ Signal sanity checks (`validation/tactile.py`) and `tests/test_tactile.py`.
- ⬜ Configuration C (distributed 8×8×3 taxels).
- ⬜ Sensor experiments T2 (sliding), T3 (contact location), T4 (multiple contacts) and T5 (robustness
  sweep over noise, latency and dropout). The repo has no record of them.
- ⬜ Pad calibration: Menagerie's `solref 0.06` pads sink ~5 mm into the cube. The README defers this
  to M1.
- ⬜ Written sensor-interface document.

## Manipulation prerequisites (Stages 1–2) ✅ / 🟡

**Stage 1 — Grasp stabilisation** ✅ (reports in `reports/grasp*`)
- ✅ `GraspEnv`: wrist reorientations and random pushes, with success defined as no layer corruption.
- ✅ Asymmetric PPO (CPU and GPU) and SAC. Passive and scripted baselines.
- ✅ Palm-down hold 100 %, random reorientations 100 %.
- 🟡 With pushes: 86 % (PPO GPU), against a 95 % target. 12–18 % of pushed episodes turn a layer. Next
  steps: a recurrent actor, alignment shaping, more seeds.
- 🟡 Single seed per trainer. SAC varies across seeds (38–74 % at 6M steps).

**Stage 2 — Rigid cube rotation** ✅ for x and y
- ✅ `RotationEnv` with locked layers, a privileged teacher observation, and a goal curriculum.
- ✅ Single quarter-turns about x and y at 97–100 %. Chains of ~7.8 goals per episode at 97 % per goal.
  Drop rate is ~1.5 % per turn ([`reports/rotation`](../reports/rotation/rotation_report.md)).
- ⬜ z axis (about the fingers): 0 %. Options listed in the README: a pinch grasp, help from the arm, a
  longer exploration phase.
- 🟡 This is a teacher on a locked cube, palm-up, without pushes, from one seed. It is not yet a tactile
  student.

## M2 — Single-turn policy ⬜

**Gate:** at least 90 % success executing a commanded layer turn in the nominal simulation.

**Stage 3 — Single-layer rotation**
- ⬜ Scripted or manually driven face turn on the unlocked cube (Appendix B, task 4). This is the
  reproducibility check before any learning.
- ⬜ `FaceTurnEnv`: a layer and an angle as the command, with the reward from draft section 13 (progress,
  completion, stability, drop, unwanted motion, effort).
- ⬜ Face-turn physics gate: the hand must overcome inter-cubie friction and gap play without squaring
  the cube externally.
- ⬜ Teacher policy with privileged layer angle. Use the reduced action space (stabilising fingers fixed,
  turning fingers learned). Add demonstrations or BC if RL stalls (Risk 3).
- ⬜ Clockwise and counterclockwise turns, then randomised starting grasps and friction.
- ⬜ Benchmarks B1 and B2.

## M3 — Blindfolded estimator ⬜

**Gate:** mean absolute layer-angle error below 10° with calibrated uncertainty, and no privileged
observations in the deployed controller.

**Stage 4 — Tactile estimation**
- ⬜ Dataset collection: tactile, proprioception, actions, and ground-truth angle and pose (Appendix B,
  task 5). The privileged readout in `info["privileged"]` already exists.
- ⬜ GRU estimator (`perception/`), predicting layer angle, whole-cube pose, contact and slip.
- ⬜ Experiment 1 (observability): proprioception only, against fingertip force, against taxels. This
  covers research question RQ1.
- ⬜ Classification of outcomes: complete, incomplete, failed, whole-cube move, wrong layer.
- ⬜ Uncertainty calibration.

**Stage 5 — Blindfolded single turns**
- ⬜ Student policy distilled from the teacher, using only tactile and proprioceptive input and the
  estimated angle (Experiment 2: ground-truth angle, against estimated angle, against none).
- ⬜ Execution monitor that decides if a turn is done (draft section 22).
- ⬜ Experiment 3: disturbance and recovery for partial turns, slip, and reduced friction.
- ⬜ Benchmark B7 (disturbed execution), first version.

## M4 — Multi-turn execution ⬜

**Gate:** at least 80 % complete-sequence success on five consecutive turns, under nominal conditions.

**Stage 6 — Multi-turn sequences**
- ⬜ `SequenceEnv` and the skill controller that maps symbolic moves to physical goals.
- ⬜ Regrasp and whole-cube reorientation between faces, reusing the Stage 2 rotation skills. The z-axis
  gap probably matters here.
- ⬜ Symbolic belief with confirmed and uncertain moves, so the plan never advances after an uncertain
  turn (`perception/belief_filter.py`, `planning/recovery_planner.py`).
- ⬜ Recovery cases A–E.
- ⬜ Benchmarks B3 and B4, with sequences of 2, 3, 5 and more moves and the first-divergence record.
- ⬜ Research questions RQ2 (explicit estimation) and RQ5 (scaling with sequence length).

## M5 — Restricted solving ⬜

**Gate:** reliable completion of three-move scrambles, before increasing complexity.

**Stage 7 — Restricted cube solving**
- ✅ Oracle initial state: `HandCubeEnv` initialises a known symbolic state, as the draft's baseline
  inspection method does.
- ⬜ 2×2×2 solver wrapper. The code has the Kociemba orientation convention and `to_perm_ori`, but no
  solver yet.
- ⬜ `BlindfoldedSolveEnv` and the end-to-end loop: solver, skill controller, estimator, monitor,
  recovery.
- ⬜ Solves of 1-, 2- and 3-move scrambles. Benchmark B5.

**Stage 8 — General 2×2×2 solving** (stretch goal)
- ⬜ Random scrambles. Benchmark B6.
- ⬜ Randomised physical parameters. Benchmark B8.
- ⬜ Full domain randomisation (mass, friction, stiffness, latency).

## M6 — World-model comparison ⬜

**Gate:** reproducible baselines, comparable training budgets, multiple seeds, and a measurable
improvement on at least one predefined metric. A negative result is acceptable.

**Stage 9 — Learned world model**
- ⬜ Action-conditioned latent dynamics with 1-, 5- and 10-step prediction (`world_models/`).
- ⬜ Predictive features as auxiliary inputs to the policy (approach A).
- ⬜ Optional: MPC (approach B) and a particle-filter belief.
- ⬜ RQ3 comparison: model-free, model-free with predictive features, world-model-assisted.

## Extensions (draft priorities 7–8 and Part XVI) ⬜

- Camera-based inspection, replacing the oracle initial state.
- Risk-aware symbolic planning, using measured per-move success rates.
- Active tactile perception.
- Franka arm integration and table-assisted regrasping.
- 3×3×3 cube.
- VLA integration, which the draft keeps last.

## How the status was derived

From `git log` (70 commits), `README.md`, `reports/m0`, `reports/grasp*`, `reports/rotation`, the
module list under `src/manip_rubiks/` (no `perception/`, `world_models/` or solver module), and the
test names. The test suite and benchmarks were not re-run; the numbers are the ones the reports state.
