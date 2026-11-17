# Graph Report - sac  (2026-11-17)

## Corpus Check
- 71 files · ~51,620 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1099 nodes · 1981 edges · 69 communities (57 shown, 12 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 341 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `34dd3394`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- HandCubeScene
- 3. Task specification
- RobotInterface
- .solved
- SAC
- grasp_benchmark.py
- ._execute_full
- 13. Learning an individual face turn
- 28. Evaluation metrics
- PrivilegedObservation
- 23. Recovery strategies
- 24. Recommended curriculum
- 10. Sensor validation experiments
- Gate M0 report -- PASSED
- 33. Development milestones and acceptance gates
- 6. Technology stack
- 37. Priority order
- 1. Relevant existing research
- 31. Critical software interfaces
- assets.py
- GraspEnvGPU
- Part VII — World Model
- 36. First three experiments
- graphify.js
- CLAUDE.md
- __init__.py
- __init__.py
- __init__.py
- manip-rubiks
- HandCubeEnv
- TactileSensors
- SceneConfig
- 34. Major technical risks
- Part XVII — Final Recommendations
- Part VI — Tactile State Estimation
- GraspEnv
- project-draft.md
- PPO
- _train
- test_grasp_env.py
- CubeMath
- __init__.py
- test_gpu_grasp_env.py
- __init__.py
- Move
- manip-rubiks
- .__init__
- TactileConfig
- artifacts.py
- ._initial_cube_state
- CubeState
- test_tactile.py
- _train
- __init__.py
- __init__.py
- __init__.py
- GraspConfig
- CubeInterface
- axis_angle_matrix
- up.sh
- ArtifactLogger
- GraspEvalCallback
- normalizer_stats
- check_episode_logic
- 11. Hierarchical control architecture
- Appendix A — Essential Resources
- test_checkpoint_deploys_on_the_cpu_reference

## God Nodes (most connected - your core abstractions)
1. `GraspEnv` - 53 edges
2. `HandCubeScene` - 41 edges
3. `CubeState` - 38 edges
4. `GraspEnvGPU` - 37 edges
5. `HandCubeEnv` - 33 edges
6. `ArtifactLogger` - 25 edges
7. `GraspConfig` - 23 edges
8. `Move` - 23 edges
9. `CubeInterface` - 23 edges
10. `EnvConfig` - 21 edges

## Surprising Connections (you probably didn't know these)
- `test_bank_reproduces_cpu_resets()` --calls--> `GraspEnv`  [INFERRED]
  tests/test_gpu_grasp_env.py → src/manip_rubiks/environments/grasp_env.py
- `test_short_rollout_tracks_cpu()` --calls--> `GraspEnv`  [INFERRED]
  tests/test_gpu_grasp_env.py → src/manip_rubiks/environments/grasp_env.py
- `env()` --calls--> `GraspEnv`  [INFERRED]
  tests/test_grasp_env.py → src/manip_rubiks/environments/grasp_env.py
- `test_passes_gymnasium_checker()` --calls--> `GraspEnv`  [INFERRED]
  tests/test_grasp_env.py → src/manip_rubiks/environments/grasp_env.py
- `test_registered()` --indirect_call--> `GraspEnv`  [INFERRED]
  tests/test_grasp_env.py → src/manip_rubiks/environments/grasp_env.py

## Import Cycles
- None detected.

## Communities (69 total, 12 thin omitted)

### Community 0 - "HandCubeScene"
Cohesion: 0.05
Nodes (55): HandCubeScene, Compiled scene with robot and cube interfaces and deterministic resets., Deterministic reset: hand to home, cube onto the palm, then settle passively., Any penetrating contact involving a hand geom (with itself or the cube)., audit_filters(), check_collisions(), _Geoms, _max_pen() (+47 more)

### Community 1 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 2 - "RobotInterface"
Cohesion: 0.16
Nodes (8): MjData, MjModel, ndarray, Joint-level interface to the Sharpa Wave hand (docs/project-draft.md section 31), Read joint state and command position targets for the hand.      Actions follow, Actuator forces (the simulated analogue of joint-torque feedback)., Delta joint-target action from section 12 (``k_a`` in rad per control step)., RobotInterface

### Community 3 - ".solved"
Cohesion: 0.07
Nodes (42): parse_moves(), Random sequence avoiding consecutive moves on the same axis.          If all all, cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml() (+34 more)

### Community 4 - "SAC"
Cohesion: 0.06
Nodes (33): Normal, Optimizer, _mlp(), device, Sequential, Tensor, Soft Actor-Critic with an asymmetric critic, entirely on the GPU (docs/project-d, ``(time, world)`` ring buffer of raw transitions with n-step sampling. (+25 more)

### Community 5 - "grasp_benchmark.py"
Cohesion: 0.05
Nodes (38): build_policy(), condition_options(), EpisodeResult, main(), policy_label(), Any, Path, Grasp-stabilisation benchmark: sustained blindfolded holds under wrist reorienta (+30 more)

### Community 6 - "._execute_full"
Cohesion: 0.16
Nodes (12): CubeReading, LayerDriver, LayerDriverConfig, MoveResult, Ground-truth access to the simulated cube.  Everything here reads privileged sim, Largest penetration depth [m] among active contacts involving the cube., Executes face turns by applying external torques to the cubies.      This stands, Snapped configuration of the physical cube. (+4 more)

### Community 7 - "13. Learning an individual face turn"
Cohesion: 0.20
Nodes (10): 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Initial algorithm choice, Part V — Manipulation Control, Recommended initial approach (+2 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

### Community 9 - "PrivilegedObservation"
Cohesion: 0.15
Nodes (13): MultiInputActorCriticPolicy, AsymmetricActorCriticPolicy, PrivilegedObservation, Any, ndarray, PPO policy whose actor reads ``actor_keys`` and whose critic also reads ``critic, Expose ``info["privileged"]`` as an observation entry, for the critic only., model() (+5 more)

### Community 10 - "23. Recovery strategies"
Cohesion: 0.13
Nodes (15): 20. Visual inspection, 21. Symbolic cube solver, 22. Execution verification, 23. Recovery strategies, Baseline inspection method, Case A — Incomplete rotation, Case B — No effective rotation, Case C — Whole-cube movement (+7 more)

### Community 11 - "24. Recommended curriculum"
Cohesion: 0.13
Nodes (15): 24. Recommended curriculum, 25. Training with privileged information, 26. Domain randomization, Part IX — Training Strategy, Recommended procedure, Stage 0 — Pure simulation validation, Stage 1 — Grasp stabilization, Stage 2 — Rigid cube rotation (+7 more)

### Community 12 - "10. Sensor validation experiments"
Cohesion: 0.14
Nodes (14): 10. Sensor validation experiments, 8. Sensor requirements, 9. Sensor configurations, Acceptance criteria, Configuration A — Contact-only, Configuration B — Fingertip force measurements, Configuration C — Distributed tactile taxels, Experiment T1 — Static contact (+6 more)

### Community 13 - "Gate M0 report -- PASSED"
Cohesion: 0.20
Nodes (9): actuators -- pass, collisions -- pass, cube_geometry -- pass, episode_logic -- pass, Gate M0 report -- PASSED, reproducibility -- pass, robot_loads -- pass, stability -- pass (+1 more)

### Community 14 - "33. Development milestones and acceptance gates"
Cohesion: 0.18
Nodes (11): 32. Timeline, 33. Development milestones and acceptance gates, Milestone M0 — Simulation foundation, Milestone M1 — Tactile foundation, Milestone M2 — Single-turn policy, Milestone M3 — Blindfolded estimator, Milestone M4 — Multi-turn execution, Milestone M5 — Restricted solving (+3 more)

### Community 15 - "6. Technology stack"
Cohesion: 0.18
Nodes (11): 6.1 Sharpa Wave model, 6.2 Robot arm, 6.3 Cube model, 6. Technology stack, 7. Recommended cube-modeling strategy, Important modeling issue, Model A — Single-axis articulated cube, Model B — Restricted multi-axis cube (+3 more)

### Community 16 - "37. Priority order"
Cohesion: 0.20
Nodes (10): 37. Priority order, Part XV — Recommended Research Priorities, Priority 1 — Manipulation feasibility, Priority 2 — Tactile observability, Priority 3 — Closed-loop control, Priority 4 — Long-horizon memory, Priority 5 — Recovery, Priority 6 — World models (+2 more)

### Community 17 - "1. Relevant existing research"
Cohesion: 0.22
Nodes (9): 1.1 OpenAI Dactyl — Solving Rubik's Cube (2019), 1.2 Learning Purely Tactile In-Hand Manipulation (2022), 1.3 Dextrous Tactile In-Hand Manipulation Using a Modular Reinforcement Learning Architecture (2023), 1.4 Rotating without Seeing: Towards In-Hand Dexterity through Touch (2023), 1.5 FINGR — Learning Dexterous Hand Control for Real-World Rubik's Cube Solving (2026), 1.6 DexTouch-WM — Action-Conditioned Tactile World Models (2026), 1. Relevant existing research, 2. Research contribution and novelty (+1 more)

### Community 18 - "31. Critical software interfaces"
Cohesion: 0.22
Nodes (9): 30. Proposed repository structure, 31. Critical software interfaces, Cube interface, Part XI — Software Architecture, Planner interface, Policy interface, Robot interface, State-estimator interface (+1 more)

### Community 19 - "assets.py"
Cohesion: 0.44
Nodes (8): cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), Path, Third-party robot assets.  The Sharpa Wave meshes are too large to vendor into t, Return the directory of a Menagerie model, fetching it on first use., sharpa_wave_xml()

### Community 20 - "GraspEnvGPU"
Cohesion: 0.11
Nodes (17): GraspEnvGPU, _min_jerk(), Tensor, Batched grasp-stabilisation environment on MuJoCo Warp.  :class:`GraspEnvGPU` re, Scale tilt range and pushes; takes effect at each world's next reset., Gravity direction after tilting the wrist (``GraspEnv.tilted_direction``)., (N, 3) gravity direction in the hand frame at per-world time ``t``., (N, R, 3) net contact force on each tactile region, world frame (``TactileSensor (+9 more)

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 30 - "HandCubeEnv"
Cohesion: 0.12
Nodes (20): GraspRewardWeights, Grasp stabilisation: keep hold of the cube while the wrist reorients (docs/proje, WristMove, EnvConfig, HandCubeEnv, Gymnasium environment for the fixed-wrist hand holding a 2x2x2 cube.  This is th, Fixed-wrist Sharpa Wave with a cube resting in the palm.      Actions are normal, assert_obs_equal() (+12 more)

### Community 31 - "TactileSensors"
Cohesion: 0.21
Nodes (6): ndarray, (n_regions, 3) net contact force on each region, in world coordinates., Noiseless, undelayed reading for the current simulator state., Corrupted, delayed reading; call once per control step., Per-region contact or force readings for the Sharpa Wave hand.      Call :meth:`, TactileSensors

### Community 32 - "SceneConfig"
Cohesion: 0.15
Nodes (10): build_spec(), is_pad(), mount_rotation(), MjSpec, Hand + cube scene with a fixed wrist (docs/project-draft.md sections 6 and 12)., Fingertip elastomer pads (compliant contact, the tactile surface)., SceneConfig, hand_scene() (+2 more)

### Community 33 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 34 - "Part XVII — Final Recommendations"
Cohesion: 0.50
Nodes (4): 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Part XVII — Final Recommendations

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "GraspEnv"
Cohesion: 0.14
Nodes (10): GraspEnv, Any, ndarray, Scale tilt range and pushes; takes effect at the next reset., Gravity direction after tilting the wrist by ``tilt`` towards in-palm direction, Whole-cube orientation, fitted to the cubies (the core may spin inside them)., Total reward; the signed per-term contributions are stored in ``info["reward_ter, Great-circle interpolation between unit vectors ``a`` and ``b``. (+2 more)

### Community 37 - "project-draft.md"
Cohesion: 0.15
Nodes (12): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, Appendix B — Immediate Action Plan, Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Conclusion, Executive Summary, Expected outcomes (+4 more)

### Community 38 - "PPO"
Cohesion: 0.10
Nodes (22): Critic, GaussianActor, _mlp(), PPO, PPOConfig, device, Sequential, Tensor (+14 more)

### Community 39 - "_train"
Cohesion: 0.06
Nodes (50): slice, SACConfig, check_server(), docker_env_file(), flatten_params(), _main_checkout(), Any, Path (+42 more)

### Community 40 - "test_grasp_env.py"
Cohesion: 0.13
Nodes (11): env(), Grasp-stabilisation environment: wrist schedule, pushes, rewards, success and te, run(), test_observations_do_not_depend_on_the_scramble(), test_passes_gymnasium_checker(), test_passive_hand_drops_the_cube_palm_down(), test_pushes_hit_the_cube_and_can_be_disabled(), test_registered() (+3 more)

### Community 41 - "CubeMath"
Cohesion: 0.09
Nodes (24): dtype, BatchedReading, CubeMath, _group_tables(), indices_to_state(), device, ndarray, Tensor (+16 more)

### Community 43 - "test_gpu_grasp_env.py"
Cohesion: 0.18
Nodes (12): flatten_observation(), ndarray, Flatten a CPU ``GraspEnv`` observation into the GPU actor layout., copy_state(), The GPU grasp environment must reproduce the CPU reference (GraspEnv)., Copy CPU grasp states (with fingertip contacts) into the GPU worlds and compare, test_bank_reproduces_cpu_resets(), test_gravity_schedule_matches_cpu() (+4 more)

### Community 45 - "Move"
Cohesion: 0.13
Nodes (11): axis_rotation(), format_moves(), Move, ndarray, Symbolic 2x2x2 cube state.  State is stored the same way the physics model is pa, Rotation angle about the outward face normal, in (-pi, pi]., Indices of the cubies currently in ``face``., All 24 proper rotation matrices with integer entries (signed permutations). (+3 more)

### Community 46 - "manip-rubiks"
Cohesion: 0.20
Nodes (8): GPU simulation, Grasp stabilisation, Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), Grasp stabilisation benchmark

### Community 47 - ".__init__"
Cohesion: 0.12
Nodes (13): BaseFeaturesExtractor, Env, Module, Space, AsymmetricMlpExtractor, KeyedFlattenExtractor, _mlp(), device (+5 more)

### Community 48 - "TactileConfig"
Cohesion: 0.19
Nodes (10): MjData, MjModel, Simulated tactile sensing derived from MuJoCo contacts (docs/project-draft.md se, Region, TactileConfig, check_tactile_signals(), probe_fingertips(), Tactile signal sanity checks (contact-derived sensors).  These establish that th (+2 more)

### Community 49 - "artifacts.py"
Cohesion: 0.11
Nodes (22): _csv_value(), DistributionStats, _jsonable(), pick_video_episodes(), Any, Artifacts uploaded to MLflow while a grasp policy trains.  Both trainers write t, Accumulate action and tactile statistics over evaluation steps., Indices of the first success and the first failure, when present. (+14 more)

### Community 50 - "._initial_cube_state"
Cohesion: 0.26
Nodes (5): Any, ndarray, Task reward; the base environment has no task., Task-specific success or failure termination., Flat ground-truth cube state for critics and rewards (never for the actor).

### Community 51 - "CubeState"
Cohesion: 0.15
Nodes (7): CubeState, Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``COR, Current position signs (8, 3) of every cubie., Rotate the whole cube (does not change the puzzle configuration)., Whole-cube rotation that brings cubie DBL to its home pose.          A 2x2x2 cub, Every cubie occupies a distinct slot and the total twist is 0 mod 3., Kociemba corner representation.          ``perm[slot]`` is the cubie in ``slot``

### Community 52 - "test_tactile.py"
Cohesion: 0.26
Nodes (12): Contact-derived tactile readings: layout, physical consistency and corruption., Hand-hand contacts cancel, so the net force on the hand is the cube's weight., sensors(), settled(), test_contact_appears_when_cube_is_pressed_onto_a_fingertip(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom() (+4 more)

### Community 53 - "_train"
Cohesion: 0.12
Nodes (18): BaseCallback, KVWriter, MLflowOutputFormat, Write the scalar values of each SB3 logger dump to the active MLflow run., DifficultyCurriculum, EpisodeMetrics, main(), make_env() (+10 more)

### Community 58 - "GraspConfig"
Cohesion: 0.27
Nodes (12): GraspConfig, range, build_reset_bank(), _config_key(), _generate(), load_reset_bank(), ndarray, range (+4 more)

### Community 59 - "CubeInterface"
Cohesion: 0.15
Nodes (10): CubeInterface, MjData, MjModel, ndarray, Variant, Write a symbolic configuration into ``qpos`` (call ``mj_forward`` afterwards)., Activate the fixture weld holding cubie ``index`` at a rotation relative to the, (8, 3) world-frame angular velocity of each cubie relative to the core (full var (+2 more)

### Community 60 - "axis_angle_matrix"
Cohesion: 0.18
Nodes (16): axis_angle_matrix(), chordal_mean(), log_rotation(), mat_to_quat(), ndarray, Small rotation helpers (numpy only)., Rotation vector (axis * angle) of a rotation matrix, robust near pi., Signed twist of rotation ``r`` about unit ``axis`` (swing-twist decomposition), (+8 more)

### Community 62 - "ArtifactLogger"
Cohesion: 0.18
Nodes (8): ArtifactLogger, Path, Upload run files to the active MLflow run and decide when periodic artifacts are, Snapshot code versions (``repro/``) and tag the run with the git commit., Count an evaluation; return its index and whether it records videos., Milestones crossed since the last call (each is reported once)., Upload the latest checkpoint; keep a copy when a milestone was crossed since the, Upload the run directory under ``run/``, except what was already uploaded period

### Community 63 - "GraspEvalCallback"
Cohesion: 0.20
Nodes (7): EvalCallback, outcome(), Single label for how an episode ended (``misaligned``: kept the cube, but cubies, GraspEvalCallback, Path, :class:`EvalCallback` that also writes the periodic training artifacts.      Whi, test_outcome_labels()

### Community 64 - "normalizer_stats"
Cohesion: 0.22
Nodes (8): normalizer_stats(), ndarray, ``actions``: (batch, act_dim), ``tactile``: (batch, tactile_dim) raw (unnormalis, ``{key: (mean, var)}`` -> ``{key: {"mean": ..., "std": ...}}`` (flattened)., Replay joint positions in a ``render_mode="rgb_array"`` environment and save an, render_gif(), test_normalizer_stats(), test_render_gif()

### Community 65 - "check_episode_logic"
Cohesion: 0.28
Nodes (7): check_episode_logic(), _max_obs_diff(), _obs_equal(), Episode logic of the Gymnasium environment: seeding, resets, termination, leakag, Tactile-signal and episode-logic validation checks., test_episode_logic_check(), test_tactile_signals_check()

### Community 66 - "11. Hierarchical control architecture"
Cohesion: 0.29
Nodes (7): 11. Hierarchical control architecture, Execution monitor, High-level planner, Low-level manipulation policy, Recovery controller, Skill controller, State estimator

### Community 67 - "Appendix A — Essential Resources"
Cohesion: 0.50
Nodes (4): Additional research directions, Appendix A — Essential Resources, Core research papers, Simulation and robot models

### Community 68 - "test_checkpoint_deploys_on_the_cpu_reference"
Cohesion: 0.67
Nodes (3): observation_layout(), (key, shape) of each actor observation entry, in flattening order., test_checkpoint_deploys_on_the_cpu_reference()

## Knowledge Gaps
- **158 isolated node(s):** `up.sh script`, `manip-rubiks`, `graphify`, `Versioning`, `Setup` (+153 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraspEnv` connect `GraspEnv` to `normalizer_stats`, `test_checkpoint_deploys_on_the_cpu_reference`, `grasp_benchmark.py`, `_train`, `test_grasp_env.py`, `PrivilegedObservation`, `test_gpu_grasp_env.py`, `GraspEnvGPU`, `GraspConfig`, `HandCubeEnv`, `GraspEvalCallback`?**
  _High betweenness centrality (0.282) - this node is a cross-community bridge._
- **Why does `HandCubeEnv` connect `HandCubeEnv` to `HandCubeScene`, `SceneConfig`, `check_episode_logic`, `GraspEnv`, `TactileConfig`, `._initial_cube_state`, `CubeState`, `GraspConfig`, `TactileSensors`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._
- **Why does `CubeState` connect `CubeState` to `HandCubeScene`, `SceneConfig`, `.solved`, `._execute_full`, `CubeMath`, `Move`, `._initial_cube_state`, `CubeInterface`, `axis_angle_matrix`, `HandCubeEnv`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `GraspEnv` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`GraspEnv` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `HandCubeScene` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`HandCubeScene` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `CubeState` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`CubeState` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `GraspEnvGPU` (e.g. with `GraspConfig` and `GraspEnv`) actually correct?**
  _`GraspEnvGPU` has 5 INFERRED edges - model-reasoned connections that need verification._