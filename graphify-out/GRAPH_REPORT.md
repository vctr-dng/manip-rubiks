# Graph Report - grasp  (2026-11-14)

## Corpus Check
- 75 files · ~44,771 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 951 nodes · 1629 edges · 62 communities (50 shown, 12 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 258 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `65fd9034`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Move
- 3. Task specification
- test_observation_leakage.py
- CubeState
- m0.py
- grasp_benchmark.py
- RobotInterface
- 11. Hierarchical control architecture
- 28. Evaluation metrics
- SceneConfig
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
- test_scene_stability.py
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
- test_tactile.py
- 34. Major technical risks
- project-draft.md
- Part VI — Tactile State Estimation
- GraspEnv
- Executive Summary
- Part XVI — Further Research Directions
- _Geoms
- Appendix A — Essential Resources
- train_grasp.py
- __init__.py
- PrivilegedObservation
- HandCubeScene
- manip-rubiks
- AsymmetricActorCriticPolicy
- __init__.py
- __init__.py
- __init__.py
- g2.py
- g3.py

## God Nodes (most connected - your core abstractions)
1. `GraspEnv` - 46 edges
2. `HandCubeScene` - 41 edges
3. `CubeState` - 38 edges
4. `HandCubeEnv` - 33 edges
5. `GraspEnvGPU` - 32 edges
6. `Move` - 23 edges
7. `CubeInterface` - 23 edges
8. `EnvConfig` - 21 edges
9. `GraspConfig` - 19 edges
10. `TactileSensors` - 18 edges

## Surprising Connections (you probably didn't know these)
- `settled()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_tactile.py → src/manip_rubiks/simulation/scene_builder.py
- `test_episode_logic_check()` --calls--> `check_episode_logic()`  [INFERRED]
  tests/test_validation_checks.py → src/manip_rubiks/validation/episodes.py
- `test_gravity_schedule_matches_cpu()` --calls--> `GraspConfig`  [INFERRED]
  tests/test_gpu_grasp_env.py → src/manip_rubiks/environments/grasp_env.py
- `test_pushes_hit_the_cube_and_can_be_disabled()` --calls--> `GraspConfig`  [INFERRED]
  tests/test_grasp_env.py → src/manip_rubiks/environments/grasp_env.py
- `test_zero_difficulty_keeps_the_palm_up_without_pushes()` --calls--> `GraspConfig`  [INFERRED]
  tests/test_grasp_env.py → src/manip_rubiks/environments/grasp_env.py

## Import Cycles
- None detected.

## Communities (62 total, 12 thin omitted)

### Community 0 - "Move"
Cohesion: 0.25
Nodes (11): Random sequence avoiding consecutive moves on the same axis.          If all all, LayerDriver, Executes face turns by applying external torques to the cubies.      This stands, make_fixture(), Physical cube models: structure, state I/O and externally driven legal moves., test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved(), test_read_is_invariant_to_whole_set_rotation() (+3 more)

### Community 1 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 2 - "test_observation_leakage.py"
Cohesion: 0.06
Nodes (26): MjData, MjModel, ndarray, Joint-level interface to the Sharpa Wave hand (docs/project-draft.md section 31), Read joint state and command position targets for the hand.      Actions follow, Actuator forces (the simulated analogue of joint-torque feedback)., Delta joint-target action from section 12 (``k_a`` in rad per control step)., RobotInterface (+18 more)

### Community 3 - "CubeState"
Cohesion: 0.13
Nodes (8): CubeState, Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``COR, Current position signs (8, 3) of every cubie., Indices of the cubies currently in ``face``., Rotate the whole cube (does not change the puzzle configuration)., Whole-cube rotation that brings cubie DBL to its home pose.          A 2x2x2 cub, Every cubie occupies a distinct slot and the total twist is 0 mod 3., Kociemba corner representation.          ``perm[slot]`` is the cubie in ``slot``

### Community 4 - "m0.py"
Cohesion: 0.05
Nodes (55): HandCubeScene, Compiled scene with robot and cube interfaces and deterministic resets., Deterministic reset: hand to home, cube onto the palm, then settle passively., Any penetrating contact involving a hand geom (with itself or the cube)., audit_filters(), check_collisions(), _Geoms, _max_pen() (+47 more)

### Community 5 - "grasp_benchmark.py"
Cohesion: 0.07
Nodes (31): build_policy(), condition_options(), EpisodeResult, main(), policy_label(), Any, Path, Grasp-stabilisation benchmark: sustained blindfolded holds under wrist reorienta (+23 more)

### Community 6 - "RobotInterface"
Cohesion: 0.06
Nodes (37): CubeInterface, CubeReading, LayerDriverConfig, MoveResult, MjData, MjModel, ndarray, Variant (+29 more)

### Community 7 - "11. Hierarchical control architecture"
Cohesion: 0.12
Nodes (17): 11. Hierarchical control architecture, 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Execution monitor, High-level planner (+9 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

### Community 9 - "SceneConfig"
Cohesion: 0.21
Nodes (15): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), MjSpec (+7 more)

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
Cohesion: 0.22
Nodes (9): 37. Priority order, Priority 1 — Manipulation feasibility, Priority 2 — Tactile observability, Priority 3 — Closed-loop control, Priority 4 — Long-horizon memory, Priority 5 — Recovery, Priority 6 — World models, Priority 7 — Visual inspection (+1 more)

### Community 17 - "1. Relevant existing research"
Cohesion: 0.22
Nodes (9): 1.1 OpenAI Dactyl — Solving Rubik's Cube (2019), 1.2 Learning Purely Tactile In-Hand Manipulation (2022), 1.3 Dextrous Tactile In-Hand Manipulation Using a Modular Reinforcement Learning Architecture (2023), 1.4 Rotating without Seeing: Towards In-Hand Dexterity through Touch (2023), 1.5 FINGR — Learning Dexterous Hand Control for Real-World Rubik's Cube Solving (2026), 1.6 DexTouch-WM — Action-Conditioned Tactile World Models (2026), 1. Relevant existing research, 2. Research contribution and novelty (+1 more)

### Community 18 - "31. Critical software interfaces"
Cohesion: 0.22
Nodes (9): 30. Proposed repository structure, 31. Critical software interfaces, Cube interface, Part XI — Software Architecture, Planner interface, Policy interface, Robot interface, State-estimator interface (+1 more)

### Community 19 - "assets.py"
Cohesion: 0.44
Nodes (8): cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), Path, Third-party robot assets.  The Sharpa Wave meshes are too large to vendor into t, Return the directory of a Menagerie model, fetching it on first use., sharpa_wave_xml()

### Community 20 - "test_scene_stability.py"
Cohesion: 0.07
Nodes (39): GraspConfig, GraspEnvGPU, _min_jerk(), observation_layout(), range, Tensor, Batched grasp-stabilisation environment on MuJoCo Warp.  :class:`GraspEnvGPU` re, Scale tilt range and pushes; takes effect at each world's next reset. (+31 more)

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 30 - "HandCubeEnv"
Cohesion: 0.07
Nodes (31): GraspRewardWeights, EnvConfig, HandCubeEnv, Any, ndarray, Gymnasium environment for the fixed-wrist hand holding a 2x2x2 cube.  This is th, Task reward; the base environment has no task., Task-specific success or failure termination. (+23 more)

### Community 31 - "TactileSensors"
Cohesion: 0.09
Nodes (22): MjData, MjModel, ndarray, Simulated tactile sensing derived from MuJoCo contacts (docs/project-draft.md se, (n_regions, 3) net contact force on each region, in world coordinates., Noiseless, undelayed reading for the current simulator state., Corrupted, delayed reading; call once per control step., Per-region contact or force readings for the Sharpa Wave hand.      Call :meth:` (+14 more)

### Community 32 - "test_tactile.py"
Cohesion: 0.15
Nodes (10): axis_rotation(), format_moves(), Move, ndarray, Symbolic 2x2x2 cube state.  State is stored the same way the physics model is pa, Rotation angle about the outward face normal, in (-pi, pi]., All 24 proper rotation matrices with integer entries (signed permutations)., Integer rotation matrix for ``quarter_turns`` * 90 degrees about a unit axis. (+2 more)

### Community 33 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 34 - "project-draft.md"
Cohesion: 0.17
Nodes (11): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, Additional research directions, Appendix A — Essential Resources, Appendix B — Immediate Action Plan, Conclusion, Core research papers (+3 more)

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "GraspEnv"
Cohesion: 0.06
Nodes (35): GraspEnv, Any, ndarray, Grasp stabilisation: keep hold of the cube while the wrist reorients (docs/proje, Scale tilt range and pushes; takes effect at the next reset., Gravity direction after tilting the wrist by ``tilt`` towards in-palm direction, Whole-cube orientation, fitted to the cubies (the core may spin inside them)., Great-circle interpolation between unit vectors ``a`` and ``b``. (+27 more)

### Community 37 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

### Community 38 - "Part XVI — Further Research Directions"
Cohesion: 0.10
Nodes (23): Normal, Critic, GaussianActor, _mlp(), PPO, PPOConfig, device, Sequential (+15 more)

### Community 39 - "_Geoms"
Cohesion: 0.07
Nodes (30): BaseCallback, KVWriter, check_server(), flatten_params(), MLflowOutputFormat, Any, Experiment tracking with MLflow.  The MLflow server runs in Docker (``docker/up., Fail fast with a useful message when an HTTP tracking server is down.      MLflo (+22 more)

### Community 40 - "Appendix A — Essential Resources"
Cohesion: 0.23
Nodes (12): parse_moves(), test_initial_state_options(), test_inverse_and_parsing(), test_kociemba_orientation_convention(), test_layer_contains_four_cubies(), test_opposite_faces_are_equivalent_up_to_whole_cube_rotation(), test_quarter_turn_has_order_four(), test_random_sequences_stay_valid() (+4 more)

### Community 41 - "train_grasp.py"
Cohesion: 0.18
Nodes (10): dtype, BatchedReading, CubeMath, device, Tensor, Group tables on a device, and the batched readout., Index of the closest group element for each rotation in ``r`` (..., 3, 3)., Rotation closest to each ``(N, 3, 3)`` matrix sum (Frobenius). (+2 more)

### Community 43 - "PrivilegedObservation"
Cohesion: 0.16
Nodes (15): _group_tables(), indices_to_state(), ndarray, Batched cube readout in PyTorch, mirroring :meth:`CubeInterface.read` and :meth:, state_to_indices(), Return (index into ROTATIONS, angular distance in radians) of the closest group, snap_rotation(), physical_cube() (+7 more)

### Community 46 - "manip-rubiks"
Cohesion: 0.20
Nodes (8): GPU simulation, Grasp stabilisation, Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), Grasp stabilisation benchmark

### Community 47 - "AsymmetricActorCriticPolicy"
Cohesion: 0.05
Nodes (33): BaseFeaturesExtractor, Env, Module, MultiInputActorCriticPolicy, Space, LearnedPolicy, ndarray, Path (+25 more)

### Community 60 - "g2.py"
Cohesion: 0.50
Nodes (4): 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Part XVII — Final Recommendations

## Knowledge Gaps
- **158 isolated node(s):** `up.sh script`, `manip-rubiks`, `graphify`, `Versioning`, `Setup` (+153 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraspEnv` connect `GraspEnv` to `test_scene_stability.py`, `grasp_benchmark.py`, `HandCubeEnv`, `AsymmetricActorCriticPolicy`?**
  _High betweenness centrality (0.206) - this node is a cross-community bridge._
- **Why does `HandCubeScene` connect `m0.py` to `test_observation_leakage.py`, `CubeState`, `RobotInterface`, `SceneConfig`, `HandCubeEnv`, `TactileSensors`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `HandCubeEnv` connect `HandCubeEnv` to `test_observation_leakage.py`, `CubeState`, `m0.py`, `GraspEnv`, `test_scene_stability.py`, `TactileSensors`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `GraspEnv` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`GraspEnv` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `HandCubeScene` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`HandCubeScene` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `CubeState` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`CubeState` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `HandCubeEnv` (e.g. with `GraspConfig` and `GraspEnv`) actually correct?**
  _`HandCubeEnv` has 20 INFERRED edges - model-reasoned connections that need verification._