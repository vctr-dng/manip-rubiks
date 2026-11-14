# Graph Report - manip-rubiks  (2026-11-14)

## Corpus Check
- 58 files · ~35,946 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 761 nodes · 1240 edges · 64 communities (52 shown, 12 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 191 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `68508ed2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Move
- 3. Task specification
- .solved
- CubeState
- HandCubeScene
- CubeParams
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
- Path
- _Geoms
- Part VII — World Model
- 36. First three experiments
- graphify.js
- AGENTS.md
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
- check_episode_logic
- Executive Summary
- Part XVI — Further Research Directions
- Part XVII — Final Recommendations
- manip-rubiks
- up.sh
- __init__.py
- act.py
- speed.py
- .__init__
- manip-rubiks
- AsymmetricActorCriticPolicy
- DifficultyCurriculum
- asymmetric.py
- AsymmetricMlpExtractor
- test_cube_kinematics.py
- check_episode_logic
- test_collisions.py
- __init__.py
- __init__.py
- __init__.py
- .positions
- check_tactile_signals
- Part XVII — Final Recommendations
- up.sh

## God Nodes (most connected - your core abstractions)
1. `HandCubeScene` - 41 edges
2. `CubeState` - 34 edges
3. `GraspEnv` - 33 edges
4. `HandCubeEnv` - 33 edges
5. `Move` - 23 edges
6. `CubeInterface` - 22 edges
7. `EnvConfig` - 21 edges
8. `TactileSensors` - 18 edges
9. `SceneConfig` - 17 edges
10. `RobotInterface` - 16 edges

## Surprising Connections (you probably didn't know these)
- `scene()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_scene_stability.py → src/manip_rubiks/simulation/scene_builder.py
- `test_reset_is_deterministic()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_scene_stability.py → src/manip_rubiks/simulation/scene_builder.py
- `settled()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_tactile.py → src/manip_rubiks/simulation/scene_builder.py
- `test_collision_audit_passes()` --calls--> `check_collisions()`  [INFERRED]
  tests/test_collisions.py → src/manip_rubiks/validation/collisions.py
- `test_tactile_signals_check()` --calls--> `check_tactile_signals()`  [INFERRED]
  tests/test_validation_checks.py → src/manip_rubiks/validation/tactile.py

## Import Cycles
- None detected.

## Communities (64 total, 12 thin omitted)

### Community 0 - "Move"
Cohesion: 0.06
Nodes (41): grav(), grav(), Move, Rotation angle about the outward face normal, in (-pi, pi]., CubeInterface, CubeReading, LayerDriver, LayerDriverConfig (+33 more)

### Community 1 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 2 - ".solved"
Cohesion: 0.14
Nodes (10): build_spec(), is_pad(), mount_rotation(), MjSpec, Hand + cube scene with a fixed wrist (docs/project-draft.md sections 6 and 12)., Fingertip elastomer pads (compliant contact, the tactile surface)., SceneConfig, hand_scene() (+2 more)

### Community 3 - "CubeState"
Cohesion: 0.17
Nodes (6): CubeState, parse_moves(), Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``COR, Indices of the cubies currently in ``face``., Rotate the whole cube (does not change the puzzle configuration)., Whole-cube rotation that brings cubie DBL to its home pose.          A 2x2x2 cub

### Community 4 - "HandCubeScene"
Cohesion: 0.13
Nodes (27): HandCubeScene, Compiled scene with robot and cube interfaces and deterministic resets., CheckResult, actuator_speed_bounds(), benchmark_throughput(), check_actuators(), check_cube_geometry(), check_reproducibility() (+19 more)

### Community 5 - "CubeParams"
Cohesion: 0.07
Nodes (32): build_policy(), condition_options(), EpisodeResult, main(), policy_label(), Any, Path, Grasp-stabilisation benchmark: sustained blindfolded holds under wrist reorienta (+24 more)

### Community 6 - "RobotInterface"
Cohesion: 0.16
Nodes (8): MjData, MjModel, ndarray, Joint-level interface to the Sharpa Wave hand (docs/project-draft.md section 31), Read joint state and command position targets for the hand.      Actions follow, Actuator forces (the simulated analogue of joint-torque feedback)., Delta joint-target action from section 12 (``k_a`` in rad per control step)., RobotInterface

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

### Community 19 - "Path"
Cohesion: 0.44
Nodes (8): cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), Path, Third-party robot assets.  The Sharpa Wave meshes are too large to vendor into t, Return the directory of a Menagerie model, fetching it on first use., sharpa_wave_xml()

### Community 20 - "_Geoms"
Cohesion: 0.18
Nodes (4): Hand + cube scene: resets, reproducibility and short interaction rollouts., scene(), test_reset_is_deterministic(), test_reset_with_scrambled_state()

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 30 - "HandCubeEnv"
Cohesion: 0.08
Nodes (28): GraspRewardWeights, WristMove, EnvConfig, HandCubeEnv, Any, ndarray, Gymnasium environment for the fixed-wrist hand holding a 2x2x2 cube.  This is th, Task reward; the base environment has no task. (+20 more)

### Community 31 - "TactileSensors"
Cohesion: 0.09
Nodes (22): MjData, MjModel, ndarray, Simulated tactile sensing derived from MuJoCo contacts (docs/project-draft.md se, (n_regions, 3) net contact force on each region, in world coordinates., Noiseless, undelayed reading for the current simulator state., Corrupted, delayed reading; call once per control step., Per-region contact or force readings for the Sharpa Wave hand.      Call :meth:` (+14 more)

### Community 32 - "test_tactile.py"
Cohesion: 0.18
Nodes (11): axis_rotation(), format_moves(), ndarray, Symbolic 2x2x2 cube state.  State is stored the same way the physics model is pa, Return (index into ROTATIONS, angular distance in radians) of the closest group, All 24 proper rotation matrices with integer entries (signed permutations)., Integer rotation matrix for ``quarter_turns`` * 90 degrees about a unit axis., _rotation_group() (+3 more)

### Community 33 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 34 - "project-draft.md"
Cohesion: 0.25
Nodes (7): Additional research directions, Appendix A — Essential Resources, Appendix B — Immediate Action Plan, Conclusion, Core research papers, Part XV — Recommended Research Priorities, Simulation and robot models

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "check_episode_logic"
Cohesion: 0.08
Nodes (23): GraspConfig, GraspEnv, Any, ndarray, Grasp stabilisation: keep hold of the cube while the wrist reorients (docs/proje, Scale tilt range and pushes; takes effect at the next reset., Gravity direction after tilting the wrist by ``tilt`` towards in-palm direction, Whole-cube orientation, fitted to the cubies (the core may spin inside them). (+15 more)

### Community 37 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

### Community 38 - "Part XVI — Further Research Directions"
Cohesion: 0.50
Nodes (4): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, Part XVI — Further Research Directions

### Community 39 - "Part XVII — Final Recommendations"
Cohesion: 0.16
Nodes (20): audit_filters(), check_collisions(), _Geoms, _max_pen(), _min_distance(), pinch(), press(), MjData (+12 more)

### Community 40 - "manip-rubiks"
Cohesion: 0.24
Nodes (11): test_initial_state_options(), test_inverse_and_parsing(), test_kociemba_orientation_convention(), test_layer_contains_four_cubies(), test_opposite_faces_are_equivalent_up_to_whole_cube_rotation(), test_quarter_turn_has_order_four(), test_random_sequences_stay_valid(), test_reachable_state_count_with_fixed_corner() (+3 more)

### Community 41 - "up.sh"
Cohesion: 0.19
Nodes (12): BaseCallback, Namespace, EpisodeMetrics, main(), make_env(), make_vec_env(), parse_args(), Train a blindfolded grasp-stabilisation policy with PPO.  The actor sees proprio (+4 more)

### Community 43 - "act.py"
Cohesion: 0.22
Nodes (8): PrivilegedObservation, ndarray, Expose ``info["privileged"]`` as an observation entry, for the critic only., model(), obs_tensor(), Asymmetric PPO: the actor must never depend on privileged state; deployment must, test_actor_ignores_privileged_state_but_critic_uses_it(), test_wrapper_exposes_privileged_state()

### Community 44 - "speed.py"
Cohesion: 0.25
Nodes (4): ndarray, Nominal cube pose above the palm, aligned with the hand., Deterministic reset: hand to home, cube onto the palm, then settle passively., Any penetrating contact involving a hand geom (with itself or the cube).

### Community 45 - ".__init__"
Cohesion: 0.25
Nodes (5): BaseFeaturesExtractor, Env, Space, KeyedFlattenExtractor, Concatenate the flattened Dict entries in an explicit key order.

### Community 46 - "manip-rubiks"
Cohesion: 0.22
Nodes (7): Grasp stabilisation, Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), Grasp stabilisation benchmark

### Community 47 - "AsymmetricActorCriticPolicy"
Cohesion: 0.25
Nodes (5): MultiInputActorCriticPolicy, AsymmetricActorCriticPolicy, Any, PPO policy whose actor reads ``actor_keys`` and whose critic also reads ``critic, test_actor_keys_may_not_include_privileged_state()

### Community 48 - "DifficultyCurriculum"
Cohesion: 0.29
Nodes (3): DifficultyCurriculum, Path, Linear difficulty ramp, applied at the start of every rollout.

### Community 49 - "asymmetric.py"
Cohesion: 0.33
Nodes (5): device, Module, Sequential, _mlp(), Asymmetric actor-critic for Stable-Baselines3 PPO (docs/project-draft.md section

### Community 50 - "AsymmetricMlpExtractor"
Cohesion: 0.60
Nodes (3): AsymmetricMlpExtractor, Policy MLP on the actor slice of the features, value MLP on all of them., Tensor

### Community 51 - "test_cube_kinematics.py"
Cohesion: 0.29
Nodes (9): Random sequence avoiding consecutive moves on the same axis.          If all all, make_fixture(), Physical cube models: structure, state I/O and externally driven legal moves., test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved(), test_read_is_invariant_to_whole_set_rotation(), test_set_state_round_trip(), test_single_axis_sequence() (+1 more)

### Community 52 - "check_episode_logic"
Cohesion: 0.28
Nodes (7): check_episode_logic(), _max_obs_diff(), _obs_equal(), Episode logic of the Gymnasium environment: seeding, resets, termination, leakag, Tactile-signal and episode-logic validation checks., test_episode_logic_check(), test_tactile_signals_check()

### Community 58 - ".positions"
Cohesion: 0.33
Nodes (3): Current position signs (8, 3) of every cubie., Every cubie occupies a distinct slot and the total twist is 0 mod 3., Kociemba corner representation.          ``perm[slot]`` is the cubie in ``slot``

### Community 59 - "check_tactile_signals"
Cohesion: 0.47
Nodes (5): check_tactile_signals(), probe_fingertips(), Tactile signal sanity checks (contact-derived sensors).  These establish that th, Drop a small sphere onto each fingertip pad of the open hand.      Returns the f, _sensors()

### Community 60 - "Part XVII — Final Recommendations"
Cohesion: 0.50
Nodes (4): 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Part XVII — Final Recommendations

## Knowledge Gaps
- **157 isolated node(s):** `up.sh script`, `manip-rubiks`, `graphify`, `Versioning`, `Setup` (+152 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraspEnv` connect `check_episode_logic` to `act.py`, `CubeParams`, `HandCubeEnv`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `HandCubeScene` connect `HandCubeScene` to `Move`, `.solved`, `CubeState`, `RobotInterface`, `Part XVII — Final Recommendations`, `SceneConfig`, `speed.py`, `_Geoms`, `check_tactile_signals`, `HandCubeEnv`, `TactileSensors`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **Why does `HandCubeEnv` connect `HandCubeEnv` to `.solved`, `CubeState`, `check_episode_logic`, `HandCubeScene`, `check_episode_logic`, `TactileSensors`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `HandCubeScene` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`HandCubeScene` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `CubeState` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`CubeState` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `GraspEnv` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`GraspEnv` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `HandCubeEnv` (e.g. with `GraspConfig` and `GraspEnv`) actually correct?**
  _`HandCubeEnv` has 20 INFERRED edges - model-reasoned connections that need verification._