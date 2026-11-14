# Graph Report - sim-foundation-rest  (2026-11-14)

## Corpus Check
- 58 files · ~31,306 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 615 nodes · 967 edges · 57 communities (50 shown, 7 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 141 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8629697d`
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
- Gate M0 report -- PASSED
- Executive Summary
- Part XVI — Further Research Directions
- Part XVII — Final Recommendations
- Appendix A — Essential Resources
- __init__.py

## God Nodes (most connected - your core abstractions)
1. `HandCubeScene` - 43 edges
2. `CubeState` - 34 edges
3. `HandCubeEnv` - 29 edges
4. `Move` - 23 edges
5. `CubeInterface` - 22 edges
6. `SceneConfig` - 19 edges
7. `TactileSensors` - 18 edges
8. `EnvConfig` - 17 edges
9. `RobotInterface` - 16 edges
10. `_Geoms` - 14 edges

## Surprising Connections (you probably didn't know these)
- `scene()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_scene_stability.py → src/manip_rubiks/simulation/scene_builder.py
- `test_reset_is_deterministic()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_scene_stability.py → src/manip_rubiks/simulation/scene_builder.py
- `settled()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_tactile.py → src/manip_rubiks/simulation/scene_builder.py
- `test_episode_logic_check()` --calls--> `check_episode_logic()`  [INFERRED]
  tests/test_validation_checks.py → src/manip_rubiks/validation/episodes.py
- `run()` --calls--> `HandCubeScene`  [INFERRED]
  .scratch/clearance_probe.py → src/manip_rubiks/simulation/scene_builder.py

## Import Cycles
- None detected.

## Communities (57 total, 7 thin omitted)

### Community 0 - "Move"
Cohesion: 0.05
Nodes (43): Move, Rotation angle about the outward face normal, in (-pi, pi]., CubeInterface, CubeReading, LayerDriver, LayerDriverConfig, MoveResult, MjData (+35 more)

### Community 1 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 2 - ".solved"
Cohesion: 0.11
Nodes (24): parse_moves(), Random sequence avoiding consecutive moves on the same axis.          If all all, make_fixture(), Physical cube models: structure, state I/O and externally driven legal moves., test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved(), test_read_is_invariant_to_whole_set_rotation(), test_set_state_round_trip() (+16 more)

### Community 3 - "CubeState"
Cohesion: 0.08
Nodes (19): axis_rotation(), CubeState, format_moves(), ndarray, Symbolic 2x2x2 cube state.  State is stored the same way the physics model is pa, Return (index into ROTATIONS, angular distance in radians) of the closest group, Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``COR, Current position signs (8, 3) of every cubie. (+11 more)

### Community 4 - "HandCubeScene"
Cohesion: 0.07
Nodes (49): HandCubeScene, Compiled scene with robot and cube interfaces and deterministic resets., audit_filters(), check_collisions(), _Geoms, _max_pen(), _min_distance(), pinch() (+41 more)

### Community 5 - "CubeParams"
Cohesion: 0.21
Nodes (15): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), MjSpec (+7 more)

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
Cohesion: 0.07
Nodes (23): clearances(), Initial (pre-settle) clearances at reset: hand-cube and hand-hand., run(), classify(), evaluate(), build_spec(), is_pad(), mount_rotation() (+15 more)

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
Cohesion: 0.12
Nodes (15): Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), Status, actuators -- pass, collisions -- pass (+7 more)

### Community 14 - "33. Development milestones and acceptance gates"
Cohesion: 0.25
Nodes (8): 33. Development milestones and acceptance gates, Milestone M0 — Simulation foundation, Milestone M1 — Tactile foundation, Milestone M2 — Single-turn policy, Milestone M3 — Blindfolded estimator, Milestone M4 — Multi-turn execution, Milestone M5 — Restricted solving, Milestone M6 — World-model comparison

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
Nodes (8): Path, cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), Third-party robot assets.  The Sharpa Wave meshes are too large to vendor into t, Return the directory of a Menagerie model, fetching it on first use., sharpa_wave_xml()

### Community 20 - "test_scene_stability.py"
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
Nodes (27): Any, EnvConfig, HandCubeEnv, ndarray, Gymnasium environment for the fixed-wrist hand holding a 2x2x2 cube.  This is th, Task reward; the base environment has no task., Task-specific success or failure termination., Flat ground-truth cube state for critics and rewards (never for the actor). (+19 more)

### Community 31 - "TactileSensors"
Cohesion: 0.14
Nodes (10): MjData, MjModel, ndarray, Simulated tactile sensing derived from MuJoCo contacts (docs/project-draft.md se, (n_regions, 3) net contact force on each region, in world coordinates., Noiseless, undelayed reading for the current simulator state., Corrupted, delayed reading; call once per control step., Per-region contact or force readings for the Sharpa Wave hand.      Call :meth:` (+2 more)

### Community 32 - "test_tactile.py"
Cohesion: 0.26
Nodes (12): Contact-derived tactile readings: layout, physical consistency and corruption., Hand-hand contacts cancel, so the net force on the hand is the cube's weight., sensors(), settled(), test_contact_appears_when_cube_is_pressed_onto_a_fingertip(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom() (+4 more)

### Community 33 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 34 - "project-draft.md"
Cohesion: 0.29
Nodes (6): 32. Timeline, Appendix B — Immediate Action Plan, Conclusion, Overall estimate, Part XII — Practical Development Plan, Part XV — Recommended Research Priorities

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "Gate M0 report -- PASSED"
Cohesion: 0.29
Nodes (6): actuators -- pass, cube_geometry -- pass, Gate M0 report -- PASSED, reproducibility -- pass, robot_loads -- pass, stability -- pass

### Community 37 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

### Community 38 - "Part XVI — Further Research Directions"
Cohesion: 0.50
Nodes (4): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, Part XVI — Further Research Directions

### Community 39 - "Part XVII — Final Recommendations"
Cohesion: 0.50
Nodes (4): 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Part XVII — Final Recommendations

### Community 40 - "Appendix A — Essential Resources"
Cohesion: 0.50
Nodes (4): Additional research directions, Appendix A — Essential Resources, Core research papers, Simulation and robot models

## Knowledge Gaps
- **159 isolated node(s):** `manip-rubiks`, `graphify`, `Status`, `Setup`, `Layout` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HandCubeScene` connect `HandCubeScene` to `Move`, `test_tactile.py`, `CubeState`, `CubeParams`, `RobotInterface`, `SceneConfig`, `test_scene_stability.py`, `HandCubeEnv`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Why does `CubeState` connect `CubeState` to `Move`, `.solved`, `HandCubeScene`, `SceneConfig`, `HandCubeEnv`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `SceneConfig` connect `SceneConfig` to `Move`, `test_tactile.py`, `CubeState`, `HandCubeScene`, `CubeParams`, `RobotInterface`, `HandCubeEnv`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `HandCubeScene` (e.g. with `run()` and `evaluate()`) actually correct?**
  _`HandCubeScene` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `CubeState` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`CubeState` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `HandCubeEnv` (e.g. with `CubeState` and `HandCubeScene`) actually correct?**
  _`HandCubeEnv` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Move` (e.g. with `CubeInterface` and `CubeReading`) actually correct?**
  _`Move` has 7 INFERRED edges - model-reasoned connections that need verification._