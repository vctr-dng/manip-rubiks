# Graph Report - m0-sim-foundation  (2026-11-13)

## Corpus Check
- 26 files · ~21,431 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 448 nodes · 684 edges · 30 communities (24 shown, 6 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `40f1cc3d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- CubeInterface
- project-draft.md
- Move
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
- manip-rubiks
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

## God Nodes (most connected - your core abstractions)
1. `CubeState` - 31 edges
2. `Move` - 23 edges
3. `CubeInterface` - 23 edges
4. `HandCubeScene` - 23 edges
5. `RobotInterface` - 15 edges
6. `LayerDriver` - 13 edges
7. `CubeParams` - 13 edges
8. `check_cube_geometry()` - 12 edges
9. `CheckResult` - 11 edges
10. `24. Recommended curriculum` - 11 edges

## Surprising Connections (you probably didn't know these)
- `scene()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_scene_stability.py → src/manip_rubiks/simulation/scene_builder.py
- `test_reset_is_deterministic()` --calls--> `HandCubeScene`  [INFERRED]
  tests/test_scene_stability.py → src/manip_rubiks/simulation/scene_builder.py
- `test_axis_rotation_right_hand_rule()` --calls--> `axis_rotation()`  [INFERRED]
  tests/test_symbolic_transitions.py → src/manip_rubiks/planning/cube_state.py
- `test_snap_rotation_reports_angle()` --calls--> `snap_rotation()`  [INFERRED]
  tests/test_symbolic_transitions.py → src/manip_rubiks/planning/cube_state.py
- `test_single_corner_twist_is_detected_as_illegal()` --calls--> `CubeState`  [INFERRED]
  tests/test_symbolic_transitions.py → src/manip_rubiks/planning/cube_state.py

## Import Cycles
- None detected.

## Communities (30 total, 6 thin omitted)

### Community 0 - "CubeInterface"
Cohesion: 0.06
Nodes (37): CubeInterface, CubeReading, LayerDriverConfig, MoveResult, MjData, MjModel, ndarray, Variant (+29 more)

### Community 1 - "project-draft.md"
Cohesion: 0.04
Nodes (44): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, 34. Major technical risks, 38. Active tactile perception, 39. Risk-aware symbolic planning, 3. Task specification, 40. Generalization beyond Rubik's Cubes (+36 more)

### Community 2 - "Move"
Cohesion: 0.09
Nodes (30): format_moves(), Move, parse_moves(), Symbolic 2x2x2 cube state.  State is stored the same way the physics model is pa, Return (index into ROTATIONS, angular distance in radians) of the closest group, Rotation angle about the outward face normal, in (-pi, pi]., Random sequence avoiding consecutive moves on the same axis.          If all all, snap_rotation() (+22 more)

### Community 3 - "CubeState"
Cohesion: 0.10
Nodes (14): axis_rotation(), CubeState, ndarray, Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``COR, Current position signs (8, 3) of every cubie., Indices of the cubies currently in ``face``., Rotate the whole cube (does not change the puzzle configuration)., Whole-cube rotation that brings cubie DBL to its home pose.          A 2x2x2 cub (+6 more)

### Community 4 - "HandCubeScene"
Cohesion: 0.14
Nodes (28): HandCubeScene, Compiled scene with robot and cube interfaces and deterministic resets., benchmark_throughput(), check_actuators(), check_cube_geometry(), check_reproducibility(), check_robot_loads(), check_stability() (+20 more)

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
Cohesion: 0.17
Nodes (8): build_spec(), mount_rotation(), MjSpec, Hand + cube scene with a fixed wrist (docs/project-draft.md sections 6 and 12)., SceneConfig, hand_scene(), Sharpa Wave integration: loading, actuator mapping and action interface., test_mount_is_palm_up()

### Community 10 - "23. Recovery strategies"
Cohesion: 0.13
Nodes (15): 20. Visual inspection, 21. Symbolic cube solver, 22. Execution verification, 23. Recovery strategies, Baseline inspection method, Case A — Incomplete rotation, Case B — No effective rotation, Case C — Whole-cube movement (+7 more)

### Community 11 - "24. Recommended curriculum"
Cohesion: 0.13
Nodes (15): 24. Recommended curriculum, 25. Training with privileged information, 26. Domain randomization, Part IX — Training Strategy, Recommended procedure, Stage 0 — Pure simulation validation, Stage 1 — Grasp stabilization, Stage 2 — Rigid cube rotation (+7 more)

### Community 12 - "10. Sensor validation experiments"
Cohesion: 0.14
Nodes (14): 10. Sensor validation experiments, 8. Sensor requirements, 9. Sensor configurations, Acceptance criteria, Configuration A — Contact-only, Configuration B — Fingertip force measurements, Configuration C — Distributed tactile taxels, Experiment T1 — Static contact (+6 more)

### Community 13 - "manip-rubiks"
Cohesion: 0.14
Nodes (12): Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), Status, actuators -- pass, cube_geometry -- pass (+4 more)

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

### Community 19 - "Path"
Cohesion: 0.44
Nodes (8): Path, cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), Third-party robot assets.  The Sharpa Wave meshes are too large to vendor into t, Return the directory of a Menagerie model, fetching it on first use., sharpa_wave_xml()

### Community 20 - "test_scene_stability.py"
Cohesion: 0.22
Nodes (4): Hand + cube scene: resets, reproducibility and short interaction rollouts., scene(), test_reset_is_deterministic(), test_reset_with_scrambled_state()

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

## Knowledge Gaps
- **151 isolated node(s):** `manip-rubiks`, `graphify`, `Status`, `Setup`, `Layout` (+146 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CubeState` connect `CubeState` to `CubeInterface`, `SceneConfig`, `Move`, `HandCubeScene`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `HandCubeScene` connect `HandCubeScene` to `CubeInterface`, `CubeState`, `CubeParams`, `RobotInterface`, `SceneConfig`, `test_scene_stability.py`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `CubeInterface` connect `CubeInterface` to `SceneConfig`, `Move`, `CubeState`, `HandCubeScene`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `CubeState` (e.g. with `.__eq__()` and `CubeInterface`) actually correct?**
  _`CubeState` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Move` (e.g. with `CubeInterface` and `CubeReading`) actually correct?**
  _`Move` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `CubeInterface` (e.g. with `CubeState` and `Move`) actually correct?**
  _`CubeInterface` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `HandCubeScene` (e.g. with `CubeState` and `CubeInterface`) actually correct?**
  _`HandCubeScene` has 8 INFERRED edges - model-reasoned connections that need verification._