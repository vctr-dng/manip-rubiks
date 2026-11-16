# Graph Report - manip-rubiks  (2026-11-16)

## Corpus Check
- 72 files · ~44,606 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 948 nodes · 1633 edges · 60 communities (47 shown, 13 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 260 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `13145939`
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
- __init__.py
- __init__.py
- __init__.py
- up.sh

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

## Communities (60 total, 13 thin omitted)

### Community 0 - "Move"
Cohesion: 0.16
Nodes (20): audit_filters(), check_collisions(), _Geoms, _max_pen(), _min_distance(), pinch(), press(), MjData (+12 more)

### Community 1 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 2 - ".solved"
Cohesion: 0.16
Nodes (8): MjData, MjModel, ndarray, Joint-level interface to the Sharpa Wave hand (docs/project-draft.md section 31), Read joint state and command position targets for the hand.      Actions follow, Actuator forces (the simulated analogue of joint-torque feedback)., Delta joint-target action from section 12 (``k_a`` in rad per control step)., RobotInterface

### Community 3 - "CubeState"
Cohesion: 0.06
Nodes (38): axis_rotation(), CubeState, format_moves(), parse_moves(), ndarray, Symbolic 2x2x2 cube state.  State is stored the same way the physics model is pa, Symbolic 2x2x2 state as per-cubie rotations (index ``i`` = cubie with home ``COR, Current position signs (8, 3) of every cubie. (+30 more)

### Community 4 - "HandCubeScene"
Cohesion: 0.13
Nodes (25): CheckResult, actuator_speed_bounds(), benchmark_throughput(), check_actuators(), check_cube_geometry(), check_reproducibility(), check_robot_loads(), check_stability() (+17 more)

### Community 5 - "CubeParams"
Cohesion: 0.11
Nodes (25): build_policy(), condition_options(), EpisodeResult, main(), policy_label(), Any, Path, Grasp-stabilisation benchmark: sustained blindfolded holds under wrist reorienta (+17 more)

### Community 6 - "RobotInterface"
Cohesion: 0.06
Nodes (40): grav(), grav(), Move, Rotation angle about the outward face normal, in (-pi, pi]., CubeInterface, CubeReading, LayerDriver, LayerDriverConfig (+32 more)

### Community 7 - "11. Hierarchical control architecture"
Cohesion: 0.12
Nodes (17): 11. Hierarchical control architecture, 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Execution monitor, High-level planner (+9 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

### Community 9 - "SceneConfig"
Cohesion: 0.19
Nodes (16): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), MjSpec (+8 more)

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
Cohesion: 0.07
Nodes (39): GraspConfig, GraspEnvGPU, _min_jerk(), observation_layout(), range, Tensor, Batched grasp-stabilisation environment on MuJoCo Warp.  :class:`GraspEnvGPU` re, Scale tilt range and pushes; takes effect at each world's next reset. (+31 more)

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 30 - "HandCubeEnv"
Cohesion: 0.06
Nodes (36): GraspRewardWeights, Grasp stabilisation: keep hold of the cube while the wrist reorients (docs/proje, WristMove, EnvConfig, HandCubeEnv, Any, ndarray, Gymnasium environment for the fixed-wrist hand holding a 2x2x2 cube.  This is th (+28 more)

### Community 31 - "TactileSensors"
Cohesion: 0.09
Nodes (22): MjData, MjModel, ndarray, Simulated tactile sensing derived from MuJoCo contacts (docs/project-draft.md se, (n_regions, 3) net contact force on each region, in world coordinates., Noiseless, undelayed reading for the current simulator state., Corrupted, delayed reading; call once per control step., Per-region contact or force readings for the Sharpa Wave hand.      Call :meth:` (+14 more)

### Community 32 - "test_tactile.py"
Cohesion: 0.20
Nodes (12): build_spec(), is_pad(), mount_rotation(), MjSpec, Hand + cube scene with a fixed wrist (docs/project-draft.md sections 6 and 12)., Fingertip elastomer pads (compliant contact, the tactile surface)., SceneConfig, check_tactile_signals() (+4 more)

### Community 33 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 34 - "project-draft.md"
Cohesion: 0.17
Nodes (11): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Appendix B — Immediate Action Plan, Conclusion (+3 more)

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "check_episode_logic"
Cohesion: 0.16
Nodes (9): GraspEnv, Any, ndarray, Scale tilt range and pushes; takes effect at the next reset., Gravity direction after tilting the wrist by ``tilt`` towards in-palm direction, Whole-cube orientation, fitted to the cubies (the core may spin inside them)., Great-circle interpolation between unit vectors ``a`` and ``b``., Hold the cube through random wrist reorientations and pushes.      ``reset(optio (+1 more)

### Community 37 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

### Community 38 - "Part XVI — Further Research Directions"
Cohesion: 0.10
Nodes (23): Normal, Critic, GaussianActor, _mlp(), PPO, PPOConfig, device, Sequential (+15 more)

### Community 39 - "Part XVII — Final Recommendations"
Cohesion: 0.07
Nodes (30): BaseCallback, KVWriter, check_server(), flatten_params(), MLflowOutputFormat, Any, Experiment tracking with MLflow.  The MLflow server runs in Docker (``docker/up., Fail fast with a useful message when an HTTP tracking server is down.      MLflo (+22 more)

### Community 40 - "manip-rubiks"
Cohesion: 0.14
Nodes (11): env(), Grasp-stabilisation environment: wrist schedule, pushes, rewards, success and te, run(), test_observations_do_not_depend_on_the_scramble(), test_passes_gymnasium_checker(), test_passive_hand_drops_the_cube_palm_down(), test_pushes_hit_the_cube_and_can_be_disabled(), test_registered() (+3 more)

### Community 41 - "up.sh"
Cohesion: 0.09
Nodes (25): dtype, BatchedReading, CubeMath, _group_tables(), indices_to_state(), device, ndarray, Tensor (+17 more)

### Community 43 - "act.py"
Cohesion: 0.18
Nodes (12): flatten_observation(), ndarray, Flatten a CPU ``GraspEnv`` observation into the GPU actor layout., copy_state(), The GPU grasp environment must reproduce the CPU reference (GraspEnv)., Copy CPU grasp states (with fingertip contacts) into the GPU worlds and compare, test_bank_reproduces_cpu_resets(), test_gravity_schedule_matches_cpu() (+4 more)

### Community 45 - ".__init__"
Cohesion: 0.17
Nodes (6): PassivePolicy, ndarray, Scripted grasp baselines.  Both policies use only the non-privileged observation, Zero action: the hand keeps its home cage., Drive the joint targets to a fixed grasp pose as fast as the action bound allows, ScriptedGraspPolicy

### Community 46 - "manip-rubiks"
Cohesion: 0.20
Nodes (8): GPU simulation, Grasp stabilisation, Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), Grasp stabilisation benchmark

### Community 47 - "AsymmetricActorCriticPolicy"
Cohesion: 0.05
Nodes (33): BaseFeaturesExtractor, Env, Module, MultiInputActorCriticPolicy, Space, LearnedPolicy, ndarray, Path (+25 more)

### Community 48 - "DifficultyCurriculum"
Cohesion: 0.21
Nodes (6): HandCubeScene, ndarray, Compiled scene with robot and cube interfaces and deterministic resets., Nominal cube pose above the palm, aligned with the hand., Deterministic reset: hand to home, cube onto the palm, then settle passively., Any penetrating contact involving a hand geom (with itself or the cube).

### Community 49 - "asymmetric.py"
Cohesion: 0.18
Nodes (4): Hand + cube scene: resets, reproducibility and short interaction rollouts., scene(), test_reset_is_deterministic(), test_reset_with_scrambled_state()

### Community 50 - "AsymmetricMlpExtractor"
Cohesion: 0.22
Nodes (3): hand_scene(), Sharpa Wave integration: loading, actuator mapping and action interface., test_mount_is_palm_up()

### Community 51 - "test_cube_kinematics.py"
Cohesion: 0.50
Nodes (4): Additional research directions, Appendix A — Essential Resources, Core research papers, Simulation and robot models

## Knowledge Gaps
- **158 isolated node(s):** `up.sh script`, `manip-rubiks`, `graphify`, `Versioning`, `Setup` (+153 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraspEnv` connect `check_episode_logic` to `CubeParams`, `manip-rubiks`, `act.py`, `AsymmetricActorCriticPolicy`, `_Geoms`, `HandCubeEnv`?**
  _High betweenness centrality (0.210) - this node is a cross-community bridge._
- **Why does `HandCubeScene` connect `DifficultyCurriculum` to `test_tactile.py`, `Move`, `.solved`, `CubeState`, `HandCubeScene`, `RobotInterface`, `SceneConfig`, `asymmetric.py`, `AsymmetricMlpExtractor`, `HandCubeEnv`, `TactileSensors`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `HandCubeEnv` connect `HandCubeEnv` to `test_tactile.py`, `CubeState`, `check_episode_logic`, `DifficultyCurriculum`, `_Geoms`, `TactileSensors`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `GraspEnv` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`GraspEnv` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `HandCubeScene` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`HandCubeScene` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `CubeState` (e.g. with `EnvConfig` and `HandCubeEnv`) actually correct?**
  _`CubeState` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `HandCubeEnv` (e.g. with `GraspConfig` and `GraspEnv`) actually correct?**
  _`HandCubeEnv` has 20 INFERRED edges - model-reasoned connections that need verification._