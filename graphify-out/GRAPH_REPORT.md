# Graph Report - sac  (2026-11-18)

## Corpus Check
- 76 files · ~56,720 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 5, .example 1, .lock 1)

## Summary
- 1209 nodes · 2808 edges · 73 communities (54 shown, 19 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 330 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f5293f5c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- m0.py
- 3. Task specification
- RobotInterface
- CubeState
- sac_gpu.py
- grasp_benchmark.py
- tracking.py
- 11. Hierarchical control architecture
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
- HandCubeScene
- CLAUDE.md
- SAC
- .reset
- manip-rubiks
- test_environment.py
- TactileSensors
- numpy
- 34. Major technical risks
- mlflow
- Part VI — Tactile State Estimation
- GraspEnv
- project-draft.md
- PPO
- train_grasp_gpu.py
- test_grasp_env.py
- torch
- test_training.py
- cube_model.py
- cube_state.py
- manip-rubiks
- asymmetric.py
- .__init__
- artifacts.py
- HandCubeEnv
- ndarray
- test_tactile.py
- train_grasp.py
- json
- pytest
- _train
- CubeInterface
- cube_interface.py
- up.sh
- ArtifactLogger
- DifficultyCurriculum
- DistributionStats
- episodes.py
- test_cube_kinematics.py
- Appendix A — Essential Resources
- repos
- Executive Summary
- grasp_ppo/grasp_report.md
- grasp_sac/grasp_report.md

## God Nodes (most connected - your core abstractions)
1. `CubeState` - 63 edges
2. `GraspEnv` - 59 edges
3. `HandCubeScene` - 43 edges
4. `GraspEnvGPU` - 40 edges
5. `HandCubeEnv` - 31 edges
6. `_train()` - 28 edges
7. `_train()` - 28 edges
8. `GraspConfig` - 27 edges
9. `Move` - 24 edges
10. `CubeInterface` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Grasp stabilisation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py
- `GPU simulation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py
- `Simulation design (M0)` --references--> `HandCubeEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/hand_cube_env.py
- `GPU simulation` --references--> `GraspEnvGPU`  [INFERRED]
  README.md → src/manip_rubiks/gpu/grasp_env.py
- `Simulation design (M0)` --references--> `TactileSensors`  [INFERRED]
  README.md → src/manip_rubiks/simulation/tactile_sensors.py

## Import Cycles
- None detected.

## Communities (73 total, 19 thin omitted)

### Community 0 - "m0.py"
Cohesion: 0.11
Nodes (16): CheckResult, finite(), warnings_count(), actuator_speed_bounds(), benchmark_throughput(), check_actuators(), check_cube_geometry(), check_reproducibility() (+8 more)

### Community 1 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 3 - "CubeState"
Cohesion: 0.14
Nodes (18): CubeState, parse_moves(), test_initial_state_options(), test_observations_do_not_depend_on_the_scramble(), test_unintended_layer_turn_terminates(), test_observations_do_not_depend_on_the_scramble(), test_reset_with_scrambled_state(), test_inverse_and_parsing() (+10 more)

### Community 4 - "sac_gpu.py"
Cohesion: 0.09
Nodes (5): _mlp(), ReplayBuffer, SquashedGaussianActor, TwinQ, test_squashed_log_prob_matches_torch_distribution()

### Community 5 - "grasp_benchmark.py"
Cohesion: 0.06
Nodes (18): build_policy(), condition_options(), EpisodeResult, main(), policy_label(), run_benchmark(), run_episode(), summarise() (+10 more)

### Community 6 - "tracking.py"
Cohesion: 0.08
Nodes (16): docker_env_file(), _main_checkout(), MLflowOutputFormat, normalize_uri(), root_env_file(), tracking_uri(), uri_from_env_file(), _env() (+8 more)

### Community 7 - "11. Hierarchical control architecture"
Cohesion: 0.12
Nodes (17): 11. Hierarchical control architecture, 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Execution monitor, High-level planner (+9 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

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
Cohesion: 0.40
Nodes (5): cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), sharpa_wave_xml()

### Community 20 - "GraspEnvGPU"
Cohesion: 0.14
Nodes (4): GraspEnvGPU, _min_jerk(), _rotate_towards(), StepInfo

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 23 - "HandCubeScene"
Cohesion: 0.10
Nodes (15): _rotation_group(), HandCubeScene, is_pad(), audit_filters(), check_collisions(), _Geoms, _max_pen(), _min_distance() (+7 more)

### Community 28 - ".reset"
Cohesion: 0.13
Nodes (5): scene(), test_actuator_speed_bounds(), test_gate_m0_quick(), test_noisy_resets_spawn_without_hand_contact(), test_reset_places_cube_in_hand()

### Community 30 - "test_environment.py"
Cohesion: 0.16
Nodes (12): EnvConfig, assert_obs_equal(), env(), test_drop_terminates(), test_passes_gymnasium_checker(), test_random_scramble_is_seeded_and_valid(), test_reset_after_rollout_matches_fresh_env(), test_same_seed_same_episode_start() (+4 more)

### Community 32 - "numpy"
Cohesion: 0.12
Nodes (13): build_spec(), mount_rotation(), SceneConfig, TactileConfig, check_tactile_signals(), probe_fingertips(), _sensors(), test_free_full_cube_falls_as_rigid_body() (+5 more)

### Community 33 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 34 - "mlflow"
Cohesion: 0.10
Nodes (19): name, MLFLOW_PORT, interval, retries, test, timeout, command, container_name (+11 more)

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "GraspEnv"
Cohesion: 0.06
Nodes (24): GraspConfig, GraspEnv, slerp_direction(), WristMove, flatten_observation(), observation_layout(), build_reset_bank(), _config_key() (+16 more)

### Community 37 - "project-draft.md"
Cohesion: 0.14
Nodes (11): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Appendix B — Immediate Action Plan, Conclusion (+3 more)

### Community 38 - "PPO"
Cohesion: 0.11
Nodes (6): Critic, GaussianActor, _mlp(), PPO, PPOConfig, RunningMeanStd

### Community 39 - "train_grasp_gpu.py"
Cohesion: 0.13
Nodes (14): normalizer_stats(), SACConfig, check_server(), flatten_params(), distribution_snapshot(), EvalResult, evaluate(), _obs_slice() (+6 more)

### Community 40 - "test_grasp_env.py"
Cohesion: 0.16
Nodes (8): test_registered(), env(), run(), test_passive_hand_drops_the_cube_palm_down(), test_registered(), test_reset_after_a_palm_down_episode_matches_a_fresh_env(), test_schedule_is_smooth_and_seeded(), test_scripted_power_grasp_holds_palm_down()

### Community 41 - "torch"
Cohesion: 0.06
Nodes (28): BatchedReading, CubeMath, _group_tables(), indices_to_state(), state_to_indices(), snap_rotation(), physical_cube(), random_rotation() (+20 more)

### Community 42 - "test_training.py"
Cohesion: 0.15
Nodes (5): GraspRewardWeights, model(), obs_tensor(), test_actor_ignores_privileged_state_but_critic_uses_it(), test_actor_input_width_excludes_privileged_state()

### Community 43 - "cube_model.py"
Cohesion: 0.24
Nodes (8): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), _stickers()

### Community 45 - "cube_state.py"
Cohesion: 0.18
Nodes (5): format_moves(), Move, LayerDriver, MoveResult, min_jerk()

### Community 46 - "manip-rubiks"
Cohesion: 0.33
Nodes (6): GPU simulation, Grasp stabilisation, Known limitations / follow-ups, Layout, manip-rubiks, Setup

### Community 47 - "asymmetric.py"
Cohesion: 0.09
Nodes (4): AsymmetricActorCriticPolicy, AsymmetricMlpExtractor, KeyedFlattenExtractor, _mlp()

### Community 49 - "artifacts.py"
Cohesion: 0.24
Nodes (6): _csv_value(), _jsonable(), _run(), write_csv(), write_json(), write_repro()

### Community 52 - "test_tactile.py"
Cohesion: 0.29
Nodes (9): sensors(), test_contact_appears_when_cube_is_pressed_onto_a_fingertip(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom(), test_latency(), test_local_frame_preserves_magnitude(), test_open_hand_reads_zero() (+1 more)

### Community 53 - "train_grasp.py"
Cohesion: 0.10
Nodes (8): pick_video_episodes(), EpisodeMetrics, GraspEvalCallback, make_vec_env(), parse_args(), SaveNormalizerOnBest, _train(), test_pick_video_episodes()

### Community 55 - "json"
Cohesion: 0.22
Nodes (6): file_store(), test_json_and_csv_files(), test_normalizer_stats(), test_repro_snapshot_outside_git(), test_repro_snapshot_records_the_commit(), test_training_cli_writes_a_loadable_run()

### Community 56 - "pytest"
Cohesion: 0.15
Nodes (9): test_control_rate(), test_rgb_array_render(), test_tilt_directions_follow_the_hand_frame(), pair(), test_observation_contains_only_robot_and_tactile_signals(), test_privileged_state_layout(), test_snap_rotation_reports_angle(), test_actor_keys_may_not_include_privileged_state() (+1 more)

### Community 58 - "_train"
Cohesion: 0.21
Nodes (4): parse_args(), record_videos(), save_checkpoint(), _train()

### Community 59 - "CubeInterface"
Cohesion: 0.09
Nodes (4): Simulation design (M0), CubeInterface, CubeReading, LayerDriverConfig

### Community 60 - "cube_interface.py"
Cohesion: 0.22
Nodes (9): axis_angle_matrix(), chordal_mean(), log_rotation(), mat_to_quat(), rotation_angle(), skew(), twist_angle(), wrap_angle() (+1 more)

### Community 62 - "ArtifactLogger"
Cohesion: 0.18
Nodes (5): ArtifactLogger, main(), main(), test_logger_uploads_periodic_and_final_files(), test_schedule()

### Community 64 - "DistributionStats"
Cohesion: 0.15
Nodes (6): DistributionStats, outcome(), render_gif(), test_distribution_stats(), test_outcome_labels(), test_render_gif()

### Community 65 - "episodes.py"
Cohesion: 0.27
Nodes (5): check_episode_logic(), _max_obs_diff(), _obs_equal(), test_episode_logic_check(), test_tactile_signals_check()

### Community 66 - "test_cube_kinematics.py"
Cohesion: 0.29
Nodes (8): make_fixture(), test_free_cube_structure(), test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved(), test_read_is_invariant_to_whole_set_rotation(), test_set_state_round_trip(), test_single_axis_sequence(), test_single_axis_set_state_rejects_unreachable_states()

### Community 67 - "Appendix A — Essential Resources"
Cohesion: 0.50
Nodes (4): Additional research directions, Appendix A — Essential Resources, Core research papers, Simulation and robot models

### Community 68 - "repos"
Cohesion: 0.25
Nodes (7): exclude, args, id, repos, hooks, repo, rev

### Community 69 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

## Knowledge Gaps
- **1 isolated node(s):** `manip-rubiks`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 494 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `manip-rubiks` connect `manip-rubiks` to `CubeInterface`, `project-draft.md`?**
  _High betweenness centrality (0.227) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `CubeState` (e.g. with `HandCubeEnv` and `state_to_indices()`) actually correct?**
  _`CubeState` has 31 INFERRED edges - model-reasoned connections that need verification._
- **What connects `manip-rubiks` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `m0.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11428571428571428 - nodes in this community are weakly interconnected._
- **Why does `GraspEnv` connect `GraspEnv` to `DistributionStats`, `CubeState`, `grasp_benchmark.py`, `train_grasp_gpu.py`, `test_grasp_env.py`, `torch`, `test_training.py`, `PrivilegedObservation`, `manip-rubiks`, `HandCubeEnv`, `GraspEnvGPU`, `train_grasp.py`, `json`, `pytest`, `_train`, `test_environment.py`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `GraspEnv` (e.g. with `GPU simulation` and `Grasp stabilisation`) actually correct?**
  _`GraspEnv` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Should `CubeState` be split into smaller, more focused modules?**
  _Cohesion score 0.13548387096774195 - nodes in this community are weakly interconnected._