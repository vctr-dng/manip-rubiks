# Graph Report - manip-rubiks  (2026-11-18)

## Corpus Check
- 88 files · ~65,993 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .example 1, .lock 1)

## Summary
- 1385 nodes · 3295 edges · 83 communities (61 shown, 22 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 369 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7a8a322b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- m0.py
- project-draft.md
- TactileSensors
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
- reset_bank.py
- GraspEnvGPU
- Part VII — World Model
- 36. First three experiments
- mujoco
- CLAUDE.md
- SAC
- HandCubeScene
- manip-rubiks
- test_environment.py
- environments/grasp_env.py
- numpy
- 34. Major technical risks
- mlflow
- Part VI — Tactile State Estimation
- GraspEnv
- Part XVII — Final Recommendations
- PPO
- train_grasp_sac_gpu.py
- test_grasp_env.py
- CubeMath
- test_training.py
- cube_model.py
- Move
- RotationEnv
- asymmetric.py
- HandCubeEnvGPU
- artifacts.py
- HandCubeEnv
- ndarray
- test_tactile.py
- train_grasp.py
- LearnedPolicy
- test_observation_leakage.py
- gpu_loop.py
- CubeInterface
- cube_interface.py
- up.sh
- ArtifactLogger
- train_rotation_gpu.py
- evaluate
- episodes.py
- torch
- environments/rotation_env.py
- repos
- Executive Summary
- scene_builder.py
- grasp_ppo/grasp_report.md
- grasp_sac/grasp_report.md
- GraspEvalCallback
- SquashedGaussianActor
- cube_state.py
- .__init__
- AsymmetricActorCriticPolicy
- flatten_params
- pytest
- KeyedFlattenExtractor
- AsymmetricMlpExtractor

## God Nodes (most connected - your core abstractions)
1. `CubeState` - 63 edges
2. `GraspEnv` - 56 edges
3. `HandCubeScene` - 43 edges
4. `HandCubeEnvGPU` - 41 edges
5. `HandCubeEnv` - 38 edges
6. `RotationEnv` - 38 edges
7. `GraspEnvGPU` - 27 edges
8. `Move` - 24 edges
9. `CubeInterface` - 24 edges
10. `ArtifactLogger` - 24 edges

## Surprising Connections (you probably didn't know these)
- `In-hand reorientation` --references--> `RotationEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/rotation_env.py
- `Grasp stabilisation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py
- `Simulation design (M0)` --references--> `HandCubeEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/hand_cube_env.py
- `GPU simulation` --references--> `GraspEnvGPU`  [INFERRED]
  README.md → src/manip_rubiks/gpu/grasp_env.py
- `Simulation design (M0)` --references--> `TactileSensors`  [INFERRED]
  README.md → src/manip_rubiks/simulation/tactile_sensors.py

## Import Cycles
- None detected.

## Communities (83 total, 22 thin omitted)

### Community 0 - "m0.py"
Cohesion: 0.14
Nodes (14): CheckResult, finite(), warnings_count(), benchmark_throughput(), check_actuators(), check_cube_geometry(), check_reproducibility(), check_robot_loads() (+6 more)

### Community 1 - "project-draft.md"
Cohesion: 0.10
Nodes (19): 38. Active tactile perception, 39. Risk-aware symbolic planning, 3. Task specification, 40. Generalization beyond Rubik's Cubes, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory (+11 more)

### Community 2 - "TactileSensors"
Cohesion: 0.08
Nodes (4): RobotInterface, Region, TactileSensors, probe_fingertips()

### Community 3 - "CubeState"
Cohesion: 0.10
Nodes (18): CubeState, parse_moves(), test_initial_state_options(), test_observations_do_not_depend_on_the_scramble(), test_unintended_layer_turn_terminates(), test_observations_do_not_depend_on_the_scramble(), test_reset_with_scrambled_state(), test_inverse_and_parsing() (+10 more)

### Community 4 - "sac_gpu.py"
Cohesion: 0.12
Nodes (3): _mlp(), ReplayBuffer, TwinQ

### Community 5 - "grasp_benchmark.py"
Cohesion: 0.06
Nodes (28): RotationConfig, build_policy(), condition_options(), EpisodeResult, main(), policy_label(), run_benchmark(), run_episode() (+20 more)

### Community 6 - "tracking.py"
Cohesion: 0.11
Nodes (13): docker_env_file(), _main_checkout(), normalize_uri(), root_env_file(), tracking_uri(), uri_from_env_file(), _env(), file_store() (+5 more)

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

### Community 19 - "reset_bank.py"
Cohesion: 0.07
Nodes (14): flatten_keys(), build_reset_bank(), _config_key(), _generate(), load_reset_bank(), ResetBank, RotationEnvGPU, RotationStepInfo (+6 more)

### Community 20 - "GraspEnvGPU"
Cohesion: 0.15
Nodes (4): GraspEnvGPU, GraspStepInfo, _min_jerk(), _rotate_towards()

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 23 - "mujoco"
Cohesion: 0.13
Nodes (10): audit_filters(), check_collisions(), _Geoms, _max_pen(), _min_distance(), pinch(), press(), reset_clearance() (+2 more)

### Community 28 - "HandCubeScene"
Cohesion: 0.12
Nodes (10): HandCubeScene, check_tactile_signals(), _sensors(), scene(), test_actuator_speed_bounds(), test_gate_m0_quick(), test_noisy_resets_spawn_without_hand_contact(), test_reset_is_deterministic() (+2 more)

### Community 30 - "test_environment.py"
Cohesion: 0.17
Nodes (10): assert_obs_equal(), test_drop_terminates(), test_passes_gymnasium_checker(), test_random_scramble_is_seeded_and_valid(), test_registered(), test_reset_after_rollout_matches_fresh_env(), test_same_seed_same_episode_start(), test_passes_gymnasium_checker() (+2 more)

### Community 31 - "environments/grasp_env.py"
Cohesion: 0.14
Nodes (7): GraspConfig, GraspRewardWeights, WristMove, _config(), main(), test_pushes_hit_the_cube_and_can_be_disabled(), test_zero_difficulty_keeps_the_palm_up_without_pushes()

### Community 32 - "numpy"
Cohesion: 0.13
Nodes (8): _rotation_group(), actuator_speed_bounds(), _contact_penetrations(), test_free_full_cube_falls_as_rigid_body(), hand_scene(), test_delta_action(), test_fingertips_above_palm_and_forward(), test_position_actuator_tracks_step()

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
Nodes (22): GPU simulation, Grasp stabilisation, In-hand reorientation, Known limitations / follow-ups, Layout, manip-rubiks, Setup, Grasp stabilisation benchmark (+14 more)

### Community 37 - "Part XVII — Final Recommendations"
Cohesion: 0.50
Nodes (4): 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Part XVII — Final Recommendations

### Community 38 - "PPO"
Cohesion: 0.11
Nodes (7): Critic, GaussianActor, _mlp(), PPO, PPOConfig, RunningMeanStd, test_running_mean_std_matches_numpy()

### Community 39 - "train_grasp_sac_gpu.py"
Cohesion: 0.29
Nodes (7): SACConfig, distribution_snapshot(), main(), parse_args(), sac_config(), save_checkpoint(), _train()

### Community 40 - "test_grasp_env.py"
Cohesion: 0.22
Nodes (6): run(), test_passive_hand_drops_the_cube_palm_down(), test_reset_after_a_palm_down_episode_matches_a_fresh_env(), test_schedule_is_smooth_and_seeded(), test_scripted_power_grasp_holds_palm_down(), test_tilt_directions_follow_the_hand_frame()

### Community 41 - "CubeMath"
Cohesion: 0.12
Nodes (7): BatchedReading, CubeMath, physical_cube(), random_rotation(), test_readout_matches_cube_interface(), test_same_configuration_matches_cube_state(), test_snap_matches_cpu()

### Community 42 - "test_training.py"
Cohesion: 0.20
Nodes (5): obs_tensor(), test_actor_ignores_privileged_state_but_critic_uses_it(), test_actor_input_width_excludes_privileged_state(), test_actor_keys_may_not_include_privileged_state(), test_saved_policy_deploys_without_privileged_state()

### Community 43 - "cube_model.py"
Cohesion: 0.24
Nodes (8): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), _stickers()

### Community 45 - "Move"
Cohesion: 0.14
Nodes (12): Move, LayerDriver, MoveResult, min_jerk(), make_fixture(), test_free_cube_structure(), test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved() (+4 more)

### Community 46 - "RotationEnv"
Cohesion: 0.07
Nodes (21): allowed_commands(), command_rotation(), RotationEnv, flatten_rotation_observation(), copy_state(), match_goal(), setup(), sync() (+13 more)

### Community 49 - "artifacts.py"
Cohesion: 0.14
Nodes (12): _csv_value(), _jsonable(), render_gif(), _run(), write_csv(), write_json(), write_repro(), file_store() (+4 more)

### Community 50 - "HandCubeEnv"
Cohesion: 0.19
Nodes (3): HandCubeEnv, env(), test_truncation_without_termination()

### Community 52 - "test_tactile.py"
Cohesion: 0.29
Nodes (9): sensors(), test_contact_appears_when_cube_is_pressed_onto_a_fingertip(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom(), test_latency(), test_local_frame_preserves_magnitude(), test_open_hand_reads_zero() (+1 more)

### Community 53 - "train_grasp.py"
Cohesion: 0.10
Nodes (11): check_server(), DifficultyCurriculum, EpisodeMetrics, main(), make_env(), _init(), make_vec_env(), parse_args() (+3 more)

### Community 56 - "test_observation_leakage.py"
Cohesion: 0.33
Nodes (3): pair(), test_observation_contains_only_robot_and_tactile_signals(), test_privileged_state_layout()

### Community 58 - "gpu_loop.py"
Cohesion: 0.14
Nodes (9): normalizer_stats(), pick_video_episodes(), distribution_snapshot(), EvalResult, obs_slice(), record_videos(), save_checkpoint(), test_normalizer_stats() (+1 more)

### Community 59 - "CubeInterface"
Cohesion: 0.10
Nodes (4): Simulation design (M0), CubeInterface, CubeReading, LayerDriverConfig

### Community 60 - "cube_interface.py"
Cohesion: 0.19
Nodes (9): axis_angle_matrix(), chordal_mean(), log_rotation(), mat_to_quat(), rotation_angle(), skew(), twist_angle(), wrap_angle() (+1 more)

### Community 62 - "ArtifactLogger"
Cohesion: 0.17
Nodes (7): ArtifactLogger, main(), parse_args(), Task, _train(), test_logger_uploads_periodic_and_final_files(), test_schedule()

### Community 63 - "train_rotation_gpu.py"
Cohesion: 0.12
Nodes (7): StepInfo, grasp_like_flags(), _add_args(), _config(), _eval_step(), _flags(), main()

### Community 64 - "evaluate"
Cohesion: 0.21
Nodes (3): DistributionStats, evaluate(), test_distribution_stats()

### Community 65 - "episodes.py"
Cohesion: 0.27
Nodes (5): check_episode_logic(), _max_obs_diff(), _obs_equal(), test_episode_logic_check(), test_tactile_signals_check()

### Community 66 - "torch"
Cohesion: 0.20
Nodes (15): make_ppo(), test_actor_never_sees_privileged_state(), test_gae_matches_reference_loop(), test_truncated_episodes_are_bootstrapped(), test_update_runs_and_reports_losses(), fill_buffer(), make_sac(), reference_nstep() (+7 more)

### Community 67 - "environments/rotation_env.py"
Cohesion: 0.18
Nodes (10): EnvConfig, _rotation_env_config(), RotationRewardWeights, SceneConfig, TactileConfig, test_single_axis_variant_scrambles_u_and_d_only(), test_tactile_configurations(), squeeze() (+2 more)

### Community 68 - "repos"
Cohesion: 0.25
Nodes (7): exclude, args, id, repos, hooks, repo, rev

### Community 69 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

### Community 70 - "scene_builder.py"
Cohesion: 0.15
Nodes (4): build_spec(), is_pad(), mount_rotation(), test_mount_is_palm_up()

### Community 73 - "GraspEvalCallback"
Cohesion: 0.23
Nodes (3): outcome(), GraspEvalCallback, test_outcome_labels()

### Community 75 - "cube_state.py"
Cohesion: 0.24
Nodes (6): _group_tables(), indices_to_state(), state_to_indices(), format_moves(), snap_rotation(), test_snap_rotation_reports_angle()

### Community 78 - "flatten_params"
Cohesion: 0.22
Nodes (3): flatten_params(), MLflowOutputFormat, test_flatten_params()

### Community 79 - "pytest"
Cohesion: 0.28
Nodes (4): test_control_rate(), test_rgb_array_render(), test_condition_options_are_seeded(), test_wilson_interval()

## Knowledge Gaps
- **1 isolated node(s):** `manip-rubiks`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 555 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `manip-rubiks` connect `GraspEnv` to `CubeInterface`?**
  _High betweenness centrality (0.246) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `CubeState` (e.g. with `HandCubeEnv` and `state_to_indices()`) actually correct?**
  _`CubeState` has 31 INFERRED edges - model-reasoned connections that need verification._
- **What connects `manip-rubiks` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `m0.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13793103448275862 - nodes in this community are weakly interconnected._
- **Why does `GraspEnv` connect `GraspEnv` to `torch`, `environments/rotation_env.py`, `CubeState`, `grasp_benchmark.py`, `train_grasp_sac_gpu.py`, `test_grasp_env.py`, `GraspEvalCallback`, `test_training.py`, `PrivilegedObservation`, `AsymmetricActorCriticPolicy`, `artifacts.py`, `HandCubeEnv`, `reset_bank.py`, `GraspEnvGPU`, `train_grasp.py`, `LearnedPolicy`, `test_environment.py`, `environments/grasp_env.py`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `GraspEnv` (e.g. with `GPU simulation` and `Grasp stabilisation`) actually correct?**
  _`GraspEnv` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Should `project-draft.md` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._