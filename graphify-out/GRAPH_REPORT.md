# Graph Report - manip-rubiks  (2026-11-24)

## Corpus Check
- 140 files · ~102,531 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .example 1, .lock 1)

## Summary
- 1697 nodes · 3930 edges · 122 communities (74 shown, 48 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 438 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8aa11da6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- m0.py
- 34. Major technical risks
- TactileSensors
- grasp_benchmark.py
- sac_gpu.py
- test_grasp_env.py
- tracking.py
- 11. Hierarchical control architecture
- 28. Evaluation metrics
- GraspEnvGPU
- 23. Recovery strategies
- 24. Recommended curriculum
- 10. Sensor validation experiments
- Milestones and stages
- 33. Development milestones and acceptance gates
- 6. Technology stack
- 37. Priority order
- 1. Relevant existing research
- 31. Critical software interfaces
- GraspEnvGPU
- PPO
- Part VII — World Model
- 36. First three experiments
- mujoco
- CLAUDE.md
- SAC
- HandCubeScene
- manip-rubiks
- test_environment.py
- RobotInterface
- pytest
- tactile_m1.py
- mlflow
- Part VI — Tactile State Estimation
- GraspEnv
- mujoco
- ppo_gpu.py
- test_rotation_env.py
- test_tactile.py
- CubeMath
- test_training.py
- cube_model.py
- cube_state.py
- RotationEnv
- RotationEnvGPU
- Tensor
- artifacts.py
- cube_math.py
- .solved
- evaluate
- flatten_params
- LearnedPolicy
- numpy
- parallel.py
- gpu_loop.py
- CubeInterface
- cube_interface.py
- up.sh
- train_grasp_sac_gpu.py
- train_rotation_gpu.py
- torch
- run_episode
- torch
- scene_builder.py
- repos
- 3. Task specification
- GraspEvalCallback
- grasp_ppo/grasp_report.md
- grasp_sac/grasp_report.md
- test_grasp_env.py
- SquashedGaussianActor
- project-draft.md
- rotation_benchmark.py
- test_gpu_rotation_env.py
- core.py
- asymmetric.py
- test_gpu_tactile_taxels.py
- compose_command
- cube_state.py
- TactileRig
- tactile_rig.py
- numpy
- gpu/hand_cube_env.py
- assets.py
- EpisodeTrace
- torch_policy.py
- command_rotation
- test_locked_cube.py
- build_spec
- _slide
- Appendix A — Essential Resources
- PadFrame
- pushA/grasp_report.md
- pushB_wrench/grasp_report.md
- pushC_wrench_barrier/grasp_report.md
- pushD/grasp_report.md
- s0_v0/rotation_report.md
- s0_v1/rotation_report.md
- s0_v2/rotation_report.md
- s0_v3/rotation_report.md
- s1_v0/rotation_report.md
- s1_v1/rotation_report.md
- s1_v2/rotation_report.md
- s1_v3/rotation_report.md
- s2_v0/rotation_report.md
- s2_v1/rotation_report.md
- s2_v2/rotation_report.md
- s2_v3/rotation_report.md

## God Nodes (most connected - your core abstractions)
1. `CubeState` - 59 edges
2. `GraspEnv` - 53 edges
3. `HandCubeScene` - 43 edges
4. `RotationEnv` - 41 edges
5. `HandCubeEnvGPU` - 40 edges
6. `HandCubeEnv` - 39 edges
7. `TactileSensors` - 27 edges
8. `TactileRig` - 26 edges
9. `GraspEnvGPU` - 26 edges
10. `SceneConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `M5 — Restricted solving ⬜` --references--> `HandCubeEnv`  [INFERRED]
  docs/milestones.md → src/manip_rubiks/environments/hand_cube_env.py
- `Simulation design (M0)` --references--> `CubeInterface`  [INFERRED]
  README.md → src/manip_rubiks/simulation/cube_interface.py
- `Simulation design (M0)` --references--> `TactileSensors`  [INFERRED]
  README.md → src/manip_rubiks/simulation/tactile_sensors.py
- `Grasp stabilisation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py
- `In-hand reorientation` --references--> `RotationEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/rotation_env.py

## Import Cycles
- None detected.

## Communities (122 total, 48 thin omitted)

### Community 0 - "m0.py"
Cohesion: 0.15
Nodes (13): CheckResult, check_episode_logic(), _max_obs_diff(), _obs_equal(), benchmark_throughput(), check_actuators(), check_cube_geometry(), check_reproducibility() (+5 more)

### Community 1 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 3 - "grasp_benchmark.py"
Cohesion: 0.22
Nodes (11): condition_options(), EpisodeResult, main(), run_benchmark(), summarise(), _worker(), write_report(), test_benchmark_scores_baselines_and_writes_report() (+3 more)

### Community 4 - "sac_gpu.py"
Cohesion: 0.12
Nodes (3): _mlp(), ReplayBuffer, TwinQ

### Community 6 - "tracking.py"
Cohesion: 0.11
Nodes (13): docker_env_file(), _main_checkout(), normalize_uri(), root_env_file(), tracking_uri(), uri_from_env_file(), _env(), file_store() (+5 more)

### Community 7 - "11. Hierarchical control architecture"
Cohesion: 0.12
Nodes (17): 11. Hierarchical control architecture, 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Execution monitor, High-level planner (+9 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

### Community 9 - "GraspEnvGPU"
Cohesion: 0.13
Nodes (4): GraspEnvGPU, GraspStepInfo, _min_jerk(), _rotate_towards()

### Community 10 - "23. Recovery strategies"
Cohesion: 0.13
Nodes (15): 20. Visual inspection, 21. Symbolic cube solver, 22. Execution verification, 23. Recovery strategies, Baseline inspection method, Case A — Incomplete rotation, Case B — No effective rotation, Case C — Whole-cube movement (+7 more)

### Community 11 - "24. Recommended curriculum"
Cohesion: 0.13
Nodes (15): 24. Recommended curriculum, 25. Training with privileged information, 26. Domain randomization, Part IX — Training Strategy, Recommended procedure, Stage 0 — Pure simulation validation, Stage 1 — Grasp stabilization, Stage 2 — Rigid cube rotation (+7 more)

### Community 12 - "10. Sensor validation experiments"
Cohesion: 0.14
Nodes (14): 10. Sensor validation experiments, 8. Sensor requirements, 9. Sensor configurations, Acceptance criteria, Configuration A — Contact-only, Configuration B — Fingertip force measurements, Configuration C — Distributed tactile taxels, Experiment T1 — Static contact (+6 more)

### Community 13 - "Milestones and stages"
Cohesion: 0.05
Nodes (38): Extensions (draft priorities 7–8 and Part XVI) ⬜, How the status was derived, M1 — Tactile foundation ✅, M2 — Single-turn policy ⬜, M3 — Blindfolded estimator ⬜, M4 — Multi-turn execution ⬜, M5 — Restricted solving ⬜, M6 — World-model comparison ⬜ (+30 more)

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

### Community 19 - "GraspEnvGPU"
Cohesion: 0.25
Nodes (5): build_reset_bank(), _config_key(), _generate(), load_reset_bank(), ResetBank

### Community 21 - "Part VII — World Model"
Cohesion: 0.25
Nodes (8): 17. Purpose of the world model, 18. World-model architecture, 19. Using the world model for control, Approach A — Predictive features for the RL policy, Approach B — Model-predictive control, Part VII — World Model, Prediction horizons, World-model training loss

### Community 22 - "36. First three experiments"
Cohesion: 0.25
Nodes (8): 35. Initial project scope, 36. First three experiments, Experiment 1 — Observability, Experiment 2 — Closed-loop manipulation, Experiment 3 — Disturbance recovery, Part XIV — Recommended Minimum Viable Project, Proposed MVP, Why this is the correct starting point

### Community 23 - "mujoco"
Cohesion: 0.12
Nodes (13): is_pad(), audit_filters(), check_collisions(), _Geoms, _max_pen(), _min_distance(), pinch(), press() (+5 more)

### Community 28 - "HandCubeScene"
Cohesion: 0.10
Nodes (11): HandCubeScene, test_fingertips_above_palm_and_forward(), test_position_actuator_tracks_step(), scene(), test_actuator_speed_bounds(), test_gate_m0_quick(), test_noisy_resets_spawn_without_hand_contact(), test_reset_is_deterministic() (+3 more)

### Community 30 - "test_environment.py"
Cohesion: 0.12
Nodes (16): EnvConfig, assert_obs_equal(), env(), test_control_rate(), test_drop_terminates(), test_passes_gymnasium_checker(), test_random_scramble_is_seeded_and_valid(), test_registered() (+8 more)

### Community 32 - "pytest"
Cohesion: 0.20
Nodes (4): hand_scene(), test_delta_action(), test_mount_is_palm_up(), test_curriculum_tightens_the_tolerance()

### Community 33 - "tactile_m1.py"
Cohesion: 0.20
Nodes (13): centroid(), check_contact_location(), check_pad_calibration(), check_robustness(), check_sliding(), count_blobs(), _fmt(), _grid() (+5 more)

### Community 34 - "mlflow"
Cohesion: 0.10
Nodes (19): name, MLFLOW_PORT, interval, retries, test, timeout, command, container_name (+11 more)

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "GraspEnv"
Cohesion: 0.05
Nodes (13): GPU simulation, Grasp stabilisation, Known limitations / follow-ups, Layout, manip-rubiks, Setup, Simulation design (M0), GraspEnv (+5 more)

### Community 37 - "mujoco"
Cohesion: 0.15
Nodes (7): check_tactile_signals(), probe_fingertips(), _sensors(), test_free_full_cube_falls_as_rigid_body(), test_unintended_layer_turn_terminates(), test_episode_logic_check(), test_tactile_signals_check()

### Community 38 - "ppo_gpu.py"
Cohesion: 0.15
Nodes (5): Critic, _mlp(), PPOConfig, RunningMeanStd, test_running_mean_std_matches_numpy()

### Community 39 - "test_rotation_env.py"
Cohesion: 0.20
Nodes (5): env(), teleport_to_goal(), test_first_goal_is_a_quarter_turn_from_the_start(), test_observation_keys_with_and_without_teacher(), test_reaching_a_goal_pays_progress_and_bonus_then_issues_the_next()

### Community 40 - "test_tactile.py"
Cohesion: 0.15
Nodes (15): sensors(), test_contact_appears_when_cube_is_pressed_onto_a_fingertip(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom(), test_latency(), test_local_frame_preserves_magnitude(), test_open_hand_reads_zero() (+7 more)

### Community 42 - "test_training.py"
Cohesion: 0.12
Nodes (6): parse_args(), obs_tensor(), test_actor_ignores_privileged_state_but_critic_uses_it(), test_actor_input_width_excludes_privileged_state(), test_actor_keys_may_not_include_privileged_state(), test_saved_policy_deploys_without_privileged_state()

### Community 43 - "cube_model.py"
Cohesion: 0.24
Nodes (8): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), _stickers()

### Community 45 - "cube_state.py"
Cohesion: 0.14
Nodes (9): format_moves(), Move, parse_moves(), LayerDriver, LayerDriverConfig, MoveResult, min_jerk(), wrap_angle() (+1 more)

### Community 46 - "RotationEnv"
Cohesion: 0.19
Nodes (3): RotationEnv, test_bank_reproduces_cpu_resets_with_locked_layers(), test_goal_timeout_terminates_with_failure_penalty()

### Community 49 - "artifacts.py"
Cohesion: 0.11
Nodes (14): _csv_value(), _jsonable(), pick_video_episodes(), _run(), write_csv(), write_json(), write_repro(), file_store() (+6 more)

### Community 50 - "cube_math.py"
Cohesion: 0.18
Nodes (9): _group_tables(), indices_to_state(), state_to_indices(), snap_rotation(), physical_cube(), random_rotation(), test_readout_matches_cube_interface(), test_same_configuration_matches_cube_state() (+1 more)

### Community 51 - ".solved"
Cohesion: 0.11
Nodes (22): make_fixture(), test_free_cube_structure(), test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved(), test_read_is_invariant_to_whole_set_rotation(), test_set_state_round_trip(), test_single_axis_sequence(), test_single_axis_set_state_rejects_unreachable_states() (+14 more)

### Community 52 - "evaluate"
Cohesion: 0.16
Nodes (6): DistributionStats, normalizer_stats(), render_gif(), test_distribution_stats(), test_normalizer_stats(), test_render_gif()

### Community 53 - "flatten_params"
Cohesion: 0.22
Nodes (3): flatten_params(), MLflowOutputFormat, test_flatten_params()

### Community 55 - "LearnedPolicy"
Cohesion: 0.14
Nodes (3): build_policy(), LearnedPolicy, TorchGraspPolicy

### Community 56 - "numpy"
Cohesion: 0.10
Nodes (9): rig(), test_centroid_recovers_the_press_position(), test_count_blobs_and_centroid_helpers(), test_full_gate_passes(), test_pressing_a_probe_lights_only_that_fingertip(), test_quick_gate_passes_and_writes_a_report(), test_rig_pad_frames_are_orthonormal_and_face_the_probe(), test_slip_channels_separate_along_and_across_motion() (+1 more)

### Community 57 - "parallel.py"
Cohesion: 0.13
Nodes (10): map_resilient(), spawn_pool(), _crash_once(), _fail(), _square(), test_map_resilient_does_not_retry_errors_raised_by_the_job(), test_map_resilient_keeps_order(), test_map_resilient_reruns_the_jobs_a_crashed_pool_left_unfinished() (+2 more)

### Community 58 - "gpu_loop.py"
Cohesion: 0.12
Nodes (12): distribution_snapshot(), EvalResult, evaluate(), main(), obs_slice(), parse_args(), record_videos(), save_checkpoint() (+4 more)

### Community 60 - "cube_interface.py"
Cohesion: 0.15
Nodes (9): CubeReading, axis_angle_matrix(), chordal_mean(), log_rotation(), mat_to_quat(), rotation_angle(), skew(), twist_angle() (+1 more)

### Community 62 - "train_grasp_sac_gpu.py"
Cohesion: 0.15
Nodes (9): ArtifactLogger, SACConfig, main(), main(), parse_args(), sac_config(), _train(), test_logger_uploads_periodic_and_final_files() (+1 more)

### Community 63 - "train_rotation_gpu.py"
Cohesion: 0.13
Nodes (8): StepInfo, grasp_like_flags(), _add_args(), _add_args(), _config(), _eval_step(), _flags(), main()

### Community 64 - "torch"
Cohesion: 0.13
Nodes (18): flatten_observation(), observation_layout(), copy_state(), test_bank_reproduces_cpu_resets(), test_gravity_schedule_matches_cpu(), test_history_wrapper_stacks_frames_newest_first_and_refills_on_reset(), test_layer_turn_terminates_and_world_resets(), test_push_scale_critic_wrench_and_barrier_on_the_gpu() (+10 more)

### Community 65 - "run_episode"
Cohesion: 0.16
Nodes (10): RotationConfig, command_plan(), PassivePolicy, test_command_plan_is_seeded_and_covers_all_axes(), test_composed_chain_counts_commands_not_goals(), test_composed_report_has_command_success(), test_composed_z_turn_counts_one_command_for_three_goals(), test_passive_hand_fails_a_composed_z_turn_on_its_first_sub_goal() (+2 more)

### Community 66 - "torch"
Cohesion: 0.27
Nodes (9): fill_buffer(), make_sac(), reference_nstep(), test_actor_never_sees_privileged_state(), test_normalizer_freezes_after_the_configured_transitions(), test_nstep_sampling_matches_reference(), test_rewards_are_scaled_before_storage(), test_sampling_respects_ring_buffer_wraparound() (+1 more)

### Community 67 - "scene_builder.py"
Cohesion: 0.25
Nodes (3): _rotation_env_config(), RotationRewardWeights, SceneConfig

### Community 68 - "repos"
Cohesion: 0.25
Nodes (7): exclude, args, id, repos, hooks, repo, rev

### Community 69 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 70 - "GraspEvalCallback"
Cohesion: 0.13
Nodes (6): outcome(), EpisodeMetrics, GraspEvalCallback, SaveNormalizerOnBest, _train(), test_outcome_labels()

### Community 73 - "test_grasp_env.py"
Cohesion: 0.05
Nodes (27): GraspConfig, GraspRewardWeights, HistoryEnvGPU, _config(), _eval_config(), main(), _wrap_env(), setup() (+19 more)

### Community 75 - "project-draft.md"
Cohesion: 0.12
Nodes (16): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Appendix B — Immediate Action Plan, Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation (+8 more)

### Community 77 - "rotation_benchmark.py"
Cohesion: 0.14
Nodes (17): base_parser(), build_report(), cached(), policy_label(), summary_table(), wilson_interval(), write_json_report(), EpisodeResult (+9 more)

### Community 78 - "test_gpu_rotation_env.py"
Cohesion: 0.23
Nodes (8): copy_state(), match_goal(), setup(), sync(), test_goal_timeout_terminates_and_resets(), test_reaching_a_goal_issues_the_next(), test_reset_observations_match_cpu(), test_short_rollout_tracks_cpu()

### Community 79 - "core.py"
Cohesion: 0.20
Nodes (7): make_jobs(), run_jobs(), Row, test_jobs_cover_every_policy_condition_and_held_out_seed_once(), test_report_layout(), test_run_jobs_keeps_job_order_in_process(), test_wilson_interval()

### Community 80 - "asymmetric.py"
Cohesion: 0.05
Nodes (11): AsymmetricActorCriticPolicy, AsymmetricMlpExtractor, KeyedFlattenExtractor, _mlp(), PrivilegedObservation, DifficultyCurriculum, make_env(), _init() (+3 more)

### Community 82 - "test_gpu_tactile_taxels.py"
Cohesion: 0.22
Nodes (3): TactileConfig, setup(), test_taxel_corruption_runs_on_the_gpu()

### Community 83 - "compose_command"
Cohesion: 0.25
Nodes (6): Manipulation prerequisites (Stages 1–2) ✅ / 🟡, In-hand reorientation, Composed z from the retrained x/y teachers, compose_command(), test_x_and_y_commands_compose_to_themselves_and_bad_variants_are_rejected(), test_z_turns_are_three_x_and_y_turns()

### Community 84 - "cube_state.py"
Cohesion: 0.11
Nodes (4): M0 — Simulation foundation ✅, axis_rotation(), CubeState, test_axis_rotation_right_hand_rule()

### Community 85 - "TactileRig"
Cohesion: 0.18
Nodes (6): check_multiple_contacts(), check_repeatability(), check_static_contact(), pad_curve(), _press(), TactileRig

### Community 86 - "tactile_rig.py"
Cohesion: 0.22
Nodes (7): add_taxel_sensors(), _grid_faces(), _mat_to_quat(), pad_frame(), _quat_to_mat(), sensor_name(), skin_vertices()

### Community 87 - "numpy"
Cohesion: 0.17
Nodes (7): _rotation_group(), actuator_speed_bounds(), _contact_penetrations(), random_target_trajectory(), rollout(), test_observation_contains_only_robot_and_tactile_signals(), test_privileged_state_layout()

### Community 89 - "assets.py"
Cohesion: 0.40
Nodes (5): cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), sharpa_wave_xml()

### Community 90 - "EpisodeTrace"
Cohesion: 0.46
Nodes (4): EpisodeTrace, run_episode(), run_episode(), test_episode_trace_accumulates_reward_effort_and_misalignment()

### Community 91 - "torch_policy.py"
Cohesion: 0.33
Nodes (4): make_policy(), obs(), test_checkpoints_without_history_use_one_frame(), test_policy_feeds_newest_first_and_refills_after_reset()

### Community 96 - "command_rotation"
Cohesion: 0.25
Nodes (4): allowed_commands(), command_rotation(), test_commands_are_quarter_turns_within_the_24_orientations(), test_z_has_no_shorter_x_y_route()

### Community 97 - "test_locked_cube.py"
Cohesion: 0.40
Nodes (3): squeeze(), test_lock_needs_full_variant_and_solved_cube(), test_locked_cube_stays_rigid_under_power_grasp()

### Community 99 - "_slide"
Cohesion: 0.60
Nodes (4): _slide(), drive(), pos(), record()

### Community 100 - "Appendix A — Essential Resources"
Cohesion: 0.50
Nodes (4): Additional research directions, Appendix A — Essential Resources, Core research papers, Simulation and robot models

## Knowledge Gaps
- **1 isolated node(s):** `manip-rubiks`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 697 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **48 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `manip-rubiks` connect `GraspEnv` to `compose_command`, `Milestones and stages`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `CubeState` (e.g. with `M0 — Simulation foundation ✅` and `state_to_indices()`) actually correct?**
  _`CubeState` has 27 INFERRED edges - model-reasoned connections that need verification._
- **What connects `manip-rubiks` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `TactileSensors` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._
- **Why does `GraspEnv` connect `GraspEnv` to `torch`, `scene_builder.py`, `test_grasp_env.py`, `GraspEvalCallback`, `GraspEnvGPU`, `test_training.py`, `test_grasp_env.py`, `asymmetric.py`, `artifacts.py`, `GraspEnvGPU`, `compose_command`, `evaluate`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `GraspEnv` (e.g. with `Manipulation prerequisites (Stages 1–2) ✅ / 🟡` and `GPU simulation`) actually correct?**
  _`GraspEnv` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Should `sac_gpu.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11578947368421053 - nodes in this community are weakly interconnected._