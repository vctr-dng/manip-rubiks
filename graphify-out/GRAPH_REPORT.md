# Graph Report - manip-rubiks  (2026-11-24)

## Corpus Check
- 105 files · ~85,991 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .example 1, .lock 1)

## Summary
- 1642 nodes · 3878 edges · 104 communities (68 shown, 36 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 445 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c980ecc9`
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
- HandCubeEnv
- RobotInterface
- pytest
- tactile_m1.py
- mlflow
- Part VI — Tactile State Estimation
- GraspEnv
- SceneConfig
- ppo_gpu.py
- test_rotation_env.py
- test_tactile.py
- CubeMath
- test_training.py
- cube_model.py
- Move
- RotationEnv
- RotationEnvGPU
- Tensor
- artifacts.py
- cube_math.py
- CubeState
- evaluate
- train_grasp.py
- LearnedPolicy
- numpy
- parallel.py
- gpu_loop.py
- CubeInterface
- cube_interface.py
- up.sh
- train_grasp_sac_gpu.py
- train_rotation_gpu.py
- test_gpu_grasp_env.py
- run_episode
- torch
- environments/rotation_env.py
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
- TorchGraspPolicy
- test_gpu_tactile_taxels.py
- manip-rubiks
- cube_state.py
- test_scene_stability.py
- train_rotation_gpu.py
- numpy
- ndarray
- base_parser
- EpisodeTrace
- Part XVI — Further Research Directions
- diagnosis.md
- test_push_scale_critic_wrench_and_barrier_on_the_gpu
- test_gpu_training_cli_deploys_on_the_cpu_benchmark

## God Nodes (most connected - your core abstractions)
1. `CubeState` - 59 edges
2. `GraspEnv` - 53 edges
3. `HandCubeScene` - 43 edges
4. `RotationEnv` - 41 edges
5. `HandCubeEnvGPU` - 40 edges
6. `HandCubeEnv` - 39 edges
7. `TactileSensors` - 27 edges
8. `GraspEnvGPU` - 26 edges
9. `TactileRig` - 26 edges
10. `SceneConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Grasp stabilisation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py
- `In-hand reorientation` --references--> `RotationEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/rotation_env.py
- `M5 — Restricted solving ⬜` --references--> `HandCubeEnv`  [INFERRED]
  docs/milestones.md → src/manip_rubiks/environments/hand_cube_env.py
- `Manipulation prerequisites (Stages 1–2) ✅ / 🟡` --references--> `GraspEnv`  [INFERRED]
  docs/milestones.md → src/manip_rubiks/environments/grasp_env.py
- `GPU simulation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py

## Import Cycles
- None detected.

## Communities (104 total, 36 thin omitted)

### Community 0 - "m0.py"
Cohesion: 0.13
Nodes (12): CheckResult, actuator_speed_bounds(), check_actuators(), check_reproducibility(), check_robot_loads(), check_stability(), _contact_penetrations(), main() (+4 more)

### Community 1 - "34. Major technical risks"
Cohesion: 0.25
Nodes (8): 34. Major technical risks, Part XIII — Risk Analysis, Risk 1 — Cube physics is unstable, Risk 2 — Tactile sensing is insufficient, Risk 3 — RL fails to discover stable turns, Risk 4 — GPU acceleration is incompatible with sensor features, Risk 5 — State estimation becomes unreliable, Risk 6 — End-to-end solving is too difficult

### Community 3 - "grasp_benchmark.py"
Cohesion: 0.22
Nodes (12): condition_options(), EpisodeResult, main(), run_benchmark(), run_episode(), summarise(), _worker(), write_report() (+4 more)

### Community 4 - "sac_gpu.py"
Cohesion: 0.12
Nodes (3): _mlp(), ReplayBuffer, TwinQ

### Community 6 - "tracking.py"
Cohesion: 0.10
Nodes (13): docker_env_file(), _main_checkout(), normalize_uri(), root_env_file(), tracking_uri(), uri_from_env_file(), _env(), file_store() (+5 more)

### Community 7 - "11. Hierarchical control architecture"
Cohesion: 0.12
Nodes (17): 11. Hierarchical control architecture, 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Execution monitor, High-level planner (+9 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

### Community 9 - "GraspEnvGPU"
Cohesion: 0.15
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
Nodes (37): Extensions (draft priorities 7–8 and Part XVI) ⬜, How the status was derived, M1 — Tactile foundation ✅, M2 — Single-turn policy ⬜, M3 — Blindfolded estimator ⬜, M4 — Multi-turn execution ⬜, M5 — Restricted solving ⬜, M6 — World-model comparison ⬜ (+29 more)

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
Cohesion: 0.14
Nodes (7): flatten_keys(), build_reset_bank(), _config_key(), _generate(), load_reset_bank(), ResetBank, test_goal_timeout_terminates_and_resets()

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
Cohesion: 0.19
Nodes (9): HandCubeScene, benchmark_throughput(), test_position_actuator_tracks_step(), test_noisy_resets_spawn_without_hand_contact(), test_reset_is_deterministic(), test_reset_places_cube_in_hand(), settled(), taxel_scene() (+1 more)

### Community 30 - "HandCubeEnv"
Cohesion: 0.06
Nodes (27): EnvConfig, HandCubeEnv, _rotation_env_config(), check_episode_logic(), _max_obs_diff(), _obs_equal(), assert_obs_equal(), env() (+19 more)

### Community 32 - "pytest"
Cohesion: 0.24
Nodes (5): test_push_kind_is_validated(), test_tilt_directions_follow_the_hand_frame(), test_delta_action(), test_fingertips_above_palm_and_forward(), test_mount_is_palm_up()

### Community 33 - "tactile_m1.py"
Cohesion: 0.06
Nodes (32): add_taxel_sensors(), _grid_faces(), _mat_to_quat(), pad_frame(), _quat_to_mat(), sensor_name(), skin_vertices(), centroid() (+24 more)

### Community 34 - "mlflow"
Cohesion: 0.10
Nodes (19): name, MLFLOW_PORT, interval, retries, test, timeout, command, container_name (+11 more)

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 37 - "SceneConfig"
Cohesion: 0.25
Nodes (8): build_spec(), mount_rotation(), SceneConfig, TactileConfig, check_tactile_signals(), probe_fingertips(), _sensors(), hand_scene()

### Community 38 - "ppo_gpu.py"
Cohesion: 0.12
Nodes (4): Critic, GaussianActor, _mlp(), RunningMeanStd

### Community 39 - "test_rotation_env.py"
Cohesion: 0.09
Nodes (12): allowed_commands(), command_rotation(), compose_command(), env(), teleport_to_goal(), test_commands_are_quarter_turns_within_the_24_orientations(), test_curriculum_tightens_the_tolerance(), test_first_goal_is_a_quarter_turn_from_the_start() (+4 more)

### Community 40 - "test_tactile.py"
Cohesion: 0.15
Nodes (14): sensors(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom(), test_latency(), test_local_frame_preserves_magnitude(), test_open_hand_reads_zero(), test_pad_compliance_knob() (+6 more)

### Community 42 - "test_training.py"
Cohesion: 0.20
Nodes (5): obs_tensor(), test_actor_ignores_privileged_state_but_critic_uses_it(), test_actor_input_width_excludes_privileged_state(), test_actor_keys_may_not_include_privileged_state(), test_saved_policy_deploys_without_privileged_state()

### Community 43 - "cube_model.py"
Cohesion: 0.24
Nodes (8): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), _stickers()

### Community 45 - "Move"
Cohesion: 0.14
Nodes (12): Move, LayerDriver, MoveResult, check_cube_geometry(), make_fixture(), test_free_cube_structure(), test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved() (+4 more)

### Community 46 - "RotationEnv"
Cohesion: 0.18
Nodes (5): RotationEnv, rotation_angle(), test_bank_reproduces_cpu_resets_with_locked_layers(), test_goal_timeout_terminates_with_failure_penalty(), test_reaching_a_goal_pays_progress_and_bonus_then_issues_the_next()

### Community 49 - "artifacts.py"
Cohesion: 0.10
Nodes (14): _csv_value(), _jsonable(), normalizer_stats(), _run(), write_csv(), write_json(), write_repro(), file_store() (+6 more)

### Community 50 - "cube_math.py"
Cohesion: 0.17
Nodes (10): _group_tables(), indices_to_state(), state_to_indices(), snap_rotation(), physical_cube(), random_rotation(), test_readout_matches_cube_interface(), test_same_configuration_matches_cube_state() (+2 more)

### Community 51 - "CubeState"
Cohesion: 0.12
Nodes (16): M0 — Simulation foundation ✅, CubeState, parse_moves(), test_initial_state_options(), test_observations_do_not_depend_on_the_scramble(), test_reset_with_scrambled_state(), test_inverse_and_parsing(), test_kociemba_orientation_convention() (+8 more)

### Community 52 - "evaluate"
Cohesion: 0.22
Nodes (4): DistributionStats, render_gif(), test_distribution_stats(), test_render_gif()

### Community 53 - "train_grasp.py"
Cohesion: 0.08
Nodes (12): flatten_params(), MLflowOutputFormat, DifficultyCurriculum, EpisodeMetrics, main(), make_env(), _init(), make_vec_env() (+4 more)

### Community 56 - "numpy"
Cohesion: 0.13
Nodes (8): rig(), test_count_blobs_and_centroid_helpers(), test_full_gate_passes(), test_pressing_a_probe_lights_only_that_fingertip(), test_quick_gate_passes_and_writes_a_report(), test_rig_pad_frames_are_orthonormal_and_face_the_probe(), test_slip_channels_separate_along_and_across_motion(), test_taxel_depth_grows_with_the_commanded_depth()

### Community 57 - "parallel.py"
Cohesion: 0.10
Nodes (15): map_resilient(), spawn_pool(), cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), sharpa_wave_xml(), _crash_once() (+7 more)

### Community 58 - "gpu_loop.py"
Cohesion: 0.12
Nodes (14): distribution_snapshot(), EvalResult, evaluate(), obs_slice(), record_videos(), save_checkpoint(), PPOConfig, SACConfig (+6 more)

### Community 59 - "CubeInterface"
Cohesion: 0.10
Nodes (5): Simulation design (M0), CubeInterface, CubeReading, LayerDriverConfig, test_free_full_cube_falls_as_rigid_body()

### Community 60 - "cube_interface.py"
Cohesion: 0.17
Nodes (9): axis_angle_matrix(), chordal_mean(), log_rotation(), mat_to_quat(), min_jerk(), skew(), twist_angle(), wrap_angle() (+1 more)

### Community 62 - "train_grasp_sac_gpu.py"
Cohesion: 0.15
Nodes (7): ArtifactLogger, main(), parse_args(), Task, _train(), test_logger_uploads_periodic_and_final_files(), test_schedule()

### Community 63 - "train_rotation_gpu.py"
Cohesion: 0.27
Nodes (4): StepInfo, grasp_like_flags(), _eval_step(), _flags()

### Community 64 - "test_gpu_grasp_env.py"
Cohesion: 0.16
Nodes (9): flatten_observation(), copy_state(), setup(), test_bank_reproduces_cpu_resets(), test_layer_turn_terminates_and_world_resets(), test_reset_observations_match_cpu(), test_sensing_matches_cpu_in_a_grasp(), test_short_rollout_tracks_cpu() (+1 more)

### Community 65 - "run_episode"
Cohesion: 0.18
Nodes (10): command_plan(), run_episode(), PassivePolicy, test_command_plan_is_seeded_and_covers_all_axes(), test_composed_chain_counts_commands_not_goals(), test_composed_report_has_command_success(), test_composed_z_turn_counts_one_command_for_three_goals(), test_passive_hand_fails_a_composed_z_turn_on_its_first_sub_goal() (+2 more)

### Community 66 - "torch"
Cohesion: 0.20
Nodes (15): make_ppo(), test_actor_never_sees_privileged_state(), test_gae_matches_reference_loop(), test_running_mean_std_matches_numpy(), test_truncated_episodes_are_bootstrapped(), test_update_runs_and_reports_losses(), fill_buffer(), make_sac() (+7 more)

### Community 67 - "environments/rotation_env.py"
Cohesion: 0.12
Nodes (9): GraspRewardWeights, WristMove, RotationConfig, RotationRewardWeights, _add_args(), _config(), _eval_config(), main() (+1 more)

### Community 68 - "repos"
Cohesion: 0.25
Nodes (7): exclude, args, id, repos, hooks, repo, rev

### Community 69 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 70 - "GraspEvalCallback"
Cohesion: 0.16
Nodes (5): outcome(), pick_video_episodes(), GraspEvalCallback, test_outcome_labels(), test_pick_video_episodes()

### Community 73 - "test_grasp_env.py"
Cohesion: 0.12
Nodes (16): GraspConfig, env(), _largest_push_force(), _push_wrenches(), run(), test_alignment_barrier_is_zero_below_onset_and_full_at_the_limit(), test_critic_wrench_is_appended_to_the_privileged_state_only(), test_observations_do_not_depend_on_the_scramble() (+8 more)

### Community 75 - "project-draft.md"
Cohesion: 0.12
Nodes (16): 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Additional research directions, Appendix A — Essential Resources, Appendix B — Immediate Action Plan, Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Conclusion (+8 more)

### Community 77 - "rotation_benchmark.py"
Cohesion: 0.29
Nodes (10): build_report(), policy_label(), EpisodeResult, main(), _outcome(), _pct(), run_benchmark(), summarise() (+2 more)

### Community 78 - "test_gpu_rotation_env.py"
Cohesion: 0.19
Nodes (8): flatten_rotation_observation(), copy_state(), match_goal(), setup(), sync(), test_reaching_a_goal_issues_the_next(), test_reset_observations_match_cpu(), test_short_rollout_tracks_cpu()

### Community 79 - "core.py"
Cohesion: 0.13
Nodes (11): cached(), make_jobs(), run_jobs(), summary_table(), wilson_interval(), Row, test_cached_builds_once(), test_jobs_cover_every_policy_condition_and_held_out_seed_once() (+3 more)

### Community 80 - "asymmetric.py"
Cohesion: 0.07
Nodes (6): AsymmetricActorCriticPolicy, AsymmetricMlpExtractor, KeyedFlattenExtractor, _mlp(), PrivilegedObservation, test_wrapper_exposes_privileged_state()

### Community 81 - "TorchGraspPolicy"
Cohesion: 0.20
Nodes (3): observation_layout(), TorchGraspPolicy, test_checkpoint_deploys_on_the_cpu_reference()

### Community 83 - "manip-rubiks"
Cohesion: 0.29
Nodes (7): GPU simulation, Grasp stabilisation, In-hand reorientation, Known limitations / follow-ups, Layout, manip-rubiks, Setup

### Community 84 - "cube_state.py"
Cohesion: 0.11
Nodes (4): axis_rotation(), format_moves(), _rotation_group(), test_axis_rotation_right_hand_rule()

### Community 85 - "test_scene_stability.py"
Cohesion: 0.18
Nodes (3): scene(), test_actuator_speed_bounds(), test_gate_m0_quick()

### Community 86 - "train_rotation_gpu.py"
Cohesion: 0.20
Nodes (3): _add_args(), _config(), main()

### Community 87 - "numpy"
Cohesion: 0.32
Nodes (5): test_barrier_term_exists_only_when_switched_on(), test_reset_after_a_palm_down_episode_matches_a_fresh_env(), test_observation_contains_only_robot_and_tactile_signals(), test_privileged_state_layout(), test_rotation_group_has_24_proper_rotations()

### Community 89 - "base_parser"
Cohesion: 0.29
Nodes (3): base_parser(), write_json_report(), test_policy_label_and_parser_defaults()

### Community 96 - "Part XVI — Further Research Directions"
Cohesion: 0.50
Nodes (4): 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, Part XVI — Further Research Directions

## Knowledge Gaps
- **1 isolated node(s):** `manip-rubiks`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 660 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **36 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `manip-rubiks` connect `manip-rubiks` to `CubeInterface`, `Milestones and stages`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `CubeState` (e.g. with `M0 — Simulation foundation ✅` and `state_to_indices()`) actually correct?**
  _`CubeState` has 27 INFERRED edges - model-reasoned connections that need verification._
- **What connects `manip-rubiks` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `m0.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Why does `GraspEnv` connect `GraspEnv` to `TactileSensors`, `test_grasp_env.py`, `GraspEnvGPU`, `Milestones and stages`, `GraspEnvGPU`, `PPO`, `HandCubeEnv`, `test_training.py`, `artifacts.py`, `evaluate`, `train_grasp.py`, `test_gpu_grasp_env.py`, `environments/rotation_env.py`, `GraspEvalCallback`, `test_grasp_env.py`, `asymmetric.py`, `TorchGraspPolicy`, `manip-rubiks`, `ndarray`, `test_push_scale_critic_wrench_and_barrier_on_the_gpu`, `test_gpu_training_cli_deploys_on_the_cpu_benchmark`, `.close`, `.wrist_moving`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `GraspEnv` (e.g. with `Manipulation prerequisites (Stages 1–2) ✅ / 🟡` and `GPU simulation`) actually correct?**
  _`GraspEnv` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Should `TactileSensors` be split into smaller, more focused modules?**
  _Cohesion score 0.11692307692307692 - nodes in this community are weakly interconnected._