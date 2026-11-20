# Graph Report - manip-rubiks  (2026-11-20)

## Corpus Check
- 99 files · ~79,071 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .example 1, .lock 1)

## Summary
- 1573 nodes · 3737 edges · 96 communities (64 shown, 32 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 422 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6da8944d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- m0.py
- project-draft.md
- TactileSensors
- CubeState
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
- RunningMeanStd
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
- CubeState
- evaluate
- train_grasp.py
- LearnedPolicy
- numpy
- assets.py
- gpu_loop.py
- CubeInterface
- cube_interface.py
- up.sh
- train_grasp_sac_gpu.py
- train_rotation_gpu.py
- test_gpu_grasp_env.py
- HandCubeEnv
- torch
- environments/rotation_env.py
- repos
- 3. Task specification
- flatten_params
- grasp_ppo/grasp_report.md
- grasp_sac/grasp_report.md
- test_grasp_env.py
- SquashedGaussianActor
- Executive Summary
- GraspConfig
- AsymmetricActorCriticPolicy
- rotation_benchmark.py
- PrivilegedObservation
- asymmetric.py
- test_gpu_tactile_taxels.py
- manip-rubiks
- ndarray
- KeyedFlattenExtractor
- AsymmetricMlpExtractor
- test_observation_leakage.py
- Appendix A — Essential Resources
- PadFrame
- CubeReading

## God Nodes (most connected - your core abstractions)
1. `CubeState` - 62 edges
2. `GraspEnv` - 54 edges
3. `HandCubeScene` - 43 edges
4. `HandCubeEnvGPU` - 40 edges
5. `HandCubeEnv` - 39 edges
6. `RotationEnv` - 38 edges
7. `TactileSensors` - 27 edges
8. `TactileRig` - 26 edges
9. `EnvConfig` - 25 edges
10. `SceneConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `M5 — Restricted solving ⬜` --references--> `HandCubeEnv`  [INFERRED]
  docs/milestones.md → src/manip_rubiks/environments/hand_cube_env.py
- `Grasp stabilisation` --references--> `GraspEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/grasp_env.py
- `In-hand reorientation` --references--> `RotationEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/rotation_env.py
- `M0 — Simulation foundation ✅` --references--> `HandCubeEnv`  [INFERRED]
  docs/milestones.md → src/manip_rubiks/environments/hand_cube_env.py
- `Simulation design (M0)` --references--> `HandCubeEnv`  [INFERRED]
  README.md → src/manip_rubiks/environments/hand_cube_env.py

## Import Cycles
- None detected.

## Communities (96 total, 32 thin omitted)

### Community 0 - "m0.py"
Cohesion: 0.12
Nodes (14): HandCubeScene, actuator_speed_bounds(), benchmark_throughput(), check_actuators(), check_reproducibility(), check_robot_loads(), check_stability(), _contact_penetrations() (+6 more)

### Community 1 - "project-draft.md"
Cohesion: 0.11
Nodes (18): 34. Major technical risks, 38. Active tactile perception, 39. Risk-aware symbolic planning, 40. Generalization beyond Rubik's Cubes, 41. Recommended implementation decisions, 42. Recommended order of execution, 43. Final project definition, Appendix B — Immediate Action Plan (+10 more)

### Community 3 - "CubeState"
Cohesion: 0.13
Nodes (21): make_fixture(), test_full_cube_random_sequence_matches_symbolic(), test_full_cube_single_move_from_solved(), test_read_is_invariant_to_whole_set_rotation(), test_set_state_round_trip(), test_single_axis_sequence(), test_single_axis_set_state_rejects_unreachable_states(), test_initial_state_options() (+13 more)

### Community 4 - "sac_gpu.py"
Cohesion: 0.12
Nodes (3): _mlp(), ReplayBuffer, TwinQ

### Community 5 - "test_grasp_env.py"
Cohesion: 0.13
Nodes (4): PassivePolicy, ScriptedGraspPolicy, test_passive_hand_times_out_on_a_single_turn(), test_report_is_written()

### Community 6 - "tracking.py"
Cohesion: 0.09
Nodes (13): docker_env_file(), _main_checkout(), normalize_uri(), root_env_file(), tracking_uri(), uri_from_env_file(), _env(), file_store() (+5 more)

### Community 7 - "11. Hierarchical control architecture"
Cohesion: 0.12
Nodes (17): 11. Hierarchical control architecture, 12. Action representation, 13. Learning an individual face turn, A practical alternative, Alternative: reduced action space, Curriculum, Execution monitor, High-level planner (+9 more)

### Community 8 - "28. Evaluation metrics"
Cohesion: 0.12
Nodes (16): 27. Primary research questions, 28.1 Individual turn success rate, 28.2 Sequence success rate, 28.3 Complete solve success rate, 28.4 State-estimation error, 28.5 Error detection performance, 28.6 Recovery performance, 28.7 Training efficiency (+8 more)

### Community 9 - "GraspEnvGPU"
Cohesion: 0.14
Nodes (5): GraspEnvGPU, GraspStepInfo, _min_jerk(), _rotate_towards(), setup()

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
Cohesion: 0.10
Nodes (10): _group_tables(), indices_to_state(), state_to_indices(), flatten_keys(), build_reset_bank(), _config_key(), _generate(), load_reset_bank() (+2 more)

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
Nodes (10): test_fingertips_above_palm_and_forward(), test_position_actuator_tracks_step(), scene(), test_actuator_speed_bounds(), test_gate_m0_quick(), test_noisy_resets_spawn_without_hand_contact(), test_reset_is_deterministic(), test_reset_places_cube_in_hand() (+2 more)

### Community 30 - "test_environment.py"
Cohesion: 0.13
Nodes (15): EnvConfig, assert_obs_equal(), env(), test_control_rate(), test_drop_terminates(), test_passes_gymnasium_checker(), test_random_scramble_is_seeded_and_valid(), test_registered() (+7 more)

### Community 32 - "pytest"
Cohesion: 0.13
Nodes (9): snap_rotation(), test_rgb_array_render(), squeeze(), test_lock_needs_full_variant_and_solved_cube(), test_locked_cube_stays_rigid_under_power_grasp(), test_delta_action(), test_mount_is_palm_up(), test_snap_rotation_reports_angle() (+1 more)

### Community 33 - "tactile_m1.py"
Cohesion: 0.10
Nodes (25): CheckResult, centroid(), check_contact_location(), check_multiple_contacts(), check_pad_calibration(), check_repeatability(), check_robustness(), check_sliding() (+17 more)

### Community 34 - "mlflow"
Cohesion: 0.10
Nodes (19): name, MLFLOW_PORT, interval, retries, test, timeout, command, container_name (+11 more)

### Community 35 - "Part VI — Tactile State Estimation"
Cohesion: 0.29
Nodes (7): 14. State-estimation objectives, 15. Baseline estimator: recurrent neural network, 16. Advanced estimator: belief-state tracking, Hybrid symbolic and continuous belief, Initial training objective, Part VI — Tactile State Estimation, Training data

### Community 36 - "GraspEnv"
Cohesion: 0.15
Nodes (3): GraspEnv, WristMove, test_bank_reproduces_cpu_resets()

### Community 37 - "mujoco"
Cohesion: 0.12
Nodes (14): build_spec(), mount_rotation(), SceneConfig, TactileConfig, add_taxel_sensors(), _grid_faces(), _mat_to_quat(), pad_frame() (+6 more)

### Community 38 - "RunningMeanStd"
Cohesion: 0.19
Nodes (4): Critic, _mlp(), RunningMeanStd, test_running_mean_std_matches_numpy()

### Community 39 - "test_rotation_env.py"
Cohesion: 0.18
Nodes (5): env(), teleport_to_goal(), test_curriculum_tightens_the_tolerance(), test_first_goal_is_a_quarter_turn_from_the_start(), test_observation_keys_with_and_without_teacher()

### Community 40 - "test_tactile.py"
Cohesion: 0.14
Nodes (15): sensors(), test_contact_appears_when_cube_is_pressed_onto_a_fingertip(), test_dropout_and_noise(), test_fingertip_layout(), test_hand_coverage_includes_every_hand_collision_geom(), test_latency(), test_local_frame_preserves_magnitude(), test_open_hand_reads_zero() (+7 more)

### Community 41 - "CubeMath"
Cohesion: 0.12
Nodes (7): BatchedReading, CubeMath, physical_cube(), random_rotation(), test_readout_matches_cube_interface(), test_same_configuration_matches_cube_state(), test_snap_matches_cpu()

### Community 42 - "test_training.py"
Cohesion: 0.20
Nodes (5): obs_tensor(), test_actor_ignores_privileged_state_but_critic_uses_it(), test_actor_input_width_excludes_privileged_state(), test_actor_keys_may_not_include_privileged_state(), test_saved_policy_deploys_without_privileged_state()

### Community 43 - "cube_model.py"
Cohesion: 0.21
Nodes (9): cube_spec(), cube_xml(), CubeParams, cubie_center(), _cubie_geoms(), fixture_cube_xml(), _fmt(), _stickers() (+1 more)

### Community 45 - "Move"
Cohesion: 0.15
Nodes (7): format_moves(), Move, parse_moves(), LayerDriver, MoveResult, check_cube_geometry(), test_sexy_move_has_order_six()

### Community 46 - "RotationEnv"
Cohesion: 0.12
Nodes (12): RotationEnv, copy_state(), match_goal(), setup(), sync(), test_bank_reproduces_cpu_resets_with_locked_layers(), test_goal_timeout_terminates_and_resets(), test_reaching_a_goal_issues_the_next() (+4 more)

### Community 49 - "artifacts.py"
Cohesion: 0.14
Nodes (12): _csv_value(), _jsonable(), render_gif(), _run(), write_csv(), write_json(), write_repro(), file_store() (+4 more)

### Community 51 - "CubeState"
Cohesion: 0.11
Nodes (4): M0 — Simulation foundation ✅, Simulation design (M0), CubeState, test_observations_do_not_depend_on_the_scramble()

### Community 52 - "evaluate"
Cohesion: 0.22
Nodes (4): DistributionStats, normalizer_stats(), test_distribution_stats(), test_normalizer_stats()

### Community 53 - "train_grasp.py"
Cohesion: 0.08
Nodes (10): outcome(), pick_video_episodes(), DifficultyCurriculum, EpisodeMetrics, GraspEvalCallback, parse_args(), SaveNormalizerOnBest, _train() (+2 more)

### Community 56 - "numpy"
Cohesion: 0.17
Nodes (8): _rotation_group(), rig(), test_count_blobs_and_centroid_helpers(), test_full_gate_passes(), test_pressing_a_probe_lights_only_that_fingertip(), test_quick_gate_passes_and_writes_a_report(), test_rig_pad_frames_are_orthonormal_and_face_the_probe(), test_taxel_depth_grows_with_the_commanded_depth()

### Community 57 - "assets.py"
Cohesion: 0.40
Nodes (5): cache_dir(), ensure_menagerie_model(), _git(), menagerie_dir(), sharpa_wave_xml()

### Community 58 - "gpu_loop.py"
Cohesion: 0.09
Nodes (15): distribution_snapshot(), EvalResult, evaluate(), obs_slice(), parse_args(), record_videos(), save_checkpoint(), Task (+7 more)

### Community 59 - "CubeInterface"
Cohesion: 0.11
Nodes (3): CubeInterface, LayerDriverConfig, test_free_full_cube_falls_as_rigid_body()

### Community 60 - "cube_interface.py"
Cohesion: 0.17
Nodes (11): slerp_direction(), axis_angle_matrix(), chordal_mean(), log_rotation(), mat_to_quat(), min_jerk(), rotation_angle(), skew() (+3 more)

### Community 62 - "train_grasp_sac_gpu.py"
Cohesion: 0.18
Nodes (8): ArtifactLogger, main(), check_server(), main(), main(), test_logger_uploads_periodic_and_final_files(), test_schedule(), test_unreachable_server_fails_fast()

### Community 63 - "train_rotation_gpu.py"
Cohesion: 0.17
Nodes (6): StepInfo, grasp_like_flags(), _config(), _eval_step(), _flags(), main()

### Community 64 - "test_gpu_grasp_env.py"
Cohesion: 0.18
Nodes (8): flatten_observation(), copy_state(), setup(), test_gravity_schedule_matches_cpu(), test_layer_turn_terminates_and_world_resets(), test_reset_observations_match_cpu(), test_sensing_matches_cpu_in_a_grasp(), test_short_rollout_tracks_cpu()

### Community 65 - "HandCubeEnv"
Cohesion: 0.15
Nodes (6): HandCubeEnv, check_episode_logic(), _max_obs_diff(), _obs_equal(), test_episode_logic_check(), test_tactile_signals_check()

### Community 66 - "torch"
Cohesion: 0.15
Nodes (16): test_truncation_after_time_limit(), make_ppo(), test_actor_never_sees_privileged_state(), test_gae_matches_reference_loop(), test_gpu_training_cli_deploys_on_the_cpu_benchmark(), test_truncated_episodes_are_bootstrapped(), test_update_runs_and_reports_losses(), fill_buffer() (+8 more)

### Community 67 - "environments/rotation_env.py"
Cohesion: 0.14
Nodes (7): GraspRewardWeights, allowed_commands(), command_rotation(), _rotation_env_config(), RotationConfig, RotationRewardWeights, test_commands_are_quarter_turns_within_the_24_orientations()

### Community 68 - "repos"
Cohesion: 0.25
Nodes (7): exclude, args, id, repos, hooks, repo, rev

### Community 69 - "3. Task specification"
Cohesion: 0.22
Nodes (9): 3. Task specification, 4. Mathematical formulation, 5.1 Symbolic memory, 5.2 Physical-state memory, 5. Two different kinds of memory, Evaluation, Part II — Formal Problem Definition, Phase A — Visual inspection (+1 more)

### Community 70 - "flatten_params"
Cohesion: 0.22
Nodes (3): flatten_params(), MLflowOutputFormat, test_flatten_params()

### Community 73 - "test_grasp_env.py"
Cohesion: 0.16
Nodes (8): env(), run(), test_passive_hand_drops_the_cube_palm_down(), test_registered(), test_reset_after_a_palm_down_episode_matches_a_fresh_env(), test_schedule_is_smooth_and_seeded(), test_scripted_power_grasp_holds_palm_down(), test_tilt_directions_follow_the_hand_frame()

### Community 74 - "SquashedGaussianActor"
Cohesion: 0.20
Nodes (4): observation_layout(), SquashedGaussianActor, test_checkpoint_deploys_on_the_cpu_reference(), test_squashed_log_prob_matches_torch_distribution()

### Community 75 - "Executive Summary"
Cohesion: 0.33
Nodes (6): Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation, Executive Summary, Expected outcomes, Project objective, Recommended research strategy, Scope recommendation

### Community 77 - "GraspConfig"
Cohesion: 0.20
Nodes (8): GraspConfig, _config(), main(), make_env(), _init(), make_vec_env(), test_pushes_hit_the_cube_and_can_be_disabled(), test_zero_difficulty_keeps_the_palm_up_without_pushes()

### Community 79 - "rotation_benchmark.py"
Cohesion: 0.05
Nodes (41): base_parser(), build_policy(), build_report(), cached(), EpisodeTrace, make_jobs(), policy_label(), run_jobs() (+33 more)

### Community 83 - "manip-rubiks"
Cohesion: 0.29
Nodes (7): GPU simulation, Grasp stabilisation, In-hand reorientation, Known limitations / follow-ups, Layout, manip-rubiks, Setup

### Community 87 - "test_observation_leakage.py"
Cohesion: 0.33
Nodes (3): pair(), test_observation_contains_only_robot_and_tactile_signals(), test_privileged_state_layout()

### Community 88 - "Appendix A — Essential Resources"
Cohesion: 0.50
Nodes (4): Additional research directions, Appendix A — Essential Resources, Core research papers, Simulation and robot models

## Knowledge Gaps
- **1 isolated node(s):** `manip-rubiks`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 634 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `manip-rubiks` connect `manip-rubiks` to `CubeState`, `Milestones and stages`?**
  _High betweenness centrality (0.179) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `CubeState` (e.g. with `M0 — Simulation foundation ✅` and `state_to_indices()`) actually correct?**
  _`CubeState` has 30 INFERRED edges - model-reasoned connections that need verification._
- **What connects `manip-rubiks` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `m0.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11612903225806452 - nodes in this community are weakly interconnected._
- **Why does `GraspEnv` connect `GraspEnv` to `TactileSensors`, `CubeState`, `GraspEnvGPU`, `Milestones and stages`, `GraspEnvGPU`, `test_environment.py`, `test_training.py`, `artifacts.py`, `train_grasp.py`, `LearnedPolicy`, `cube_interface.py`, `test_gpu_grasp_env.py`, `HandCubeEnv`, `torch`, `environments/rotation_env.py`, `test_grasp_env.py`, `SquashedGaussianActor`, `GraspConfig`, `AsymmetricActorCriticPolicy`, `PrivilegedObservation`, `test_gpu_tactile_taxels.py`, `manip-rubiks`, `.set_difficulty`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `GraspEnv` (e.g. with `Manipulation prerequisites (Stages 1–2) ✅ / 🟡` and `GPU simulation`) actually correct?**
  _`GraspEnv` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Should `project-draft.md` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._