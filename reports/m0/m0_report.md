# Gate M0 report -- PASSED

Environment: MuJoCo 4.0.1, NumPy 2.5.4, Python 3.12.13, Menagerie `370c43965323`
Throughput (hand + cube, CPU, 1 thread): 14582 steps/s (29.16x real time)

| Check | Result |
|---|---|
| robot_loads | pass |
| actuators | pass |
| cube_geometry | pass |
| stability | pass |
| collisions | pass |
| tactile_signals | pass |
| episode_logic | pass |
| reproducibility | pass |

## robot_loads -- pass

- nq: 61
- nv: 52
- nu: 22
- hand_joints: 22
- hand_bodies: 23
- cube_bodies: 9
- timestep: 0.002
- warnings_after_reset: 0
- hand_mass_kg: 1.3151100000000002
- thresholds: {'expected': {'nu': 22, 'hand_joints': 22, 'hand_bodies': 23, 'cube_bodies': 9, 'warnings_after_reset': 0}}

## actuators -- pass

- worst_final_error_rad: 0.00958
- worst_coupling_rad: 5e-05
- all_directions_correct: True
- forces_and_targets_within_limits: True
- max_transient_limit_violation_rad: 0.02054
- max_steady_limit_violation_rad: 0.0
- delta_action_saturates_at_limits: True
- warnings: 0
- info_gravity_sag_home_max_rad: 0.0093
- info_gravity_sag_worst_joint: pinky_CMC
- info_all_joints_at_limits_violation_rad: 0.058

| joint | step | final err | settle [s] | coupling | force ratio |
|---|---|---|---|---|---|
| thumb_CMC_FE | 0.3 | 0.00958 | 0.088 | 1e-05 | 0.632 |
| thumb_CMC_AA | 0.1746 | 0.00259 | 0.05 | 0.0 | 0.698 |
| thumb_MCP_FE | 0.3 | 0.00223 | 0.054 | 0.0 | 0.766 |
| thumb_MCP_AA | 0.1746 | 0.0007 | 0.036 | 0.0 | 0.62 |
| thumb_IP | 0.3 | 0.00461 | 0.068 | 0.0 | 0.423 |
| index_MCP_FE | 0.3 | 0.00261 | 0.056 | 0.0 | 0.766 |
| index_MCP_AA | 0.1746 | 0.00089 | 0.038 | 0.0 | 0.62 |
| index_PIP | 0.3 | 0.00474 | 0.07 | 0.0 | 0.423 |
| index_DIP | 0.3 | 0.0 | 0.016 | 5e-05 | 1.0 |
| middle_MCP_FE | 0.3 | 0.00261 | 0.056 | 0.0 | 0.766 |
| middle_MCP_AA | 0.1746 | 0.00089 | 0.038 | 0.0 | 0.62 |
| middle_PIP | 0.3 | 0.00474 | 0.07 | 0.0 | 0.423 |
| middle_DIP | 0.3 | 0.0 | 0.016 | 5e-05 | 1.0 |
| ring_MCP_FE | 0.3 | 0.00261 | 0.056 | 0.0 | 0.766 |
| ring_MCP_AA | 0.1746 | 0.00089 | 0.038 | 0.0 | 0.62 |
| ring_PIP | 0.3 | 0.00474 | 0.07 | 0.0 | 0.423 |
| ring_DIP | 0.3 | 0.0 | 0.016 | 5e-05 | 1.0 |
| pinky_CMC | 0.1309 | 0.0001 | 0.026 | 0.0 | 0.342 |
| pinky_MCP_FE | 0.3 | 0.00261 | 0.056 | 0.0 | 0.766 |
| pinky_MCP_AA | 0.1746 | 0.00088 | 0.038 | 0.0 | 0.62 |
| pinky_PIP | 0.3 | 0.00474 | 0.07 | 0.0 | 0.423 |
| pinky_DIP | 0.3 | 0.0 | 0.016 | 5e-05 | 1.0 |
- thresholds: {'final_error_rad': 0.05, 'coupling_rad': 0.05, 'transient_limit_violation_rad': 0.05, 'steady_limit_violation_rad': 0.01, 'step_rad': 0.3, 'duration_s': 0.6}
- note: Accuracy tests use an open hand with gravity and contacts off; gravity sag and self-collision limit violations are reported as info_* values.

## cube_geometry -- pass

- **full**: moves=200, moves_successful=200, states_matching_symbolic=200, first_divergence=None, max_snap_error_deg=2.289, mean_snap_error_deg=1.871, max_internal_penetration_mm=0.2925, max_rotor_swing_deg=2.805, max_hold_angle_error_deg=0.686, warnings=0, finite=True, sim_time_s=171.0, wall_time_s=14.15
- **single_axis**: moves=50, moves_successful=50, states_matching_symbolic=50, first_divergence=None, max_snap_error_deg=0.326, mean_snap_error_deg=0.151, max_internal_penetration_mm=0.0, max_rotor_swing_deg=0.0, max_hold_angle_error_deg=0.076, warnings=0, finite=True, sim_time_s=38.8, wall_time_s=0.11
- set_state_round_trip: 50/50
- thresholds: {'max_snap_error_deg': 3.0, 'max_internal_penetration_mm': 1.0, 'match_rate': 1.0}
- note: Full cube = 8 box cubies on ball joints about the cube centre; legal motion is enforced by contacts, not by the kinematic tree. Turns are driven by axial torques on the turning layer while the other layer is held by fixture welds (stand-in for a gripping hand).

## stability -- pass

- max_joint_speed_ratio: 0.4039
- max_cube_speed_m_s: 0.4183
- max_cube_angular_speed_rad_s: 9.7088
- max_pen_in_hand_hand_cube_mm: 0.7658
- max_pen_in_hand_cube_internal_mm: 0.786
- max_cube_speed_any_m_s: 2.5104
- max_pen_any_cube_internal_mm: 1.5258
- info_max_joint_speed_rad_s: 58.425
- info_max_pen_in_hand_pad_cube_mm: 5.118
- info_max_pen_in_hand_hand_hand_mm: 13.114
- info_max_pen_any_environment_mm: 12.505
- episodes: 10
- episode_duration_s: 4.0
- long_rollout_s: 20.0
- long_rollout_realtime_factor: 9.87
- all_finite: True
- total_warnings: 0
- all_final_states_valid: True
- episodes_cube_still_in_hand: 9
- long_rollout_cube_in_hand: False
- thresholds: {'max_joint_speed_ratio': 1.0, 'max_cube_speed_m_s': 1.0, 'max_cube_angular_speed_rad_s': 60.0, 'max_pen_in_hand_hand_cube_mm': 5.0, 'max_pen_in_hand_cube_internal_mm': 1.0, 'max_cube_speed_any_m_s': 3.639, 'max_pen_any_cube_internal_mm': 5.0}
- note: Cube drops are counted but are not instability; floor-impact penetration is reported as info.
- note: Joint speed is judged against the actuator-attainable bound, not a fixed rad/s value: a distal phalanx (4.7 g) released after being pressed past its limit by another finger snaps back at up to ~60 rad/s under its force-limited actuator and settles within ~30 ms. That is fast but physically consistent; the bound is 145 rad/s for DIP joints and 30-73 rad/s for the others.
- note: info_max_pen_in_hand_pad_cube_mm and info_max_pen_in_hand_hand_hand_mm reflect Menagerie's very compliant fingertip pads (elastomer solref 0.06). Pad indentation of several mm is not an instability but is unrealistically deep; pad compliance must be calibrated in M1 (tactile).

## collisions -- pass

- **filters**: hand_geoms=53, pad_geoms=5, hand_cube_pairs_collidable=424/424, cube_floor_collidable=True, stickers_visual_only=True, finger_distal_pairs_collidable=10/10, hand_self_pairs_enabled=639
- **reset_clearance**: seeds=20, min_hand_cube_clearance_mm=0.893, min_rigid_self_clearance_mm=0.027, info_min_self_clearance_incl_pads_mm=0.027
- **settled_home**: cube_touches_hand=True, cube_touches_floor=False, rigid_self_penetration_mm=0.0, info_pad_self_penetration_mm=0.0, warnings=0
- **pinch**: thumb_index_contact_detected=True, max_rigid_self_penetration_mm=2.841, info_max_pad_penetration_mm=9.823, finite=True, warnings=0
- **press**: force_n=20.0, max_rigid_hand_cube_penetration_mm=0.367, cube_displacement_mm=4.34, finite=True, warnings=0
- thresholds: {'min_hand_cube_clearance_mm': 0.0, 'min_rigid_self_clearance_mm': 0.0, 'home_rigid_self_penetration_mm': 0.5, 'pinch_max_rigid_self_penetration_mm': 5.0, 'press_max_rigid_hand_cube_penetration_mm': 2.0}
- note: Rigid = palm and phalanx geoms; pads = compliant fingertip elastomer geoms, reported as info until their compliance is calibrated together with the tactile sensor.
- note: Self pairs exclude parent-child bodies (skipped by MuJoCo) and the Menagerie <exclude> list.

## tactile_signals -- pass

- hand_regions: 22
- coverage_complete: True
- open_hand_silent: True
- resting_force_balance_rel_error: 0.0001
- **first_region_touched_per_target**: thumb=thumb_tip, index=index_tip, middle=middle_tip, ring=ring_tip, pinky=pinky_tip
- pads_silent_after_release: True
- thresholds: {'resting_force_balance_rel_error': 0.02}
- note: Readings are noiseless (TactileSensors.measure); noise, dropout and latency are unit-tested.

## episode_logic -- pass

- gymnasium_env_checker: pass
- same_seed_identical_start: True
- same_seed_identical_scramble: True
- different_seed_differs: True
- reset_after_rollout_equals_fresh_env: True
- max_obs_diff_solved_vs_scrambled: 0.0
- privileged_state_sees_scramble: True
- truncates_at_horizon: True
- drop_terminates: True
- info_reset_ms: 18.3
- info_control_steps_per_s: 360
- thresholds: {'max_obs_diff_solved_vs_scrambled': 0.0001}
- note: Observation leakage: cubies are physically identical, so solved and scrambled cubes must give the same observations under the same actions; only float rounding (different cubie order) may differ.

## reproducibility -- pass

- same_seed_bitwise_identical: True
- different_seed_differs: True
- reset_after_rollout_equals_fresh_reset: True
