# Gate M1 report -- PASSED

Environment: MuJoCo 4.0.1, NumPy 2.5.4, Python 3.12.16, Menagerie `370c43965323`

| Check | Result |
|---|---|
| T1_static_contact | pass |
| T2_sliding | pass |
| T3_contact_location | pass |
| T4_multiple_contacts | pass |
| T5_robustness | pass |
| repeatability | pass |
| pad_calibration | pass |

## T1_static_contact -- pass

- commanded_depth_mm: [0.5, 1.0, 2.0, 3.0]

**per_finger**

| | pass | force_N | taxel_depth_sum_mm | taxel_depth_max_mm | monotonic | crosstalk_on_other_fingers | released_to_zero | binary_contact_at_deepest |
|---|---|---|---|---|---|---|---|---|
| thumb | True | [0.0108, 0.0343, 0.081, 0.1271] | [1.149, 2.999, 8.223, 11.13] | [0.203, 0.66, 2.03, 3.342] | {"force": true, "taxel_depth": true} | 0.0 | True | 1.0 |
| index | True | [0.0138, 0.0371, 0.0828, 0.1282] | [1.8, 6.075, 11.503, 15.852] | [0.232, 0.814, 1.961, 3.141] | {"force": true, "taxel_depth": true} | 0.0 | True | 1.0 |
| middle | True | [0.0138, 0.0371, 0.0828, 0.1282] | [1.8, 6.075, 11.503, 15.852] | [0.232, 0.814, 1.961, 3.141] | {"force": true, "taxel_depth": true} | 0.0 | True | 1.0 |
| ring | True | [0.0138, 0.0371, 0.0828, 0.1282] | [1.8, 6.075, 11.503, 15.852] | [0.232, 0.814, 1.961, 3.141] | {"force": true, "taxel_depth": true} | 0.0 | True | 1.0 |
| pinky | True | [0.0138, 0.0371, 0.0828, 0.1281] | [1.8, 6.075, 11.502, 15.853] | [0.232, 0.814, 1.961, 3.141] | {"force": true, "taxel_depth": true} | 0.0 | True | 1.0 |

- thresholds: {"monotonic": "non-decreasing in commanded depth", "crosstalk": "< 1e-9", "release": "all readings zero"}
- note: Depth is the commanded penetration of a flat probe below the unloaded pad surface; the fingers are position controlled, so the realised indentation is smaller than the command.
- note: The binary contact flag needs 0.05 N, so it stays off for the lightest touches while taxel depth already responds.

## T2_sliding -- pass


**index_sweep**

| | rows | B_tangential_force_increases_with_speed |
|---|---|---|
| along | [{"speed_m_s": 0.01, "slip_measured_m_s": 0.01, "rel_error": 0.0002, "cross_channel_ratio": 0.00117, "sign_blind_diff_ratio": 6e-05, "B_tangential_force_N": 0.0359, "B_sign_flips_with_direction": true}, {"speed_m_s": 0.03, "slip_measured_m_s": 0.02999, "rel_error": 0.0003, "cross_channel_ratio": 0.00154, "sign_blind_diff_ratio": 0.00013, "B_tangential_force_N": 0.0719, "B_sign_flips_with_direction": true}, {"speed_m_s": 0.06, "slip_measured_m_s": 0.05996, "rel_error": 0.0006, "cross_channel_ratio": 0.00188, "sign_blind_diff_ratio": 0.00044, "B_tangential_force_N": 0.1217, "B_sign_flips_with_direction": true}, {"speed_m_s": 0.1, "slip_measured_m_s": 0.09988, "rel_error": 0.0012, "cross_channel_ratio": 0.00235, "sign_blind_diff_ratio": 0.0013, "B_tangential_force_N": 0.1847, "B_sign_flips_with_direction": true}] | True |
| across | [{"speed_m_s": 0.01, "slip_measured_m_s": 0.01, "rel_error": 0.0001, "cross_channel_ratio": 0.00085, "sign_blind_diff_ratio": 0.0, "B_tangential_force_N": 0.0144, "B_sign_flips_with_direction": true}, {"speed_m_s": 0.03, "slip_measured_m_s": 0.02975, "rel_error": 0.0082, "cross_channel_ratio": 0.00142, "sign_blind_diff_ratio": 0.0, "B_tangential_force_N": 0.0426, "B_sign_flips_with_direction": true}, {"speed_m_s": 0.06, "slip_measured_m_s": 0.05923, "rel_error": 0.0128, "cross_channel_ratio": 0.00143, "sign_blind_diff_ratio": 0.0, "B_tangential_force_N": 0.0769, "B_sign_flips_with_direction": true}, {"speed_m_s": 0.1, "slip_measured_m_s": 0.09852, "rel_error": 0.0148, "cross_channel_ratio": 0.00089, "sign_blind_diff_ratio": 0.0, "B_tangential_force_N": 0.1163, "B_sign_flips_with_direction": true}] | True |

- static_slip_max_m_s: 2.8928565787122036e-05

**all_fingertips_along_0.03_m_s**

| | slip_along_m_s | rel_error | cross_ratio |
|---|---|---|---|
| thumb | 0.02995 | 0.0016 | 0.01392 |
| index | 0.02999 | 0.0003 | 0.00154 |
| middle | 0.02999 | 0.0003 | 0.00164 |
| ring | 0.02999 | 0.0003 | 0.00144 |
| pinky | 0.02999 | 0.0003 | 0.00164 |

- signed_slip_channels: False
- binary_contact_static_vs_sliding: [1.0, 1.0]
- binary_contact_sees_sliding: False
- thresholds: {"slip_rel_error": "<= 0.1", "cross_channel_ratio": "<= 0.05", "static_slip_m_s": "< 2e-3"}
- note: Configuration C reports slip speed magnitudes along (channel 1) and across (channel 2) the pad: both channels are non-negative, so the direction of slip is not observable from C alone (sign_blind_diff_ratio ~ 0). Configuration B's tangential force is signed.
- note: Configuration A is a single bit per fingertip and does not change between a static press and a slide.

## T3_contact_location -- pass

- grid_points: 25
- positions_mm: [-4.0, -2.0, 0.0, 2.0, 4.0]
- taxel_pitch_mm_along_across: {"thumb": [1.94, 2.33], "index": [1.53, 2.0], "middle": [1.53, 2.0], "ring": [1.53, 2.0], "pinky": [1.53, 2.0]}

**per_finger**

| | pass | C_mean_error_mm | C_max_error_mm | C_corr_along_across | B_linear_fit_R2_along_across | A_distinct_values |
|---|---|---|---|---|---|---|
| thumb | True | 0.395 | 0.82 | [0.9998, 0.9885] | [0.084, 0.996] | 1 |
| index | True | 0.169 | 0.353 | [0.9986, 0.9995] | [0.623, 0.993] | 1 |
| middle | True | 0.169 | 0.353 | [0.9986, 0.9995] | [0.623, 0.993] | 1 |
| ring | True | 0.169 | 0.353 | [0.9986, 0.9995] | [0.623, 0.993] | 1 |
| pinky | True | 0.169 | 0.353 | [0.9986, 0.9995] | [0.624, 0.993] | 1 |

- thresholds: {"C_mean_error": "<= 1.0 mm", "C_corr": ">= 0.98"}
- note: C localises the contact as the depth-weighted centroid of the taxel map. A is one bit per fingertip, so it carries no location. B is a net force vector, but in this noiseless set-up a linear map of it recovers the across-pad position almost perfectly and the along-pad position only partly (B_linear_fit_R2): the pad is a curved capsule, so the force direction depends on where it is pressed across it. That is not a reliable location sensor on a real contact patch or under noise, and B cannot separate two contacts (T4) while C can.

## T4_multiple_contacts -- pass

- finger: index
- probe_width_mm: 3.0
- blobs_single_probe: 1
- blobs_by_centre_separation: {"2mm": 1, "3mm": 1, "4mm": 1, "6mm": 2, "8mm": 2}
- resolved_from_mm: 6.0
- taxel_pitch_along_mm: 1.53
- thresholds: {"single_probe_blobs": 1, "two_blobs_from_separation": "6 mm"}
- note: Two probes 3 mm wide on one pad. A blob is a 4-connected region of taxels above half the peak depth. A and B return a single value per fingertip and cannot tell one contact from two.

## T5_robustness -- pass

- finger: index
- samples: 50
- estimator: median of 5 reads, detection threshold 0.02 of full scale (0.3 mm)

**sweep**

| | precision | recall | F1 | localisation_error_mm |
|---|---|---|---|---|
| noise=0.0,dropout=0.0,latency=0 | 1.0 | 1.0 | 1.0 | 0.485 |
| noise=0.002,dropout=0.0,latency=0 | 1.0 | 1.0 | 1.0 | 0.489 |
| noise=0.005,dropout=0.0,latency=0 | 1.0 | 1.0 | 1.0 | 0.491 |
| noise=0.01,dropout=0.0,latency=0 | 1.0 | 1.0 | 1.0 | 0.469 |
| noise=0.02,dropout=0.0,latency=0 | 0.568 | 1.0 | 0.725 | 0.485 |
| noise=0.0,dropout=0.1,latency=0 | 1.0 | 0.96 | 0.98 | 0.485 |
| noise=0.0,dropout=0.3,latency=0 | 1.0 | 0.84 | 0.913 | 0.471 |
| noise=0.0,dropout=0.5,latency=0 | 1.0 | 0.56 | 0.718 | 0.496 |
| noise=0.005,dropout=0.1,latency=2 | 1.0 | 1.0 | 1.0 | 0.496 |

- nominal_corruption: {"noise_std": 0.005, "dropout": 0.1, "latency": 2}
- detection_delay_in_reads_by_latency: {"0": 0, "1": 1, "2": 2, "4": 4}
- latency_adds_exactly_its_value: True
- thresholds: {"nominal_F1": ">= 0.9", "nominal_localisation_error": "<= 2 mm", "clean_F1": ">= 0.99"}
- note: noise_std is in normalised taxel units (1.0 = max_depth 15 mm / max_speed 0.15 m/s), so 0.005 is 0.075 mm of depth noise. Dropout zeroes a whole fingertip for one read.
- note: Detection delay is counted in control steps (one read per 20 physics steps) after the probe appears.

## repeatability -- pass

- repeats: 5
- max_deviation_between_repeats: 0.0
- thresholds: {"max_deviation": "< 1e-9"}
- note: Identical press, repeated after a full reset: the physical readings are deterministic.

## pad_calibration -- pass


**penetration_mm_by_load**

| | 1 N | 2 N | 5 N | 10 N | 20 N |
|---|---|---|---|---|---|
| default (solref 0.06 0.9) | 0.846 | 1.708 | None | None | None |
| solref 0.04 1.0 | 0.641 | 1.033 | 3.055 | None | None |
| solref 0.02 1.0 | 0.395 | 0.605 | 1.23 | 3.11 | None |
| solref 0.01 1.0 | 0.251 | 0.432 | 0.757 | 1.573 | None |
| solref 0.005 1.0 | 0.181 | 0.332 | 0.615 | 1.049 | None |
| solref 0.01 1.0, solimp 0.95 0.99 0.001 0.5 2 | 0.191 | 0.343 | 0.601 | 0.906 | None |

- default_pad_cannot_carry_load_from: 5 N
- target: <= 2 mm at 10 N
- recommended_pad: solref 0.01 1.0
- default_pad_unchanged: True
- thresholds: {"penetration_increases_with_load": true, "a_candidate_meets_the_target": true}
- note: None means the pad could not carry the load: the steady contact force stayed below 90% of the applied load because the probe sank through the pad. The load is a 70 g flat probe pressed on the index pad with the hand clamped, so the pad is rigidly backed.
- note: MuJoCo's soft contacts scale with the effective mass of the pair, so these penetrations belong to a contact between the finger and a light object such as the cube; they are not a material property.
- note: ASSUMPTION: the target of 2 mm at 10 N is a typical figure for a silicone fingertip pad. It is not taken from the Sharpa Wave datasheet and should be confirmed.
- note: The recommendation is not applied: SceneConfig keeps Menagerie's pad so trained grasp and rotation policies stay valid. Set SceneConfig.pad_solref / pad_solimp to use a candidate.
