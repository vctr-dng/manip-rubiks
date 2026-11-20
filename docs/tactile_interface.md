# Tactile sensor interface

This documents the simulated tactile sensors that the learning code consumes (draft sections 8–10, milestone M1).
Every number below comes from the gate M1 run in [`reports/m1/m1_report.md`](../reports/m1/m1_report.md)
(`uv run manip-rubiks-m1`) or from the tests named in the text.

## Configurations

`TactileConfig.mode` selects what `TactileSensors.read()` returns. `coverage="fingertips"` (default) gives
five regions in the order `thumb, index, middle, ring, pinky`. `coverage="hand"` (22 regions, A and B only)
adds the palm and every phalanx.

| Mode | Draft | Shape (fingertips) | Values | Source |
|---|---|---|---|---|
| `contact` | A | `(5,)` | 1.0 if the net contact force on the region exceeds `contact_threshold` (0.05 N) | MuJoCo contact list |
| `force` (default) | B | `(5, 3)` | net contact force on the region in its local frame [N], clipped to ±`max_force` (50 N) | MuJoCo contact list |
| `taxel` | C | `(5, 8, 8, 3)` = 960 | per taxel: depth, slip along, slip across; normalised to `[0, 1]` | native `<sensor><tactile>` |

**Frames (A, B).** The force is the force *applied to* the region (so a pad pressed from outside reads
negative `x`). A fingertip's local frame is its `*_fingertip` site: `x` is the outward palmar surface normal,
`y` runs across the pad, `z` points to the fingertip. Other regions use the body frame.

**Contacts with anything count.** A and B are computed from every contact that touches the region's geoms,
including other fingers, the palm and the floor, as a skin would.

## Configuration C: taxels

Each fingertip has a non-colliding *skin mesh* that follows the pad capsule on its palmar side, with an
8 × 8 grid of vertices (`simulation/taxels.py`). Row `i` runs along the finger (base to tip), column `j`
runs across the pad over ±60° of the capsule. The pad capsule remains the collider, so the dynamics do not
change when taxels are on. Taxel pitch is 1.5 mm × 2.0 mm for the index, middle, ring and pinky and
1.9 mm × 2.3 mm for the thumb.

Channels, last axis of the `(5, 8, 8, 3)` array:

| Index | Name | Physical quantity | Normalised by (`TactileConfig`) |
|---|---|---|---|
| 0 | `depth` | how far the vertex lies inside any other geom touching the fingertip body [m]; since the vertices sit on the unloaded pad surface, this is the indentation | `max_depth` = 15 mm |
| 1 | `slip_along` | relative speed between the vertex and the touching body, along the finger [m/s] | `max_speed` = 0.15 m/s |
| 2 | `slip_across` | the same, across the pad [m/s] | `max_speed` |

Values are clipped to `[0, 1]`. The defaults come from scripted grasps, where the soft default pads indent 10–15 mm
(99th percentile 14.8 mm) and slip speeds reach 0.14 m/s (99th percentile).

**Limits you need to know.**

- **Slip is unsigned.** Both slip channels are magnitudes. T2 shows that a slide at +v and one at −v read
  identically (difference ≤ 0.13 % of v). The direction of slip is only recoverable from configuration B's force
  or from how the contact moves over time.
- **Only the palmar face is covered** (±60° around the pad, along the cylindrical part of the capsule). A
  contact on the back or at the very end of the fingertip reads zero in C, while A and B still see it.
- **Depth also responds to other hand geoms.** Any geom that touches the fingertip body counts, for example
  a neighbouring finger pressed into the pad.
- **Contacts with a mesh geom without an octree are ignored** by MuJoCo's sensor. The cube cubies are boxes, so
  this does not affect them.

**Enabling it.**

```python
from manip_rubiks.environments.hand_cube_env import EnvConfig, HandCubeEnv
from manip_rubiks.simulation.tactile_sensors import TactileConfig

env = HandCubeEnv(EnvConfig(tactile=TactileConfig(mode="taxel")))  # turns on SceneConfig.tactile_taxels
```

`HandCubeScene(SceneConfig(tactile_taxels=True))` adds the skins and sensors without an environment.
`TactileSensors(..., TactileConfig(mode="taxel"))` raises a `ValueError` if the scene lacks them. The GPU
environments (`HandCubeEnvGPU` and subclasses) read `d.sensordata` directly, and their output matches the CPU
reference to ~1e-5 in a loaded grasp (`tests/test_gpu_tactile_taxels.py`).

In the observation dictionary the key is still `tactile`; flattened actor observations contain the 960 values
in C order (tip, row, column, channel).

## Corruption model

`noise_std`, `noise_rel`, `dropout` and `latency` apply to B and C (dropout also to A), with a seeded generator
(`TactileSensors.reset(seed)`).

| Parameter | Meaning | Units for C |
|---|---|---|
| `noise_std` | additive Gaussian noise | normalised units (0.005 is 0.075 mm of depth) |
| `noise_rel` | multiplicative noise | fraction of the reading |
| `dropout` | probability that a whole fingertip reads zero for one read | per read |
| `latency` | delay in `read()` calls (control steps) | exact: detection delay grows by exactly `latency` |

## What each configuration can observe

Measured on the rig (`validation/tactile_rig.py`): a flat probe driven kinematically in the pad frame.

| Question (draft section 10) | A | B | C |
|---|---|---|---|
| T1 contact present, graded with indentation | one bit, 0.05 N threshold | yes | yes, from the lightest touch (0.2 mm) |
| T1 right finger, no cross-talk | yes | yes | yes (exactly zero on the other four) |
| T2 sliding | no (same bit static and sliding) | tangential force grows with speed and flips sign with direction | speed magnitude along and across, relative error ≤ 1.5 % for 0.01–0.1 m/s, cross-channel leakage ≤ 1.4 % |
| T3 contact location | none | across the pad nearly recovered by a linear fit of the force (R² ≈ 0.99), along the pad only partly (0.08–0.62) | mean centroid error 0.17–0.40 mm, correlation ≥ 0.988 |
| T4 two contacts | cannot tell | cannot tell | two blobs from 6 mm centre separation (3 mm probes); not at 4 mm |
| T5 robustness | – | – | with a 5-read median: F1 ≥ 0.98 up to noise 0.01 (0.15 mm) or dropout 0.1, F1 = 0.73 at noise 0.02 and 0.72 at dropout 0.5 |

B's location information comes from the curved pad and is noiseless here; it is not a reliable location sensor on a
real contact patch.

## Pad compliance

Menagerie's elastomer pad (`solref 0.06 0.9`) is the default and is unchanged, so earlier grasp and rotation
results remain valid. `SceneConfig.pad_solref` and `pad_solimp` override it for experiments.

Measured pad penetration under a known normal load (70 g flat probe, hand clamped, `pad_calibration` check):

| Pad | 1 N | 2 N | 5 N | 10 N | 20 N |
|---|---|---|---|---|---|
| default `0.06 0.9` | 0.85 mm | 1.71 mm | cannot carry | cannot carry | cannot carry |
| `0.04 1.0` | 0.64 | 1.03 | 3.06 | cannot carry | cannot carry |
| `0.02 1.0` | 0.40 | 0.61 | 1.23 | 3.11 | cannot carry |
| `0.01 1.0` | 0.25 | 0.43 | 0.76 | 1.57 | cannot carry |
| `0.005 1.0` | 0.18 | 0.33 | 0.62 | 1.05 | cannot carry |
| `0.01 1.0`, solimp `0.95 0.99 0.001 0.5 2` | 0.19 | 0.34 | 0.60 | 0.91 | cannot carry |

"Cannot carry" means the steady contact force stayed under 90 % of the load because the probe sank through the pad.
The default pad therefore cannot hold 5 N against a light object, which matches the 10–17 mm taxel depths in
scripted grasps. MuJoCo's soft contact scales with the effective mass of the colliding pair, so these figures
describe a contact with a light object such as the cube and are not a material property.

The gate recommends `solref 0.01 1.0` as the softest candidate that stays within 2 mm at 10 N. **The 2 mm at 10 N
target is an assumption**, not a Sharpa Wave specification, and the recommendation is not applied anywhere.
Adopting it would change grasp and rotation behaviour and means retraining.

## Two MuJoCo pitfalls this design avoids

- The builtin `plate` mesh cannot be used for the skin. MuJoCo takes the slip channels from the second and third
  vertex normals, and the plate stores the surface normal third, so its second slip channel measured motion
  along the surface normal instead of along the finger. The skin is a custom mesh whose normals are
  (surface normal, along tangent, across tangent).
- Slip is measured against the other body's velocity. A mocap body has none, so sliding experiments read zero;
  the rig uses real free bodies whose pose and velocity it sets every step.
