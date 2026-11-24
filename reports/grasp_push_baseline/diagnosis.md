# Why pushed grasps fail: torque, during the push

Re-baseline of the two 6M PPO reference runs (`runs/grasp_gpu/ppo_ref6m_s{0,1}`) on the CPU reference,
300 held-out episodes per cell (seeds 100000..100299), same seeds in every condition. Reproduce with:

    uv run manip-rubiks-eval-grasp --policy ppo_ref6m_s0=runs/grasp_gpu/ppo_ref6m_s0 \
        --policy ppo_ref6m_s1=runs/grasp_gpu/ppo_ref6m_s1 --condition reorient_push \
        --condition reorient_push_0.5x --condition reorient_push_1.5x --condition reorient_push_force \
        --condition reorient_push_torque --episodes 300 --out reports/grasp_push_baseline

Full table: [`grasp_report.md`](grasp_report.md). Success, layer-turn rate:

| Condition | `ppo_ref6m_s0` | `ppo_ref6m_s1` |
|---|---|---|
| `reorient_push` (1x force and torque) | 73 %, 20 % | 80 %, 16 % |
| `reorient_push_0.5x` | 86 %, 11 % | 86 %, 12 % |
| `reorient_push_1.5x` | 72 %, 22 % | 75 %, 21 % |
| `reorient_push_force` (force only) | **94 %, 0 %** | **94 %, 0 %** |
| `reorient_push_torque` (torque only) | 76 %, 23 % | 83 %, 16 % |

What the table says:

- **The torque of the push turns layers; the force does not.** With force only, no layer turns in 599 of
  600 episodes and success is 94 % (the rest are drops, 4-5 %). Torque alone reproduces the full failure.
- **It happens during the push.** Of the 110 layer turns under `reorient_push`, 81 were still being pushed
  at that step and 95 were within 5 control steps (0.2 s) of a push. The median torque of the last push was
  13.7 mN m (limit 20 mN m). The wrist was moving in 61 of the 110.
- Failure grows with push strength (0.5x: 86 %, 1x: 73-80 %, 1.5x: 72-75 %), and the misalignment p95 is
  about 45 degrees in every pushed condition: either aligned, or a layer has turned.

Caveats:

- The pushes act on the cube's core body (`xfrc_applied`), not on its surface. A torque there is carried to
  the cubies through the ball joints, which may shear layers more easily than a contact disturbance would.
  Whether 20 mN m of torque is the right target is a modelling choice to confirm before more compute is spent
  chasing it.
- These two runs trained for 6M steps with `--curriculum-fraction 1.6`, so difficulty only reached 0.7 by the
  end and they never trained at full push strength. The 86 % `gpu_s0` run quoted in the README is not
  available locally (no checkpoint in `runs/` or MLflow), so it could not be re-measured.
- Per-seed spread is large (73 % vs 80 % at 1x), so a single seed cannot show a 5-point gain.
