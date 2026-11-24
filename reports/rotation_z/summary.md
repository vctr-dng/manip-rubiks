# Composed z from the retrained x/y teachers

Three seeds of `manip-rubiks-train-rotate-gpu --axes xy` (60M steps each, `runs/rotation_gpu/xy60m_s{0,1,2}`),
benchmarked on the CPU reference, 100 episodes per cell, held-out seeds from 100000. A z command is realised as
three x/y sub-goals (`compose_command`); the four `--z-variant` routes differ only in which x/y turns they use.
Reports per seed and variant: `s<seed>_v<variant>/rotation_report.md`. Success is per command, so a z command
counts only when all three sub-goals are reached; `chain_composed` is 12 random xyz commands per episode.

Plain quarter-turns (variant 0 runs): +-x and +-y are 99-100 % per goal on all three seeds.

Command success, `turn_+z_composed` / `turn_-z_composed` / `chain_composed`:

| Seed | Variant 0 | Variant 1 | Variant 2 | Variant 3 |
|---|---|---|---|---|
| `xy60m_s0` | 98 / 99 / 92 % | 100 / 100 / 94 % | 98 / **88** / 96 % | 98 / 99 / 95 % |
| `xy60m_s1` | 100 / 100 / 100 % | 99 / 99 / 98 % | 100 / 98 / 99 % | 100 / 100 / 99 % |
| `xy60m_s2` | 100 / 100 / 100 % | 100 / 100 / 100 % | 100 / 100 / 99 % | 99 / 100 / 99 % |

Gate (+-z >= 95 % per command, chain >= 90 % per command): variant 1 passes on all three seeds, and so do
variants 0 and 3. Variant 2 fails once (`xy60m_s0`, -z: 88 %), so the route matters for a weaker x/y policy.
Chain episodes of `xy60m_s0` time out more often (16-34 %), meaning slower, not wrong.
