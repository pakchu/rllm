# MFIC frozen selection result — 2026-07-14

## Verdict

**MFIC v1 is rejected.** Neither frozen candidate passed the preregistered train/selection gates. Per protocol, no threshold, branch, horizon, or hold repair is allowed, and the 2024/2025/2026 outcomes remain sealed.

- preregistration commit: `81ea71c`
- evaluator commit: `f34fb10`
- result: `results/metaorder_fragmentation_impact_curvature_selection_2026-07-14.json`
- result SHA256: `2e33ac7e76c8212dcd0b3f919c6bb912647251cfcfb71264fe7702175af47de3`
- execution: next 5-minute open, scheduled-open exit, `0.5x`, `6 bp` per side
- CAGR clock: complete split including idle cash
- MDD: complete held path with favorable extreme first and adverse extreme second

## `mfic_fast`

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly-cluster p |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | -60.67% | -26.73% | 60.67% | -0.44 | 1,348 | 1.0000 |
| select 2023 | -13.51% | -13.52% | 13.62% | -0.99 | 218 | 1.0000 |
| 2023 H1 | -9.21% | -17.71% | 9.42% | -1.88 | 139 | 1.0000 |
| 2023 H2 | -4.74% | -9.20% | 4.87% | -1.89 | 79 | 0.9997 |

Net mean trade return was `-0.0690%` in train and `-0.0665%` in full 2023.

## `mfic_slow`

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly-cluster p |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | -53.33% | -22.43% | 54.91% | -0.41 | 1,392 | 1.0000 |
| select 2023 | -12.66% | -12.67% | 13.17% | -0.96 | 243 | 1.0000 |
| 2023 H1 | -7.46% | -14.48% | 7.92% | -1.83 | 164 | 0.9986 |
| 2023 H2 | -5.62% | -10.85% | 5.83% | -1.86 | 79 | 1.0000 |

Net mean trade return was `-0.0543%` in train and `-0.0555%` in full 2023.

## Gate failures

Both candidates failed all economically important gates:

- negative absolute return in train and full 2023;
- CAGR/strict-MDD below `3.0` in both windows;
- train strict MDD above `15%`;
- one-sided weekly-cluster p-value not below `0.10`;
- negative absolute return in both 2023 halves.

The samples were large enough that lack of trade count is not the explanation. The frozen mechanism selected approximately 1,350–1,400 train trades and 218–243 2023 trades, yet the mean net trade return was consistently negative.

## Consequence

MFIC v1 cannot be promoted, inverted, branch-pruned, or retuned using these outcomes. Any investigation of gross-versus-cost drag or continuation-versus-fade behavior is post-hoc failure analysis only. A successor must be a separately named and separately preregistered hypothesis; it cannot claim the MFIC v1 OOS status.
