# Stable ensemble conditional-pullback OOS — 2026-07-15

**REJECTED_FROZEN_OOS**

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

| Window | 6 bp/side result | 10 bp/side stress | Gate pass |
|---|---:|---:|---:|
| test_2024 | 10.75% / 10.73% / 4.54% / 2.36 / 24 | 9.69% / 9.67% / 4.66% / 2.08 / 24 | False |
| eval_2025 | 3.39% / 3.39% / 7.41% / 0.46 / 18 | 2.65% / 2.65% / 7.45% / 0.36 / 18 | False |
| holdout_2026h1 | 11.73% / 30.55% / 5.63% / 5.43 / 24 | 10.66% / 27.57% / 5.78% / 4.77 / 24 | True |
| oos_2024_2025 | 14.51% / 7.00% / 7.41% / 0.94 / 42 | 12.60% / 6.11% / 7.45% / 0.82 / 42 | False |
| oos_all | 27.94% / 10.73% / 7.41% / 1.45 / 66 | 24.61% / 9.53% / 7.45% / 1.28 / 66 | False |

## Integrity

- Pinned manifest `ebf5e4602ac1cfd18d4c98a8955839f88df0ad358ded0d37ae911cf0c4aa20be` was validated before the future builder ran.
- Pre-2024 feature, activation, and all schedule hashes replayed exactly.
- Models were fit only on frozen 2020-07-01..2022-12-31 examples; 2023 was selection; 2024+ was not used for thresholds, exits, or model choice.
- Execution uses next-open entry, 6 bp/notional/side, realized funding, stop-before-take, split-contained exits, wall-clock CAGR, and strict path MDD.
- This is candidate-level implementation-clean OOS. It is not globally epistemically pristine because the broader repository previously researched related feature families on later periods.
