# BFMWD-144 train strict result — 2026-07-20

Absolute return and CAGR use the full declared calendar. Strict MDD uses the global/pre-entry HWM, costs, exact realized funding and every held five-minute favorable-then-adverse path.

| Variant | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | RW p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bfmwd_w12_d3_z10_h12 | 2.74% | 1.36% | 29.32% | 0.05 | 177 | 97/80 | 18.26bp | 0.6344 |
| bfmwd_w24_d3_z10_h12 | 17.40% | 8.36% | 17.21% | 0.49 | 147 | 83/64 | 37.83bp | 0.4106 |
| bfmwd_w12_d6_z10_h12 | 0.39% | 0.19% | 18.45% | 0.01 | 152 | 81/71 | 15.81bp | 0.6344 |
| bfmwd_w24_d6_z10_h12 | -5.07% | -2.57% | 20.23% | -0.13 | 131 | 68/63 | 7.70bp | 0.6344 |

- Stage passed: **False**
- Passing variants: `[]`
- Disposition: `REJECT_NO_REPAIR`

## Per-variant gates

### `bfmwd_w12_d3_z10_h12`

- passed: **False**
- failed gates: `['cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'each_contained_calendar_half_positive', 'fUSD_long_contribution_positive', 'mean_gross_side_adjusted_move_at_least_30bp', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'one_bar_delay_absolute_return_positive', 'romano_wolf_adjusted_p_at_most_10pct']`
| 10bp stress | -4.28% | -2.16% | 32.65% | -0.07 | 177 | 97/80 | 18.26bp | - |
| one-bar delay | -2.04% | -1.02% | 30.54% | -0.03 | 177 | 97/80 | 12.84bp | - |

### `bfmwd_w24_d3_z10_h12`

- passed: **False**
- failed gates: `['cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'each_contained_calendar_half_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'romano_wolf_adjusted_p_at_most_10pct']`
| 10bp stress | 10.70% | 5.22% | 19.98% | 0.26 | 147 | 83/64 | 37.83bp | - |
| one-bar delay | 13.26% | 6.43% | 18.45% | 0.35 | 147 | 83/64 | 32.81bp | - |

### `bfmwd_w12_d6_z10_h12`

- passed: **False**
- failed gates: `['cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'each_contained_calendar_half_positive', 'fBTC_short_contribution_positive', 'mean_gross_side_adjusted_move_at_least_30bp', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'romano_wolf_adjusted_p_at_most_10pct']`
| 10bp stress | -5.54% | -2.81% | 21.23% | -0.13 | 152 | 81/71 | 15.81bp | - |
| one-bar delay | 0.93% | 0.46% | 17.68% | 0.03 | 152 | 81/71 | 16.47bp | - |

### `bfmwd_w24_d6_z10_h12`

- passed: **False**
- failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'each_contained_calendar_half_positive', 'fUSD_long_contribution_positive', 'fBTC_short_contribution_positive', 'mean_gross_side_adjusted_move_at_least_30bp', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'one_bar_delay_absolute_return_positive', 'romano_wolf_adjusted_p_at_most_10pct']`
| 10bp stress | -9.92% | -5.09% | 22.79% | -0.22 | 131 | 68/63 | 7.70bp | - |
| one-bar delay | -4.37% | -2.21% | 19.87% | -0.11 | 131 | 68/63 | 8.76bp | - |

## Integrity

- evaluator SHA-256: `4ef73db88d45c903aa26bff7e8676f3d8eae393922b6bdf1b02938657984ebf0`
- report manifest: `c37909cff6604fa70c0745efcd5eda535e6f49f0593ec37c8a4648540c3f18a0`
- physical source window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- still sealed: `['selection']`
- post-2023 rows read: `0`
