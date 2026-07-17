# FADC-21 stage1_2021_2022 evaluation — 2026-07-17

| Cost | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 6 bp/notional-side | -4.9902% | -2.6482% | 27.5473% | -0.0961 | 30 |
| 10 bp/notional-side | -7.2538% | -3.8712% | 27.7157% | -0.1397 | 30 |

Disposition: **REJECT_STAGE1_KEEP_2023_AND_2024_PLUS_SEALED**.

Gates: `{'combined_absolute_return_positive': False, 'each_calendar_year_absolute_return_positive': False, 'at_least_three_of_four_halves_positive': False, 'combined_cagr_to_strict_mdd_at_least_2': False, 'strict_mdd_at_most_15pct': False, 'stress_cost_absolute_return_positive': False, 'both_carry_gap_directions_positive': False, 'weekly_cluster_signflip_pvalue_at_most_10pct': False}`

The parser stopped before the first 2023 execution value. Therefore 2023 PnL
and all 2024+ outcomes remain sealed unless this stage passes. Absolute return
uses the full wall-clock period, including warm-up and idle time.

Manifest hash: `9c6043f579cb2e3678f53d543d821a7d520706df7a8cf7f3c817dcaecc591038`
