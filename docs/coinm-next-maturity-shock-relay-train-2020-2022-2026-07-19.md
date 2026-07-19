# CMSR-36 train_2020_2022 result — 2026-07-19

All returns use the full declared calendar, exact funding, 6 bp per side at base cost, and intratrade strict MDD. Absolute return is always shown.

| Clock | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross bp | Mean net bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | -16.94% | -7.39% | 18.56% | -0.40 | 93 | 41/52 | -13.93 | -19.79 | 0.0001 |
| primary @ 10bp/side | -19.98% | -8.81% | 21.49% | -0.41 | 93 | 41/52 | -13.93 | -23.79 | 0.0000 |
| no_share_transition | -34.38% | -15.99% | 38.25% | -0.42 | 385 | 161/224 | -4.60 | -10.57 | 0.0036 |
| no_lead_shock | -26.46% | -11.94% | 27.44% | -0.44 | 171 | 72/99 | -11.99 | -17.83 | 0.0000 |
| front_led_mirror | -6.52% | -2.75% | 7.67% | -0.36 | 63 | 33/30 | -4.55 | -10.58 | 0.1022 |
| direction_flip | 7.36% | 2.98% | 3.18% | 0.94 | 93 | 52/41 | 13.93 | 7.79 | 0.0927 |
| extra_latency_1h | -18.97% | -8.33% | 20.53% | -0.41 | 93 | 41/52 | -16.65 | -22.42 | 0.0001 |
| deterministic_random_side | -10.31% | -4.40% | 11.05% | -0.40 | 93 | 54/39 | -5.54 | -11.54 | 0.0442 |

## Subperiods

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross bp | Mean net bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020_h2 | -1.23% | -2.90% | 1.96% | -1.48 | 19 | 8/11 | -0.43 | -6.47 | 0.0714 |
| 2021_h1 | -4.67% | -9.21% | 6.69% | -1.38 | 16 | 5/11 | -23.69 | -29.42 | 0.0793 |
| 2021_h2 | -4.23% | -8.22% | 5.06% | -1.62 | 21 | 9/12 | -14.90 | -20.45 | 0.0834 |
| 2022_h1 | -4.80% | -9.46% | 5.13% | -1.85 | 12 | 4/8 | -34.84 | -40.86 | 0.0037 |
| 2022_h2 | -3.25% | -6.34% | 5.91% | -1.07 | 25 | 15/10 | -7.10 | -13.08 | 0.2107 |

## Decision

- Qualified: **False**
- Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'each_subperiod_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'mechanism_control_margin_at_least_0_25']`
- Minimum mechanism-control margin: `-0.0399`
- Disposition: `REJECT_KEEP_2023_SEALED`

## Integrity

- Evaluator source SHA-256: `8638aad9eaa1e8a2d6a89fadf98d7a893e8e83907c58bc044ba0a7191d5ab95e`
- Freeze manifest: `1004cc198dde1f8c20010f4b8dae0242c863d23afdc0528920557eb6324f67c5`
- Report manifest: `19147ff20d5decef5297f91b4be633e193bb956f751a74b14b48730152e3c66a`
- Physical execution window: `['2020-08-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
