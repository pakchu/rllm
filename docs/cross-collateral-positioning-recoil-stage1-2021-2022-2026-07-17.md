# CCPR-1 stage1_2021_2022 result — 2026-07-17

Metrics use full-calendar time, realized funding, next-open fills, and intratrade strict MDD. Absolute return is always shown with CAGR.

| Candidate | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| CCPR-H4 | -0.65% | -0.44% | 20.64% | -0.02 | 96 | 0.9390 |
| CCPR-H8 | -2.97% | -2.01% | 21.39% | -0.09 | 83 | 0.7256 |

- Stage1 pass: **False**
- Selected: `None`
- Disposition: `REJECT_KEEP_2023_SEALED`
- 2023 execution outcomes remain sealed unless Stage1 passed.

## CCPR-H4 subperiods

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| 2021_partial | -3.77% | -7.63% | 20.64% | -0.37 | 32 | 0.4732 |
| 2022_h1 | 0.44% | 0.90% | 1.82% | 0.49 | 11 | 0.7490 |
| 2022_h2 | 2.79% | 5.62% | 2.72% | 2.06 | 53 | 0.2968 |

Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'weekly_cluster_signflip_p_within_limit', 'each_subperiod_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'mechanism_control_margin_at_least_0_25']`

## CCPR-H8 subperiods

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| 2021_partial | -1.16% | -2.37% | 21.39% | -0.11 | 29 | 0.8055 |
| 2022_h1 | 1.59% | 3.24% | 2.06% | 1.57 | 10 | 0.4983 |
| 2022_h2 | -3.38% | -6.59% | 8.09% | -0.81 | 44 | 0.6194 |

Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'weekly_cluster_signflip_p_within_limit', 'each_subperiod_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'mechanism_control_margin_at_least_0_25']`

## Integrity

- Evaluator source SHA-256: `ab918bae12237056b413c506a0bac8508efcb75dbd07a84a46cd6755f11b4132`
- Report manifest: `95f7d74048b8ed8e5199ad3cd6456e58ddf569dee6530673e0584ba9bd9504c0`
- Physical execution window: `['2021-07-08T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
