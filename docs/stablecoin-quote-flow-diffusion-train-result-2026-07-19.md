# SQFD-6 train strict result — 2026-07-19

Absolute return uses the full declared calendar. Strict MDD includes the global/pre-entry HWM, entry and virtual/actual exit costs, conservative funding boundaries and every held 5m favorable-then-adverse path.

| Clock | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | -1.72% | -3.39% | 8.59% | -0.39 | 55 | 32/23 | 6.17bp | 0.7002 |
| primary 10bp stress | -3.86% | -7.52% | 9.49% | -0.79 | 55 | 32/23 | 6.17bp | 0.3329 |
| no_alt_breadth | -8.87% | -16.83% | 12.10% | -1.39 | 163 | 95/68 | 0.94bp | 0.1210 |
| no_usdt_lag | -7.20% | -13.79% | 11.03% | -1.25 | 114 | 64/50 | -0.74bp | 0.0353 |
| no_participation | -4.93% | -9.55% | 6.36% | -1.50 | 89 | 44/45 | 0.99bp | 0.3384 |
| usdt_only | -31.29% | -52.53% | 31.84% | -1.65 | 441 | 236/205 | -4.69bp | 0.0015 |
| direction_flip | -4.84% | -9.37% | 10.44% | -0.90 | 55 | 23/32 | -6.17bp | 0.3197 |
| extra_latency_1h | -2.86% | -5.60% | 8.48% | -0.66 | 55 | 32/23 | 1.74bp | 0.3332 |
| deterministic_random_side | -0.52% | -1.02% | 7.25% | -0.14 | 55 | 23/32 | 9.90bp | 0.8637 |

- Stage passed: **False**
- Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'weekly_cluster_signflip_p_at_most_10pct', 'mean_gross_underlying_at_least_20bp', 'each_contained_half_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'mechanism_control_margin_at_least_0_25']`
- Disposition: `REJECT_NO_REPAIR`

## Contained halves

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023_q3 | -1.94% | -7.48% | 2.33% | -3.21 | 15 | 9/6 | -13.95bp | 0.0625 |
| 2023_q4 | 0.22% | 0.90% | 6.98% | 0.13 | 40 | 23/17 | 13.71bp | 0.9551 |

## Integrity

- evaluator SHA-256: `0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b`
- report manifest: `9333bffca33062a478e7748e8eae2d25142026a78d7e8c24b127dc4cd73465cc`
- physical source window: `['2023-07-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00']`
- still sealed: `['test', 'eval', 'final']`
