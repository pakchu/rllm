# FCIR-12 train strict result — 2026-07-19

Absolute return and CAGR use the full declared calendar. Strict MDD uses the global/pre-entry HWM, costs, exact funding boundaries and every held five-minute favorable-then-adverse path.

| Clock | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | -2.42% | -2.42% | 6.07% | -0.40 | 62 | 26/36 | 4.16bp | 0.6849 |
| primary 10bp stress | -4.81% | -4.81% | 6.78% | -0.71 | 62 | 26/36 | 4.16bp | 0.4052 |
| direction_flip | -5.15% | -5.15% | 9.87% | -0.52 | 62 | 36/26 | -4.16bp | 0.3354 |
| equal_weight_side | -5.11% | -5.12% | 7.28% | -0.70 | 62 | 27/35 | -4.84bp | 0.3584 |
| stale_network_24h | -2.42% | -2.42% | 6.07% | -0.40 | 62 | 26/36 | 4.16bp | 0.6849 |
| symbol_permuted_network | -9.02% | -9.02% | 12.28% | -0.73 | 62 | 30/32 | -18.16bp | 0.0525 |
| deterministic_random_side | -3.18% | -3.18% | 8.96% | -0.35 | 62 | 31/31 | 2.23bp | 0.5410 |
| extra_latency_1h | -2.35% | -2.35% | 5.49% | -0.43 | 62 | 26/36 | 4.41bp | 0.6650 |

- Stage passed: **False**
- Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'weekly_cluster_signflip_p_at_most_10pct', 'mean_gross_underlying_at_least_20bp', 'each_contained_half_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'mechanism_control_margin_at_least_0_25']`
- Disposition: `REJECT_NO_REPAIR`

## Contained halves

| Window | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023_h1 | -0.34% | -0.69% | 6.07% | -0.11 | 36 | 14/22 | 10.53bp | 0.9602 |
| 2023_h2 | -2.08% | -4.09% | 4.38% | -0.93 | 26 | 12/14 | -4.67bp | 0.5078 |

## Integrity

- evaluator SHA-256: `036b22442a2080e7ea5ffe914c605a9b1b1a55b128a315a2f2f05be7b37a736d`
- report manifest: `a7e8d1bcf4acf5ae352f752c44f961359b1932543ee87e15741a48ea25898a33`
- physical source window: `['2023-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00']`
- still sealed: `['test', 'eval', 'final']`
