# DTAC-8 train strict result — 2026-07-19

Absolute return and CAGR use the full declared calendar. Strict MDD uses the global/pre-entry HWM, costs, exact funding boundaries and every held five-minute favorable-then-adverse path.

| Clock | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | -18.33% | -18.35% | 19.71% | -0.93 | 143 | 84/59 | -15.76bp | 0.0072 |
| primary 10bp stress | -22.89% | -22.90% | 24.11% | -0.95 | 143 | 84/59 | -15.76bp | 0.0006 |
| direction_flip | 2.48% | 2.48% | 8.66% | 0.29 | 143 | 59/84 | 15.76bp | 0.7286 |
| all_six_premium_side | -21.79% | -21.80% | 23.11% | -0.94 | 143 | 81/62 | -21.81bp | 0.0017 |
| all_six_flow_fade_side | -19.13% | -19.15% | 20.50% | -0.93 | 143 | 85/58 | -17.12bp | 0.0055 |
| deterministic_random_side | -15.92% | -15.93% | 19.35% | -0.82 | 143 | 72/71 | -11.83bp | 0.0409 |
| symbol_permuted_premium_pairing | -18.08% | -18.09% | 21.59% | -0.84 | 170 | 102/68 | -10.88bp | 0.0199 |
| stale_premium_pairing_24h | -7.59% | -7.60% | 14.74% | -0.52 | 167 | 99/68 | 3.00bp | 0.2692 |
| extra_latency_1h | -16.55% | -16.56% | 18.84% | -0.88 | 143 | 84/59 | -12.80bp | 0.0034 |

- Stage passed: **False**
- Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'mean_gross_underlying_at_least_20bp', 'each_contained_half_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5', 'mechanism_control_margin_at_least_0_25']`
- Disposition: `REJECT_NO_REPAIR`

## Contained halves

| Window | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023_h1 | -12.47% | -23.57% | 15.41% | -1.53 | 71 | 41/30 | -24.82bp | 0.0865 |
| 2023_h2 | -6.70% | -12.86% | 8.40% | -1.53 | 72 | 43/29 | -6.83bp | 0.0083 |

## Integrity

- evaluator SHA-256: `f2d87eb64f40c4c0e55cf1f670193a803b3268172eab42dc33781253dac4d0c1`
- report manifest: `2b56eaebadbd206733986bc419d5ea514a02815744c98997f93f701ef639c30f`
- physical source window: `['2023-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00']`
- still sealed: `['test', 'eval', 'final']`
