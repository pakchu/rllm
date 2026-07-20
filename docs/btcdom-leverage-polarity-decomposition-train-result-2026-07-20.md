# DLPD-12 train strict result — 2026-07-20

Absolute return and CAGR use the full declared calendar. Strict MDD uses the global/pre-entry HWM, exact funding, costs, and every held five-minute favorable-then-adverse path.

| Clock | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | -26.29% | -26.31% | 34.39% | -0.76 | 237 | 122/115 | -11.54bp | 0.1432 |
| primary 10bp stress | -32.96% | -32.98% | 36.74% | -0.90 | 237 | 122/115 | -11.54bp | 0.0529 |
| btc_only_tail | -51.55% | -51.58% | 58.92% | -0.88 | 566 | 270/296 | -11.84bp | 0.0083 |
| dom_only_mirror | -35.46% | -35.48% | 38.43% | -0.92 | 530 | 279/251 | -3.01bp | 0.1925 |
| same_sign | -23.51% | -23.52% | 36.56% | -0.64 | 240 | 100/140 | -8.46bp | 0.2602 |
| stale_btc_1h | -37.10% | -37.12% | 38.72% | -0.96 | 233 | 115/118 | -25.73bp | 0.0353 |
| stale_dom_1h | -30.20% | -30.22% | 34.53% | -0.87 | 242 | 121/121 | -15.50bp | 0.0873 |
| direction_flip | -2.11% | -2.11% | 28.34% | -0.07 | 237 | 115/122 | 11.54bp | 0.9982 |
| deterministic_random_side | 7.38% | 7.38% | 21.29% | 0.35 | 237 | 110/127 | 19.58bp | 0.8094 |
| extra_latency_1h | -23.95% | -23.97% | 36.65% | -0.65 | 237 | 122/115 | -9.04bp | 0.2174 |

- Stage passed: **False**
- Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'ten_bp_stress_absolute_return_positive', 'contained_subperiods_positive', 'weekly_cluster_signflip_p_at_most_10pct', 'direction_flip_inferior']`
- Disposition: `REJECT_NO_REPAIR`
- Component controls are diagnostic only and cannot repair primary.

## Contained subperiods

| Window | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022_h1 | -24.91% | -43.90% | 34.39% | -1.28 | 124 | 66/58 | -31.31bp | 0.1022 |
| 2022_h2 | -3.37% | -6.58% | 16.20% | -0.41 | 112 | 55/57 | 7.32bp | 0.7872 |

## Integrity

- evaluator SHA-256: `748fae0511cfe3f3eca48f43627bc1a6b728253c225ee1fc6e2aba14f390b17c`
- report manifest: `3a714ad877e6e2e613ffd7c8168d747b118bb10bf948a8d9d294acab5abb4693`
- physical source window: `['2022-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- still sealed: `['test', 'eval', 'final']`
