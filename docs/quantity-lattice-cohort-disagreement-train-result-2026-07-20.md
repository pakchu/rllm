# QLCD-288 train economic result — 2026-07-20

## Verdict

**REJECT_NO_REPAIR.** The frozen phase-one evaluator was applied without parameter repair.

## Strict metrics

| Slice | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | L/S | Mean gross | Weekly nominal p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base 6bp/side | -47.24% | -19.19% | 48.10% | -0.40 | 377 | 180/197 | -19.52bp | 0.0841 |
| stress 10bp/side | -54.64% | -23.16% | 55.35% | -0.42 | 377 | 180/197 | -19.52bp | 0.0287 |
| 2020 | -20.50% | -20.46% | 31.43% | -0.65 | 112 | 46/66 | -25.90bp | 0.2341 |
| 2021 | -17.96% | -17.97% | 34.05% | -0.53 | 125 | 57/68 | -16.73bp | 0.5470 |
| 2022 | -17.75% | -17.76% | 26.40% | -0.67 | 138 | 76/62 | -14.84bp | 0.1308 |

## Mandatory report-only falsification controls

| exact_side_flip | 8.30% | 2.69% | 30.81% | 0.09 | 377 | 197/180 | 19.52bp | 0.6976 |
| medium_vs_fine | -39.60% | -15.47% | 53.35% | -0.29 | 421 | 197/224 | -8.67bp | 0.2340 |
| remove_opposition | -49.40% | -20.31% | 52.03% | -0.39 | 392 | 191/201 | -20.37bp | 0.0535 |
| all_quantity_imbalance | -42.71% | -16.94% | 55.57% | -0.30 | 386 | 199/187 | -15.00bp | 0.1076 |
| stale_one_hour | -51.43% | -21.39% | 52.59% | -0.41 | 377 | 180/197 | -24.08bp | 0.0325 |
| stale_twenty_four_hours | -38.09% | -14.77% | 50.25% | -0.29 | 377 | 180/197 | -11.12bp | 0.2045 |

These controls were frozen before outcomes and are always reported. The preregistration set no control-margin gate, so they are diagnostic rather than promotion gates; none may repair a failed primary.
The weekly sign-flip value is a frozen nominal clustered randomization diagnostic used only as one preregistered gate; it is not presented as a standalone discovery p-value or multiple-search-adjusted inference.

## Frozen gates

- `base_absolute_return_positive`: **fail**
- `base_cagr_to_strict_mdd_at_least_3`: **fail**
- `strict_mdd_at_most_15pct`: **fail**
- `stress_absolute_return_positive`: **fail**
- `stress_cagr_to_strict_mdd_at_least_2_5`: **fail**
- `mean_gross_underlying_at_least_24bp`: **fail**
- `weekly_cluster_signflip_p_strictly_below_10pct`: **pass**
- `each_train_year_absolute_return_positive`: **fail**

## Accounting and boundary

- Absolute return and CAGR include the full declared calendar, including idle cash.
- Strict MDD keeps the global pre-entry high-water mark and marks every held bar favorable then adverse after costs and funding.
- Entry/exit use the frozen five-minute opens; exposure is 0.5x and hold is exactly 288 bars.
- Exact entry/exit funding credits are dropped while debits are retained.
- Test, eval, and recent-report sources remain sealed unless both phase-one stages pass and a phase-two evaluator is committed first.
- No threshold, direction, delay, hold, cost, split, or gate may repair a failed result.

## Artifact binding

- evaluator freeze manifest: `9ea2049f8c4ae02350b241a716284cc01463a5658c2469d0cab2adef2e12f992`
- evaluator source SHA-256: `1ed4ffa7aca2bbe3c84dc7dca05c537ab020a11ee5bde2d4219770721d755f2d`
- stage manifest: `b3f28024c0eb1b6b6229e834d4fcbe688994a0eedaa56363df3020838cf42b84`
