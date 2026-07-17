# ORFR-1 stage1_2021_2022 result — 2026-07-17

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | 57.8571% | 25.6609% | 17.9620% | 1.4286 | 111 | 102.6723 | 0.0175 |
| 10 bp/notional/side | 51.0072% | 22.9023% | 18.9961% | 1.2056 | 111 | 102.6723 | 0.0308 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 | 72.6922% | 72.7569% | 15.1162% | 4.8132 | 58 |
| 2022 | -8.9438% | -8.9497% | 15.2350% | -0.5874 | 52 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | PASS |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | FAIL |
| `weekly_cluster_signflip_p_at_most_10pct` | PASS |
| `minimum_trades` | PASS |
| `mean_gross_underlying_at_least_35bp` | PASS |
| `stress_cost_absolute_return_positive` | PASS |
| `each_subperiod_absolute_return_positive` | FAIL |
| `each_subperiod_minimum_trades` | PASS |
| `minimum_each_side_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |
| `falsification_controls_do_not_fully_qualify` | PASS |

## Integrity

- Physically opened window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `8bd60256d065da1750c9852b7c7b47375ad1cd65842b4b8e256cc43b470a8567`
- Result manifest: `db7e3333913a0f2d1eb2c38fdca7144121b957ad980c25479c7267b8d3fce939`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
