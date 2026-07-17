# CITA-1 stage1_2020_2022 result — 2026-07-17

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | 32.3633% | 9.7943% | 45.2891% | 0.2163 | 98 | 96.2739 | 0.4291 |
| 10 bp/notional/side | 27.2422% | 8.3600% | 46.2153% | 0.1809 | 98 | 96.2739 | 0.4755 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2020 | 6.4436% | 6.4300% | 35.8932% | 0.1791 | 34 |
| 2021 | 8.5856% | 8.5917% | 39.1390% | 0.2195 | 36 |
| 2022 | 14.5185% | 14.5291% | 23.1263% | 0.6283 | 28 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | PASS |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | FAIL |
| `weekly_cluster_signflip_p_at_most_10pct` | FAIL |
| `minimum_75_trades` | PASS |
| `mean_gross_underlying_at_least_35bp` | PASS |
| `stress_cost_absolute_return_positive` | PASS |
| `each_calendar_year_absolute_return_positive` | PASS |
| `each_calendar_year_minimum_20_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |

## Integrity

- Physically opened window: `['2020-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `39cabad32e5374b2f918d640c78d00c30a204959366db28ce5bd519c5f6bcff3`
- Result manifest: `10cbd385ef3118f70a43f2cf2488624a5dce578834954264389251c4abcdc65b`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
