# IBRD-7 stage1_2020_2022 result — 2026-07-17

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | 32.0203% | 9.6994% | 12.8864% | 0.7527 | 20 | 296.8853 | 0.1816 |
| 10 bp/notional/side | 30.9947% | 9.4147% | 13.0449% | 0.7217 | 20 | 296.8853 | 0.1951 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2020 | 6.0450% | 6.0323% | 8.2239% | 0.7335 | 6 |
| 2021 | 32.2185% | 32.2438% | 9.2418% | 3.4889 | 7 |
| 2022 | -5.8418% | -5.8457% | 12.8864% | -0.4536 | 7 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | PASS |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | PASS |
| `weekly_cluster_signflip_p_at_most_10pct` | FAIL |
| `minimum_20_trades` | PASS |
| `mean_gross_underlying_at_least_35bp` | PASS |
| `stress_cost_absolute_return_positive` | PASS |
| `each_calendar_year_absolute_return_positive` | FAIL |
| `each_calendar_year_minimum_4_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |

## Integrity

- Physically opened window: `['2020-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `e8c1b7aae889c2c58df65383867be5d1f09d171d68a9322c456ccb99b39ed879`
- Result manifest: `3dd8b2fa6c5acdcf97deca0e0d54b369d04cef0b4285416e81e4d0241f0d9e06`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
