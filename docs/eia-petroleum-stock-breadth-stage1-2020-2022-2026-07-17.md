# EPSB-1 stage1_2020_2022 result — 2026-07-17

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | 29.2116% | 8.9161% | 12.0345% | 0.7409 | 37 | 157.3845 | 0.1405 |
| 10 bp/notional/side | 27.3219% | 8.3826% | 12.2023% | 0.6870 | 37 | 157.3845 | 0.1629 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2020 | -5.7377% | -5.7263% | 12.0345% | -0.4758 | 12 |
| 2021 | 18.9572% | 18.9713% | 6.1708% | 3.0744 | 10 |
| 2022 | 14.9302% | 14.9411% | 4.9278% | 3.0320 | 14 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | PASS |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | PASS |
| `weekly_cluster_signflip_p_at_most_10pct` | FAIL |
| `minimum_30_trades` | PASS |
| `mean_gross_underlying_at_least_35bp` | PASS |
| `stress_cost_absolute_return_positive` | PASS |
| `each_calendar_year_absolute_return_positive` | FAIL |
| `each_calendar_year_minimum_8_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | PASS |

## Integrity

- Physically opened window: `['2020-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `bad1b5b33f185b241e611d55ff4c40ac91de21b8d848295b02be94990883d5ba`
- Result manifest: `9e0e0e5eabe6eeeddbd46ff88bbf45453f08d76ca6488192d8f309f84c4b9869`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
