# CIHM-1 stage1_2021_2022 result — 2026-07-18

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -2.3058% | -1.1604% | 35.3130% | -0.0329 | 151 | 8.3366 | 0.9622 |
| 10 bp/notional/side | -8.0354% | -4.1046% | 36.7039% | -0.1118 | 151 | 8.3366 | 0.8487 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 | -24.2735% | -24.2879% | 30.6416% | -0.7926 | 74 |
| 2022 | 29.0092% | 29.0317% | 9.8206% | 2.9562 | 77 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | FAIL |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | FAIL |
| `weekly_cluster_signflip_p_at_most_10pct` | FAIL |
| `minimum_trades` | PASS |
| `mean_gross_underlying_at_least_35bp` | FAIL |
| `stress_cost_absolute_return_positive` | FAIL |
| `each_subperiod_absolute_return_positive` | FAIL |
| `each_subperiod_minimum_trades` | PASS |
| `minimum_short_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |
| `falsification_controls_do_not_performance_qualify` | PASS |

## Integrity

- Physically opened window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `b02b68acf1f2a57e9a55a57e76380e3984c68d49f1b872de7e3608058235e9e5`
- Result manifest: `e39d43f7d485a1f55fa45699c28a99137a99bac7657abfce7e92fceb4e6a66cf`
- Full-calendar CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
