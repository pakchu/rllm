# CTHD-1 stage1_2021_2022 result — 2026-07-18

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -13.5192% | -7.0096% | 25.3489% | -0.2765 | 156 | -14.2696 | 0.6701 |
| 10 bp/notional/side | -18.7645% | -9.8757% | 26.2134% | -0.3767 | 156 | -14.2696 | 0.4875 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 | -11.9319% | -11.9395% | 25.3489% | -0.4710 | 123 |
| 2022 | -1.8024% | -1.8036% | 9.3345% | -0.1932 | 33 |

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
- Evaluator SHA-256: `7bdb67fc82b46cfbcca8bdd076b196cf84a9bca9662dd12223b8508939ec6fd5`
- Result manifest: `22b07be2336bc56e92ff36f96cf87cfd4695298e36fe94035304c166192a2b69`
- Full-calendar CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
