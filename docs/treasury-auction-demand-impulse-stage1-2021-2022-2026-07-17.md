# TADI-1 stage1_2021_2022 result — 2026-07-17

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -4.2508% | -2.1500% | 13.0136% | -0.1652 | 28 | -12.6028 | 0.7541 |
| 10 bp/notional/side | -5.3187% | -2.6975% | 13.8666% | -0.1945 | 28 | -12.6028 | 0.6842 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 | 0.0122% | 0.0122% | 8.5115% | 0.0014 | 13 |
| 2022 | -4.2625% | -4.2653% | 10.1369% | -0.4208 | 15 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | FAIL |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | PASS |
| `weekly_cluster_signflip_p_at_most_10pct` | FAIL |
| `minimum_trades` | PASS |
| `mean_gross_underlying_at_least_35bp` | FAIL |
| `stress_cost_absolute_return_positive` | FAIL |
| `each_subperiod_absolute_return_positive` | FAIL |
| `each_subperiod_minimum_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |

## Integrity

- Physically opened window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `994f96f261b23718aef74ecd21c053b01c8f656072f9ddff6264a43a42f34984`
- Result manifest: `4d1525e039745619826e4ae0c8a5716730e7857a83b237e4386bdaacfb54b921`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
