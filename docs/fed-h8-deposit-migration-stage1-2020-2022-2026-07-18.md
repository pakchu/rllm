# H8DM-1 stage1_2020_2022 result — 2026-07-18

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long | Short | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -2.2954% | -0.7709% | 19.7683% | -0.0390 | 74 | 28 | 46 | 10.9870 | 0.9446 |
| 10 bp/notional/side | -5.1435% | -1.7444% | 20.7193% | -0.0842 | 74 | 28 | 46 | 10.9870 | 0.7953 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2020 | -10.2749% | -10.2550% | 17.9243% | -0.5721 | 27 |
| 2021 | 3.5104% | 3.5128% | 8.6405% | 0.4066 | 18 |
| 2022 | 0.7240% | 0.7244% | 10.6556% | 0.0680 | 27 |


## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | FAIL |
| `cagr_to_strict_mdd_at_least` | FAIL |
| `strict_mdd_pct_at_most` | FAIL |
| `weekly_cluster_signflip_p_at_most` | FAIL |
| `minimum_trades` | PASS |
| `minimum_long_trades` | PASS |
| `minimum_short_trades` | PASS |
| `mean_gross_underlying_bp_at_least` | FAIL |
| `stress_cost_absolute_return_positive` | FAIL |
| `each_calendar_year_absolute_return_positive` | FAIL |
| `each_calendar_year_minimum_trades` | PASS |
| `best_single_component_ratio_margin_at_least` | FAIL |

## Integrity

- Physically opened window: `['2020-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `39044584b07376e08db930354e67e3f56c55c9d6653f5ff43bd58138a1a1cbf4`
- Result manifest: `d6f52f185a539f2ad822613cd2c1482f13f137f57339a21bef42c47c80afe4f9`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
