# CVTR-1 stage1_2021_2022 result — 2026-07-17

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -11.4374% | -5.8962% | 39.5061% | -0.1492 | 281 | 13.1826 | 0.9156 |
| 10 bp/notional/side | -20.8611% | -11.0471% | 42.4158% | -0.2604 | 281 | 13.1826 | 0.6961 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 | -20.5455% | -20.5580% | 32.5769% | -0.6311 | 147 |
| 2022 | 12.6845% | 12.6938% | 22.0857% | 0.5747 | 133 |

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
| `minimum_each_side_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |
| `falsification_controls_do_not_fully_qualify` | PASS |

## Integrity

- Physically opened window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `1bb47f6d704c2f977e44e378bf57acf4d4f6ab6455346e7b720149132f2f1f0e`
- Result manifest: `9f5a5f42d4686c04566b2a1916bfe7959b3e0359e6bd9db3b464ae70a0cfd120`
- Full-calendar CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
