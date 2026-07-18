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
- Evaluator SHA-256: `8f66ec75c20eba83af12f8531382b5c450c7c0ba1f2340f82e27915d09f168ac`
- Result manifest: `fba386377571abf79c15c2888541695c8d7b7e828481c3c8bf0753c644356607`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
