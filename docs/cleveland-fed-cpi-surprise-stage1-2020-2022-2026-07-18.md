# CFCS-1 stage1_2020_2022 result — 2026-07-18

## Decision

**REJECT_KEEP_2023_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long | Short | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | 6.5035% | 2.1220% | 4.3509% | 0.4877 | 26 | 10 | 16 | 61.3281 | 0.3055 |
| 10 bp/notional/side | 5.4047% | 1.7696% | 4.4468% | 0.3980 | 26 | 10 | 16 | 61.3281 | 0.3904 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2020 | 0.7592% | 0.7577% | 2.2186% | 0.3415 | 9 |
| 2021 | 3.7401% | 3.7428% | 2.5422% | 1.4722 | 8 |
| 2022 | 1.8901% | 1.8914% | 4.3509% | 0.4347 | 9 |

## Gates

| Gate | Result |
|---|:---:|
| `absolute_return_positive` | PASS |
| `cagr_to_strict_mdd_at_least_3` | FAIL |
| `strict_mdd_at_most_15pct` | PASS |
| `weekly_cluster_signflip_p_at_most_10pct` | FAIL |
| `minimum_24_trades` | PASS |
| `minimum_8_long_trades` | PASS |
| `minimum_8_short_trades` | PASS |
| `mean_gross_underlying_at_least_50bp` | PASS |
| `stress_cost_absolute_return_positive` | PASS |
| `each_calendar_year_absolute_return_positive` | PASS |
| `each_calendar_year_minimum_8_trades` | PASS |
| `mechanism_control_margin_at_least_0_25` | FAIL |

## Integrity

- Physically opened window: `['2020-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Evaluator SHA-256: `92aba5e648ee4a0ac7119d37a271edd86df99f4177b8a533d4338d0e88bb5ff2`
- Result manifest: `bc47514ad06ad3e4a422d078a7436f13077c3634fe3234fc9b4ba04c416a08d6`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
