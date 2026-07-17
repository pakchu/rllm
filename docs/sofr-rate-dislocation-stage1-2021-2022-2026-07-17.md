# SFRD-1 stage1_2021_2022 result

## Decision

**REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp |
|---|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -13.1405% | -6.8061% | 42.5349% | -0.1600 | 48 | -26.5540 |
| 10 bp/notional/side | -14.7960% | -7.6991% | 43.1247% | -0.1785 | 48 | -26.5540 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 | -4.1228% | -4.1255% | 27.7839% | -0.1485 | 12 |
| 2022 | -9.4056% | -9.4117% | 27.4977% | -0.3423 | 36 |

## Statistical and falsification evidence

- Weekly-cluster sign-flip: `p = 0.641768` over 48 UTC-ISO-week clusters;
  this is not statistically significant under the frozen `p <= 0.10` gate.
- The SOFR-level-tail mechanism control produced +29.5079% absolute return,
  13.8116% CAGR, 10.9539% strict MDD, and a 1.2609 ratio, but only 12 trades.
  Controls are falsification evidence and cannot replace the preregistered
  singleton.
- The one-observation-delay control produced +9.2744% absolute return but only
  a 0.1678 ratio with 27.0456% strict MDD; it did not fully qualify.
- The primary failed to beat both the level-tail and month-turn mechanism
  controls. No threshold, side, feature, delay, or hold repair is permitted.

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
| `single_entry_month_share_at_most_15pct` | PASS |
| `primary_ratio_strictly_beats_each_mechanism_control` | FAIL |
| `one_observation_delay_and_random_do_not_fully_qualify` | PASS |

## Isolation

- Candidate class: `source-only-screened exploratory singleton`
- Evaluator freeze: `a4f5eac7bfd433cbc311c23ee2de594116c5e989bfebf5d3d419f82f294e782e`
- Physically opened window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Still sealed: stage2_2023, 2024, 2025, 2026_ytd
- Full-clock CAGR includes warm-up and idle cash.
- Strict MDD uses favorable-before-adverse held OHLC, global high-water,
  entry/hypothetical-exit/realized-exit costs, and exact realized funding.
