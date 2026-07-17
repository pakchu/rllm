# FQPR-3 stage1_2021_2022 result

## Decision

**REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp |
|---|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -26.1665% | -14.0825% | 42.3807% | -0.3323 | 44 | -102.1560 |
| 10 bp/notional/side | -27.4562% | -14.8367% | 42.9434% | -0.3455 | 44 | -102.1560 |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021_after_warmup | 7.6458% | 15.6573% | 20.1912% | 0.7754 | 23 |
| 2022 | -31.3989% | -31.4166% | 38.0698% | -0.8252 | 20 |

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
| `primary_ratio_strictly_beats_each_mechanism_control` | FAIL |
| `one_day_delay_and_random_do_not_fully_qualify` | PASS |

## Isolation

- Evaluator freeze: `35131ea4975abe6800aa66b784f97c7cb4e493e96c25c38ec1172c857603df1b`
- Physically opened window: `['2021-01-01T00:00:00+00:00', '2023-01-01T00:00:00+00:00']`
- Still sealed: stage2_2023, 2024, 2025, 2026_ytd
- Full-clock CAGR includes idle cash.
- Strict MDD uses favorable-before-adverse held OHLC, global high-water,
  entry/hypothetical-exit/realized-exit costs, and exact realized funding.

## Interpretation

- The fixed long thesis was not stable: the contained 2021 window earned
  `+7.6458%`, but 2022 lost `-31.3989%`.
- The mean gross underlying move was `-102.1560 bp`, so transaction-cost tuning
  cannot repair the economic sign.
- The direction-flip control earned `+20.9390%`, but its full-clock CAGR/MDD was
  only `0.3945` with weekly-cluster `p=0.1833`; it is not promoted or reused as
  a replacement alpha.
- A one-day stale signal did better than the primary (`+8.4370%` versus
  `-26.1665%`), directly contradicting the proposed immediate participation
  mechanism.

The singleton is rejected without changing `Q`, direction, hold, execution
delay, cost, or any gate. No 2023 outcome or portfolio correlation is opened.
