# Delta-neutral basis-compression alpha: pre-2024

- Decision: **reject_pre2024**
- 2024+ opened: **no**
- All pre-2024 selected statistics and p-values are post-selection descriptive, not independent validation.
- Equal BTC quantities are fixed from entry to exit; gross is fixed only at entry.
- Proxy spot bars may mark strict MDD but cannot create or close a signal.

## Frozen candidate

```json
{
  "entry_z": 2.0,
  "lookback_minutes": 10080,
  "max_hold_minutes": 1440
}
```

## Statistics

| window | absolute return | CAGR | strict MDD | CAGR/MDD | episodes |
|---|---:|---:|---:|---:|---:|
| fit_2020_2022 | 4.0915% | 1.3453% | 7.9501% | 0.1692 | 23 |
| select_2023h1 | 0.3235% | 0.6539% | 1.9805% | 0.3302 | 1 |
| select_2023h2 | 0.0940% | 0.1867% | 3.4957% | 0.0534 | 1 |
| select_2023 | 0.4178% | 0.4181% | 3.4957% | 0.1196 | 2 |

## Promotion/control gates

```json
{
  "bonferroni_weekly_rademacher_p_below_0p10": false,
  "double_cost_positive": true,
  "primary_beats_stale_delay_inversion_controls": true,
  "zero_funding_positive": true
}
```

## Carry orthogonality comparator

```json
{
  "policy": {
    "entry_threshold": 5e-05,
    "exit_threshold": 0.0,
    "lookback_events": 21,
    "min_hold_events": 3
  },
  "windows": {
    "fit_2020_2022": {
      "basis_stats": {
        "absolute_return_pct": 4.091513270497038,
        "active_days": 23,
        "cagr_pct": 1.3453398261457972,
        "cagr_to_strict_mdd": 0.16922197655821086,
        "calendar_years": 3.0006844626967832,
        "close_mdd_pct": 1.1553075303715632,
        "daily_btc_beta": -0.0019804826554786457,
        "entry_times": [
          "2020-03-12 10:40:00",
          "2020-04-02 17:05:00",
          "2020-05-30 22:45:00",
          "2020-06-01 23:10:00",
          "2020-07-26 10:05:00",
          "2020-07-27 22:25:00",
          "2020-12-17 09:40:00",
          "2021-01-02 21:00:00",
          "2021-01-04 06:40:00",
          "2021-01-04 09:40:00",
          "2021-01-06 04:25:00",
          "2021-01-07 18:50:00",
          "2021-01-29 08:55:00",
          "2021-02-08 12:50:00",
          "2021-02-08 13:00:00",
          "2021-02-11 03:10:00",
          "2021-02-22 14:15:00",
          "2021-03-02 01:05:00",
          "2021-03-02 01:30:00",
          "2021-04-18 03:40:00",
          "2021-05-19 12:55:00",
          "2021-09-07 14:30:00",
          "2022-05-28 17:05:00"
        ],
        "episodes": 23,
        "exit_times": [
          "2020-03-12 10:45:00",
          "2020-04-02 18:20:00",
          "2020-05-30 23:00:00",
          "2020-06-02 14:55:00",
          "2020-07-27 10:05:00",
          "2020-07-28 22:25:00",
          "2020-12-17 13:00:00",
          "2021-01-02 21:20:00",
          "2021-01-04 09:35:00",
          "2021-01-04 10:15:00",
          "2021-01-06 07:55:00",
          "2021-01-07 19:00:00",
          "2021-01-29 12:05:00",
          "2021-02-08 12:55:00",
          "2021-02-08 13:30:00",
          "2021-02-12 03:10:00",
          "2021-02-22 14:25:00",
          "2021-03-02 01:20:00",
          "2021-03-02 01:55:00",
          "2021-04-18 03:45:00",
          "2021-05-19 13:00:00",
          "2021-09-07 15:00:00",
          "2022-05-28 17:20:00"
        ],
        "funding_cash_pct_initial": 0.664725596632809,
        "funding_events_received": 12,
        "gross_turnover_x_initial": 47.035246873533886,
        "nonzero_daily_pnl_days": 23,
        "strict_mdd_pct": 7.950148399803214,
        "transaction_cost_pct_initial": 3.9953248546245304
      },
      "carry_stats": {
        "absolute_return_pct": 24.4777004176459,
        "active_days": 874,
        "cagr_pct": 7.569699484289982,
        "cagr_to_strict_mdd": 0.6711135067445142,
        "calendar_years": 3.0006844626967832,
        "close_mdd_pct": 2.7438947099800326,
        "daily_btc_beta": -0.0023482908538114827,
        "entry_times": [
          "2020-01-08 00:05:00",
          "2020-04-29 08:05:00",
          "2020-09-28 08:05:00",
          "2020-11-07 08:05:00",
          "2021-05-29 00:05:00",
          "2021-08-02 00:05:00",
          "2022-02-10 00:05:00",
          "2022-03-07 00:05:00",
          "2022-05-04 08:05:00",
          "2022-05-20 00:05:00",
          "2022-07-08 08:05:00",
          "2022-09-12 16:05:00",
          "2022-12-18 00:05:00"
        ],
        "episodes": 13,
        "exit_times": [
          "2020-03-13 16:05:00",
          "2020-09-19 16:05:00",
          "2020-11-01 00:05:00",
          "2021-05-26 08:05:00",
          "2021-06-23 16:05:00",
          "2022-02-01 00:05:00",
          "2022-02-24 08:05:00",
          "2022-04-18 08:05:00",
          "2022-05-12 16:05:00",
          "2022-06-17 08:05:00",
          "2022-08-24 16:05:00",
          "2022-11-10 00:05:00",
          "2023-01-01 00:00:00"
        ],
        "funding_cash_pct_initial": 29.398540521736315,
        "funding_events_received": 2592,
        "gross_turnover_x_initial": 57.02329828291721,
        "nonzero_daily_pnl_days": 874,
        "strict_mdd_pct": 11.27931327296574,
        "transaction_cost_pct_initial": 4.847112579978377
      },
      "daily_pnl_pearson": 0.1296133638817743,
      "overlap_days": 1096
    },
    "select_2023": {
      "basis_stats": {
        "absolute_return_pct": 0.4178377270771305,
        "active_days": 3,
        "cagr_pct": 0.4181245147850232,
        "cagr_to_strict_mdd": 0.11960952736337654,
        "calendar_years": 0.999315537303217,
        "close_mdd_pct": 0.33099944068082365,
        "daily_btc_beta": 0.0012614284161636103,
        "entry_times": [
          "2023-01-14 00:40:00",
          "2023-10-16 13:30:00"
        ],
        "episodes": 2,
        "exit_times": [
          "2023-01-15 00:40:00",
          "2023-10-16 13:45:00"
        ],
        "funding_cash_pct_initial": 0.034570956661668796,
        "funding_events_received": 3,
        "gross_turnover_x_initial": 3.9512317871677705,
        "nonzero_daily_pnl_days": 3,
        "strict_mdd_pct": 3.495745899193725,
        "transaction_cost_pct_initial": 0.3356778759202904
      },
      "carry_stats": {
        "absolute_return_pct": 1.1450111489317072,
        "active_days": 67,
        "cagr_pct": 1.1457998790784574,
        "cagr_to_strict_mdd": 0.38158806541444706,
        "calendar_years": 0.999315537303217,
        "close_mdd_pct": 0.6540114402037944,
        "daily_btc_beta": -0.0003779241704666394,
        "entry_times": [
          "2023-10-26 16:05:00"
        ],
        "episodes": 1,
        "exit_times": [
          "2024-01-01 00:00:00"
        ],
        "funding_cash_pct_initial": 1.4362896977590138,
        "funding_events_received": 198,
        "gross_turnover_x_initial": 3.0644335257953306,
        "nonzero_daily_pnl_days": 67,
        "strict_mdd_pct": 3.0027141384361467,
        "transaction_cost_pct_initial": 0.260461230946006
      },
      "daily_pnl_pearson": -0.013754075383996681,
      "overlap_days": 365
    }
  }
}
```

## Constraints

- Historical DB rows are backfilled/non-PIT; frozen OOS and live forward parity are still mandatory.
- Unified margin or automatic collateral transfer/liquidation guard is required before live promotion.
- The earlier directional spot-perp residual family failed OOS; this two-leg family must remain profitable with funding removed.
