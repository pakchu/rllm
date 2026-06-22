# Feature coverage audit — 2026-06-23

## Purpose

After the analyzer/trader reset and alpha gate failures, the next question was
whether the data actually contains the feature sources we think it contains. This
audit measures coverage, variance, and yearly availability using the same causal
feature builders used by alpha scans.

## Command

```bash
.venv/bin/python -m training.feature_coverage_audit \
  --input-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/feature_coverage_audit_2020_2026_wave_2026-06-23.json \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --external-tolerance 30min
```

## Result summary

Input coverage:

- rows: `674,785`
- period: `2019-12-31 15:00:00` to `2026-05-31 15:00:00`
- feature count: `111`
- external attach error: `null`

Feature family usability:

| family | features | usable |
|---|---:|---:|
| market | 20 | 20 |
| flow_volume | 13 | 13 |
| derivatives_aux | 4 | 0 |
| external_macro_kimchi | 22 | 22 |
| higher_timeframe | 25 | 25 |
| wave | 27 | 27 |

## Important finding

DXY, Kimchi Premium, USDKRW, higher-timeframe, and flow/volume features are
present and non-constant enough to scan:

- `mkt__dxy`: usable, nonzero fraction `0.764`, std `2.123`
- `mkt__kimchi_premium`: usable, nonzero fraction `0.650`, std `0.0227`
- `mkt__usdkrw_zscore`: usable, nonzero fraction `0.633`, std `1.069`
- `mkt__weekly_return_4w`: usable, nonzero fraction `0.983`, std `0.186`
- `mkt__taker_imbalance`: usable, nonzero fraction `1.000`, std `0.227`

But BTC single-asset derivative auxiliary features are completely unusable in
this dataset:

- `mkt__funding_rate`: nonzero fraction `0.0`, std `0.0`
- `mkt__funding_zscore`: nonzero fraction `0.0`, std `0.0`
- `mkt__oi_change`: nonzero fraction `0.0`, std `0.0`
- `mkt__oi_zscore`: nonzero fraction `0.0`, std `0.0`

## Interpretation

The previous alpha failures were not caused by missing DXY/Kimchi/HTF/flow
features; those are available. However, the BTC single-asset scan has not really
used funding/open-interest information because those columns are neutral zero.

Next useful expansion is to either:

1. join BTC funding/open-interest/premium-index data into the single-asset BTC
   feature frame, or
2. continue multi-asset futures experiments where funding/premium auxiliary data
   already exists per symbol.

Do not spend new LLM/RL cycles assuming funding/OI was tested on BTC unless this
coverage audit turns those features usable.
