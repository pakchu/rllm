# Rolling alpha scan with wave_trading FX component features — 2026-06-25

## Purpose
The previous symbolic/event/market scans repeatedly failed because validation edges inverted under strict walk-forward replay. This run tested whether `wave_trading`'s individual FX component bars add a new no-leak lead-lag source beyond the existing DXY/Kimchi/USDKRW aggregate features.

## Change tested
- Added optional `wave_trading` FX component features:
  - raw point-in-time component closes `fx_eurusd`, `fx_usdjpy`, `fx_gbpusd`, `fx_usdcad`, `fx_usdsek`, `fx_usdchf`.
  - derived history-only `*_zscore` and `*_momentum` features.
  - `btckrw_zscore` and `btckrw_momentum` from the Kimchi premium join.
- `training.rolling_alpha_feature_discovery` can now use `--include-external-components` with `--wave-trading-root`.

## Command
```bash
.venv/bin/python -m training.rolling_alpha_feature_discovery \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/rolling_alpha_feature_discovery_fx_components_2026-06-25/report.json \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --external-tolerance 30min \
  --include-external-components \
  --window-size 144 \
  --horizons 12,24,36 \
  --quantiles 0.05,0.10,0.20 \
  --min-train-rows 20000 \
  --min-eval-events 200 \
  --top-event-candidates 40 \
  --max-strict-candidates 10 \
  --leverage 1.0
```

## Result
No deployable candidate met the target. The best strict replay still came from existing `wave__mom_288` / higher-timeframe return features and remained unstable:

- `wave__mom_288`, h=36, q=0.05:
  - 2024H2: CAGR 35.81 / strict MDD 13.15 / ratio 2.72 / 123 trades
  - 2025H1: CAGR 62.75 / strict MDD 10.65 / ratio 5.89 / 116 trades
  - 2025H2: CAGR -10.85 / strict MDD 12.06 / ratio -0.90 / 55 trades
  - 2026H1: CAGR 3.23 / strict MDD 22.08 / ratio 0.15 / 91 trades
- `mkt__htf_1d_return_4`, h=36, q=0.05:
  - 2026H1: CAGR 14.58 / strict MDD 13.52 / ratio 1.08 / 48 trades
  - but 2025H1 and 2025H2 were negative.
- `mkt__usdkrw_zscore`, h=36, q=0.05 appeared in strict candidates only as a strong negative candidate:
  - all 2023H1–2026H1 folds negative, worst CAGR -75.69, max strict MDD 52.48.

## Interpretation
Adding individual FX component features did not solve the alpha problem. The current univariate quantile approach still selects unstable regime-specific edges, and the extra macro/FX components do not provide a stable standalone trigger under strict no-leak replay.

The useful conclusion is negative: continue treating current event/feature selection as a candidate generator only, not a deployable selector. Next productive direction is to mine interaction/composite patterns or change label/action construction, not to keep widening single-feature gates.

## Leakage guard
The report records:
- features use rows at or before signal time;
- each fold rule fit uses only data before eval start;
- strict replay uses actual OHLC bar-by-bar execution;
- external joins are backward-asof with no future rows.
