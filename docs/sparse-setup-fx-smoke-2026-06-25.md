# Sparse setup smoke with macro/FX/price-action predicates — 2026-06-25

## Purpose
Dense linear-combo predictors failed under no-leak test/eval. This run moved back toward sparse event construction: mine two-predicate setup events, then select an ensemble using only prior fold outcomes.

## Implementation changes
- `rolling_sparse_setup_miner` now supports:
  - `--include-external-components` for wave_trading FX component features;
  - `--feature-include-regex` to run bounded, reproducible feature-pool scans.
- `sparse_setup_walkforward_selector` now accepts `--include-external-components` when rebuilding the same feature space.

## Runtime finding
Full 82-feature and reduced 30-feature sparse scans were too slow for the current miner implementation:
- the miner recomputes full-length boolean masks across ~674k rows for every predicate pair/fold;
- the wave_trading external component path also reloads large 1m FX shards;
- both long runs were stopped after about 5-6 minutes without producing a report.

This is a tooling bottleneck, not an alpha result. Future larger sparse scans need mask caching/vectorization or prebuilt enriched caches.

## Smoke command
A 12-feature cached-macro smoke run completed in ~23 seconds:

```bash
.venv/bin/python -m training.rolling_sparse_setup_miner \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/sparse_setup_cached_macro_smoke_2026-06-25/report.json \
  --feature-include-regex '^(mkt__(dxy_|kimchi_|usdkrw_|htf_1d)|wave__(mom_))' \
  --window-size 144 \
  --horizons 72 \
  --quantiles 0.05 \
  --min-fold-events 20 \
  --max-fold-events 180 \
  --max-strict-candidates 10 \
  --leverage 1.0 \
  --max-features 12
```

## Candidate result
Best strict candidate:
- predicates: `mkt__dxy_momentum low` AND `wave__mom_12 high`
- horizon: 72 bars, q=0.05
- strict summary: 6/7 positive folds, 149 trades, median CAGR 4.62, median strict MDD 10.23, worst fold CAGR -1.94
- but ratio3/MDD15 folds: 0, so it does not meet the target.

## Past-only walk-forward ensemble
Command used the smoke report as candidate pool and selected each fold from prior fold outcomes only.

Final result:
- CAGR: 2.23%
- strict MDD: 15.94%
- CAGR/strict MDD: 0.14
- trades: 426
- p-value approx: 0.612

Fold results:
- 2023H1: CAGR 1.81 / MDD 11.11 / ratio 0.16 / 88 trades
- 2023H2: CAGR 7.47 / MDD 5.49 / ratio 1.36 / 71 trades
- 2024H1: CAGR -0.85 / MDD 9.17 / ratio -0.09 / 52 trades
- 2024H2: CAGR 2.63 / MDD 9.77 / ratio 0.27 / 102 trades
- 2025H1: CAGR 5.89 / MDD 5.81 / ratio 1.01 / 42 trades
- 2025H2: CAGR -0.82 / MDD 2.77 / ratio -0.29 / 24 trades
- 2026H1: CAGR 0.40 / MDD 5.94 / ratio 0.07 / 47 trades

## Interpretation
Sparse setup construction is structurally better than dense continuous trading because it lowers drawdown and preserves statistical trade count, but current predicates do not produce meaningful return. The direction is viable for LLM/RL only if the event generator is upgraded: cache masks, add asymmetric path labels, and rank sparse event contexts instead of trying to trade every weak quantile.

## Leakage guard
- Fold thresholds and sides are fit before each eval fold.
- Walk-forward selector uses only previous fold outcomes.
- Current fold metrics are not used for current selection.
- Strict replay uses bar-by-bar OHLC execution.
