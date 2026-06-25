# Optimized sparse setup scan — 2026-06-25

## Purpose
The previous sparse setup direction was structurally promising but too slow to scale beyond a 12-feature smoke run. This pass optimized the miner enough to run a wider macro/price-action sparse scan and then validated candidates with past-only walk-forward selection.

## Miner optimization
`rolling_sparse_setup_miner` now caches per-feature/per-fold predicate threshold masks for each horizon/quantile. This avoids recomputing the same quantile threshold and full-row boolean comparison for every feature pair.

Observed runtime on the 12-feature smoke scan:
- before cache: about 23s
- after cache: 11.43s
- output matched the previous smoke result
- max RSS: ~3.9GB

## Wider optimized scan
Command shape:
```bash
.venv/bin/python -m training.rolling_sparse_setup_miner \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/sparse_setup_cached_macro_optimized_2026-06-25/report.json \
  --feature-include-regex '^(mkt__(dxy_|kimchi_|usdkrw_|htf_1d|htf_3d)|wave__(mom_|cvd_|flow_))' \
  --horizons 36,72 --quantiles 0.05,0.10 --max-features 30
```
Actual selected feature count after regex/std filters: 27.
Runtime: 50.27s.

## Best individual sparse candidate
Predicate:
- `wave__mom_12 low`
- `wave__mom_288 low`
- horizon 36, q=0.05

Strict fold summary:
- 7/7 positive folds
- 187 total trades
- median CAGR 26.01%
- median strict MDD 11.20%
- worst fold CAGR +11.04%
- ratio>=3 and MDD<=15 folds: 3/7

Fold details:
- 2023H1: CAGR 11.04 / MDD 3.62 / ratio 3.05 / 8 trades
- 2023H2: CAGR 30.31 / MDD 4.05 / ratio 7.49 / 7 trades
- 2024H1: CAGR 48.79 / MDD 11.20 / ratio 4.36 / 46 trades
- 2024H2: CAGR 25.67 / MDD 15.47 / ratio 1.66 / 34 trades
- 2025H1: CAGR 19.20 / MDD 8.85 / ratio 2.17 / 35 trades
- 2025H2: CAGR 35.19 / MDD 12.02 / ratio 2.93 / 24 trades
- 2026H1: CAGR 26.01 / MDD 15.17 / ratio 1.71 / 33 trades

This is the first candidate in this branch with all folds positive, but trade count per half-year is still low and ratio target is not consistently met.

## Past-only walk-forward selector
Selector config selected by 2023-2025 history, not by 2026:
- candidate_limit 16
- max_ensemble_size 4
- leverage 1.0
- stop/take-profit 6%
- execution horizon keeps candidate horizon (`0`)
- sizing `prior_sharpe`

Final 2023H1-2026H1:
- CAGR 15.10%
- strict MDD 9.57%
- CAGR/MDD 1.58
- trades 428
- p-value approx 0.031

Untouched 2026H1 fold:
- CAGR 14.31%
- strict MDD 8.62%
- CAGR/MDD 1.66
- trades 65

## Selector sweeps
Small no-leak config sweeps selected by 2023-2025 history found:
- best historical ratio config: leverage 1.0, ensemble 4, stop/tp 6
- increasing leverage raised or destabilized MDD and did not improve ratio enough
- overriding execution horizon to 24/36/72 worsened historical selection vs keeping each candidate horizon

## Interpretation
This is real progress but not the stated target. The sparse event route now shows statistically positive, no-leak performance, but the return density is too low for CAGR 50 / strict MDD 15. The next useful step is not dense ridge/LLM prediction; it is to strengthen sparse event generation and path-aware action ranking around the profitable event family, especially `wave__mom_12 low & wave__mom_288 low`.

## Leakage guard
- Candidate thresholds and side are fit before each eval fold.
- Selector uses only previous fold outcomes for each current fold.
- 2026H1 is not used to pick selector config in the reported best config.
- Strict replay uses actual OHLC bar-by-bar execution.
