# Rolling alpha discovery small run (2026-06-25)

## Goal
After symbolic/event candidate approaches repeatedly inverted, test whether a simpler market-path feature rule can provide stable alpha outside the current LLM policy pool.

## Run
Script: `training/rolling_alpha_feature_discovery.py`  
Report: `results/rolling_alpha_feature_discovery_strict_small_2026-06-25/report.json`

Parameters:
- horizons: 72, 144, 288
- quantiles: 0.10, 0.20, 0.30
- top event candidates: 30
- strict candidates: 8
- leverage: 1.0

## Result
No stable strict candidate survived broad folds including 2026.

Examples:

### `wave__mom_144`, horizon 288, q=0.10
- 2025H1: CAGR 81.39%, strict MDD 20.65%, ratio 3.94
- 2025H2: CAGR 4.08%, strict MDD 21.04%, ratio 0.19
- 2026H1: CAGR -44.03%, strict MDD 30.50%, ratio -1.44

### `mkt__htf_1w_return_4` / `mkt__weekly_return_4w`, horizon 288, q=0.20
- 2024H1: CAGR 47.32%, strict MDD 18.67%, ratio 2.53
- 2024H2: CAGR -9.30%, strict MDD 14.88%, ratio -0.62
- 2025H1: CAGR -12.43%, strict MDD 15.01%, ratio -0.83
- 2026H1: CAGR -14.79%, strict MDD 16.35%, ratio -0.90

### DXY availability/value candidates
Some event scores looked high, but strict replay was broadly negative across most folds and 2026 remained negative.

## Conclusion
The repeated failure is not confined to LLM/symbolic policies. Simple market-path rules also exhibit fold-specific spikes and next-regime inversion. Current available feature set does not yet expose a stable strict-replay alpha satisfying the objective.

## Next direction
The project needs a different search target:
1. regime-instability detector / abstention meta-model, or
2. a new data source / alpha source beyond current OHLCV + DXY/kimchi/USDKRW-style features, or
3. much shorter-horizon execution/microstructure features where edge can be tested with tighter path risk.
