# Prediction feature-filter audit (2026-06-25)

## Goal
After the symbolic monthly selector failed, test whether signal-time market/path features can veto bad trades without changing the LLM-style entry generator.

## Diagnostic audit
Script: `training/prediction_trade_feature_audit.py`  
Report: `results/symbolic_trade_feature_audit_2026-06-25/report.json`

Top separators on the failed symbolic trades included:
- `sma12_ratio`
- `usdkrw_zscore`
- `usdkrw_momentum`
- `kimchi_premium_zscore`
- short-term trend/SMA ratios

This audit uses realized trade returns as labels, so it is diagnostic and not deployable evidence by itself.

## Walk-forward filter test
Script: `training/prediction_feature_filter_walkforward.py`

Using prior-month validation to select a simple feature filter:
- For April eval, March validation selected `usdkrw_zscore <= 0`, scope `ALL`.
- April eval improved from the symbolic selector loss to CAGR 28.94%, strict MDD 4.96%, 26 trades.
- For May, requiring validation p-value <= 0.2 rejected the weak April validation filter and left May unfiltered.

Combined March original + April filtered + May unfiltered:
- Report: `results/symbolic_prediction_feature_filter_wf_p02_2026-06-25/combined_mar_may_backtest.json`
- CAGR: 17.63%
- Strict MDD: 7.19%
- CAGR / strict MDD: 2.45
- Trades: 107
- p-value approx: 0.644

## Interpretation
This is the first recent result that moves in the right direction after rejecting symbolic thresholds, but it is not enough:
- ratio is below the target 3.0
- p-value is weak
- feature candidates came from a posthoc audit of failed 2026 trades

## Next direction
Validate feature-filter discovery on pre-2026 data, then freeze the discovery rule before 2026. If the same USDKRW/short-trend veto family emerges from prior data and survives 2026, it can become a real component. Otherwise it is just another 2026-specific repair.
