# Pre-2026 feature-filter validation (2026-06-25)

## Goal
Check whether the 2026 USDKRW/short-trend veto result is a reusable pre-2026 feature family or a 2026-specific repair.

## Base stream
Generated a no-leak fixed symbolic policy for 2025H2:

- History: rows before 2025-07-01
- Eval: 2025-07-01 through 2026-01-01
- Target: `distributional_safety`
- Threshold: `-0.0134`
- Predictions: `results/pre2026_feature_filter_validation_2026-06-25/fixed_distributional_m0134_predictions.jsonl`
- Backtest: `results/pre2026_feature_filter_validation_2026-06-25/fixed_distributional_m0134_backtest.json`

Base result:
- CAGR: -47.88%
- Strict MDD: 30.76%
- Trades: 280
- p-value: 0.072

This provides a sufficiently active but bad trade stream for veto testing.

## Feature-filter walk-forward
Report: `results/pre2026_feature_filter_validation_2026-06-25/filter_wf_2025h2_p02_report.json`

Protocol:
- Select simple feature filter on previous month only.
- Apply selected filter to next month only.
- Candidate features match the 2026 audit family: USDKRW, DXY, kimchi, short trend/SMA.
- Validation p-value gate: <= 0.2.

Aggregate result after filters:
- CAGR: -46.95%
- Strict MDD: 24.73%
- Trades: 232
- p-value: 0.135

The filter reduced MDD but did not create profitability.

## Monthly behavior
Some filters looked strong in validation but inverted in the next month:

- 2025-09 selected `sma12_ratio <= 0.0010`, validation CAGR 107.6%, but eval CAGR -56.0%.
- 2025-11 selected `dxy_momentum <= 0` for LONG, validation CAGR 215.6%, but eval CAGR -42.0%.
- Other months either failed validation or were left unfiltered and remained weak/negative.

## Conclusion
The 2026 USDKRW veto is not yet validated as a stable pre-2026 feature family. It may still be useful as a local 2026 repair, but not as a deployable alpha/risk filter without broader evidence.

## Implication
The repeated pattern is now clear:
- validation spikes are common,
- next-period inversion is common,
- strict OHLC replay punishes these inversions,
- feature/gate tweaks improve individual months but do not provide stable alpha.

Next work should shift toward a different objective: detect and abstain during unstable feature/strategy regimes, or mine a new base alpha outside the current symbolic-event candidate pool.
