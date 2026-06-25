# Path-outcome gate experiment (2026-06-25)

## Goal
Test whether candidate-level path outcomes can fix the recent drawdown failure of the event-candidate policy without using future information at decision time.

## Method
Added `training/event_candidate_path_gate_walkforward.py`.

Per fold:
1. Train ridge scores on past candidate rows only.
2. Candidate labels are path-aware historical outcomes: `ret`, `ret_minus_mae`, `ret_minus_2mae`, `ret_plus_mfe_minus_2mae`, `win_stop1`, `win_stop2`.
3. Select target recipe, score quantile, and full-margin on validation only.
4. Refit on fit+validation and trade the next test window.

Leakage guard: test rows are not used for target/threshold selection or fitting before the test window.

## Results

### Recent 6M fit / 1M val / 1M test, loose validation
Path: `results/event_candidate_path_gate_recent_6m1m1m_targets_2026-06-25/report.json`

- CAGR: 3.61%
- Strict MDD: 10.50%
- CAGR / strict MDD: 0.34
- Trades: 67
- p-value approx: 0.839

This reduced the prior recent MDD problem but did not produce meaningful alpha.

### Recent 12M fit / 3M val / 1M test, strict validation, feature subset
Path: `results/event_candidate_path_gate_recent_12m3m1m_faststrict_subset_2026-06-25/report.json`

- CAGR: 0.78%
- Strict MDD: 17.87%
- CAGR / strict MDD: 0.04
- Trades: 89
- p-value approx: 0.893

Longer validation did not recover profitability.

### Recent 6M fit / 3M val / 1M test, strict validation, feature subset
Path: `results/event_candidate_path_gate_recent_6m3m1m_faststrict_subset_2026-06-25/report.json`

- CAGR: -17.21%
- Strict MDD: 34.09%
- CAGR / strict MDD: -0.50
- Trades: 131
- p-value approx: 0.0106

This is statistically meaningful in the wrong direction: the model selected bad trades.

### Side-flip diagnostic for the bad-trade selector
Path: `results/event_candidate_path_gate_recent_6m3m1m_faststrict_subset_2026-06-25/combined_test_backtest_inverted.json`

- CAGR: 3.72%
- Strict MDD: 12.02%
- CAGR / strict MDD: 0.31
- Trades: 131
- p-value approx: 0.564

The bad-trade signal does not become a strong tradable contrarian alpha after costs and strict path drawdown.

## Conclusion
Candidate-level path labels help reduce drawdown in some settings, but they do not solve the alpha problem. The next step should not be more gate optimization. It should search for genuinely predictive feature edges, then only use the gate as a risk filter.
