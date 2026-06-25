# Rolling ridge walk-forward audit — 2026-06-25

## Purpose

A single 2026 holdout for the ridge expected-return ranker looked strong:

- Report: `results/recent_model_family_2026-06-25/ridge_2026_eval_coef/report.json`
- Eval 2026-01 through 2026-05: CAGR 63.24%, strict MDD 9.45%, CAGR/MDD 6.69, 88 trades, p≈0.0341.

This could be a block-selection artifact because adjacent 2025 splits failed. A rolling walk-forward was added to test whether the ridge family can be selected and traded period-by-period without seeing future test rows.

## Implementation

Added `training/event_candidate_ridge_walkforward.py`.

Per fold:

1. Fit ridge expected-return score on fit window.
2. Select score quantile/full-margin on validation window only.
3. Require validation evidence gates.
4. Refit on fit+validation only.
5. Apply selected policy to test window.

Leakage guard:

- Fold windows are chronological.
- Test rows are never used for fit, threshold selection, or validation gates.
- Test model refit uses fit+validation only.

## Results

Input:

- `data/event_action_compressor_ranker_all_2022_2026_paext_rex_2026-06-24.jsonl`
- `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`

### Validation-gated monthly ridge WF

Report: `results/event_candidate_ridge_walkforward_recent_6m1m1m_alpha100_gate_v1_2026-06-25/report.json`

Config:

- start 2024-01-01
- 6M fit / 1M validation / 1M test / 1M step
- ridge alpha 100
- quantiles 0.80, 0.85, 0.90, 0.95
- margins 0, 0.5, 1.0
- validation gates: min trades 5, CAGR >= 10%, ratio >= 1, MDD <= 20%, p <= 0.5

Aggregate:

| Metric | Value |
| --- | ---: |
| CAGR | -5.83% |
| strict MDD | 19.68% |
| CAGR/MDD | -0.30 |
| Trades | 162 |
| Mean trade return | -0.055% |
| p-value | 0.6642 |

Recent folds:

| Test month | Status | Return | MDD | Trades |
| --- | --- | ---: | ---: | ---: |
| 2025-11 | TRADED | -0.79% | 5.38% | 2 |
| 2026-01 | ABSTAIN | 0.00% | 0.00% | 0 |
| 2026-02 | ABSTAIN | 0.00% | 0.00% | 0 |
| 2026-03 | ABSTAIN | 0.00% | 0.00% | 0 |
| 2026-04 | ABSTAIN | 0.00% | 0.00% | 0 |
| 2026-05 | TRADED | -0.70% | 2.37% | 8 |

### Relaxed monthly ridge WF

Report: `results/event_candidate_ridge_walkforward_recent_6m1m1m_alpha100_relaxed_2026-06-25/report.json`

All meaningful validation gates were relaxed to test whether the gates were simply too conservative.

Aggregate:

| Metric | Value |
| --- | ---: |
| CAGR | -18.21% |
| strict MDD | 37.44% |
| CAGR/MDD | -0.49 |
| Trades | 452 |
| Mean trade return | -0.071% |
| p-value | 0.2750 |

2026 monthly returns under relaxed gates:

| Test month | Return | MDD | Trades |
| --- | ---: | ---: | ---: |
| 2026-01 | -8.83% | 11.93% | 10 |
| 2026-02 | -2.90% | 12.99% | 20 |
| 2026-03 | -2.50% | 9.79% | 35 |
| 2026-04 | +2.84% | 5.40% | 27 |
| 2026-05 | -0.70% | 2.37% | 8 |

## Conclusion

The 2026 ridge holdout is not reproducible under live-like monthly rolling selection. It appears to rely on a block-level selection setup where 2025-09 through 2025-12 validation chooses one policy for the whole 2026 eval block.

This is useful research evidence but **not a deployable strategy**. The current candidate family still fails the user's recent-regime requirement.

Next viable direction: stop trying to rescue the same event-candidate ranker with filters/gates. Build a new target that explicitly learns recent-regime state transitions or a meta-selector that decides when the ridge family is allowed based on prior multi-month validation, then evaluate with untouched rolling windows.
