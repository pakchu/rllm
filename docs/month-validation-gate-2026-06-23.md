# Month validation health gate (2026-06-23)

## Purpose

The pairwise event-context preference ranker had weak but real edge, while 2026 months showed clear prior-validation decay. This gate separates side ranking from live-risk admission:

- side ranker: pairwise `WAIT/LONG/SHORT` event-context preference model;
- month gate: block all target-month trades if the prior validation-selected score is below a fixed threshold.

The gate uses only the rolling summary known before the target month. It does not inspect target-month outcomes.

## Implementation

Added:

- `training/apply_month_validation_gate.py`
- `tests/test_apply_month_validation_gate.py`

The script reads:

- prediction rows from `results/rolling_event_context_preference_predictions_2026-06-23.jsonl`;
- per-month prior-validation scores from `results/rolling_event_context_preference_summary_2026-06-23.json`.

If `selected.score < threshold`, that month's predictions are converted to `NO_TRADE`.

## Diagnostic threshold sweep

Best threshold from diagnostic sweep:

| Threshold | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 310 | 15.07% | 12.81% | 1.18 | 0.092 |

This already improves over the ungated pairwise result:

| Variant | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| pairwise preference | 488 | 8.82% | 22.06% | 0.40 | 0.367 |
| month gate threshold 0.5 | 310 | 15.07% | 12.81% | 1.18 | 0.092 |

## Reproduced gated + take-profit diagnostic

Command output:

- `results/month_validation_gate_preference_th0p5_predictions_2026-06-23.jsonl`
- `results/month_validation_gate_preference_th0p5_tp3_backtest_2026-06-23.json`

Fixed settings:

- month validation threshold: `0.5`
- take profit: `3.0%`
- leverage: `0.5x`

Result:

| Metric | Value |
| --- | ---: |
| Rows | 3,522 |
| Blocked rows | 1,206 |
| Trade signals before gate | 1,355 |
| Trade signals after gate | 860 |
| Executed trades | 316 |
| Return | 65.57% |
| CAGR | 23.28% |
| Strict MDD | 12.81% |
| CAGR/MDD | 1.82 |
| Mean trade return | 0.167% |
| p-value approx | 0.014 |

Blocked months include all 2026 target months in this run because their prior validation scores were strongly negative.

## Interpretation

This is the strongest result so far, but it still does not meet the target `CAGR/MDD >= 3`.

What improved:

- validation-decay state is highly useful;
- side ranker + causal admission gate is better than a monolithic ranker;
- p-value is finally below 0.05 in the diagnostic result;
- MDD is below 15%.

Remaining caveat:

- threshold `0.5` and take-profit `3%` are selected diagnostically on this same rolling result, not on a separate untouched overlay-selection holdout;
- CAGR is still far below the desired 50%;
- ratio is 1.82, not 3+.

## Decision

Keep the gate-first architecture:

1. Pairwise side ranker proposes candidate trades.
2. Causal validation-health gate admits/rejects months.
3. Fixed execution overlay handles per-trade exits.

Next step: make gate threshold and overlay selection leakage-safe by adding a nested walk-forward overlay-selection layer, or use 2024-2025 to choose gate/overlay and 2026 as untouched evaluation.
