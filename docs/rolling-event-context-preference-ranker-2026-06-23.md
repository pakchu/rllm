# Rolling event-context preference ranker (2026-06-23)

## Purpose

The pointwise utility ranker improved over exact context matching, but did not explicitly learn when to wait. This run adds an explicit `WAIT` candidate with zero utility and trains a pairwise preference model over:

- `WAIT`
- `LONG`
- `SHORT`

The feature surface remains compact event/regime state tokens plus candidate-action interactions.

## Protocol

- Input: `data/llm_context_regime_events_2026-06-23.jsonl`.
- Model: monthly pairwise linear ridge preference ranker.
- Train window: prior 1095 days before validation.
- Validation window: prior 180 days before target month.
- Target month: unseen during fit and threshold selection.
- Threshold selection: prior validation only.
- Final backtest: strict online overlay on all target-month predictions.

Outputs:

- `results/rolling_event_context_preference_predictions_2026-06-23.jsonl`
- `results/rolling_event_context_preference_summary_2026-06-23.json`
- `results/rolling_event_context_preference_backtest_2026-06-23.json`

## Base rolling result

| Metric | Value |
| --- | ---: |
| Months | 29 |
| Prediction rows | 3,522 |
| Trades | 488 |
| CAGR | 8.82% |
| Strict MDD | 22.06% |
| CAGR / strict MDD | 0.40 |
| Mean trade return | 0.049% |
| p-value approx | 0.367 |

Compared to pointwise utility ridge, pairwise WAIT/LONG/SHORT improved CAGR and trade count, but MDD remained too high.

## Diagnostic overlay sweep

Best diagnostic overlay on the fixed rolling predictions:

| Stop | Take | Rolling loss stop | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3.0% | 3.0% | 20 trades / 8% | 251 | 14.76% | 12.09% | 1.22 | 0.108 |

This is the best result so far in the event-context line, but it is still below the target (`CAGR / strict MDD >= 3`) and below strong statistical significance.

## Interpretation

The pairwise objective is directionally better:

- explicit WAIT candidate improves the shape of decisions;
- event-context tokens appear to contain weak learnable utility;
- risk overlay can bring MDD under 15% while preserving enough trades.

Remaining failure:

- edge magnitude is too small;
- validation can still select bad months;
- p-value remains weak;
- overlay sweep is diagnostic and not a clean untouched deployment selection.

## Decision

Keep pairwise preference framing as the next RLLM direction. Do not claim target achievement.

Next work should focus on improving feature/context edge rather than widening the same linear model:

1. Add recent performance/regime-memory tokens to detect validation decay.
2. Add multi-timeframe context summaries beyond event tokens.
3. Train Gemma on pairwise candidate ranking only after rolling non-LLM ranker reaches a stronger baseline.
