# HTF event-context preference test (2026-06-23)

## Purpose

The previous best line was pairwise `WAIT/LONG/SHORT` preference over event/regime tokens. This test adds completed higher-timeframe state:

- 4h / 1d / 3d / 1w return and range-position buckets;
- 1d / 1w drawdown buckets;
- compact derived tokens: `htf_trend_stack`, `htf_risk_state`.

All HTF fields come from completed higher-timeframe candles and are bucketed with train-only edges.

## Dataset

Generated:

- `data/llm_context_regime_events_htf_2026-06-23.jsonl`
- `results/llm_context_regime_events_htf_summary_2026-06-23.json`

Rows are unchanged from the previous event-context dataset:

| Split | Rows |
| --- | ---: |
| train | 5,844 |
| test | 2,924 |
| eval | 598 |

Prompt size increased but remains compact enough for Gemma-style SFT:

| Prompt chars | Value |
| --- | ---: |
| min | 1,572 |
| mean | 1,631 |
| max | 1,729 |

## Rolling preference result with HTF interactions

Generated:

- `results/rolling_event_context_preference_htf_interactions_predictions_2026-06-23.jsonl`
- `results/rolling_event_context_preference_htf_interactions_summary_2026-06-23.json`
- `results/rolling_event_context_preference_htf_interactions_backtest_2026-06-23.json`

Base result:

| Metric | Value |
| --- | ---: |
| Trades | 566 |
| CAGR | -4.01% |
| Strict MDD | 45.49% |
| CAGR/MDD | -0.09 |
| Mean trade return | -0.010% |
| p-value approx | 0.851 |

This is worse than the prior non-HTF pairwise result. The initial HTF attempt did not affect model decisions because common state tokens cancel in pairwise differences; after adding HTF candidate interactions, the model changed decisions but overfit/lagged regime transitions badly.

## Diagnostic overlay sweep

Generated:

- `results/rolling_event_context_preference_htf_overlay_sweep_2026-06-23.json`

Best diagnostic overlay:

| Take profit | Rolling loss stop | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4.0% | 10 trades / 5% | 71 | 12.86% | 8.27% | 1.55 | 0.011 |

This is statistically more interesting than the raw HTF run, but it is an overlay-selected diagnostic and still below the target ratio of 3.

## Interpretation

HTF context is not free alpha:

- Without risk overlay it worsens target-month generalization.
- With aggressive rolling loss stop it can isolate a smaller, cleaner trade subset.
- That suggests HTF tokens are useful for risk/regime gating, not direct directional ranking.

## Decision

Keep HTF tokens available in the RLLM context, but do not use HTF candidate interactions as the default direct preference ranker.

Next direction:

1. Use HTF as a risk/gate feature, not as a side predictor.
2. Make rolling loss / validation-decay state explicit and causal.
3. Evaluate gate-first architecture: pairwise side ranker + causal risk gate.
