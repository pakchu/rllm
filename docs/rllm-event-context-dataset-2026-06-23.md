# RLLM event-context dataset update (2026-06-23)

## Change

The single-policy RLLM context miner now includes compact causal price-action event tokens:

- `pa_event_pressure`
- `pa_downside_reclaim`
- `pa_upside_rejection`
- `pa_long_window_event`

These are derived from shifted prior-range price-action events, not from future returns. Full raw event columns are intentionally not placed in the prompt to avoid prompt bloat and overfit.

## Dataset run

Command generated:

- `data/llm_context_regime_events_2026-06-23.jsonl`
- `results/llm_context_regime_events_summary_2026-06-23.json`

Summary:

| Split | Rows | LONG | SHORT | NO_TRADE |
| --- | ---: | ---: | ---: | ---: |
| train | 5,844 | 2,297 | 2,079 | 1,468 |
| test | 2,924 | 1,121 | 992 | 811 |
| eval | 598 | 204 | 230 | 164 |

Prompt length:

- min: 1,245 chars
- mean: 1,294 chars
- max: 1,386 chars

## Stable context selection with event tokens

Context keys:

```text
trend_alignment,risk_state,pa_event_pressure,pa_downside_reclaim,pa_upside_rejection,pa_long_window_event,funding_zscore,premium_index_zscore,range_pos,window_drawdown,taker_imbalance
```

Outputs:

- `data/stable_context_events_policy_2026-06-23.jsonl`
- `results/stable_context_events_policy_summary_2026-06-23.json`

Selection summary:

| Metric | Value |
| --- | ---: |
| train contexts | 2,924 |
| test contexts | 1,886 |
| selected contexts | 17 |
| train selected rows | 168 |
| test selected rows | 72 |
| eval selected rows | 8 |

Selected target distribution:

| Split | LONG | SHORT | NO_TRADE |
| --- | ---: | ---: | ---: |
| train | 110 | 58 | 5,676 |
| test | 54 | 18 | 2,852 |
| eval | 6 | 2 | 590 |

## Oracle diagnostics

These are target-echo diagnostics, not model results.

Full train+test+eval oracle:

| Metric | Value |
| --- | ---: |
| Trades | 210 |
| CAGR | 13.64% |
| Strict MDD | 11.71% |
| CAGR/MDD | 1.16 |
| Mean trade return | 0.402% |
| p-value approx | 0.000087 |

Eval-only oracle:

| Metric | Value |
| --- | ---: |
| Trades | 8 |
| CAGR | 1.26% |
| Strict MDD | 3.91% |
| CAGR/MDD | 0.32 |
| Mean trade return | 0.067% |
| p-value approx | 0.823 |

## Interpretation

Event tokens made the RLLM prompt more structurally meaningful, but the current stable-context selector is too sparse on 2026 eval.

This is better than the previous failure mode because it does not fabricate high eval profitability. But it is not enough for model training/deployment:

- The full oracle is mostly train/test support and cannot be claimed as live edge.
- Eval has only 8 trades, far below statistical usefulness.
- The next improvement should relax or redesign context matching without returning to broad overfit.

## Next step

Move from exact context-id matching to an event-aware utility model:

1. Keep compact event tokens in the prompt.
2. Train a small utility/ranker or LLM policy on row-level event contexts.
3. Validate by rolling month, not static exact context matching.
4. Require eval/rolling trade count before SFT promotion.
