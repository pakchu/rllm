# Rolling event-context utility ranker (2026-06-23)

## Purpose

Exact stable-context matching with price-action event tokens was too sparse on 2026 eval. This run replaces exact matching with a lightweight symbolic utility ranker:

- input: `data/llm_context_regime_events_2026-06-23.jsonl`;
- features: categorical `state_tokens` plus `side` interactions;
- target: `reward_audit[LONG/SHORT].net_return_pct`;
- model: ridge regression fit monthly using prior-only rows;
- selection: threshold/min-gap chosen on the immediately previous validation window;
- target month: never used for fit or threshold selection;
- final evaluation: strict online overlay backtest over all rolling target predictions.

This is not a Gemma inference result. It is a pre-Gemma test of whether event-context tokens contain learnable utility.

## Main rolling result

Command output:

- `results/rolling_event_context_utility_predictions_2026-06-23.jsonl`
- `results/rolling_event_context_utility_summary_2026-06-23.json`
- `results/rolling_event_context_utility_backtest_2026-06-23.json`

| Metric | Value |
| --- | ---: |
| Months | 29 |
| Prediction rows | 3,522 |
| Trades | 344 |
| CAGR | 4.14% |
| Strict MDD | 19.12% |
| CAGR / strict MDD | 0.22 |
| Mean trade return | 0.036% |
| p-value approx | 0.588 |

## Diagnostic risk overlay sweep

A fixed diagnostic overlay sweep on the same rolling predictions was run to see whether the problem is primarily risk control.

Best diagnostic case:

| Stop | Take | Rolling loss stop | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3.0% | 2.0% | 20 trades / 8% | 153 | 10.15% | 11.48% | 0.88 | 0.140 |

This improves MDD materially but still does not approach the target ratio and is not statistically decisive.

## Interpretation

This is a real improvement over exact matching and static ridge bundles:

- trade count is no longer sparse;
- target-month leakage is blocked;
- final result is mildly positive before overlays;
- risk overlay can reduce MDD below 15%.

But it is not a deployable alpha:

- CAGR/MDD is far below 3;
- mean trade p-value is weak;
- 2026 target months mostly abstain because prior validation deteriorates;
- the learned utility is too diffuse, not strong enough for direct trading.

## Decision

Keep the event-context representation. Do not promote this ridge ranker as a trading policy.

Next useful step is to make the learning objective more RLLM-like:

1. Convert each timestamp into explicit `WAIT`, `LONG`, `SHORT` candidates with rationale tokens.
2. Train/evaluate a pairwise preference/value model rather than pointwise ridge.
3. Add regime-memory tokens summarizing recent validation decay so the model can abstain earlier.
4. Only after rolling utility passes should Gemma SFT be run.
