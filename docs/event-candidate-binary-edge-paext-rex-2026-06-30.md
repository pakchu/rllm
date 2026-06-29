# Event Candidate Binary Edge Surface — 2026-06-30

## Why this replaces the A/B/C policy surface

The same-signal A/B/C task forced one model answer to solve three things at once: direction selection, trade/no-trade gating, and relative candidate utility. Corrected eval showed Gemma base outperformed the small SFT adapters, so the surface was too noisy for the current training setup.

This binary-edge surface asks a smaller question per concrete candidate:

- `A` = EDGE_TRADE
- `B` = NO_EDGE

Side, hold, and execution remain explicit candidate metadata. The LLM only judges whether the described price-action setup is worth considering.

## Data

Source candidate pool with price-action extrema and rolling extrema:

- `data/event_action_compressor_ranker_all_2022_2026_paext_rex_2026-06-24.jsonl`

Chronological split generated for this experiment:

- train candidates: `results/event_action_compressor_ranker_paext_rex_train_2022_2024.jsonl`
- eval candidates: `results/event_action_compressor_ranker_paext_rex_eval_2025_2026.jsonl`

Exported binary-edge rows:

- train: `data/event_candidate_binary_edge_paext_rex_train_2022_2024_2026-06-30.jsonl`
- eval: `data/event_candidate_binary_edge_paext_rex_eval_2025_2026_2026-06-30.jsonl`
- summary: `results/event_candidate_binary_edge_paext_rex_2026-06-30/summary.json`

Label threshold: `reward utility >= 0.003` (~+0.3%) → `A`, otherwise `B`.

| split | period | rows | A/EDGE | B/NO_EDGE | mean prompt chars |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 2022–2024 | 87,680 | 17,492 | 70,188 | 2,982.5 |
| eval | 2025–2026 | 41,140 | 7,978 | 33,162 | 2,976.8 |

## Included signal-time features

Prompts include candidate side/hold/family, state buckets, and compact numeric evidence from:

- short/mid trend and range location
- taker/volume context
- DXY, kimchi premium, USDKRW context when available
- higher-timeframe returns
- price-action extrema over 36/72/144 bars
- rolling extrema over 36/144/576/2016/8640 bars

Future reward is not present in the prompt; it is used only to create the supervised target.

## Gemma base diagnostic

Corrected next-token option-logprob evaluator, balanced 512 eval sample:

- report: `results/event_candidate_binary_edge_paext_rex_2026-06-30/base_eval512_balanced_report.json`
- predictions: `results/event_candidate_binary_edge_paext_rex_2026-06-30/base_eval512_balanced_predictions.jsonl`

Result:

- accuracy: 0.51171875 (262/512)
- targets: A 256 / B 256
- predictions: A 394 / B 118
- by target: A 0.78125 / B 0.2421875

Margin analysis on the same balanced sample did not show a strong utility lift: top 5% by A-vs-B margin had target-A rate 0.621 but mean realized utility remained slightly negative. This base model is not directly tradable, but the surface is now compact enough for a controlled SFT experiment.

## Decision

Proceed with a small balanced Gemma SFT on this binary-edge surface, then evaluate on a chronological 2025–2026 holdout before any backtest. Promotion requires both:

1. Balanced classification improvement over base without predicting almost all `A`.
2. Positive realized utility/backtest after score-threshold selection on held-out rows only.
