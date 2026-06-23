# Event side-map Gemma4 SFT POC (2026-06-23)

## Purpose

After moving side-map reliability from monthly labels to 2,792 event-level labels, we ran a minimal Gemma4 LoRA SFT proof-of-concept. The goal was not target performance yet; it was to see whether Gemma can beat simple memory baselines on the event-level reliability head.

## Supporting code changes

Added/updated:

- `training/eval_text_json_key.py`
  - supports `side_map` key;
  - candidate-logprob uses lowercase JSON values for `side_map` because the SFT targets use lowercase values.
- `training/train_text_sft.py`
  - target counter now reports `side_map=*` counts.
- `training/apply_event_side_map_text_predictions.py`
  - converts text-eval `side_map` predictions into normal/inverse/unreliable trade transforms and strict replay.

## Dry-run

Command used `gemma4-e4b-it`, 512 balanced samples, LoRA r=8, max seq length 1536, max steps 8.

Dry-run summary:

- rows: 512
- target counts:
  - normal: 265
  - inverse: 223
  - unreliable: 24
- prompt chars mean: 1,386
- prompt chars max: 1,477

## Training

Checkpoint:

- `checkpoints/event_side_map_gemma4_e4b_sft8_2026-06-23`

Training stats:

- max steps: 8
- runtime: 51.54s
- train loss: 4.306
- token accuracy around 0.54-0.61 during the tiny run

## Evaluation on 2026 event rows

Eval data:

- `data/event_side_map_reliability_h288_start2022_eval2026_2026-06-23.jsonl`
- 201 event rows

Candidate-logprob results:

| Score normalization | Label accuracy | Prediction collapse |
| --- | ---: | --- |
| mean | 4.98% | all `UNRELIABLE` |
| sum | 5.97% | 195 unreliable, 5 normal, 1 inverse |

Strict replay:

| Method | Trades | CAGR | Strict MDD | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Gemma4 SFT8 mean-logprob | 0 | 0.00% | 0.00% | 0.00 |
| Gemma4 SFT8 sum-logprob | 6 | -14.53% | 6.68% | -2.17 |
| token_signature memory baseline | 79 | 9.37% | 11.38% | 0.82 |
| monthly history-majority baseline | 61 | 16.48% | 8.31% | 1.98 |

## Decision

No-go for the tiny Gemma4 SFT8 POC.

The model collapsed toward `UNRELIABLE` and did not beat either memory baseline. This does not rule out LLM use, but it shows the current prompt/label/training setup is not sufficient.

Next changes should focus on:

1. balancing or oversampling `unreliable` carefully without letting candidate scoring collapse;
2. shorter/compressed prompts with fewer noisy tokens;
3. evaluating generation and calibrated candidate scores, not just raw candidate logprob;
4. increasing steps only after a smoke config beats memory baselines.

Do not integrate this checkpoint into the trading stack.
