# Event side-pair Gemma4 POC (2026-06-23)

## Purpose

The side-map Gemma experiments failed because the 3-way candidate surface mixed two different decisions:

1. side transform: `NORMAL` vs `INVERSE`;
2. abstention/reliability: `UNRELIABLE`.

`UNRELIABLE` dominated candidate logprob. This POC removes `UNRELIABLE` from side selection and trains a pairwise `side_pair` head with only `normal` and `inverse` labels.

## Data

Builder:

- `training/build_event_side_pair_sft.py`

Source:

- `data/event_side_map_reliability_h288_start2022_2026-06-23.jsonl`

Output:

- `data/event_side_pair_h288_start2022_2026-06-23.jsonl`
- `results/event_side_pair_h288_start2022_summary_2026-06-23.json`

Counts:

| Split | Rows | Normal | Inverse | Skipped unreliable |
| --- | ---: | ---: | ---: | ---: |
| full | 2,620 | 1,328 | 1,292 | 172 |
| 2026 eval | 191 | 92 | 99 | n/a |

Leakage guard:

- labels are projected from existing event audit labels;
- eval rows are 2026 only;
- generated rows retain source metadata and add `target_projected_to_pairwise_side_map`.

## Code changes

- `training/build_event_side_pair_sft.py`
  - filters out unreliable labels;
  - writes `{"side_pair":"normal|inverse"}` targets.
- `training/eval_text_json_key.py`
  - supports `side_pair` candidate-logprob evaluation with values `NORMAL` / `INVERSE`.
- `training/train_text_sft.py`
  - reports `side_pair=*` target counts in SFT summaries.
- `tests/test_build_event_side_pair_sft.py`
  - verifies unreliable filtering and target projection.

## Training

Checkpoint:

- `checkpoints/event_side_pair_gemma4_e4b_sft16_2026-06-23`

Config:

| Field | Value |
| --- | --- |
| Model alias | `gemma4-e4b-it` (`google/gemma-4-E4B-it`) |
| Samples | 768 balanced |
| Sample labels | normal 400 / inverse 368 |
| Max sequence length | 1536 |
| LoRA | r=8, alpha=16, dropout=0.05 |
| Steps | 16 |
| Runtime | 102s |

Training loss decreased from 6.993 to 4.862, but token accuracy stayed weak around 0.44-0.51.

## 2026 eval

Both candidate-logprob normalizations collapsed to `NORMAL` for every row.

| Score normalization | Accuracy | Prediction distribution |
| --- | ---: | --- |
| mean | 48.17% | 191 normal / 0 inverse |
| sum | 48.17% | 191 normal / 0 inverse |

This exactly matches the normal share of the eval set, so it is not evidence of learned side discrimination.

## Strict replay

Applying the all-normal pairwise output to the 2026 rows:

| Trades | CAGR | Strict MDD | CAGR / strict MDD | Mean trade ret | p approx |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 77 | -30.22% | 15.98% | -1.89 | -0.179% | 0.228 |

This is worse than the current memory baselines:

| Baseline | Trades | CAGR | Strict MDD | Ratio |
| --- | ---: | ---: | ---: | ---: |
| token_signature memory | 79 | 9.37% | 11.38% | 0.82 |
| monthly history-majority | 61 | 16.48% | 8.31% | 1.98 |

## Decision

No-go for direct pairwise SFT16.

The experiment did fix the `UNRELIABLE` attractor, but exposed another problem: Gemma candidate logprob prefers the `normal` token/string prior rather than learning side-map inversion from the current prompt. This means the next useful LLM change is not more steps on this prompt. The next surface should make the model compare two explicit event narratives or rank two candidate outcomes, not classify a bare label.

## Next direction

Use LLM where it has a structural advantage:

1. create paired prompts containing `candidate_A=normal` and `candidate_B=inverse` with realized prior-training evidence summaries;
2. train/evaluate a preference/ranking target instead of a single symbolic class token;
3. keep abstention/reliability as a separate calibrated statistical head;
4. only replay candidates that beat the `always normal`, `token_signature`, and `monthly history_majority` baselines on 2026 and longer rolling windows.
