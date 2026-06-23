# Event side-pair preference Gemma4 DPO POC (2026-06-23)

## Purpose

Bare SFT classification failed twice:

- 3-way `side_map` collapsed toward `UNRELIABLE`;
- 2-way `side_pair` collapsed toward `NORMAL`.

This POC changes the training objective from symbolic class prediction to pairwise preference learning. For each event, the correct side transform is `chosen` and the opposite transform is `rejected`.

## Data

Builder:

- `training/build_event_side_pair_preference.py`

Source:

- `data/event_side_pair_h288_start2022_2026-06-23.jsonl`

Output:

- `data/event_side_pair_preference_h288_start2022_2026-06-23.jsonl`
- `results/event_side_pair_preference_h288_start2022_summary_2026-06-23.json`

Counts:

| Split | Pairs | Chosen normal | Chosen inverse |
| --- | ---: | ---: | ---: |
| full | 2,620 | 1,328 | 1,292 |
| 2026 eval | 191 | 92 | 99 |

Each row keeps the same causal event prompt. Future realized side returns are used only to construct `chosen`/`rejected` for training labels, not as prompt tokens.

## Code changes

- `training/build_event_side_pair_preference.py`
  - writes DPO-style `prompt`, `chosen`, `rejected` rows;
  - uses the opposite side transform as rejected response.
- `training/train_text_dpo.py`
  - summary buckets now understand `{"side_pair":"..."}` responses instead of reporting `gate=None`.
- `tests/test_build_event_side_pair_preference.py`
  - verifies chosen/rejected construction and leakage guard markers.

## Training

Checkpoint:

- `checkpoints/event_side_pair_preference_gemma4_e4b_dpo16_2026-06-23`

Config:

| Field | Value |
| --- | --- |
| Model alias | `gemma4-e4b-it` (`google/gemma-4-E4B-it`) |
| Samples | 768 balanced |
| Chosen counts | normal 384 / inverse 384 |
| Rejected counts | normal 384 / inverse 384 |
| Max length | 1536 |
| LoRA | r=8, alpha=16, dropout=0.05 |
| DPO beta | 0.1 |
| Steps | 16 |
| Runtime | 245.4s |

Training evidence:

- train loss stayed near 0.694;
- rewards/accuracy varied but did not create a stable margin;
- this indicates the preference objective did not separate the two responses in this smoke setting.

## 2026 eval

Evaluation reused `side_pair` candidate-logprob scoring on the 191-row 2026 side-pair eval set.

| Score normalization | Accuracy | Prediction distribution |
| --- | ---: | --- |
| mean | 48.17% | 191 normal / 0 inverse |
| sum | 48.17% | 191 normal / 0 inverse |

DPO did not fix the class-token prior. It still selected `NORMAL` for every row.

## Strict replay

Applying DPO predictions to strict 2026 replay:

| Trades | CAGR | Strict MDD | CAGR / strict MDD | Mean trade ret | p approx |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 77 | -30.22% | 15.98% | -1.89 | -0.179% | 0.228 |

This is the same all-normal replay produced by pairwise SFT and remains below memory baselines.

## Decision

No-go for direct DPO over bare `side_pair` JSON responses.

The important result is narrower than “DPO does not work”: DPO over two nearly identical short JSON strings does not overcome the model/token prior. The LLM still is not being asked to use its natural strength: comparing substantive evidence.

## Next direction

The next valid RLLM/LLM structure should not score `{"side_pair":"normal"}` vs `{"side_pair":"inverse"}` directly. It should score richer candidate rationales, for example:

- response A: trust generated side because [compact causal state + similar prior event outcomes];
- response B: invert generated side because [compact causal state + similar prior event outcomes].

Then map the selected rationale to action. This creates a real language comparison target rather than a token-prior contest.
