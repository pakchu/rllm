# Event side-rationale preference Gemma4 DPO POC (2026-06-23)

## Purpose

The previous Gemma side-map attempts failed on short symbolic responses:

- 3-way `side_map`: collapsed to `UNRELIABLE`;
- 2-way `side_pair` SFT: collapsed to `NORMAL`;
- 2-way bare DPO: still collapsed to `NORMAL`.

This POC changes the response surface. Instead of comparing two short JSON labels, the model compares two evidence-rich causal rationales:

- `normal`: trust the generated side;
- `inverse`: invert the generated side.

The goal is to test whether giving the LLM actual language/evidence to compare can break the class-token prior.

## Data

Builder:

- `training/build_event_side_rationale_preference.py`

Evaluator:

- `training/eval_event_side_rationale_preference.py`

Source:

- `data/event_side_pair_h288_start2022_2026-06-23.jsonl`

Outputs:

- `data/event_side_rationale_preference_h288_start2022_2026-06-23.jsonl`
- `results/event_side_rationale_preference_h288_start2022_summary_2026-06-23.json`

Counts:

| Split | Pairs | Chosen normal | Chosen inverse |
| --- | ---: | ---: | ---: |
| full | 2,620 | 1,328 | 1,292 |
| 2026 eval | 191 | 92 | 99 |

Length profile:

| Field | Min chars | Max chars | Mean chars |
| --- | ---: | ---: | ---: |
| prompt | 947 | 1,060 | 988.3 |
| chosen response | 693 | 844 | 731.2 |

Leakage guard:

- prompt uses only signal-time score geometry and causal state tokens;
- candidate rationales are recomputed from signal-time tokens only;
- future realized side returns choose `chosen`/`rejected` for training labels only;
- tests verify `label_audit` future values do not enter prompt/response text.

## Training

Checkpoint:

- `checkpoints/event_side_rationale_gemma4_e4b_dpo16_2026-06-23`

Config:

| Field | Value |
| --- | --- |
| Model alias | `gemma4-e4b-it` (`google/gemma-4-E4B-it`) |
| Samples | 512 balanced |
| Chosen counts | normal 256 / inverse 256 |
| Rejected counts | normal 256 / inverse 256 |
| Max length | 2048 |
| LoRA | r=8, alpha=16, dropout=0.05 |
| DPO beta | 0.1 |
| Steps | 16 |
| Runtime | 287.7s |

Training evidence:

- final train loss: 0.6946;
- token accuracy around 0.58-0.62;
- reward margins were not stable enough to claim robust preference learning.

## 2026 eval

Rationale candidates were scored by log probability. The model did break the prior direction from the previous all-`NORMAL` collapse, but it over-corrected into almost all `INVERSE`.

| Score normalization | Accuracy | Prediction distribution |
| --- | ---: | --- |
| mean | 51.83% | 191 inverse / 0 normal |
| sum | 49.74% | 187 inverse / 4 normal |

## Strict replay

Both mean and sum produced the same strict replay metrics:

| Method | Trades | CAGR | Strict MDD | CAGR / strict MDD | Mean trade ret | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rationale DPO16 mean | 80 | 20.36% | 10.43% | 1.95 | 0.101% | 0.473 |
| rationale DPO16 sum | 80 | 20.36% | 10.43% | 1.95 | 0.101% | 0.473 |
| monthly history-majority baseline | 61 | 16.48% | 8.31% | 1.98 | n/a | n/a |
| token_signature memory baseline | 79 | 9.37% | 11.38% | 0.82 | n/a | n/a |

## Decision

Not deployable, but useful.

The richer rationale response surface changed the failure mode from all-`NORMAL` losing money to near-all-`INVERSE` with positive but statistically weak returns. This is not enough for the target and is not statistically significant, but it is the first Gemma-side experiment in this chain that produced a positive strict replay while using causal candidate rationales.

The result is still mostly a constant policy, not event-level discrimination. It should not be promoted as alpha.

## Next direction

Do not train longer blindly. The next step should remove response-template length/style bias before more DPO:

1. equalize normal/inverse rationale token length and wording symmetry;
2. evaluate base-model prior scores before training and subtract/normalize candidate prior bias;
3. train/evaluate on a score margin threshold, only trading when rationale score spread is large enough;
4. compare against explicit constant `always inverse` and `always normal` baselines over longer rolling windows.
