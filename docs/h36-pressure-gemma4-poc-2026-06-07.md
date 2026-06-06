# h36/t0.5/s0.6 pressure Gemma4 POC (2026-06-07)

## Why this run

The learnability sweep found that `horizon=36`, `target=0.5%`, `stop=0.6%` is the best pressure label definition for the existing feature set:

- softmax val: 42.39% vs majority 36.05% (+6.34pp)
- softmax OOS: 46.54% vs majority 35.51% (+11.03pp)

## Data generated

Ignored local artifacts:

- `data/economic_path_shape_h36_t0p5_s0p6_{train,val,oos}.jsonl`
- `data/economic_pressure_analyzer_sft_h36_t0p5_s0p6_{train,val,oos}.jsonl`

Pressure distribution:

| split | rows | majority | LONG | SHORT | NO_TRADE | BOTH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2370 | 33.92% | 804 | 770 | 723 | 73 |
| val | 552 | 36.05% | 173 | 164 | 199 | 16 |
| OOS | 535 | 35.51% | 156 | 190 | 176 | 13 |

## Gemma4 POC

- model: `google/gemma-4-E4B-it`
- checkpoint: `checkpoints/pressure_analyzer_gemma4_e4b_h36_t0p5_s0p6_step16`
- rows: 512 balanced
- max steps: 16
- runtime: 113.5s
- train loss: 0.3497

Generation evaluation:

| split | samples | Gemma acc | full-split majority | softmax baseline |
| --- | ---: | ---: | ---: | ---: |
| val | 128 | 36.72% | 36.05% | 42.39% |
| OOS | 128 | 28.91% | 35.51% | 46.54% |

## Decision

Do not scale this verbose-summary Gemma pressure analyzer. The label is learnable by cheap structured features but not by the current long natural-language prompt in a short SFT run.

## Next move

Compress the analyzer input to the fields that the softmax baseline uses effectively, then retry SFT:

- remove long `recent_bar_sequence` and unused names
- provide compact key-value facts only
- optionally distill softmax predictions/logits as a teacher target or auxiliary confidence

The promising path is **not** more steps on the same verbose prompt; it is compact feature distillation for the LLM analyzer.
