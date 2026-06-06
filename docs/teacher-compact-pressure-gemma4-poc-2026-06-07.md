# Teacher-compact pressure Gemma4 POC (2026-06-07)

## Purpose

Close the gap between compact Gemma and the structured softmax baseline by adding a train-only teacher hint to the compact prompt.

The teacher is fit only on train split structured features, then its predicted pressure/confidence bucket is appended to train/val/OOS prompts. OOS labels remain report-only.

## Data

Ignored local artifacts:

- `data/economic_teacher_compact_pressure_analyzer_sft_h36_t0p5_s0p6_{train,val,oos}.jsonl`

Prompt mean chars:

- compact: ~1348
- teacher-compact: ~1458
- verbose original: ~2174

## Training

- model: `google/gemma-4-E4B-it`
- checkpoint: `checkpoints/teacher_compact_pressure_analyzer_gemma4_e4b_h36_t0p5_s0p6_step16`
- rows: 512 balanced
- max steps: 16
- max seq length: 2048
- runtime: 108.9s
- train loss: 0.2505

## Generation comparison on 128 balanced samples

| analyzer input | val128 | OOS128 |
| --- | ---: | ---: |
| majority baseline | 36.05% | 35.51% |
| verbose Gemma h36 | 36.72% | 28.91% |
| compact Gemma h36 | 38.28% | 36.72% |
| teacher-compact Gemma h36 | 46.09% | 44.53% |
| structured softmax full split | 42.39% | 46.54% |

## Decision

This is the first materially useful LLM analyzer result in the current branch. The LLM benefits from compact features plus train-only teacher hints and approaches the structured baseline while keeping a text reasoning/interface layer.

It is not yet a trading result. Next validation must run full val/OOS generation and then connect analyzer predictions to the trader/backtest path.

## Next step

1. Run full val/OOS teacher-compact generation.
2. Convert predicted pressure to trader prompts/actions.
3. Strict backtest the resulting stop/target template actions before any larger SFT.
