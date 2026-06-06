# Compact h36 pressure Gemma4 POC (2026-06-07)

## Purpose

The h36/t0.5/s0.6 pressure label is learnable by a structured softmax baseline, but Gemma failed on the verbose prompt. This run compresses the analyzer input to compact past-only key-value features and retries the same 16-step Gemma POC.

## Data

- source: `data/economic_path_shape_h36_t0p5_s0p6_{train,val,oos}.jsonl`
- output: `data/economic_compact_pressure_analyzer_sft_h36_t0p5_s0p6_{train,val,oos}.jsonl`
- prompt mean chars: verbose `~2174` → compact `~1348` (~38% reduction)
- target: `{"direction_pressure": ...}`

## Training

- model: `google/gemma-4-E4B-it`
- checkpoint: `checkpoints/compact_pressure_analyzer_gemma4_e4b_h36_t0p5_s0p6_step16`
- rows: 512 balanced
- max steps: 16
- max seq length: 2048
- runtime: 111.3s
- train loss: 0.2759

## Generation evaluation

| model/input | val128 | OOS128 |
| --- | ---: | ---: |
| majority baseline | 36.05% | 35.51% |
| verbose Gemma h36 | 36.72% | 28.91% |
| compact Gemma h36 | 38.28% | 36.72% |
| structured softmax full split | 42.39% | 46.54% |

## Decision

Compact prompts improve Gemma and recover above majority on both val and OOS samples, but the model still underperforms the structured softmax teacher. This is a directional improvement, not a final analyzer.

## Next step

Use compact input as the default, then close the gap with the structured teacher:

1. train longer / use more rows on compact pressure SFT;
2. add teacher prediction/confidence/logit buckets from the train-only softmax;
3. evaluate full val/OOS, not just 128 samples, before connecting to trader/backtest.
