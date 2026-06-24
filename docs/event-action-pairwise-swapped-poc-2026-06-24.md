# Event-action pairwise swapped POC — 2026-06-24

## Purpose

The first pairwise Gemma4 SFT32 POC showed a strong A-position prior:

- train1024 raw accuracy: 54.1%, predictions A/B = 878/146
- eval1024 raw accuracy: 52.3%, predictions A/B = 908/116
- train-calibrated eval1024 accuracy: 53.0%, predictions A/B = 675/349

This POC tested whether emitting both A/B orientations for each semantic pair would force the model to compare candidate content rather than rely on position.

## Data change

Added `--emit-swapped-duplicates` to `training.event_action_pairwise_rank_data`.

For each selected semantic comparison `(chosen, rejected)`, the builder emits both:

1. chosen as Candidate A, rejected as Candidate B, target `A`
2. rejected as Candidate A, chosen as Candidate B, target `B`

Dry-run sizes:

| split | pairs | signals | target A | target B |
| --- | ---: | ---: | ---: | ---: |
| train pre-2026 swapped | 70,014 | 5,835 | 35,007 | 35,007 |
| eval 2026 swapped | 7,164 | 597 | 3,582 | 3,582 |

## SFT32 setup

- model: `google/gemma-4-E4B-it`
- train jsonl: `data/event_action_pairwise_rank_train_pre2026_swapped_2026-06-24.jsonl`
- samples: 2,048 balanced
- steps: 32
- LoRA: r=16, alpha=32, dropout=0.05
- runtime: 188.2s
- train loss: 0.4918

## Results

| eval | accuracy | pred A | pred B | note |
| --- | ---: | ---: | ---: | --- |
| train1024 raw | 51.66% | 895 | 129 | worse than non-swapped |
| eval1024 raw | 48.93% | 893 | 131 | below random |
| eval1024 train-calibrated | 49.71% | 933 | 91 | calibration did not help |

Train-calibrated threshold was selected only from train1024 scores: `-0.07816171646118164`.

## Conclusion

Swapped duplicate data did **not** remove the A-position prior under the current SFT/logprob evaluation path. It likely made the prompt-label mapping harder while the model still retained a base preference for the first short label/action position.

Do not spend more time on A/B label SFT as the primary alpha path. The failure is now twofold:

1. binary TAKE/SKIP value SFT failed full 2026 strict backtest,
2. pairwise A/B SFT remains dominated by position/token prior even when semantic pairs are swapped.

## Next direction

The next LLM-compatible target should avoid direct A/B token competition. Better candidates:

1. **candidate-wise ordinal utility class**: LOW/MID/HIGH/AVOID per candidate, calibrated on train only;
2. **listwise JSON ranking** with candidate IDs that are not `A/B` and are randomly permuted, then parsed by ID;
3. **LLM feature compressor only**: have Gemma output compact qualitative tags, then train a small non-LLM ranker on those tags plus numeric price-action features;
4. **utility teacher distillation**: use future utility to train a non-LLM teacher/ranker, then ask LLM to explain/compress the teacher decision, not make the final numeric comparison directly.

The most practical next unit is candidate-wise ordinal utility class because it reuses the existing fast label scorer and avoids paired position priors.
