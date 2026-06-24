# Event-action neutral Q-code POC — 2026-06-24

## Purpose

Semantic ordinal labels failed because base Gemma4 strongly preferred `AVOID` over other labels. Prior audit showed `Q1,Q2,Q3,Q4` had much smaller mean-score spread than semantic labels, so this POC remapped:

- `AVOID -> Q1`
- `LOW -> Q2`
- `MID -> Q3`
- `HIGH -> Q4`

The prompt asks for codes, not semantic output labels.

## Data

Source: ordinal utility rows from 2026-06-24.

| split | rows | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train pre-2026 | 116,880 | 40,738 | 55,677 | 12,764 | 7,701 |
| eval 2026 | 11,940 | 4,190 | 5,700 | 1,401 | 649 |

Training used balanced sampling:

- samples: 4,096
- Q1/Q2/Q3/Q4: 1,024 each
- max sequence length: 1,792
- Gemma4 E4B LoRA r=16 alpha=32 dropout=0.05
- steps: 32

## Training

- runtime: 180.2s
- train loss: 0.7861
- token accuracy in later steps: roughly 0.75-0.91

This converged better than semantic ordinal SFT, whose train loss was 1.087 and collapsed to AVOID under logprob scoring.

## Balanced 256 evaluation

Raw logprob argmax:

| split | accuracy | prediction distribution |
| --- | ---: | --- |
| train balanced 256 | 30.86% | Q1 32 / Q2 196 / Q3 27 / Q4 1 |
| eval balanced 256 | 27.34% | Q1 52 / Q2 186 / Q3 17 / Q4 1 |

Q2 prior still dominates.

Train-only mean-score centering:

| split | accuracy | mean abs rank error | prediction distribution |
| --- | ---: | ---: | --- |
| train balanced 256 | 32.42% | 1.219 | Q1 89 / Q2 73 / Q3 61 / Q4 33 |
| eval balanced 256 | 32.03% | 1.121 | Q1 116 / Q2 99 / Q3 11 / Q4 30 |

The centering constants came only from train balanced 256 label means. Eval scores were not used to calibrate.

## Interpretation

Neutral Q-code labels are an improvement over semantic ordinal labels:

- semantic ordinal raw train-balanced accuracy: 25%, all AVOID;
- Q-code raw eval-balanced accuracy: 27.3%, still Q2-biased;
- Q-code train-centered eval-balanced accuracy: 32.0%, no longer single-label collapse.

This is still not enough to claim trading alpha. It is only a weak learnability signal. However, it is the first LLM-direct target in this round that did not immediately collapse to a single semantic/position prior after train-only correction.

## Next step

Do not run full backtest yet. The next useful unit is to turn Q-code scores into a candidate selector on a small eval slice and compare:

1. raw Q-code score selector;
2. train-centered Q-code selector;
3. oracle utility upper bound on the same candidate set;
4. random/action-family baselines.

If Q-code selector cannot improve candidate utility ranking even on balanced or small held-out slices, move to LLM-as-feature-compressor plus non-LLM ranker.
