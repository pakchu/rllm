# Event candidate A/B/C Gemma SFT PoC — 2026-06-30

## Purpose

Train a small Gemma 4 E4B LoRA SFT on the compact A/B/C option-choice surface and test whether it fixes the base
model's trade-option bias, especially the failure to choose `C = NO_TRADE`.

## Training

- Dataset: `data/event_candidate_option_choice_wavefull_ext_micro_c72_s2_train_2026-06-29.jsonl`
- Adapter: `checkpoints/event_candidate_option_choice_gemma4_sft_s64_2026-06-29`
- Samples: 1,024 balanced
- Target mix: A 341, B 341, C 342
- Steps: 64
- LoRA: r16 alpha32 dropout 0.05
- Max seq length: 2048

## Held-out option-logprob benchmark

Balanced eval256, seed 42, same rows as base benchmark.

| Model | Accuracy | Correct | Prediction A | Prediction B | Prediction C |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Gemma 4 E4B | 34.77% | 89/256 | 161 | 95 | 0 |
| A/B/C SFT s64 | 36.33% | 93/256 | 223 | 22 | 11 |

Accuracy by target:

| Model | A/LONG | B/SHORT | C/NO_TRADE |
| --- | ---: | ---: | ---: |
| Base | 63.95% | 40.00% | 0.00% |
| SFT s64 | 90.70% | 12.94% | 4.71% |

## Decision

Do not promote the s64 adapter to backtesting. It improves aggregate accuracy only slightly and worsens action
calibration by collapsing toward `A = LONG`. The important finding is that single-token option scoring is measurable
and trainable, but class priors/calibration must be handled before policy use.

## Next step

Apply score calibration on held-out validation/test logits before any backtest:

1. Estimate option prior offsets on a validation subset.
2. Apply offsets to eval option scores.
3. Require balanced non-degenerate predictions and accuracy materially above random before policy conversion.

If calibration cannot recover B/C without destroying accuracy, the prompt needs stronger option descriptions or
shorter feature summaries.
