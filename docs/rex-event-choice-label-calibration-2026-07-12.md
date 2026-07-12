# REX event choice-label calibration (2026-07-12)

## Goal
Fix JSON completion length/format bias by replacing action JSON with canonical multiple-choice labels:

- `CHOICE_A_LONG`
- `CHOICE_B_SHORT`
- `CHOICE_C_SKIP`

Then apply train-only candidate-score calibration to remove label prior bias.

## Data
Source: `data/rex_event_reasoning_policy_sft_20260712.jsonl`

Converted dataset:
- `data/rex_event_choice_label_sft_20260712.jsonl`
- test split: `data/rex_event_choice_label_sft_test_20260712.jsonl`
- eval split: `data/rex_event_choice_label_sft_eval_20260712.jsonl`

Label counts:

| split | LONG | SHORT | SKIP |
|---|---:|---:|---:|
| train | 367 | 303 | 589 |
| test | 31 | 22 | 51 |
| eval | 21 | 21 | 39 |

## SFT POC
Base model: `google/gemma-2-2b-it`

Config:
- 384 balanced samples
- 32 steps
- LoRA r=16 alpha=32 dropout=0.05
- max seq length 1536

Training result:
- runtime: 100.7 sec
- train loss: 0.8716

Checkpoint dir is local only, weights not committed:
- `checkpoints/rex_event_choice_label_gemma2_2b_lora_s32_20260712`

## Raw candidate-logprob result
Mean and sum normalization were identical here: the model collapsed mostly to SHORT.

| split | accuracy | abs return | CAGR | strict MDD | CAGR/MDD | trades | behavior |
|---|---:|---:|---:|---:|---:|---:|---|
| test 2025 | 20.19% | -4.02% | -4.67% | 9.36% | -0.50 | 49 | SHORT collapse |
| eval 2026H1 | 25.93% | 0.78% | 2.94% | 4.87% | 0.60 | 28 | all SHORT |

## Train-only score calibration
Calibration fit uses train rows only.

Mean candidate scores on train:
- `CHOICE_A_LONG`: -2.4779
- `CHOICE_B_SHORT`: -2.3121
- `CHOICE_C_SKIP`: -2.5777

This confirms strong raw preference toward SHORT.

Calibrated result:

| split | accuracy | abs return | CAGR | strict MDD | CAGR/MDD | trades | side counts |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 36.93% | 42.85% | 7.57% | 18.17% | 0.42 | 399 | L216/S183 |
| test 2025 | 33.65% | -2.60% | -3.03% | 8.26% | -0.37 | 42 | L17/S25 |
| eval 2026H1 | 35.80% | 1.98% | 7.57% | 4.10% | 1.85 | 23 | L9/S14 |

## Verdict
Equal-form labels plus train-only calibration reduced the scoring-bias collapse, but did not create a profitable OOS policy.

Important conclusion:
- The remaining bottleneck is not just JSON length bias.
- 32-step SFT on the current labels learns format and some class balancing, but not robust event/side/no-trade boundaries.
- Further work should not just increase steps blindly. The target labeling objective likely needs redesign:
  - use pairwise/DPO preferences over actions instead of single hard class;
  - use reward-ranked action candidates with margin, not noisy argmax labels;
  - add confidence/abstention calibration using train-only validation thresholds;
  - separate trade/no-trade from side selection.

## Leakage guard
- Choice labels are target-only.
- Calibration uses train candidate scores only.
- Test/eval targets are used only for metrics.
