# Wave top5 take/skip Gemma4 check (2026-07-01)

## Purpose

Move away from raw pairwise setup choice and test a smaller LLM-shaped policy surface: a pre-generated wave candidate plus a compact state card, scored as a single-token A/B decision.

- `A = TAKE_TRADE`
- `B = SKIP_TRADE`

The target is label-only from realized candidate reward (`A` when the original row target was `TAKE_FULL` or `TAKE_SMALL`; `B` for `ABSTAIN`). Prompts do not include realized rewards.

## Data

| split | rows | A | B | positive reward rate | mean candidate reward |
|---|---:|---:|---:|---:|---:|
| train 2021-2024H1 | 2,173 | 1,133 | 1,040 | 52.14% | 0.0239% |
| eval 2024H2-2026 | 181 | 82 | 99 | 45.30% | 0.2778% |

Exports:

- `data/wave_state_top5_take_skip_option_train_2021_2024h1.jsonl`
- `data/wave_state_top5_take_skip_option_eval_2024h2_2026.jsonl`
- `data/wave_state_rex_top5_take_skip_option_train_2021_2024h1.jsonl`
- `data/wave_state_rex_top5_take_skip_option_eval_2024h2_2026.jsonl`

## Base Gemma4 results

| prompt | accuracy | pred A/B | margin filter | selected | compound return | CAGR | strict MDD | CAGR/MDD | mean trade | p-value |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline state | 45.30% | 77/104 | none | 77 | -1.00% | -0.63% | 26.37% | -0.02 | 0.003% | 0.990 |
| baseline state | 45.30% | 77/104 | abs margin >= 2.0 | 30 | 20.71% | 12.55% | 8.16% | 1.54 | 0.654% | 0.122 |
| REX state | 39.78% | 109/72 | none | 109 | 8.16% | 5.05% | 23.10% | 0.22 | 0.090% | 0.623 |
| REX state | 39.78% | 109/72 | abs margin >= 2.0 | 50 | 9.22% | 5.70% | 19.23% | 0.30 | 0.196% | 0.491 |

Base Gemma4 has a small confidence-filtered positive subset on baseline prompts, but only 30 trades and p≈0.122. This is a hypothesis, not enough for deployment.

## SFT run

LoRA SFT on baseline top5 take/skip rows:

- checkpoint: `checkpoints/wave_state_top5_take_skip_gemma4_sft_s128_2026-07-01`
- model: `google/gemma-4-E4B-it`
- rows: 2,048 balanced (`A=1024`, `B=1024`)
- LoRA: r=8, alpha=16, dropout=0.10, target all-linear
- LR: 5e-6
- steps: 128
- runtime: 874.9s
- train loss: 0.6547

Eval:

| model | accuracy | pred A/B | best checked margin | selected | compound return | CAGR | strict MDD | CAGR/MDD | mean trade | p-value |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SFT128 | 40.33% | 112/69 | none | 112 | -7.38% | -4.71% | 32.61% | -0.14 | -0.054% | 0.737 |
| SFT128 | 40.33% | 112/69 | abs margin >= 1.0 | 51 | 6.96% | 4.32% | 20.21% | 0.21 | 0.152% | 0.593 |
| SFT128 | 40.33% | 112/69 | abs margin >= 2.0 | 21 | -9.44% | -7.98% | 10.25% | -0.78 | -0.466% | 0.040 |

## Decision

Reject this SFT. It overfits the train labels and worsens eval accuracy/economics relative to base Gemma4. The best current LLM-shaped signal is still the base Gemma4 baseline-state high-margin subset, but it is not statistically strong enough.

Next direction: do not increase SFT steps on this same binary target. Instead, build a larger chronological candidate surface and/or use the LLM as a feature/rationale scorer whose output is calibrated by a non-LLM fold validator.
