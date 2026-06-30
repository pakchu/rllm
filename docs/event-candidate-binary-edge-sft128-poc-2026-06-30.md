# Event Candidate Binary Edge Gemma SFT128 PoC — 2026-06-30

## Setup

Surface: candidate-level binary edge classification over paext/rex rows.

- train rows: `data/event_candidate_binary_edge_paext_rex_train_2022_2024_2026-06-30.jsonl`
- eval rows: `data/event_candidate_binary_edge_paext_rex_eval_2025_2026_2026-06-30.jsonl`
- model: `google/gemma-4-E4B-it`
- adapter: `checkpoints/event_candidate_binary_edge_gemma4_sft_s128_2026-06-30`
- train sample: 4,096 balanced rows, A 2,048 / B 2,048
- LoRA: r=8, alpha=16, dropout=0.10
- lr: 1e-5
- steps: 128
- runtime: 5,120 sec (~85 min)
- final train loss: 0.4747

## Balanced eval512 diagnostic

Balanced sampling is useful for class behavior but is not representative of deployment priors.

| model | acc | pred A/B | target A acc | target B acc |
| --- | ---: | --- | ---: | ---: |
| base | 0.5117 | 394 / 118 | 0.7813 | 0.2422 |
| sft_s128 | 0.4883 | 376 / 136 | 0.7227 | 0.2539 |

Balanced score-backtest looked promising, especially q0.85–q0.90:

| model | q | trades | CAGR | strict MDD | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.85 | 68 | 69.14 | 10.22 | 6.76 |
| base | 0.90 | 48 | 44.57 | 10.35 | 4.31 |
| sft_s128 | 0.85 | 66 | 81.78 | 10.22 | 8.00 |
| sft_s128 | 0.90 | 49 | 63.97 | 10.22 | 6.26 |

## Random eval2048 reality check

Random sampling preserves the real eval class prior: target A/B = 400 / 1,648.

| model | acc | pred A/B | target A acc | target B acc |
| --- | ---: | --- | ---: | ---: |
| base | 0.3408 | 1,594 / 454 | 0.8050 | 0.2282 |
| sft_s128 | 0.3730 | 1,492 / 556 | 0.7600 | 0.2791 |

The adapter improves random-sample classification slightly by reducing the A prior, but score-threshold backtests are negative across the sweep.

| model | q | trades | edge rate | mean label utility | CAGR | strict MDD | ratio | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.80 | 235 | 0.195 | -0.008511 | -34.68 | 47.60 | -0.73 | 0.0985 |
| base | 0.85 | 197 | 0.198 | -0.008669 | -26.66 | 38.01 | -0.70 | 0.1829 |
| base | 0.90 | 145 | 0.204 | -0.008457 | -21.96 | 34.33 | -0.64 | 0.2222 |
| base | 0.95 | 87 | 0.191 | -0.008503 | -12.25 | 25.11 | -0.49 | 0.4198 |
| sft_s128 | 0.80 | 255 | 0.207 | -0.007841 | -33.19 | 50.36 | -0.66 | 0.1276 |
| sft_s128 | 0.85 | 207 | 0.213 | -0.007475 | -19.78 | 39.46 | -0.50 | 0.4069 |
| sft_s128 | 0.90 | 152 | 0.195 | -0.008116 | -21.96 | 40.76 | -0.54 | 0.2416 |
| sft_s128 | 0.95 | 93 | 0.177 | -0.009042 | -21.01 | 35.55 | -0.59 | 0.1660 |

## Decision

Reject this adapter as a trading candidate. The balanced eval512 result was a sampling artifact: when real class priors return, the model still over-predicts `A` and the selected trades are negative after realistic backtest costs and entry delay.

## Lessons

1. Balanced classification can hide deployment-prior failure.
2. The LLM can learn the target token distribution but is not yet discovering an economic edge from the current prompt.
3. Backtest promotion must use random/chronological prior-preserving samples, not only balanced diagnostics.
4. The next useful direction is not more steps on this same target. Use either:
   - chronological/monthly score calibration with fit-only thresholds, or
   - a pairwise/ranking objective where positive and negative candidates are matched within the same regime/family so the model learns relative setup quality rather than global `EDGE` priors.
