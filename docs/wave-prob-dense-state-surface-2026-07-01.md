# Dense wave-probability state surface (2026-07-01)

## Purpose

The top5 wave-state surface was too sparse for recent validation. This builds a denser candidate-label surface directly from cached wave teacher probability rows, before online cooldown/overlay removes most trades.

## Data construction

- Train source: `results/wave_teacher_best_l1c005_2024h2_2025_predictions.jsonl`
- Eval source: `results/wave_teacher_best_l1c005_2026_jan_may_predictions.jsonl`
- Candidate inclusion:
  - LONG when `teacher_probability_long >= 0.54`
  - SHORT when `teacher_probability_long <= 0.46`
- Label reward: fixed delayed-open hold, 12 bars, entry delay 3, fee+slippage included.
- Prompt: signal-time state buckets only; reward is label/audit only.

Outputs:

- `data/wave_prob_dense_take_skip_train_2024h2_2025.jsonl`
- `data/wave_prob_dense_take_skip_eval_2026_jan_may.jsonl`
- `data/wave_prob_dense_take_skip_summary_2026-07-01.json`

| split | rows | A/take labels | B/skip labels | LONG | SHORT | mean fixed-hold reward | positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 2024H2-2025 | 8,582 | 3,836 | 4,746 | 6,217 | 2,365 | -0.0906% | 44.70% |
| eval 2026 Jan-May | 1,121 | 535 | 586 | 668 | 453 | -0.0635% | 47.73% |

This solves the sample-size problem for representation testing, but the raw dense candidate pool is negative.

## Train-only token rule sanity check

A train-only additive token scorer was fit on 2024H2-2025 and thresholds were frozen from train score quantiles. Eval rewards were reporting only.

| train-score q | eval selected | fixed-hold CAGR | fixed-hold MDD | CAGR/MDD | mean trade | p-value |
|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 96 | 57.81% | 14.75% | 3.92 | 0.151% | 0.250 |
| 0.80 | 203 | 5.57% | 21.35% | 0.26 | 0.014% | 0.848 |
| 0.70 | 289 | -0.85% | 22.35% | -0.04 | 0.003% | 0.952 |
| 0.50 | 474 | -8.39% | 24.97% | -0.34 | -0.004% | 0.924 |
| 0.00 | 1,121 | -84.07% | 53.77% | -1.56 | -0.063% | 0.010 |

The q0.90 fixed-hold subset looks promising by CAGR/MDD, but it is not statistically significant and it is based on a simplified fixed-hold reward model.

## Strict overlay check

The q0.90 eval selections were converted back into trade predictions and replayed with the stricter online overlay / ATR trailing-stop backtester.

- Predictions: `results/wave_prob_dense_token_q90_eval2026_predictions.jsonl`
- Strict replay: `results/wave_prob_dense_token_q90_eval2026_strict_overlay.json`

| selected samples | executed trades | strict CAGR | strict MDD | CAGR/MDD | mean trade | p-value |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 51 | -15.99% | 12.17% | -1.31 | -0.094% | 0.606 |

## Decision

The dense surface is useful for LLM/RL training data volume, but the current label/execution mismatch is a fundamental problem: fixed-hold labels can look profitable while strict live-style ATR/cooldown replay loses money.

Next work should align labels with the execution simulator before any Gemma fine-tune. Specifically, dense candidates need labels from the same strict execution assumptions or a candidate-level reward model that accounts for overlap/cooldown/ATR exits.
