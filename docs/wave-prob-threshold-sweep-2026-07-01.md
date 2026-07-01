# Cached wave probability threshold sweep (2026-07-01)

## Purpose

The LLM wave-state eval surface was too sparse after 2025. Before rebuilding the wave teacher, this sweep reuses cached wave probability rows and tests whether looser probability thresholds can increase candidate density while preserving out-of-sample quality.

## Inputs

- Test predictions: `results/wave_teacher_best_l1c005_2024h2_2025_predictions.jsonl` (52,704 15m rows)
- Eval predictions: `results/wave_teacher_best_l1c005_2026_jan_may_predictions.jsonl` (14,461 15m rows)
- Market: `data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz`
- Long thresholds: `0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68`
- Short thresholds: `0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46`
- Execution: hold 12 bars, entry delay 3, ATR trailing stop 3.75, 1x leverage.
- Selection: rank thresholds on 2024H2-2025 test only; 2026 Jan-May is holdout reporting only.

## Top test-ranked thresholds

| long | short | test trades | test CAGR | test MDD | test CAGR/MDD | test mean | test p | eval trades | eval CAGR | eval MDD | eval CAGR/MDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.66 | 0.32 | 52 | 12.12% | 6.91% | 1.75 | 0.342% | 0.111 | 5 | 22.34% | 1.15% | 19.46 |
| 0.68 | 0.32 | 41 | 6.27% | 6.00% | 1.05 | 0.233% | 0.317 | 4 | 4.53% | 1.07% | 4.23 |
| 0.66 | 0.34 | 65 | 7.32% | 6.91% | 1.06 | 0.175% | 0.369 | 5 | 11.10% | 2.53% | 4.38 |
| 0.64 | 0.32 | 83 | 9.49% | 10.65% | 0.89 | 0.173% | 0.252 | 6 | 22.66% | 2.75% | 8.25 |
| 0.66 | 0.36 | 72 | 6.26% | 6.91% | 0.91 | 0.138% | 0.446 | 6 | 17.82% | 2.87% | 6.21 |

## Decision

Loosening thresholds increases test trade count somewhat, but not enough in 2026. The best test-ranked threshold still yields only 5 eval trades from Jan-May 2026. This confirms the blocker: the current wave teacher probability surface is too sparse in recent data for statistically meaningful LLM/RL validation.

The next useful expansion is not another Gemma fine-tune. We need to generate denser candidate labels from raw probability rows (candidate rows before online cooldown/overlay), or rebuild the wave teacher with a wider policy family when dependencies are available. Those candidate labels can then be filtered by a separate chronological validator.
