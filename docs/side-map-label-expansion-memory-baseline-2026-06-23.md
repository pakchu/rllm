# Side-map label expansion and memory baseline (2026-06-23)

## Purpose

The initial 29-month side-map dataset was too small for Gemma fine-tuning. This expands labels back to 2022 and checks whether the target is learnable before spending time on an LLM head.

## Expanded rolling predictions

Generated h288 pairwise predictions from 2022-01 through 2026-05:

- `results/rolling_event_context_preference_h288_start2022_predictions_2026-06-23.jsonl`
- `results/rolling_event_context_preference_h288_start2022_summary_2026-06-23.json`
- `results/rolling_event_context_preference_h288_start2022_backtest_2026-06-23.json`

Full 2022-2026 replay is not deployable:

| Months | Rows | Trades | CAGR | Strict MDD | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 53 | 6,442 | 945 | -8.57% | 44.74% | -0.19 |

The purpose was label expansion, not trading this raw policy.

## Expanded monthly side-map labels

Generated:

- `results/monthly_side_map_reliability_h288_start2022_tp3_2026-06-23.json`

Distribution over 2022-01 through 2026-05:

| Label | Months |
| --- | ---: |
| normal | 23 |
| inverse | 14 |
| unreliable | 16 |

The 2024-2026 labels match the previous 29-month audit.

## Expanded SFT rows

Generated:

- `data/side_map_reliability_sft_h288_start2022_2026-06-23.jsonl`
- `results/side_map_reliability_sft_h288_start2022_summary_2026-06-23.json`

Split/class distribution:

| Split | normal | inverse | unreliable |
| --- | ---: | ---: | ---: |
| train 2022-2024 | 17 | 8 | 11 |
| val 2025 | 5 | 5 | 2 |
| eval 2026-01..2026-05 | 1 | 1 | 3 |

## Memory baseline

Added:

- `training/eval_side_map_reliability_memory.py`
- `training/apply_side_map_memory_predictions.py`

Memory classifier accuracy on 2026 eval labels:

| Method | Accuracy |
| --- | ---: |
| global_majority | 1/5 = 20% |
| last_history | 0/5 = 0% |
| history_majority | 2/5 = 40% |
| bucket_majority | 3/5 = 60% |

Strict trading replay on 2026 eval:

| Method | Transform behavior | Trades | CAGR | Strict MDD | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| bucket_majority | all unreliable/block | 0 | 0.00% | 0.00% | 0.00 |
| history_majority | mostly inverse, May unreliable | 61 | 16.48% | 8.31% | 1.98 |
| global_majority | all normal | 77 | -26.32% | 15.70% | -1.68 |

## Decision

This is not target-achieving, but it is a useful signal:

- side-map history contains more useful information than global normal or validation-score-only transforms;
- a simple history-majority head reduces drawdown and improves 2026 eval ratio to 1.98;
- still below target ratio 3 and CAGR 50, so it is not deployable.

Next step:

- add richer prior-only monthly state from wave_trading/Binance auxiliary data to the side-map head dataset;
- then test a small Gemma/LLM classifier against memory baselines.
