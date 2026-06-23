# Event-level side-map reliability labels (2026-06-23)

## Purpose

Monthly side-map labels were too coarse: 2026 contains unreliable, inverse, and normal months. This test moves the reliability target to each generated trade proposal.

For each TRADE proposal:

- `normal`: generated side is profitable and better than the flipped side;
- `inverse`: flipped side is profitable and better than generated side;
- `unreliable`: neither side is profitable enough.

The prompt/features are causal; the label uses future realized returns for supervised training/evaluation only.

## Implementation

Added:

- `training/event_side_map_reliability_dataset.py`
- `training/eval_event_side_map_memory.py`
- `tests/test_event_side_map_reliability_dataset.py`
- `tests/test_eval_event_side_map_memory.py`

Generated dataset:

- `data/event_side_map_reliability_h288_start2022_2026-06-23.jsonl`
- `results/event_side_map_reliability_h288_start2022_summary_2026-06-23.json`

Rows: 2,792 trade proposals.

Split/class distribution:

| Split | normal | inverse | unreliable |
| --- | ---: | ---: | ---: |
| train 2022-2024 | 954 | 914 | 129 |
| val 2025 | 282 | 279 | 33 |
| eval 2026-01..2026-05 | 92 | 99 | 10 |

## Memory baselines

Train memory: train+val rows. Eval: 2026 rows.

| Method | Label accuracy | Trades | CAGR | Strict MDD | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| score_only | 47.76% | 79 | 1.36% | 12.35% | 0.11 |
| core_state | 49.25% | 78 | -28.39% | 14.78% | -1.92 |
| token_signature | 49.75% | 79 | 9.37% | 11.38% | 0.82 |

## Decision

Event-level labels produce enough rows for a real LLM/SFT experiment, but simple memory baselines are not sufficient.

Important implications:

- The side-map target is not trivially learnable from current coarse token signatures.
- Event-level data is much better sized than the 53-month dataset for Gemma fine-tuning.
- However, current eval replay is still below the monthly history-majority baseline ratio 1.98 and far below target ratio 3.

Next step:

Run a small Gemma/LoRA proof-of-concept on the event-level side-map dataset, but treat it as a learnability test. It must beat the token_signature memory baseline and monthly history-majority replay before becoming part of the trading stack.
