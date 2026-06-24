# Event-action recent-window ranker tests — 2026-06-24

## Goal

Feature drift audit showed that 2026 is not globally worse, but several price-action/direction features flip sign. Test whether using only recent train windows adapts feature signs better than all-history fitting.

All runs use validation-only selection. 2026 eval is diagnostic only.

## Recent-window datasets

Generated with `training.filter_jsonl_by_date` from `data/event_action_compressor_ranker_train_pre2026_2026-06-24.jsonl`.

| Dataset | Rows | First date | Last date |
| --- | ---: | --- | --- |
| `data/event_action_compressor_ranker_train_2024_2025_2026-06-24.jsonl` | 58,480 | 2024-01-01 02:55:00 | 2025-12-31 20:55:00 |
| `data/event_action_compressor_ranker_train_2023_2025_2026-06-24.jsonl` | 87,680 | 2023-01-01 02:55:00 | 2025-12-31 20:55:00 |
| `data/event_action_compressor_ranker_train_2025_2026-06-24.jsonl` | 29,200 | 2025-01-01 02:55:00 | 2025-12-31 20:55:00 |

## Results

| Train window | Fit period | Validation period | Model | Val trades | Val CAGR/MDD | Eval trades | Eval CAGR | Eval strict MDD | Eval CAGR/MDD | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2024-2025 | 2024 | 2025 | ridge | 292 | 0.09 | 89 | -8.36% | 4.87% | -1.72 | fail |
| 2024-2025 | 2024 | 2025 | pairwise q/m grid | 41 | 0.18 | 20 | 12.27% | 15.73% | 0.78 | weak positive, not enough |
| 2024-2025 | 2024 | 2025 | pairwise high-q grid | 41 | 0.49 | 20 | 12.70% | 15.60% | 0.81 | weak positive, not enough |
| 2023-2025 | 2023-2024 | 2025 | pairwise high-q grid | 57 | 0.58 | 29 | -28.79% | 19.60% | -1.47 | fail |
| 2025 | 2025 H1 | 2025 H2 | pairwise high-q grid | 10 | -0.74 | 9 | 34.79% | 12.78% | 2.72 | reject: validation negative, tiny eval n |

## Interpretation

1. All-history and 2023-2025 fitting are stale for 2026.
2. 2024-only fit with 2025 validation is the first path producing positive 2026 eval, but it is weak: only 20 trades, CAGR/MDD < 1, and MDD slightly above 15.
3. 2025-only fitting can accidentally look strong on 2026, but the validation period is negative and eval has only 9 trades. This is not tradable evidence.
4. The signal is probably short-half-life regime adaptation, not a stable global alpha.

## Next step

Do not tune directly on 2026. Build a rolling walk-forward selector that:

- trains only on a recent lookback window before each validation/eval slice,
- requires positive validation evidence before trading the next slice,
- abstains when validation is weak/negative,
- aggregates multiple validation/eval slices to increase trade count.

This is the right structure before spending more compute on LLM compressor fine-tuning.
