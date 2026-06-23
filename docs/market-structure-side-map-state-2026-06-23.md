# Market-structure side-map state test (2026-06-23)

## Purpose

Funding/premium buckets did not improve side-map reliability prediction. This test adds prior-month BTC market-structure buckets that should be more directly related to side-map regime:

- prior-month return;
- realized volatility;
- high/low range;
- max drawdown;
- close position inside the monthly range.

All features use only the month before the target month.

## Implementation

Added:

- `training/augment_side_map_sft_with_market_structure.py`
- `tests/test_augment_side_map_sft_with_market_structure.py`

Generated:

- `data/side_map_reliability_sft_h288_start2022_market_structure_2026-06-23.jsonl`
- `results/side_map_reliability_sft_h288_start2022_market_structure_summary_2026-06-23.json`

Updated `training/eval_side_map_reliability_memory.py` with:

- `market_majority`
- `combined_market_bucket_majority`

## Memory result

Eval label accuracy on 2026-01 through 2026-05:

| Method | Accuracy |
| --- | ---: |
| bucket_majority | 3/5 = 60% |
| history_majority | 2/5 = 40% |
| market_majority | 2/5 = 40% |
| combined_market_bucket_majority | 2/5 = 40% |

The market buckets failed to identify the important exceptions:

- 2026-02 should be `inverse`, predicted `unreliable`;
- 2026-04 should be `normal`, predicted `inverse`.

Strict trading replay:

| Method | Trades | CAGR | Strict MDD | Ratio |
| --- | ---: | ---: | ---: | ---: |
| market_majority | 21 | -9.24% | 5.90% | -1.57 |
| combined_market_bucket_majority | 21 | -9.24% | 5.90% | -1.57 |
| history_majority baseline | 61 | 16.48% | 8.31% | 1.98 |

## Decision

No-go for coarse prior-month market-structure buckets.

The result reinforces that the current monthly-classification setup is too coarse. The side-map head likely needs either:

1. more granular intra-month state around the target decision period;
2. a richer sequence model over recent monthly/weekly states;
3. or a direct event-level reliability label instead of one label per month.

Do not spend Gemma fine-tune time on the current coarse monthly feature set expecting target-level performance.
