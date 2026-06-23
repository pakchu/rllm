# Binance aux side-map state test (2026-06-23)

## Purpose

The side-map reliability head needs richer prior-only state than validation score and label history. This test adds prior-month Binance BTC futures funding/premium buckets to the side-map SFT prompts.

## Data used

Local repo data:

- `data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz`
- `data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz`

`../wave_trading` was not present in this environment, so this used the already-cached aux files in `rllm/data`.

## Implementation

Added:

- `training/augment_side_map_sft_with_binance_aux.py`
- `tests/test_augment_side_map_sft_with_binance_aux.py`

Generated:

- `data/side_map_reliability_sft_h288_start2022_binance_aux_2026-06-23.jsonl`
- `results/side_map_reliability_sft_h288_start2022_binance_aux_summary_2026-06-23.json`

Prompt tokens added from the month before the target month:

- `prior_btc_premium_mean`
- `prior_btc_premium_abs`
- `prior_btc_funding_mean`
- `prior_btc_funding_abs`

## Memory baseline result

Updated `training/eval_side_map_reliability_memory.py` with aux-aware majority baselines.

Eval label accuracy on 2026-01 through 2026-05:

| Method | Accuracy |
| --- | ---: |
| global_majority | 1/5 = 20% |
| last_history | 0/5 = 0% |
| history_majority | 2/5 = 40% |
| bucket_majority | 3/5 = 60% |
| aux_majority | 2/5 = 40% |
| combined_aux_bucket_majority | 3/5 = 60% |

Strict trading replay:

| Method | Trades | CAGR | Strict MDD | Ratio |
| --- | ---: | ---: | ---: | ---: |
| history_majority | 61 | 16.48% | 8.31% | 1.98 |
| aux_majority | 35 | -25.94% | 15.69% | -1.65 |
| combined_aux_bucket_majority | 0 | 0.00% | 0.00% | 0.00 |

## Decision

No-go for these coarse BTC funding/premium monthly buckets as a side-map improvement.

They do not identify the two important 2026 exceptions:

- 2026-02 should be `inverse`;
- 2026-04 should be `normal`.

The best simple prior-only signal remains side-map history majority, but it is still below target. Next feature attempt should use richer market structure/state, not only monthly funding/premium buckets.
