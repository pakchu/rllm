# Path-shape price-action prompt augmentation (2026-06-26)

## Purpose

The path-shape oracle label is strong, but the existing past-only symbolic summary is not learnable enough. This pass adds explicit past-only price-action tokens to the trader prompt and reruns the same train/val/eval token baseline.

Implementation:

- `training/augment_path_shape_prompts_with_pa.py`
- `training/path_shape_token_policy_tte.py` now consumes `augmented_price_action_tokens` and `augmented_price_action_features` from the summary JSON.
- `tests/test_augment_path_shape_prompts_with_pa.py`

Added token families for windows `36,144,576,2016` bars:

- rolling range position
- distance to rolling high/low
- window return bucket
- rolling range width bucket
- rolling max/min age bucket
- rolling volume z bucket

All features use bars at or before `signal_pos`; targets are unchanged.

## Artifacts

Augmented data:

- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_train_paaug.jsonl`
- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_val_paaug.jsonl`
- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_oos_paaug.jsonl`

Report:

- `results/path_shape_token_policy_tte_h144_t1p0_s0p6_paaug/report.json`

Each row received `32` price-action tokens.

## Result

Baseline before augmentation:

| Split | Accuracy | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val | 42.21% | 258 | -53.13% | 32.09% | -1.66 |
| OOS | 38.13% | 279 | -48.07% | 32.16% | -1.49 |

After price-action augmentation:

| Split | Accuracy | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val | 42.75% | 244 | -46.16% | 28.21% | -1.64 |
| OOS | 39.63% | 289 | -35.64% | 24.79% | -1.44 |

## Interpretation

The added price-action tokens move in the right direction but do not create a profitable policy. This is useful negative evidence:

- Rolling max/min and range-position context matter, but coarse token buckets are insufficient.
- The model still cannot distinguish target-first from stop-first path outcomes reliably enough.
- The next improvement should add richer **micro-path trajectory** and **multi-timeframe compression/expansion** summaries, not merely more hard gates.

Practical next step:

1. Add micro-path features: recent 12/36/72-bar signed returns, wick/rejection counts, trend persistence, alternating chop count, range expansion ratio.
2. Include macro/cross-market tokens by date-joining the wavefull market cache or `../wave_trading` features, not by `signal_pos`.
3. Re-run the same token baseline before GPU fine-tuning.
