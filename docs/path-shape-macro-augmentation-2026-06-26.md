# Path-shape macro augmentation (2026-06-26)

## Purpose

Add DXY / kimchi premium / USDKRW context to path-shape prompts without `signal_pos` leakage or market-index mismatch. The path-shape dataset is aligned to the 2023-start OHLCV file, while macro context is in the 2020-start wavefull cache, so macro must be joined by timestamp.

Implementation updates:

- `training/augment_path_shape_prompts_with_pa.py`
  - New `--context-market-csv` option.
  - Backward-asof joins context rows by `date`.
  - Adds `augmented_macro_tokens` and `augmented_macro_features`.
- `training/path_shape_token_policy_tte.py`
  - Consumes macro augmented tokens/features.
- `tests/test_augment_path_shape_prompts_with_pa.py`
  - Covers macro bucketing and backward-asof behavior.

Macro columns used from `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`:

- `dxy_zscore`, `dxy_momentum`
- `kimchi_premium_zscore`, `kimchi_premium_change`
- `usdkrw_zscore`, `usdkrw_momentum`
- availability flags

## Result

Artifacts:

- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_train_pa_macro_aug.jsonl`
- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_val_pa_macro_aug.jsonl`
- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_oos_pa_macro_aug.jsonl`
- `results/path_shape_token_policy_tte_h144_t1p0_s0p6_pa_macro_aug/report.json`

All rows received macro context:

| Split | Rows with macro | Token count / row |
| --- | ---: | ---: |
| Train | 2,370 | 41 |
| Val | 552 | 41 |
| OOS | 535 | 41 |

Comparison:

| Prompt | Val accuracy | Val CAGR/MDD | OOS accuracy | OOS CAGR | OOS MDD | OOS CAGR/MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base summary | 42.21% | -1.66 | 38.13% | -48.07% | 32.16% | -1.49 |
| PA-only aug | 42.75% | -1.64 | 39.63% | -35.64% | 24.79% | -1.44 |
| PA + macro aug | 40.40% | -1.60 | 39.25% | -36.83% | 25.23% | -1.46 |

## Conclusion

The date-joined macro plumbing is valid, but this coarse macro tokenization does not improve the current path-shape token baseline. Keep the implementation for future richer prompts, but do not select the PA+macro prompt variant as the next SFT candidate.

Current best cheap baseline among these is **PA-only augmentation**, but it is still strongly loss-making. The next useful representation work should focus on micro-path trajectory features rather than broad macro buckets.
