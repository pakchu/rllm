# Path-shape micro-path augmentation (2026-06-26)

## Purpose

PA-only augmentation improved the path-shape token baseline slightly but remained loss-making. This pass adds recent micro-path trajectory tokens that are more directly tied to target-first vs stop-first path labels.

Implementation:

- `training/augment_path_shape_prompts_with_pa.py`
  - Adds `--micro-windows`, default `12,36,72`.
  - Adds `augmented_micro_path_tokens` and `augmented_micro_path_features`.
- `training/path_shape_token_policy_tte.py`
  - Consumes micro-path tokens/features.
  - Adds `--side-modes normal,invert` so val can test whether model predictions are anti-signals.
- `tests/test_augment_path_shape_prompts_with_pa.py`
- `tests/test_path_shape_token_policy_tte.py`

Micro-path token families:

- recent path return and realized volatility
- body/range efficiency
- up/down/flat counts
- alternation/chop count
- max same-direction run
- upper/lower rejection counts
- range and volume expansion ratio
- recent MFE/MAE style path buckets

All micro-path features use bars at or before `signal_pos`; labels are unchanged.

## PA + micro result

Artifacts:

- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_*_pa_micro_aug.jsonl`
- `results/path_shape_token_policy_tte_h144_t1p0_s0p6_pa_micro_aug/report.json`

Each row received `74` augmentation tokens.

| Prompt | Val accuracy | Val CAGR | Val MDD | OOS accuracy | OOS CAGR | OOS MDD | OOS CAGR/MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base summary | 42.21% | -53.13% | 32.09% | 38.13% | -48.07% | 32.16% | -1.49 |
| PA-only | 42.75% | -46.16% | 28.21% | 39.63% | -35.64% | 24.79% | -1.44 |
| PA + micro | 43.30% | -50.30% | 30.07% | 39.81% | -34.97% | 24.69% | -1.42 |
| PA + micro, mc=3/tk=24 | 43.66% | -40.45% | 25.52% | 39.81% | -30.73% | 23.25% | -1.32 |

The best cheap setting tried was `min_count=3`, `top_k_tokens=24`, still loss-making but improved OOS damage.

## Side inversion check

Rationale: some earlier experiments showed prediction inversion occasionally helped. Here the token model's val trades were significantly negative, so `normal` vs `invert` was added as a val-selected candidate.

Artifact:

- `results/path_shape_token_policy_tte_h144_t1p0_s0p6_pa_micro_aug_invert/report.json`

Selected by val:

```json
{"side_mode":"invert","prob_threshold":0.34,"margin_threshold":0.0}
```

Result:

| Split | Side mode | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | invert | 276 | -46.11% | 28.49% | -1.62 | -0.109% | 0.0147 |
| OOS | invert | 309 | -68.71% | 44.16% | -1.56 | -0.180% | 0.000012 |

Conclusion: inversion is not a stable escape hatch here. It slightly won the val scoring tie but collapsed OOS.

## Current conclusion

Micro-path tokens improve classification/strict backtest damage directionally, but still do not produce a profitable cheap policy. The current label/input pair remains below the threshold for GPU SFT/RL.

Next viable work:

1. Stop adding broad token buckets blindly.
2. Build a learnability audit that ranks which token groups create positive/negative conditional returns after val selection.
3. If no token group has positive conditional expectancy, change event sampling/label design rather than fine-tuning Gemma on this target.
