# Path-shape token policy TTE (2026-06-26)

## Purpose

Sparse setup + hard veto tuning failed under train-only validation. The next LLM/RL-compatible direction is path-shape learning: predict future target/stop/path diagnostics from past-only market summaries, then trade only when the predicted path shape is favorable.

Before spending GPU on Gemma/Gemma4 fine-tuning, this pass checks whether the existing past-only symbolic summary has cheap learnability for the path-shape trader label.

Implementation:

- `training/export_path_shape_targets_as_predictions.py`
- `training/path_shape_token_policy_tte.py`
- `tests/test_export_path_shape_targets_as_predictions.py`
- `tests/test_path_shape_token_policy_tte.py`

Market alignment note:

- These path-shape datasets use `signal_pos` aligned to `data/2023-01-01_2026-02-28_d2a88c0700504d6a5e15bc3839ad84b6.csv.gz`.
- Using the 2020-start wavefull market shifts positions by ~3 years and invalidates the backtest.

## Oracle target-echo upper bound

Target echo converts future-derived trader targets into prediction rows and backtests them with strict OHLC stop/take-profit execution. This is not deployable; it only checks label geometry.

Config:

- horizon/max hold: `144` bars
- target: `1.0%`
- stop: `0.6%`
- leverage: `1.0`
- fee: `0.04%`, slippage: `0.01%`
- strict MDD includes intrabar adverse movement

| Split | Period | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2023-01-01 to 2025-02-28 | 1,350 | 24,996.54% | 1.71% | 14,615.60 | 0.888% |
| Val | 2025-03-01 to 2025-08-31 | 282 | 14,535.08% | 1.86% | 7,835.46 | 0.890% |
| OOS | 2025-09-01 to 2026-02-26 | 296 | 22,031.38% | 1.88% | 11,706.13 | 0.893% |

Interpretation: the future-derived label is a very strong oracle because it encodes whether target or stop was hit first. It is useful as an upper bound, not as evidence that a model can infer it from past-only data.

Artifacts:

- `results/path_shape_target_echo_train_h144_t1p0_s0p6.bt.json`
- `results/path_shape_target_echo_val_h144_t1p0_s0p6.bt.json`
- `results/path_shape_target_echo_oos_h144_t1p0_s0p6.bt.json`

## Past-summary token baseline

Protocol:

1. Fit a categorical token Naive-Bayes/log-odds model on train only.
2. Tokens come only from the past-only analyzer summary: regime, symbolic features, context tags, sequence stats, binned numeric evidence, and recent bar tokens.
3. Select confidence/margin thresholds on val only by strict backtest score.
4. Evaluate selected thresholds on untouched OOS.

Artifact:

- `results/path_shape_token_policy_tte_h144_t1p0_s0p6/report.json`

Classification:

| Split | Accuracy | Target counts | Pred counts |
| --- | ---: | --- | --- |
| Train | 43.50% | NO_TRADE 872 / LONG 749 / SHORT 749 | NO_TRADE 1013 / SHORT 841 / LONG 516 |
| Val | 42.21% | LONG 162 / NO_TRADE 227 / SHORT 163 | SHORT 168 / NO_TRADE 281 / LONG 103 |
| OOS | 38.13% | LONG 151 / NO_TRADE 208 / SHORT 176 | SHORT 191 / NO_TRADE 240 / LONG 104 |

Selected by val:

```json
{"prob_threshold": 0.34, "margin_threshold": 0.0}
```

Backtest:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 258 | -53.13% | 32.09% | -1.66 | -0.144% | 0.0019 |
| OOS | 279 | -48.07% | 32.16% | -1.49 | n/a in summary | n/a |

## Conclusion

The path-shape label has a strong oracle upper bound, but the current symbolic past summary is not learnable enough for profitable deployment. This explains why direct LLM fine-tunes have repeatedly collapsed into priors, side bias, or non-generalizing gates.

Practical next step:

- Do not spend another long run fine-tuning this exact summary/label pair.
- Improve the input representation before Gemma/Gemma4 SFT: add explicit multi-timeframe price-action path tokens, rolling max/min location, compression/expansion, distance-to-extreme, DXY/USDKRW/kimchi shock tokens, and recent realized micro-path tokens directly into the prompt.
- Then rerun this cheap token baseline. Only if cheap learnability improves should GPU SFT/RL be retried.
