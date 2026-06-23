# Event side rationale train-derived abstention (2026-06-23)

## Purpose

The previous score-spread abstention pass swept thresholds directly on the 2026 eval window. That is useful as a diagnostic, but not valid for selecting a live threshold. This pass fixes the threshold source to pre-2026 rows only, then applies those fixed thresholds once to the 2026 eval replay.

## Split

The source side-pair file contains 2,620 rows across 2022 through 2026. It is not safe to treat the whole file as train data for threshold calibration.

Reproducible split command:

```bash
.venv/bin/python -m training.filter_jsonl_by_date \
  --input-jsonl data/event_side_pair_h288_start2022_2026-06-23.jsonl \
  --output-jsonl data/event_side_pair_h288_start2022_train_pre2026_2026-06-23.jsonl \
  --max-date 2026-01-01
```

Result:

- train/calibration: 2,429 rows, 2022-01-01 03:00:00 through 2025-12-31 15:00:00
- eval: 191 rows, 2026-01 through 2026-05

## Pre-2026 model behavior

Base Gemma prior on pre-2026:

- samples: 2,429
- prediction counts: all INVERSE
- accuracy: 49.11%

DPO adapter after base-prior subtraction:

- samples: 2,429
- prediction counts: INVERSE 1,268 / NORMAL 1,161
- accuracy: 49.49%

Pre-2026 score-spread quantiles did not show a reliable confidence relation:

| train quantile | threshold | kept rows | kept side accuracy |
|---:|---:|---:|---:|
| q50 | 0.004017 | 1,216 | 49.51% |
| q60 | 0.005017 | 972 | 49.38% |
| q70 | 0.006141 | 729 | 49.66% |
| q75 | 0.006838 | 608 | 49.34% |
| q80 | 0.007575 | 486 | 48.15% |
| q85 | 0.008441 | 365 | 46.30% |
| q90 | 0.009802 | 243 | 47.74% |
| q95 | 0.011823 | 122 | 46.72% |

## Fixed-threshold 2026 replay

| fixed threshold source | kept eval decisions | side accuracy | executed trades | CAGR | strict MDD | CAGR/strict MDD | mean-ret p approx | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train q75 | 38 | 57.89% | 31 | 3.36% | 6.74% | 0.50 | 0.829 | Not enough edge |
| train q80 | 31 | 58.06% | 26 | -0.29% | 5.42% | -0.05 | 0.986 | Fails |
| train q90 | 13 | 69.23% | 12 | 10.79% | 2.22% | 4.85 | 0.092 | Interesting but too few trades |
| train q95 | 6 | 83.33% | 6 | 6.61% | 1.88% | 3.52 | 0.128 | Too few trades |

## Interpretation

- The q90/q95 fixed-threshold results cross the ratio target, but only because MDD is tiny over 6-12 trades. This is not enough statistical support for a live strategy.
- The pre-2026 calibration set itself does not validate score-spread as confidence; higher spread actually has worse side accuracy in train.
- The safest conclusion is that score-spread tail behavior may contain a sparse anomaly, but it is not a robust standalone RLLM alpha.

## Next direction

Stop optimizing this gate in isolation. The next useful RLLM test should change the learned target or representation, not the threshold: train the LLM to rank a richer event/action template or predict structured price-action regimes, then use RL/backtest reward only as a downstream selector with strict out-of-sample replay.
