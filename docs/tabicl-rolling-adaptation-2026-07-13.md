# Tabular-model annual rolling adaptation

Date: 2026-07-13

## Summary

The frozen Top-10 algorithms from the TabICLv2 search were replayed with an
annual expanding-fit schedule. Before each evaluation period, the model was fit
using only data before the immediately preceding calibration year; the frozen
score quantile was recalibrated on that preceding year.

No candidate passed 2024, 2025 and 2026 together. Annual retraining did not fix
the regime drift seen in the static models.

## Causal schedule

| evaluation | fit data | score calibration |
|---|---|---|
| Test 2024 | 2020-2022 | 2023 predictions |
| Eval 2025 | 2020-2023 | 2024 predictions |
| 2026 YTD | 2020-2024 | 2025 predictions |

The adaptation manifest was frozen before fold metrics were computed. Candidate
model type, feature group, side policy and score quantile came unchanged from
the original Top-10 manifest.

## Representative failures

### ExtraTrees full, original rank 1, long top 30%

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +41.83% | 41.73% | 10.58% | 3.95 | 62 |
| Eval 2025 | +11.09% | 11.09% | 11.17% | 0.99 | 88 |
| 2026 YTD | -7.61% | -17.32% | 13.73% | -1.26 | 35 |

### TabICLv2 compact, original rank 8, long top 15%

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +60.66% | 60.50% | 5.42% | 11.16 | 53 |
| Eval 2025 | -5.99% | -6.00% | 20.58% | -0.29 | 59 |
| 2026 YTD | +10.40% | 26.85% | 4.76% | 5.64 | 16 |

The attractive 2024/2026 endpoints are interrupted by a full-year 2025 loss and
cannot be promoted.

## Model drift diagnostics

- ExtraTrees full evaluation Spearman: `0.146` in 2024, `0.111` in 2025,
  `0.043` in 2026.
- TabICLv2 compact evaluation Spearman: `0.104` in 2024, `-0.030` in 2025,
  `0.027` in 2026.
- HistGB full evaluation Spearman turns negative in 2026 (`-0.085`).

The predictive ordering itself weakens or reverses; merely refitting more recent
batch data does not create a stable edge.

## Verdict

- No alpha-pool or live-grade candidate.
- Do not select a different yearly model after viewing the failed year.
- The next justified method is a truly online learner that predicts first and
  updates only when each 48-hour label becomes available, with drift detection
  and a causal rolling score distribution.

## Artifacts

- Script: `training/evaluate_tabicl_rolling_adaptation.py`
- Frozen adaptation manifest:
  `results/tabicl_rolling_adaptation_manifest_2026-07-13.json`
- Result: `results/tabicl_rolling_adaptation_2026-07-13.json`
- Tests: `tests/test_tabicl_rolling_adaptation.py`
