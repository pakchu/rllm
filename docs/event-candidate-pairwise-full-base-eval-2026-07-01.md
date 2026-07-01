# Compact pairwise base full evaluation (2026-07-01)

## Why this run

The random-1024 compact pairwise evaluation looked mildly profitable at q0.80, while the 2025-fit/2026-test risk-filter check showed instability. To separate sampling luck from a real deployable edge, we evaluated the full chronological compact pairwise option-choice set with the base Gemma 4 E4B model and no adapter.

## Inputs

- Eval set: `data/event_candidate_regime_pairwise_option_compact_paext_rex_eval_2025_2026_2026-06-30.jsonl`
- Rows: 11,689
- Model: `google/gemma-4-E4B-it` via alias `gemma4`
- Output report: `results/event_candidate_regime_pairwise_option_compact_paext_rex_2026-06-30/base_evalfull_report.json`
- Predictions: `results/event_candidate_regime_pairwise_option_compact_paext_rex_2026-06-30/base_evalfull_predictions.jsonl`
- Backtest selector: model prediction only, absolute A/B margin quantile, 1x leverage, 1-bar delayed entry, actual OHLC sparse simulation.

## Option-choice quality

| metric | value |
|---|---:|
| rows | 11,689 |
| accuracy | 50.03% |
| target A/B | 5,847 / 5,842 |
| prediction A/B | 7,122 / 4,567 |
| accuracy on target A | 60.95% |
| accuracy on target B | 39.10% |

The model still has a strong A-side preference. Accuracy is effectively random once the full set replaces random-1024 sampling.

## Full prediction backtest

| margin q | events | trades | CAGR | strict MDD | CAGR/MDD | mean trade | p-value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 1,428 | 431 | -24.81% | 52.46% | -0.47 | -0.066% | 0.562 |
| 0.75 | 1,301 | 413 | -16.79% | 46.82% | -0.36 | -0.035% | 0.761 |
| 0.80 | 1,166 | 395 | -13.91% | 49.09% | -0.28 | -0.025% | 0.836 |
| 0.85 | 995 | 375 | -32.09% | 57.06% | -0.56 | -0.116% | 0.346 |
| 0.90 | 788 | 348 | -23.64% | 58.76% | -0.40 | -0.079% | 0.553 |
| 0.95 | 459 | 260 | -6.86% | 45.04% | -0.15 | -0.006% | 0.971 |

Inverting predictions also fails across all tested margin quantiles; this is not a clean anti-signal.

## Chronological holdout at q0.80

A full q0.80 2025-fit/2026-test sweep used the fit-period threshold only (`2.4990234375`).

| split | rows | events | trades | CAGR | strict MDD | CAGR/MDD | mean trade | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit <=2025 | 8,198 | 831 | 288 | -14.31% | 42.63% | -0.34 | -0.030% | 0.810 |
| test 2026 | 3,491 | 335 | 109 | -12.73% | 43.66% | -0.29 | -0.009% | 0.974 |

Some one-feature filters selected on 2025 happen to be positive in 2026 (`tok:rex_2016_loc=lower`, `tok:rex_144_upper_gap=far`), but they have only 36 and 30 test trades respectively and were not the top fit-transfer filters. They are hypotheses for future feature design, not deployable evidence.

## Decision

Reject the raw compact pairwise base model as a trading policy. The random-1024 positive result was sampling noise. The useful artifact is not the policy; it is the label/feature surface and the observation that REX distance/location tokens can occasionally align with profitable subsets. Next work should move away from asking the LLM to directly choose A/B from dense numeric text and toward a smaller, explicit alpha-feature/risk-feature representation where the LLM is used for regime abstraction or policy reasoning over calibrated features.
