# Pairwise option risk-filter sweep (2026-07-01)

## Purpose

The compact pairwise option-choice surface produced the strongest LLM-shaped base signal so far, but the random-1024 backtest was not stable enough to promote. This sweep checks whether simple causal state/risk filters can improve the base Gemma option-choice score without using the 2026 holdout for selection.

## Leakage guard

- Predictions: `results/event_candidate_regime_pairwise_option_compact_paext_rex_2026-06-30/base_eval1024_random_predictions.jsonl`
- Candidate metadata: `results/event_action_compressor_ranker_paext_rex_eval_2025_2026.jsonl`
- Market data: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
- Fit period: selected candidate date `<= 2025-12-31 23:59:59`
- Test period: selected candidate date `> 2025-12-31 23:59:59`
- Score threshold: absolute A/B logprob margin q0.80 fit on fit period only (`2.375`)
- Filter ranking: fit period only
- Test period: never used for threshold or filter selection

## Baseline

| split | predictions | selected events | trades | CAGR | strict MDD | CAGR/MDD | mean trade | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit (<=2025) | 735 | 143 | 111 | 47.49% | 26.45% | 1.80 | 0.383% | 0.124 |
| test (2026) | 289 | 72 | 51 | -18.08% | 18.37% | -0.98 | -0.130% | 0.697 |

## Top fit-selected filters

| filter | fit trades | fit CAGR/MDD | test trades | test CAGR/MDD |
|---|---:|---:|---:|---:|
| `range_location=upper_range` | 31 | 53.37/6.85 = 7.80 | 13 | -14.64/12.16 = -1.20 |
| `taker_flow=strong_down` | 53 | 56.91/8.73 = 6.52 | 23 | -34.43/18.09 = -1.90 |
| `tok:rex_144_loc=near_max` | 33 | 49.29/8.23 = 5.99 | 14 | -16.99/13.03 = -1.30 |
| `side_trend_96=down` | 35 | 37.43/6.97 = 5.37 | 21 | -17.30/15.57 = -1.11 |
| `trend_24=up` | 33 | 44.85/9.27 = 4.84 | 11 | 2.37/5.12 = 0.46 |
| `tok:rex_144_lower_gap=far` | 34 | 35.70/8.33 = 4.29 | 16 | 11.74/10.25 = 1.14 |
| `htf_1w=down` | 30 | 31.19/8.73 = 3.57 | 8 | -11.88/5.31 = -2.24 |
| `hold=144` | 35 | 16.66/4.95 = 3.36 | 14 | 1.75/8.13 = 0.21 |
| `htf_1d=flat` | 64 | 43.73/14.01 = 3.12 | 25 | -28.19/17.40 = -1.62 |
| `trend_96=down` | 19 | 22.98/7.73 = 2.97 | 13 | -20.36/13.35 = -1.52 |

## Decision

Reject simple one-feature risk filters for this pairwise base surface. The filters can manufacture excellent fit-period CAGR/MDD, but the same chosen filters mostly fail in 2026. This is the exact failure mode we must prevent: choosing regime descriptors on the same period where performance is reported.

Next useful step is not another filter sweep on random-1024 predictions. We need a larger chronological evaluation of the compact pairwise base model, then monthly/rolling diagnostics on the full prediction set. That will separate sampling noise from real regime instability before any further fine-tune or policy layer.
