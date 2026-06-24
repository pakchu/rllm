# Rolling extrema range-width filter (2026-06-24)

## Purpose

Rolling max/min/current-price information was important in univariate audits but harmful as raw ranker features. This pass tests the correct usage: keep `rex_*` out of ranker training and use it only as a validation-selected candidate-universe filter.

## Implementation

Changes:

- `training.event_candidate_ridge_ranker._feature_names(..., drop_prefixes=...)`
  - Allows filter-only feature families to stay in row metadata while excluded from ranker fitting.
- `training.event_candidate_ridge_ranker._write_policy(..., max_feature_name, max_feature_value)`
  - Skips candidates when a selected row's feature exceeds a threshold.
- `training.event_candidate_pairwise_ranker.EventCandidatePairwiseRankerCfg.ranker_drop_prefixes`
- `training.event_candidate_pairwise_walkforward`
  - `--ranker-drop-prefixes`
  - `--max-feature-name`
  - `--max-feature-quantiles`
  - Validation selects filter quantile; test threshold is recomputed from fit+validation best rows only.

Leakage contract:

- `rex_` features are backward-asof, past-only.
- Ranker does not train on `rex_` when `--ranker-drop-prefixes rex_` is set.
- Filter threshold is selected on validation only.
- Test threshold is recomputed from train history only, using the selected filter quantile.

## Protocol

Base protocol:

- Input: `data/event_action_compressor_ranker_all_2022_2026_paext_rex_2026-06-24.jsonl`
- PA-ext ranker features only (`--ranker-drop-prefixes rex_`)
- 6M fit / 3M validation / 3M test
- pair half-life 45d
- stats gate
- light side scaling denominator 0.5

Filter grid:

- `--max-feature-quantiles 0.50,0.60,0.70,0.80,0.90`

## Results

| Filter feature | CAGR | Strict MDD | CAGR/MDD | Trades | p approx | Mean trade |
|---|---:|---:|---:|---:|---:|---:|
| none / previous best | 15.02% | 14.10% | 1.07 | 119 | 0.066 | +0.468% |
| `rex_36_range_width_pct` | 19.43% | 13.12% | 1.48 | 126 | 0.0147 | +0.547% |
| `rex_72_range_width_pct` | 11.46% | 23.67% | 0.48 | 168 | 0.1665 | +0.267% |

Best report:

- `results/event_candidate_pairwise_walkforward_paext_rex_filter_rex36rw_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`

Fold profile for `rex_36_range_width_pct`:

- 2022Q4: `CAGR 87.7 / MDD 12.5 / ratio 6.99`
- 2023Q1: `CAGR 12.8 / MDD 8.3 / ratio 1.55`
- 2023Q2: `CAGR 60.9 / MDD 9.2 / ratio 6.62`
- 2023Q3: `CAGR 17.3 / MDD 3.8 / ratio 4.53`
- 2024Q4: `CAGR 66.5 / MDD 13.1 / ratio 5.07`
- 2025Q1: `CAGR 46.8 / MDD 7.6 / ratio 6.18`
- 2025Q2: `CAGR 47.5 / MDD 2.8 / ratio 17.08`
- 2025Q3: `CAGR -16.8 / MDD 9.4 / ratio -1.77`

## Conclusion

The user's claim is correct: rolling max/min/current price is important. The correct form in this architecture is not raw ranker input but a short-window range-expansion filter. `rex_36_range_width_pct` is the strongest improvement since the PA-ext/time-decay pivot and gives the current valid best:

- CAGR 19.43%
- Strict MDD 13.12%
- CAGR/MDD 1.48
- p approx 0.0147

This still misses the long-term target but is materially closer and statistically stronger than prior results.
