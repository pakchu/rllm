# Rolling extrema lower-bound filter audit — 2026-06-25

## Baseline before this pass

Current valid best before adding lower-bound filters:

- Report: `results/event_candidate_pairwise_walkforward_paext_rex_filter_rex36rw_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`
- Policy: pairwise ranker with `rex_` dropped from ranker inputs, `rex_36_range_width_pct <= validation-selected quantile` as filter-only gate.
- Aggregate: CAGR 19.43%, strict MDD 13.12%, CAGR/strict-MDD 1.48, 126 trades, approximate p-value 0.0147, mean trade return 0.547%.

## What changed

The previous filter implementation could only express an upper bound: `feature <= validation quantile`.
That is too narrow for price-location features. For example, `rex_36_cur_to_min_pct` should often be useful as a lower bound: current price is sufficiently far above the recent rolling low.

Implemented lower-bound filter support:

- `--min-feature-names`
- `--min-feature-quantiles`

Leakage contract remains unchanged: each fold selects filter quantiles on validation, then recomputes concrete thresholds from fit+validation only before test.

## Results

All runs used:

- Input: `data/event_action_compressor_ranker_all_2022_2026_paext_rex_2026-06-24.jsonl`
- Market: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
- Walk-forward: 6M fit / 3M validation / 3M test / 3M step
- Pair half-life: 45 days
- Side scaling: `--side-min-val-trades 3 --side-scale-val-mean-ret-pct 0.5`
- Ranker inputs: `--ranker-drop-prefixes rex_`

### Upper-bound position filters were not complementary

`rex_36_range_width_pct <= q` plus upper-bound location filters underperformed:

| Run | CAGR | strict MDD | CAGR/MDD | Trades | p-value | Mean trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rw+cur_to_min <= q` | 13.38% | 17.75% | 0.75 | 140 | 0.1289 | 0.369% |
| `rw+min_to_cur <= q` | 4.92% | 19.82% | 0.25 | 142 | 0.4630 | 0.155% |
| `rw+cur_to_max <= q` | 1.62% | 38.38% | 0.04 | 139 | 0.7339 | 0.082% |
| `rw+max_to_cur <= q` | 14.38% | 13.19% | 1.09 | 158 | 0.0774 | 0.340% |

### Lower-bound `cur_to_min` was materially better

Best new candidate:

- Report: `results/event_candidate_pairwise_walkforward_filter_rw_ge_cur2min_6m3m3m_decay45_sidescale_d0p5_2026-06-25/report.json`
- Filter policy: `rex_36_range_width_pct <= validation-selected q` and `rex_36_cur_to_min_pct >= validation-selected q`
- Aggregate: CAGR 29.35%, strict MDD 12.54%, CAGR/strict-MDD 2.34, 157 trades, approximate p-value 0.0018, mean trade return 0.632%.

Other lower-bound location filters failed:

| Run | CAGR | strict MDD | CAGR/MDD | Trades | p-value | Mean trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rw+cur_to_min >= q` | 29.35% | 12.54% | 2.34 | 157 | 0.0018 | 0.632% |
| `rw+cur_to_max >= q` | 10.17% | 13.34% | 0.76 | 130 | 0.0938 | 0.293% |
| `rw+min_to_cur >= q` | 7.66% | 26.93% | 0.28 | 130 | 0.2831 | 0.239% |
| `rw+max_to_cur >= q` | 6.21% | 29.34% | 0.21 | 133 | 0.3978 | 0.203% |

### Fine-grid overfit warning

Expanding the winning grid degraded performance:

- Report: `results/event_candidate_pairwise_walkforward_filter_rw_ge_cur2min_fine_6m3m3m_decay45_sidescale_d0p5_2026-06-25/report.json`
- Aggregate: CAGR 19.72%, strict MDD 20.82%, CAGR/MDD 0.95, 157 trades, p-value 0.0237.

A failed diagnostic requiring `cur_to_min` to be non-null in every validation policy also degraded performance:

- Report: `results/event_candidate_pairwise_walkforward_filter_rw_ge_cur2min_reqmin_6m3m3m_decay45_sidescale_d0p5_2026-06-25/report.json`
- Aggregate: CAGR 25.73%, strict MDD 23.44%, CAGR/MDD 1.10, 130 trades, p-value 0.0051.
- This option was not kept in code.

## Interpretation

- The user hypothesis was correct: rolling max/min/current-price geometry contains useful information.
- Direction matters. Treating every location feature as an upper-bound filter destroys signal.
- The useful motif found here is: compressed recent range plus current price sufficiently above the recent low.
- Wider validation grids overfit; keep the narrow grid until a separate holdout/eval confirms otherwise.
- The candidate is improved but still below target CAGR/MDD >= 3.
- Recent 2026 folds mostly abstain under this gate, so live-readiness still requires a recent-period activation/risk audit.
