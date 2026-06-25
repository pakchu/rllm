# Recent-regime ridge holdout audit — 2026-06-25

## Why this was run

The previously applied rolling-extrema pairwise policy is not live-usable because recent monthly walk-forward failed:

- Report: `results/event_candidate_pairwise_walkforward_filter_rw_ge_cur2min_recent_6m1m1m_decay45_sidescale_d0p5_2026-06-25/report.json`
- Aggregate: CAGR -14.96%, strict MDD 37.72%, CAGR/MDD -0.40, 97 trades.

Several recent-regime filter attempts also failed or remained weak:

- Long-range `rex_*_range_width + cur_to_max` filters: best was `rw2016_ge_cur2max`, CAGR 6.06%, MDD 17.49%, ratio 0.35.
- Side-positive gating: best was `rw2016cur2max_sidepos`, CAGR 4.75%, MDD 17.49%, ratio 0.27.
- Price-action primitive filters: best was `pa36_maxhighspread_max`, CAGR 3.96%, MDD 16.12%, ratio 0.25.
- Raw `rex_` pairwise ranker input: best was raw-rex no-filter, CAGR 1.53%, MDD 18.45%, ratio 0.08.
- 3M adaptive pairwise fit: best was no-filter, CAGR 7.49%, MDD 18.35%, ratio 0.41.
- Overlay exit sweep could not fix the signal; best post-hoc diagnostic was CAGR 9.41%, MDD 15.12%, ratio 0.62.

## Ridge holdout result

A lower-capacity expected-return ridge ranker was tested as a model-family change.

Protocol for the strongest 2026 holdout:

- Train rows: 2024-01-01 through 2025-12-31.
- Validation selection: 2025-09-01 through 2025-12-31.
- Eval: 2026-01-01 through 2026-05-30.
- Selection: validation chooses score quantile/full margin; eval is not used for selection.
- Report: `results/recent_model_family_2026-06-25/ridge_2026_eval_coef/report.json`

2026 eval result:

| Metric | Value |
| --- | ---: |
| CAGR | 63.24% |
| strict MDD | 9.45% |
| CAGR/MDD | 6.69 |
| Trades | 88 |
| Mean trade return | 0.233% |
| Approx p-value | 0.0341 |

2026 monthly executed-trade sums:

| Month | Trades | Sum return | Mean return |
| --- | ---: | ---: | ---: |
| 2026-01 | 16 | +7.21% | +0.450% |
| 2026-02 | 33 | +6.66% | +0.202% |
| 2026-03 | 6 | +4.70% | +0.783% |
| 2026-04 | 11 | -1.39% | -0.126% |
| 2026-05 | 22 | +3.30% | +0.150% |

Top final ridge coefficients are dominated by side-interacted rolling-extrema position features, especially current-to-high geometry:

- `signed:rex_8640_cur_to_max_pct`
- `raw:rex_8640_cur_to_max_pct_x_side`
- `signed:rex_8640_max_to_cur_pct`
- `signed:rex_4032_max_to_cur_pct`
- `signed:rex_576_cur_to_max_pct`
- `signed:rex_72_cur_to_max_pct`
- `signed:window_drawdown`

## Generalization warning

The same ridge family did **not** generalize across adjacent eval splits:

| Split | Eval | CAGR | strict MDD | CAGR/MDD | Trades | p-value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `eval_2025h2` | 2025-07 to 2025-12 | -33.97% | 21.76% | -1.56 | 95 | 0.0593 |
| `eval_2025full` | 2025-01 to 2025-12 | -8.19% | 9.29% | -0.88 | 123 | 0.0938 |
| `eval_2026` | 2026-01 to 2026-05 | 63.24% | 9.45% | 6.69 | 88 | 0.0341 |

Therefore this is **not yet a deployable strategy**. It is a strong clue that 2026 has a different relation where direct ridge expected-return scoring over rolling-extrema position features works better than pairwise ranking.

## Next implication

The next valid step is not to apply this to live. It is to build a rolling ridge walk-forward that selects/runs the ridge family with the same leak guards across multiple contiguous periods, then checks whether the 2026 edge can be selected from prior validation without failing earlier adjacent periods.
