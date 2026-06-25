# Price-action extreme sparse scan — 2026-06-25

## Purpose

Re-test the user's requested price-action idea: for multiple lookbacks, use the candle that made the rolling highest high / lowest low, including the paired low/high of that extreme candle, age since extreme, range position, and overlap structure.

## Implementation

Integrated existing `build_extreme_bar_features` into sparse setup mining/replay surfaces:

- `training/rolling_sparse_setup_miner.py`
  - `--include-price-action-extremes`
  - `--price-action-lookbacks`
- `training/sparse_setup_ensemble_audit.py`
- `training/sparse_setup_walkforward_selector.py`

This keeps miner and replay feature construction aligned. Features are prefixed as `pa__pa_ext_*` and are leak-safe: they use only candles at or before the signal bar.

## Feature audit

Command output:

`results/price_action_extreme_audit_2026-06-25/report.json`

Top diagnostic features included:

- `pa_ext_144_to_max_high_pct`
- `pa_ext_144_range_pos`
- `pa_ext_576_max_high_bar_spread_pct`
- `pa_ext_288_extreme_bar_overlap_pct`

The feature audit found statistically visible relationships, but split signs are not perfectly stable across years. This means they should be used as weak event components, not standalone predictive truth.

## Sparse scan

Report:

`results/sparse_setup_price_action_extreme_2026-06-25/report.json`

Targeted feature regex:

`^pa__pa_ext_(144|288|576)_(to_max_high_pct|range_pos|to_low_of_max_high_pct|max_high_bar_spread_pct|extreme_bar_overlap_pct|age_diff|max_high_age_frac|min_low_age_frac)`

Some strict candidates looked attractive but had low total trades, e.g. 14-25 trades across 2023H1-2026H1. This is not statistically enough for production confidence.

## Walk-forward selector result

Report:

`results/sparse_setup_price_action_extreme_2026-06-25/walkforward_selector.json`

Final replay:

- CAGR: `9.79%`
- strict MDD: `10.37%`
- CAGR/MDD: `0.94`
- trades: `162`
- approximate p-value: `0.067`
- power gap: needs about `380` trades for 80% power; observed `162`

Fold highlights:

| fold | trades | CAGR | strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2023H1 | 21 | 23.94% | 6.90% | 3.47 |
| 2024H2 | 84 | 5.50% | 8.44% | 0.65 |
| 2025H1 | 31 | 51.65% | 5.60% | 9.23 |
| 2025H2 | 8 | -4.66% | 3.06% | -1.52 |
| 2026H1 | 15 | 37.51% | 10.37% | 3.62 |

## Conclusion

Price-action extreme features are real enough to keep, especially because 2026H1 is strong and the family is different from the previous momentum setup. However, standalone price-action sparse setups are too low-frequency and not strong enough to meet the target. The next useful step is a combined candidate pool where robust momentum top1 is the core and price-action extreme setups are secondary diversifiers under stricter trade-count/statistical constraints.
