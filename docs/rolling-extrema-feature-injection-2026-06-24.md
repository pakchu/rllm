# Rolling max/min/current-price feature injection (2026-06-24)

## Purpose

The user pointed out that rolling max/min versus current price is important. This pass separated that idea from broad market-regime features and tested it directly.

Feature family:

- rolling high/low windows: `36,72,144,288,576,2016,4032,8640` 5m bars
- current vs rolling max/min distances
- rolling range position
- rolling range width
- upper/lower gap over width
- side interactions for the numeric extrema features

Implementation: `training/augment_event_candidate_rolling_extrema.py`.

## Data

Input: `data/event_action_compressor_ranker_all_2022_2026_paext_2026-06-24.jsonl`

Output: `data/event_action_compressor_ranker_all_2022_2026_paext_rex_2026-06-24.jsonl`

- Rows matched: 128,820 / 128,820
- Added numeric features: 128
- Join: backward-asof, 5m tolerance

## Full rex walk-forward

Report: `results/event_candidate_pairwise_walkforward_paext_rex_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`

- CAGR: 4.29%
- Strict MDD: 19.44%
- CAGR/MDD: 0.22
- Trades: 53
- p approx: 0.400

This is worse than the current best.

## Stability audit

Audit report: `results/event_candidate_rex_feature_stability_audit_2026-06-24.json`

The user's intuition is supported by the univariate evidence. The strongest stable features are rolling range width and short-window current-to-extreme distances:

- `rex_36_range_width_pct`
- `rex_72_range_width_pct`
- `rex_2016_range_width_pct`
- `rex_288_range_width_pct`
- `rex_576_range_width_pct`
- `rex_144_range_width_pct`
- `rex_4032_range_width_pct`
- `rex_8640_range_width_pct`
- `rex_36_cur_to_min_pct`
- `rex_36_min_to_cur_pct`
- `rex_36_cur_to_max_pct`
- `rex_36_max_to_cur_pct`
- `rex_72_cur_to_min_pct`
- `rex_72_min_to_cur_pct`
- `rex_72_cur_to_max_pct`
- `rex_72_max_to_cur_pct`

Example: `rex_36_range_width_pct` has negative IC/spread in every year from 2022 through 2026, meaning high recent range expansion is consistently bad for candidate reward.

## Stable rex subset walk-forward

Data: `data/event_action_compressor_ranker_all_2022_2026_paext_rex_stable_2026-06-24.jsonl`

- Feature count per row: 100
- Includes base + PA-ext + stable rex features only

Report: `results/event_candidate_pairwise_walkforward_paext_rex_stable_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`

- CAGR: -6.97%
- Strict MDD: 30.86%
- CAGR/MDD: -0.23
- Trades: 61

## Conclusion

Rolling max/min/current-price information is important, but direct ranker feature injection is harmful in the current pairwise model. The stable signal is mostly a **bad-environment indicator**: high rolling range width / volatility expansion lowers candidate reward across years.

Next step should use this family as a no-leak candidate-universe filter or abstain rule, not as raw model capacity. Specifically: validation/fold-safe filtering of trades when selected candidate occurs under high `rex_36/rex_72_range_width_pct` regimes.
