# Rolling-extrema feature scan (2026-07-01)

## Change

Added history-only rolling extrema (`rex_*`) features to the shared market feature frame so alpha scans and future LLM policy prompts can use the same price-action information that had only been present in event-candidate text augmentation.

Windows: 36, 144, 576, 2016, 8640 bars. For each window:

- `range_width_pct`: `(rolling_high - rolling_low) / close`
- `range_pos`: normalized current close location inside the rolling high/low range (`-1` low, `+1` high)
- `cur_to_max_pct`: `close / rolling_high - 1`
- `cur_to_min_pct`: `close / rolling_low - 1`
- `max_to_cur_pct`: `rolling_high / close - 1`

All values use rows `<= t` only; forward returns are diagnostic labels only.

## Validation

- `py_compile`: `preprocessing/market_features.py`, `training/alpha_feature_scan.py`, `training/rolling_alpha_feature_discovery.py`
- Smoke build on first 10,000 market rows: 25 `rex_*` columns, zero NaNs, every declared `rex_*` column exists in the built frame.
- Full univariate diagnostic: `results/alpha_feature_scan_rex_2026-07-01.json`
- Rolling strict discovery: `results/rolling_alpha_rex_features_2026-07-01.json`
  - input rows: 674,785 (`2019-12-31 15:00:00` to `2026-05-31 15:00:00`)
  - feature count after adding market + wave features: 132

## Result

The new rolling-extrema features are informative but not yet a standalone stable trading rule. The best strict candidate was long-horizon distance-to-rolling-high:

| feature | horizon | q | positive folds | total trades | fold summary |
|---|---:|---:|---:|---:|---|
| `mkt__rex_8640_cur_to_max_pct` | 288 | 0.10 | 4/7 | 324 | 2023H1 +10.52 / 16.69 MDD, 2023H2 -13.23 / 16.46, 2024H1 +4.90 / 19.11, 2024H2 +10.59 / 18.64, 2025H1 +2.30 / 14.05, 2025H2 -16.43 / 17.69, 2026H1 -14.94 / 23.78 |
| `mkt__rex_8640_max_to_cur_pct` | 288 | 0.10 | 4/7 | 324 | duplicate/inverse magnitude of the same information |

Top univariate diagnostics also surfaced `rex_36_range_width_pct`, `rex_144_range_width_pct`, and `rex_8640_range_width_pct`, but they flipped sign in the 2025Sep-2026Feb eval split. That reinforces the current lesson: rolling extrema are useful context, but direct threshold rules overfit regime direction.

## Decision

Keep `rex_*` features as a core price-action context for the next LLM/RL surface, but do not promote a raw `rex_*` threshold strategy. The next policy surface should ask the LLM to reason over a compact state card containing extrema location/width plus regime/flow context, while the backtest still requires chronological fold validation.
