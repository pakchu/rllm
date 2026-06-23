# Price-action extreme-bar feature significance (2026-06-23)

## Feature definition

For each rolling lookback window `w`, using candles at or before signal time `t` only:

- `low_at_window_high`: the low price of the candle that made the rolling maximum high.
- `high_at_window_low`: the high price of the candle that made the rolling minimum low.

Derived scan features:

- `pa_w{w}_high_candle_low_dist = (close_t - low_at_window_high) / close_t`
- `pa_w{w}_low_candle_high_dist = (close_t - high_at_window_low) / close_t`
- `pa_w{w}_extreme_body_gap = (high_at_window_low - low_at_window_high) / close_t`
- `pa_w{w}_high_age_frac`
- `pa_w{w}_low_age_frac`

Windows scanned: `36, 72, 144, 288, 576, 2016` bars.
Horizons scanned: `36, 72, 144, 288` bars.
Quantiles scanned: `0.05, 0.10, 0.20`.

## Protocol

- Train fit: `2020-01-01` through `2023-12-31`.
- Test: `2024-01-01` through `2025-12-31`.
- Eval: `2026-01-01` through `2026-06-01`.
- Rule: train-only quantile thresholds and train-only high/low side mapping.
- Backtest: strict bar-by-bar MDD, 0.5x leverage, fee 4bp, slippage 1bp, entry next 5m open.
- Output: `results/price_action_extreme_feature_scan_2026-06-23.json`.

## Result summary

Scanned: 30 features × 4 horizons × 3 quantiles = 360 rules.

Strict filters:

| Filter | Passing rules |
| --- | ---: |
| test ratio > 1 and eval ratio > 1, with minimum trade counts | 0 |
| test ratio >= 3 and eval ratio >= 3 | 0 |
| test/eval ratio both positive | 3, all weak and statistically insignificant |

Top test-ranked candidate:

| Feature | Horizon | q | Test CAGR | Test MDD | Test ratio | Test trades | Test p | Eval CAGR | Eval MDD | Eval ratio | Eval trades | Eval p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pa_w288_high_age_frac` | 288 | 0.20 | 20.97% | 15.98% | 1.31 | 627 | 0.169 | -19.47% | 18.03% | -1.08 | 133 | 0.533 |

Only positive test+eval ratio candidates:

| Feature | Horizon | q | Test ratio | Eval ratio | Comment |
| --- | ---: | ---: | ---: | ---: | --- |
| `pa_w576_extreme_body_gap` | 144 | 0.05 | 1.08 | 0.09 | weak, eval trades only 13, p=0.938 |
| `pa_w576_extreme_body_gap` | 72 | 0.05 | 0.51 | 0.03 | weak, p-values poor |
| `pa_w36_extreme_body_gap` | 288 | 0.05 | 0.12 | 0.42 | weak, p-values poor |

## Interpretation

As a standalone univariate trading rule, this price-action extreme-bar feature family is **not significant enough**.

The interesting observation is not direct tradability but regime sensitivity:

- Some features invert sharply between test and eval.
- `low_candle_high_dist` and age features show strong eval periods when test is negative, implying regime-dependent behavior rather than stable unconditional edge.
- `extreme_body_gap` is the only family with weak same-sign test/eval behavior, but magnitude and p-values are too weak for deployment.

## Next use

Do not add these as direct rule signals. Use them as **context tokens/interactions**:

1. Add bucketized price-action level features to the RLLM context miner.
2. Test them inside rolling context selection, not as standalone quantile rules.
3. Require rolling symbolic context edge before any Gemma SFT.
