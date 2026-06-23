# Price-action weak-alpha bundle scan (2026-06-23)

## Why this scan exists

Single price-action features are not expected to be profitable alone. The useful hypothesis is that several weak alphas can become tradable when combined with regularization and causal validation.

This scan tests that hypothesis directly by fitting train-only ridge bundles over price-action extreme-bar features plus selected market / external / derivatives context.

## Feature bundles

Base price-action features come from the extreme-bar scan:

- `pa_w{w}_high_candle_low_dist`
- `pa_w{w}_low_candle_high_dist`
- `pa_w{w}_extreme_body_gap`
- `pa_w{w}_high_age_frac`
- `pa_w{w}_low_age_frac`

Windows: `36, 72, 144, 288, 576, 2016` 5m bars.

Bundle groups scanned:

| Group | Feature count |
| --- | ---: |
| `pa_only` | 30 |
| `pa_trend` | 41 |
| `pa_market` | 50 |
| `pa_external` | 42 |
| `pa_derivatives` | 38 |
| `pa_market_external_derivatives` | 70 |

Auxiliary context includes the existing leakage-safe market regime fields, wave-trading external fields, and Binance futures funding/premium fields when available.

## Protocol

- Train fit: `2020-01-01` through `2023-12-31`.
- Test / model-selection diagnostic: `2024-01-01` through `2025-12-31`.
- Untouched eval: `2026-01-01` through `2026-06-01`.
- Model: train-only standardized ridge regression on forward return labels.
- Rule: train-only score quantile thresholds and train-only side mapping.
- Backtest: strict bar-by-bar MDD, 0.5x leverage, fee 4bp, slippage 1bp, entry next 5m open.
- Output: `results/price_action_combo_scan_2026-06-23.json`.

Grid size: 6 groups × 4 horizons × 4 quantiles × 4 ridge penalties = 384 bundle rules.

## Result summary

Strict filters:

| Filter | Passing rules |
| --- | ---: |
| test ratio > 1 and eval ratio > 1 | 0 |
| test ratio >= 3 and eval CAGR > 0 | 0 |
| test ratio >= 3 and eval ratio >= 3 | 0 |
| test/eval CAGR > 0 and both strict MDD <= 15 | 2 |

Best test-ranked bundle:

| Group | Horizon | q | L2 | Test CAGR | Test MDD | Test ratio | Test trades | Test p | Eval CAGR | Eval MDD | Eval ratio | Eval trades | Eval p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pa_derivatives` | 144 | 0.10 | 1000 | 17.00% | 13.70% | 1.24 | 528 | 0.146 | -22.85% | 11.49% | -1.99 | 105 | 0.352 |

Only test/eval positive-CAGR and MDD<=15 candidates:

| Group | Horizon | q | L2 | Test CAGR | Test MDD | Test ratio | Test trades | Test p | Eval CAGR | Eval MDD | Eval ratio | Eval trades | Eval p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pa_only` | 288 | 0.05 | 100 | 4.05% | 14.83% | 0.27 | 249 | 0.621 | 20.93% | 7.96% | 2.63 | 50 | 0.447 |
| `pa_derivatives` | 288 | 0.10 | 1000 | 1.65% | 14.90% | 0.11 | 355 | 0.798 | 21.46% | 8.56% | 2.51 | 74 | 0.499 |

## Interpretation

The bundle framing is correct, but this specific static ridge setup is not yet a deployable alpha:

1. Combining weak price-action features improves some 2024-2025 test cases versus univariate rules.
2. The strongest test-ranked cases flip negative on untouched 2026 eval.
3. The only positive test/eval cases are too weak on test and statistically insignificant.
4. External/derivatives features can help interactions, but they also increase regime-overfit risk.

Conclusion: price-action extreme features should remain **candidate weak-alpha inputs**, not standalone trading rules and not fixed static ridge policies.

## Next decision

Do not fine-tune Gemma on this static combo output yet. The next useful test is rolling bundle validation:

1. Refit bundles causally on a rolling schedule.
2. Select horizon / quantile / regularization only from prior train-validation windows.
3. Evaluate the next unseen month without using that month for selection.
4. Promote features into RLLM context only if rolling out-of-sample keeps positive expectancy with enough trades.

This preserves the user's core point: profitability should come from many weak features, but the selection mechanism must be causal and robust rather than static eval-lucky.
