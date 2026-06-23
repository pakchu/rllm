# Rolling price-action weak-alpha bundle validation (2026-06-23)

## Purpose

The correct hypothesis is not that a single price-action feature should trade profitably. Weak alpha features may only become useful when bundled.

This validation therefore tests bundled price-action features under causal rolling selection:

1. Fit ridge model and thresholds using past train rows only.
2. Select candidate using only the immediately preceding validation window.
3. Trade the next unseen month.
4. Aggregate all selected monthly signals through one global strict bar-by-bar simulator.

## Protocol

- Market: BTCUSDT futures 5m, `2020-01-01` to `2026-06-01`.
- Rolling target: `2024-01-01` to `2026-06-01`.
- Per target month:
  - train window: previous 1095 days before validation.
  - validation window: previous 180 days before target month.
  - target month: not used in selection.
- Candidate grid per month:
  - groups: `pa_only`, `pa_trend`, `pa_market`, `pa_external`, `pa_derivatives`, `pa_market_external_derivatives`.
  - horizons: `72, 144, 288` bars.
  - quantiles: `0.05, 0.10, 0.20`.
  - ridge L2: `10, 100, 1000`.
  - 162 candidates per month.
- Validation score: fast non-overlapping forward-return candidate scorer.
- Final backtest: global strict bar-by-bar MDD, no monthly reset, 0.5x leverage, 4bp fee, 1bp slippage, next-bar entry.
- Output: `results/rolling_price_action_combo_scan_2026-06-23.json`.

## Result

| Metric | Value |
| --- | ---: |
| Months | 29 |
| Selected months | 29 |
| Trades | 726 |
| Long / Short | 354 / 372 |
| Return | -39.26% |
| CAGR | -18.68% |
| Strict MDD | 41.98% |
| CAGR / strict MDD | -0.44 |
| Mean trade return | -0.063% |
| Mean-return p-value approx | 0.112 |

Selection distribution:

| Selected group | Months |
| --- | ---: |
| `pa_market` | 10 |
| `pa_derivatives` | 9 |
| `pa_market_external_derivatives` | 3 |
| `pa_trend` | 3 |
| `pa_external` | 2 |
| `pa_only` | 2 |

Horizon distribution:

| Horizon bars | Months |
| ---: | ---: |
| 288 | 17 |
| 144 | 11 |
| 72 | 1 |

## Interpretation

This is a hard no-go for the current weak-alpha bundle implementation.

The user's point is correct: weak alphas should be combined. But this specific combination method does not extract a stable tradable edge:

- The rolling selector sees many viable validation candidates each month, but those do not transfer into next-month realized trades.
- The result has enough trades to reject “too few samples” as the primary excuse.
- Long/short balance is reasonable, so this is not just one-sided bias.
- The failure mode is selection overfit / non-stationarity: validation edge decays before target execution.

## Decision

Do not promote current price-action ridge bundle labels into Gemma/RLLM training.

Use price-action features only as raw context candidates until a more robust objective is found. The next useful change is not more grid search; it is changing the target/objective:

1. Replace directional forward-return labels with event/structure labels that match price action: breakout continuation, failed breakout, liquidity sweep, range reclaim.
2. Require context persistence across multiple rolling windows before trading.
3. Let the LLM consume compact symbolic state, not ridge-selected action labels.
4. Keep final evaluation as rolling unseen-month strict backtest.
