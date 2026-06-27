# Episode Reward Focus Oracle Policy Upper Bound (2026-06-27)

## Question
Before spending more time extracting full Gemma focus scores for train/test/eval, does the focused label surface itself contain enough trading value to justify a downstream policy?

## Method
Added:

`training/export_reward_focus_oracle_policy_predictions.py`

It converts focused reward rows into single-policy prediction rows using an explicit oracle rule:

```text
TRADE if utility_bucket == UTILITY_HIGH and path_shape == CLEAN_WIN_PATH
side = candidate.side
exit_profile = candidate.horizon bucket
otherwise NO_TRADE
```

This is an upper-bound diagnostic only. It uses future-derived target labels and is not deployable.

Backtest command path:

```bash
training.export_reward_focus_oracle_policy_predictions
training.backtest_single_policy_predictions
```

Execution uses actual OHLC bar-by-bar strict MDD with next-bar entry, leverage 0.5, fee 0.0004, slippage 0.0001.

## Results

### Test period: 2024-01-01 to 2025-12-31
Input rows: 41,274
Oracle actions:
- LONG: 8,140
- SHORT: 4,311
- NO_TRADE: 28,823

Strict backtest:
- trades: 529
- CAGR: 621.73%
- strict MDD: 4.62%
- CAGR / strict MDD: 134.70
- mean trade return: 0.753%
- p-value approximation: 0.0

### Eval period: 2026-01-01 to 2026-05-30
Input rows: 8,304
Oracle actions:
- LONG: 1,519
- SHORT: 1,073
- NO_TRADE: 5,712

Strict backtest:
- trades: 103
- CAGR: 708.36%
- strict MDD: 2.42%
- CAGR / strict MDD: 292.25
- mean trade return: 0.828%
- p-value approximation: 0.0

## Interpretation
The oracle result is intentionally impossible to deploy because it uses future-derived labels. The value is diagnostic:

- The `UTILITY_HIGH + CLEAN_WIN_PATH` surface has a very strong upper-bound under strict OHLC execution.
- Therefore the focused label target is worth trying to approximate with a causal model.
- This supports the current direction: Gemma should act as an auxiliary path/utility annotator feeding a calibrated downstream policy.

## Caveats
- This is not a live-valid strategy.
- The label is future-derived, so the high CAGR/MDD numbers are not evidence of deployable alpha.
- Backtest rows contain many candidate rows per signal; the strict simulator enforces one active trade at a time through position overlap handling.
- The next real gate is causal approximation: train-only model/scores, test selection, eval untouched.

## Next step
Use Gemma focus outputs/scores or cheaper causal approximations of those labels as features in a calibrated downstream model, then validate with train/test/eval without target echo.
