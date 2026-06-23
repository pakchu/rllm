# Rolling stable-context policy checkpoint (2026-06-23)

## Why

Static train/test context selection produced a strong test backtest but failed untouched 2026 eval. To remove that split-specific overfit, context selection was changed to rolling monthly selection.

## Protocol

For each target month from 2024-07 through 2026-05:

1. Use only rows before the target month.
2. Train window: 730 days before the validation window.
3. Validation window: 180 days immediately before the target month.
4. Select stable contexts on train/validation only.
5. Transform the target month with the frozen selected context map.
6. Strict backtest all monthly predictions together.

Command output:

- Predictions: `data/rolling_stable_context_policy_h288_2024_2026_2026-06-23.jsonl`
- Summary: `results/rolling_stable_context_policy_h288_2024_2026_2026-06-23.summary.json`
- Backtest: `results/rolling_stable_context_policy_h288_2024_2026_2026-06-23.backtest.json`

## Result

| Period | Samples | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-07-01 to 2026-05-30 | 2794 | 109 | -2.40% | 14.19% | -0.17 | 0.7822 |

## Interpretation

This confirms the earlier suspicion: the static stable-context result was overfit to the train/test split. Once context selection is rolled forward using only prior data, the edge disappears.

The current valid conclusion is **NO_GO** for this feature/context family. More Gemma training on these labels is not justified until the symbolic rolling context selector has positive expectancy.

Next research branch should change the information source, not the optimizer:

- broaden context features beyond the current 6 key buckets;
- use longer timeframe regime state or multiasset relative state;
- require consistency across multiple validation windows before emitting a trade label;
- only then train Gemma to compress/imitate the surviving rolling policy.
