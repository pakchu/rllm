# Stable context policy eval checkpoint (2026-06-23)

## Purpose

The row-level context SFT failed to learn action labels, so labels were smoothed into train/test-stable token contexts. This checkpoint verifies whether the frozen context map itself has deployable edge before spending more LLM training time.

## Context selection

Builder:

```bash
.venv/bin/python -m training.stable_context_policy_dataset \
  --input-jsonl data/llm_context_regime_policy_h288_2026-06-23.jsonl \
  --output data/stable_context_policy_h288_2026-06-23.jsonl \
  --context-keys trend_alignment,risk_state,dxy_zscore,kimchi_premium_zscore,funding_zscore,premium_index_zscore \
  --min-train-rows 8 \
  --min-test-rows 3 \
  --min-train-mean-pct 0.05 \
  --min-test-mean-pct 0.00 \
  --min-train-gap-pct 0.05 \
  --min-test-gap-pct -0.05
```

Selection is train/test only. Eval is transformed by the frozen selected-context map.

Selected contexts:

- Train contexts: 1811
- Test contexts: 934
- Selected contexts: 35
- Selected rows: train 705, test 266, eval 59

Action distribution after smoothing:

| Split | LONG | SHORT | NO_TRADE |
| --- | ---: | ---: | ---: |
| train | 555 | 150 | 5867 |
| test | 207 | 59 | 1930 |
| eval | 45 | 14 | 539 |

## Gemma SFT smoke

Stable-context SFT48:

- Model: `google/gemma-4-E4B-it`
- Train split only.
- Balanced rows: 450 = 150 LONG / 150 SHORT / 150 NO_TRADE.
- Runtime: 320.6s.
- Train loss: 1.063.

Test balanced generation, 60 rows:

- Accuracy: 31.7%.
- Better than the row-level SFT64 generation result (26.7%) but still not useful.

## Symbolic frozen-context backtest

This checks the selected context map directly by allowing target echo on the transformed stable-context labels. It is not an LLM claim, but it is a valid diagnostic for whether the selected contexts have edge.

| Split | Period | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | 2024-07-01 to 2025-12-31 | 174 | 32.08% | 6.30% | 5.10 | 0.0069 |
| eval | 2026-01-01 to 2026-05-30 | 41 | -6.62% | 7.58% | -0.87 | 0.7474 |

## Conclusion

This path is **NO_GO** as currently selected. The test result is strong and statistically meaningful, but the untouched 2026 eval result is negative. That means the context map overfit the train/test regime even after smoothing.

The useful finding is the failure boundary:

- Row-level oracle labels: not learnable by Gemma.
- Stable context labels: learnability slightly improves, but underlying context alpha fails untouched eval.
- Therefore the next valid direction is **rolling context selection**, not another static test-selected context map.

Next implementation target:

1. Monthly rolling context selector: for each month, fit context map only on data before that month.
2. Enforce minimum support and side consistency over multiple historical subwindows.
3. Backtest rolling context map before additional Gemma training.
4. Only if rolling symbolic context edge survives should Gemma be trained to imitate/compress it.
