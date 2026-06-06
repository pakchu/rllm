# Economic label/action drift diagnostic (2026-06-06)

## Purpose

Explain why live-safe event detection can look promising while fixed action policies fail OOS.

## Configuration

- Value data: `h36,72,144,288,432`
- Positive utility threshold: `0.003`
- Memory fields: `trend_alignment,location,momentum`
- Event threshold: `0.7`
- Leakage guard: event detector and action memory fit on train only; val/OOS labels are reporting-only.

## Split-level oracle opportunity labels

Best-action labels remain abundant:

| split | signals | positive best-action rate | mean best utility |
| --- | ---: | ---: | ---: |
| train | 2370 | 54.35% | +0.768% |
| val | 552 | 55.98% | +0.689% |
| OOS | 535 | 60.19% | +0.779% |

This says there are many hindsight opportunities. It does **not** prove tradability.

## Fixed candidate actions are all negative on average

Every individual trade action has negative mean utility in train, val, and OOS. Examples:

| action | train mean | val mean | OOS mean |
| --- | ---: | ---: | ---: |
| LONG 432 | -1.031% | -0.973% | -1.613% |
| SHORT 432 | -1.535% | -1.143% | -0.806% |
| LONG 288 | -0.912% | -0.846% | -1.278% |
| SHORT 288 | -1.218% | -0.919% | -0.722% |
| LONG 36 | -0.369% | -0.308% | -0.366% |
| SHORT 36 | -0.364% | -0.310% | -0.311% |

## Train-memory action choice failure

Train-memory event-filtered choices were dominated by LONG 432:

| split | event signals | chosen mean utility | win rate | dominant action |
| --- | ---: | ---: | ---: | --- |
| train | 873 | -1.093% | 34.14% | LONG 432 |
| val | 147 | -1.061% | 37.41% | LONG 432 |
| OOS | 179 | -1.781% | 22.91% | LONG 432 |

The strict backtest can look less bad or temporarily positive due to cooldown/path sampling, but the underlying chosen-action utility is not a stable positive edge.

## Conclusion

The main issue is not just overfitting of a gate. The current action-label formulation creates a hindsight best-action oracle, while live-safe action selection maps to actions whose unconditional/conditional averages are negative. Optimizing gates or memory buckets around these labels is structurally wrong.

## Next direction

Move away from `pick the future-best action among many candidate holds` as the primary supervised target. The next candidate should learn an analyzer output that describes **risk/reward path shape and invalidation**, then a trader should act only when a pre-declared action template has positive train/val stability. In other words: first prove a template has stable positive expectancy, then let LLM/RL specialize when to use it; do not ask the model to infer a hindsight action lottery.
