# Path-shape analyzer/trader data (2026-06-06)

## Why this exists

The prior future-best hold labels create oracle-like action targets. This generator replaces the target with explicit future path diagnostics:

- long/short MFE and MAE
- first target/stop event timing
- risk/reward ratio from MFE-to-MAE
- path grade (`CLEAN_TARGET`, `NOISY_TARGET`, `STOP_FIRST`, `DRIFT_POSITIVE`, `NO_EDGE`)
- direction pressure (`LONG_FAVORED`, `SHORT_FAVORED`, `NO_TRADE_FAVORED`, `BOTH_SIDES_VOLATILE`)

This is still future-derived training data, but it is a better representation for an analyzer/trader stack because it exposes path shape and invalidation rather than hiding them behind a single hindsight best hold.

## Generated template

- horizon: 144 bars
- target: 1.0%
- stop: 0.6%
- entry delay: 1 bar

## Generated split counts

| split | rows | LONG_FAVORED | SHORT_FAVORED | NO_TRADE_FAVORED |
| --- | ---: | ---: | ---: | ---: |
| train | 2370 | 749 | 749 | 872 |
| val | 552 | 162 | 163 | 227 |
| OOS | 535 | 151 | 176 | 208 |

The pressure distribution is much more balanced and stable than the previous likelihood-action collapse.

## Leakage discipline

- Generated labels may use future bars because they are training targets.
- Model fitting must remain train-only.
- Hyperparameter/model selection may use val only.
- OOS must remain report-only.

## Next step

Use these rows to build a two-stage LLM dataset:

1. Analyzer SFT: prompt summary → path-shape JSON.
2. Trader/RL: prompt summary + analyzer path-shape → stop/target template action, rewarded by strict path PnL.
