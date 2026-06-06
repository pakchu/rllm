# Event-memory action policy validation (2026-06-06)

## Hypothesis

After a live-safe opportunity detector fires, select an action from train-only memory buckets keyed by past-only analyzer fields. This tests whether the previously identified bottleneck is mostly action selection rather than event detection.

## Leakage guard

- Event detector is fit on `data/economic_value_h36_72_144_288_432_train.jsonl` only.
- Action memory is built from train positive events only (`utility >= 0.003`).
- Validation/OOS utilities are not used for action selection.
- Large oracle opportunity results remain non-tradable upper bounds and must not be reported as live performance.

## Fixed selection

Val sweep selected the strongest non-oracle configuration:

- fields: `trend_alignment,location,momentum`
- min bucket: `3`
- event threshold: `0.7`
- val: 68 trades, CAGR `25.69%`, strict MDD `10.80%`, CAGR/MDD `2.38`, mean trade `+0.1773%`, p `0.2689`

## OOS result

Same fixed config on OOS:

- 79 trades
- CAGR `-43.44%`
- strict MDD `31.95%`
- CAGR/MDD `-1.36`
- mean trade `-0.3428%`
- p `0.0164` against zero mean trade return

## Decision

Reject event-memory action selection as a standalone trader. It improved val over the prior majority-action baseline but failed OOS badly, suggesting regime drift / unstable action labels rather than a reusable action policy.

## Next implication

The next useful unit is a drift/stability diagnostic: compare train/val/OOS conditional edge by analyzer field, action, hold period, and event score bucket before trying another optimizer. A strategy that looks good only through val-selected action routing is not acceptable for live trading.
