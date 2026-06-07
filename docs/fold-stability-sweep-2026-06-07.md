# Fold-stability config sweep (2026-06-07)

## Purpose
The `h144 / target 1.8% / stop 1.5%` template produced a single validation fold with `CAGR/strict MDD > 3`, but that was not stable enough. This sweep selects trader configs by multi-fold stability rather than one validation winner.

## Fixed economics
```json
{"horizon_bars":144,"target_pct":1.8,"stop_pct":1.5}
```

## Selection rule
- Fit calibration tables independently per fold using only rows at or before that fold's train end.
- Evaluate the same config over all 4 chronological folds.
- Rank by positive fold count, strong fold count, worst-fold ratio, average ratio/CAGR.
- Penalize no-trade and low-trade folds so `inf` ratio cannot win.

## Selected stable config
```json
{"level":"teacher_only","min_n":20,"min_score":0.0005,"score_mode":"mean","side_gate":"free"}
```

## Fold results
| Fold | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2024 H1 | 144 | +1.00% | 9.20% | 0.109 |
| 2024 H2→2025 Feb | 175 | +4.97% | 7.78% | 0.639 |
| 2025 H1 val | 82 | +0.97% | 4.52% | 0.213 |
| 2025 H2 OOS | 88 | +0.64% | 8.57% | 0.074 |

Aggregate:
- Positive folds: 4/4.
- Strong folds (`ratio >= 3`): 0/4.
- Minimum fold ratio: 0.074.
- Average ratio: 0.259.
- Minimum trades per fold: 82.

## Interpretation
This is a meaningful improvement in stability: the selected policy no longer relies on a single lucky validation fold and stays slightly profitable in all chronological folds. However, the returns are far too weak for the user's target. It is not a deployable strategy.

## Decision
Keep this stable config as the baseline floor. Next work should improve return magnitude while preserving the 4/4 positive-fold property. The correct next direction is not more single-period gate optimization; it is uncertainty-aware value modeling or richer LLM-derived regime/state features that increase average trade return without sacrificing fold coverage.
