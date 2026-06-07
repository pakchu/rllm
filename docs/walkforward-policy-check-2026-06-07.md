# Walk-forward fixed-policy check (2026-06-07)

## Purpose
The action-space sweep found a validation winner (`h144 target 1.8 stop 1.5`) but it failed the fixed OOS check when trained only through 2025-02. This walk-forward check asks whether that template is broadly stable when recalibrated only on information available before each fold.

## Fixed policy
```json
{
  "horizon_bars": 144,
  "target_pct": 1.8,
  "stop_pct": 1.5,
  "level": "teacher_only",
  "min_n": 50,
  "min_score": 0.0005,
  "score_mode": "mean",
  "side_gate": "free"
}
```

## Fold results
| Fold | Train rows | Test rows | Trades | Return | CAGR | Strict MDD | CAGR/MDD | Mean trade | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 H1 | 1095 | 546 | 76 | -5.72% | -11.20% | 7.92% | -1.42 | -0.0748% | 0.375 |
| 2024 H2→2025 Feb | 1641 | 729 | 135 | +0.29% | +0.43% | 6.65% | 0.07 | +0.0043% | 0.940 |
| 2025 H1 val | 2370 | 552 | 68 | +6.41% | +13.20% | 4.19% | 3.15 | +0.0932% | 0.202 |
| 2025 H2 OOS | 2922 | 535 | 90 | +0.26% | +0.54% | 8.57% | 0.06 | +0.0051% | 0.942 |

## Interpretation
- Recalibrating with all prior data improves the 2025 H2 OOS from the earlier fixed-training result, but only to near-flat.
- 3 of 4 folds are non-negative, but only 1 of 4 reaches the target ratio, and no fold has a statistically persuasive mean trade return.
- The template is more promising than the 36-bar pressure setup, but it is not yet a money-making bot.

## Decision
Keep `h144 target 1.8 stop 1.5` as the current research anchor, not as a deployable strategy. The next work should optimize for fold stability and uncertainty-aware abstention rather than single-period validation score.
