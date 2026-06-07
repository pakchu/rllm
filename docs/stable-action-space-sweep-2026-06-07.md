# Stable action-space sweep (2026-06-07)

## Purpose
After finding a weak but stable `h144 / target 1.8 / stop 1.5` policy, this run checked whether another economic template could improve fold-stable return magnitude without returning to single-period overfit.

## Method
- Sweep 36 economic templates:
  - horizons: 72, 144, 288 bars
  - targets: 1.2%, 1.8%, 2.5%, 3.5%
  - stops: 1.0%, 1.5%, 2.0%
- Quick config grid around the stable `teacher_only` baseline:
  - min_n: 20, 35, 50
  - min_score: 0.0002, 0.0005, 0.0010
  - score_mode: mean
  - side_gate: free
- Rank every template/config by 4-fold stability, not by a single validation split.

## Selected template
```json
{
  "economics": {"horizon_bars": 144, "target_pct": 1.8, "stop_pct": 1.5},
  "config": {"level":"teacher_only","min_n":20,"min_score":0.0005,"score_mode":"mean","side_gate":"free"}
}
```

## Selected fold results
| Fold | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2024 H1 | 144 | +1.00% | 9.20% | 0.109 |
| 2024 H2→2025 Feb | 175 | +4.97% | 7.78% | 0.639 |
| 2025 H1 val | 82 | +0.97% | 4.52% | 0.213 |
| 2025 H2 OOS | 88 | +0.64% | 8.57% | 0.074 |

## Top template comparison
The selected template is the only one in the quick economic sweep with 4/4 positive folds. The next templates had only 2/4 positive folds and at least one negative fold:
1. `h144/t1.8/s1.5`: 4 positive folds, min ratio 0.074, avg ratio 0.259.
2. `h288/t1.2/s1.5`: 2 positive folds, min ratio -1.353.
3. `h72/t1.8/s1.0`: 2 positive folds, min ratio -1.698.

## Interpretation
The `144/1.8/1.5` anchor survived a broader economic-template search. That is useful, but the absolute edge is still too weak. The next improvement should not search for a new stop/target by single-period score. It should improve per-trade return while preserving 4/4 positive-fold stability.

## Decision
Use `h144 / target 1.8% / stop 1.5% / teacher_only min_n20 score>=0.0005` as the current stable baseline for subsequent LLM+RL feature/value-model work.
