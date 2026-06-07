# Linear value policy rejection (2026-06-07)

## Purpose
The stable bucket baseline is weak but positive in 4/4 chronological folds. This experiment tested whether a richer value model over compact LLM/analyzer context could improve per-trade return while preserving fold stability.

## Method
- Fixed economics: `h144 / target 1.8% / stop 1.5%`.
- Features: compact analyzer symbolic/state/context tokens plus side interactions.
- Model: dependency-free ridge linear regression over hypothetical LONG/SHORT net returns after fee+slippage.
- Policy: choose the side with highest predicted value after uncertainty penalty; abstain below threshold or insufficient side gap.
- Selection: 560 configs ranked by 4-fold stability, not by a single validation or OOS period.

## Best selected config
```json
{"alpha":10.0,"threshold":-0.0005,"risk_penalty":1.0,"min_gap":0.0}
```

## Best fold results
| Fold | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2024 H1 | 112 | +1.50% | 6.51% | 0.230 |
| 2024 H2→2025 Feb | 201 | -18.71% | 16.63% | -1.125 |
| 2025 H1 val | 97 | -7.62% | 8.37% | -0.911 |
| 2025 H2 OOS | 87 | +3.40% | 4.31% | 0.788 |

Aggregate:
- Positive folds: 2/4.
- Strong folds: 0/4.
- Minimum ratio: -1.125.
- Average ratio: -0.255.
- Minimum trades: 87.

## Interpretation
The linear value model increases OOS return relative to the weak stable baseline, but it loses the stability property and fails badly in the middle folds. This is worse than the current bucket baseline for the user's objective because it reintroduces regime overfit.

## Decision
Reject the current one-hot ridge value model as the next trading policy. Keep the implementation as a diagnostic scaffold, but the next model needs either:
1. better sequence-aware features, or
2. LLM-generated reasoning/state compression that is trained to be stable across folds, or
3. an objective that explicitly penalizes mid-fold loss during training/selection.

Do not replace the stable baseline with this model.
