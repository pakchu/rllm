# Multi-horizon event-context check (2026-06-23)

## Purpose

The clean nested eval showed that 2026 side edge collapsed. One plausible cause was the fixed 288-bar label/horizon. This test rebuilds the same event-context dataset and pairwise preference ranker for shorter horizons:

- 36 bars
- 72 bars
- 144 bars
- baseline 288 bars from the previous run

All runs keep the same causal token construction and rolling prior-only protocol.

## Full rolling results

| Horizon | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 251 | -13.25% | 35.84% | -0.37 | 0.077 |
| 72 | 278 | -8.24% | 34.39% | -0.24 | 0.380 |
| 144 | 328 | 3.09% | 27.50% | 0.11 | 0.669 |
| 288 | 488 | 8.82% | 22.06% | 0.40 | 0.367 |

Shorter horizons do not fix the side-edge problem. The 144-bar horizon is mildly positive over the full rolling window but weak and high-drawdown.

## 2026-only eval by horizon

| Horizon | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 41 | -4.59% | 11.46% | -0.40 | 0.831 |
| 72 | 56 | -15.34% | 14.46% | -1.06 | 0.539 |
| 144 | 71 | -13.79% | 17.07% | -0.81 | 0.646 |
| 288 | 77 | -21.00% | 14.47% | -1.45 | 0.423 |

Every horizon loses in 2026 when it actually trades.

## Overlay diagnostic for h144

The best full-window h144 overlay diagnostic:

| Stop | Take | Rolling loss stop | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 4% | 10 trades / 5% | 176 | 18.47% | 9.01% | 2.05 | 0.028 |

This is interesting, but it does not solve the clean 2026 issue because h144 2026-only remains negative before eval leakage.

## Interpretation

The 2026 failure is not just a horizon mismatch. The side model is selecting the wrong side/action in 2026 across 36/72/144/288 horizons.

The only robust behavior so far is abstention from 2026 when prior validation health is poor. That is useful risk control, not a profit source.

## Decision

Do not spend more iterations on simple horizon swaps in the same event-context preference setup.

Next work should target 2026-specific alpha discovery directly:

1. Diagnose which side/events lose in 2026 and whether inversion helps.
2. Search for a separate 2026-like regime branch instead of one shared side model.
3. Consider short-specific features/action families rather than symmetric LONG/SHORT ranking.
