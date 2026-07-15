# Delayed conditional-pullback eval — 2026-07-15

**REJECTED_FROZEN_EVAL**

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

| Window | 6 bp/side result | 10 bp/side stress | Gate pass |
|---|---:|---:|---:|
| eval_2025 | 9.82% / 9.83% / 7.22% / 1.36 / 18 | 9.04% / 9.04% / 7.22% / 1.25 / 18 | False |
| holdout_2026h1 | 8.76% / 22.37% / 5.25% / 4.26 / 24 | 7.72% / 19.57% / 5.47% / 3.58 / 24 | True |
| eval_all | 19.45% / 13.38% / 7.22% / 1.85 / 42 | 17.46% / 12.04% / 7.22% / 1.67 / 42 | False |

## Integrity

- Pinned eval manifest `2cc5d9f5837188372b3b9239c04f2873e7a43389aa6d0a385f0b100f028839bc` was validated before 2025+ was opened.
- The delayed feature matrix, activation, thresholds, and every train/selection/test schedule through 2024 replayed exactly.
- No eval quantile, model choice, exit, or threshold is computed from 2025+.
- Execution is next-open, 6 bp/notional/side, realized funding, stop-before-take, split-contained, wall-clock CAGR, and strict path MDD.
- Candidate-level eval is implementation clean; global epistemic purity remains false because related feature families were researched previously.
