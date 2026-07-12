# Advanced state-model alpha comparison

Date: 2026-07-13

> Superseded for promotion purposes by
> `docs/top10-state-model-generalization-2026-07-13.md`. This document preserves
> the earlier strict rank-1-only interpretation.

## Result

Four causal state-model approaches were compared on the same fixed
funding/premium long setup. Only the simple observable Markov transition gate
survived both Eval 2025 and 2026 YTD above CAGR/strict-MDD 5.

| frozen model winner | split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---|---:|---:|---:|---:|---:|
| Observable Markov transition | Test 2024 | +37.88% | 37.79% | 3.27% | 11.56 | 22 |
|  | Eval 2025 | +18.33% | 18.34% | 2.83% | 6.48 | 19 |
|  | 2026 YTD | +8.82% | 22.51% | 3.74% | 6.03 | 23 |
| Local-linear Kalman state | Test 2024 | +24.85% | 24.79% | 2.54% | 9.75 | 17 |
|  | Eval 2025 | +9.55% | 9.56% | 4.27% | 2.24 | 16 |
|  | 2026 YTD | +1.73% | 4.21% | 4.20% | 1.00 | 18 |
| BOCPD run-length state | Test 2024 | +22.68% | 22.63% | 3.94% | 5.74 | 22 |
|  | Eval 2025 | -3.68% | -3.68% | 7.72% | -0.48 | 14 |
|  | 2026 YTD | +8.52% | 21.70% | 4.84% | 4.48 | 19 |
| Explicit-duration semi-Markov | Untouched Test 2024 | +27.84% | 27.78% | 1.78% | 15.64 | 20 |
|  | Eval 2025 | +1.63% | 1.63% | 7.33% | 0.22 | 11 |
|  | 2026 YTD | +3.61% | 8.90% | 4.98% | 1.79 | 18 |

All CAGRs count the entire calendar window, including idle periods. All MDDs
include intraposition adverse excursion.

## Interpretation

### Why the simpler Markov model won

The successful Markov overlay asks one low-capacity question: did a coarse,
observable hourly state persist for a second hour? Its selected transition set
keeps 19-23 trades per later window and remains interpretable.

The more flexible models create many ways to partition the same sparse setup
events:

- Kalman: slope x innovation state;
- BOCPD: segment mean x short-run posterior mass x context;
- semi-Markov: observable state x duration bucket.

With only roughly 10-25 trades per later annual block, these additional state
degrees of freedom mainly improve a selected historical path. They do not add
a new information source.

### Why attractive lower rows are rejected

Each failed scan contains lower-ranked variants with good Eval or 2026 values.
Choosing those rows after seeing later windows would turn report-only periods
into selection data. No such row was promoted.

### Model-specific findings

- Fixed-noise Kalman covariance converges deterministically and is not a market
  uncertainty feature; it was removed before the final scan.
- Under constant BOCPD hazard, immediate reset probability is nearly constant;
  short-run posterior mass is the meaningful causal statistic.
- Explicit duration generated a real-looking 2024 effect, including a positive
  trade-return bootstrap interval, but the interval crossed zero in 2025 and
  the CAGR/MDD collapsed.

## Decision

1. Keep `markov_persistent_funding_premium_long_20260712` as the only promoted
   state-model research alpha from this series.
2. Do not add the Kalman, BOCPD or semi-Markov winners to the portfolio.
3. Do not continue increasing state-model capacity on the same OHLCV/flow/base
   trigger. The marginal feature information is exhausted relative to the
   available trade count.
4. The next genuinely different advanced-model search should use a new event
   source—liquidations, order-book imbalance, options surface or richer OI
   events—or use state models for conservative sizing/uncertainty rather than
   binary alpha selection.

## Primary references

- Kalman filtering: <https://doi.org/10.1115/1.3662552>
- Bayesian online change-point detection: <https://arxiv.org/abs/0710.3742>
- Hidden semi-Markov models: <https://www.cs.ubc.ca/~murphyk/papers/segment.pdf>

## Artifacts

- Markov report: `docs/markov-regime-alpha-search-2026-07-12.md`
- Kalman report: `docs/kalman-state-alpha-search-2026-07-13.md`
- BOCPD report: `docs/bocpd-state-alpha-search-2026-07-13.md`
- Semi-Markov report: `docs/semimarkov-duration-alpha-search-2026-07-13.md`
