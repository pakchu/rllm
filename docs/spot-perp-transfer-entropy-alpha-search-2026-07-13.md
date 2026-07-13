# Spot→Perp Directional Transfer-Entropy Alpha — Preflight Rejection

Date: 2026-07-13

## Hypothesis

If completed Binance spot returns contain more recent conditional information about
perpetual returns than the reverse direction, a large spot-minus-perp move may be an
unpaid information transfer. The attempted policy traded the perpetual in the sign of
that gap while the causal rolling transfer-entropy advantage favored spot→perp.

This is an information-direction hypothesis rather than a basis or correlation gate.

## Causal protocol

- Physical source cutoff: rows strictly before `2024-01-01` only.
- Spot and perpetual returns are discretized with volatility estimated from prior bars.
- Rolling transfer entropy at decision bar `t` contains transitions ending no later
  than `t-1`; the current and future states are excluded.
- Thresholds are fitted on `2020-06-01..2022-12-31` only.
- `2023` is selection/robustness data; frozen `2024+` OOS remained unopened.
- Hourly decision cadence, next-bar-open execution, 0.5x exposure, 6 bp round-trip
  implementation cost, split-contained forced exits, and conservative strict MDD.
- Search: 96 preflight combinations across two state thresholds, two TE windows, two
  TE tails, two gap tails, three holds, and two direction-consistency modes.

## Strongest adequately populated attempted policy

Parameters: state threshold `1.0`, TE window `8640` bars, TE fit quantile `0.70`,
absolute gap fit quantile `0.80`, hold `48` bars, `gap_only` mapping.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | -41.51% | -18.74% | 45.81% | -0.41 | 695 |
| Selection 2023 | -8.00% | -8.01% | 14.26% | -0.56 | 327 |
| 2023 H1 | -10.38% | -19.83% | 13.25% | -1.50 | 243 |
| 2023 H2 | +2.83% | +5.70% | 3.73% | +1.53 | 83 |

Only two of seven half-year robustness segments were profitable.

## Negative controls

| Control | Fit return / ratio | 2023 return / ratio | Interpretation |
|---|---:|---:|---|
| Exact direction flip | -28.97% / -0.39 | -27.49% / -0.97 | The opposite mapping also loses. |
| Remove TE gate | -42.00% / -0.40 | -33.39% / -0.91 | Gap catch-up alone is not an edge. |
| Require reverse TE | +15.61% / +0.57 | -16.01% / -0.97 | Historical fit does not generalize to 2023. |

## Decision

Reject the exact static usage. Conditional information-flow magnitude changes event
selection, but it does not provide a stable tradable direction after costs. Neither
the intended mapping nor its exact flip generalizes, and the reverse-TE control
collapses in selection. Do not retune this TE-tail/gap-tail policy. A materially
different synchronization-event mechanism must be tested as a fresh hypothesis.

Artifacts:

- `training/search_spot_perp_transfer_entropy_alpha.py`
- `results/spot_perp_transfer_entropy_alpha_scan_2026-07-13.json`
- `tests/test_search_spot_perp_transfer_entropy_alpha.py`
