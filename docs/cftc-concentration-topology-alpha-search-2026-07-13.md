# CFTC Concentration-Topology Alpha — Preflight

Date: 2026-07-13

## Hypothesis

The CFTC report discloses concentration among the largest four and eight traders on
each side. Their difference is a coarse rank curve: it estimates whether positions
are distributed across ranks five through eight or isolated in the top four.

Two signed topology measures were tested:

- **rank odds**: log odds of top-four versus ranks-five-to-eight concentration on
  the long side, minus the same odds on the short side;
- **rank curvature**: `(2 × top4 − top8)` long concentration minus the same short
  curvature.

Large absolute prior-standardized topology multiplied by contracting total-trader
breadth is interpreted as a fragile one-sided convoy. Both fading and following the
concentrated side were predeclared. This uses neither Binance ownership/OI handoff
nor price-state/microstructure inputs.

The initially considered stablecoin-supply collision was not tested: the cached
DefiLlama history is a presently reconstructed series without point-in-time vintages,
so using it would introduce revision-leakage risk.

## Protocol

- Physical market and CFTC release rows strictly before `2024-01-01`; OOS unopened.
- Conservative availability at CFTC report date plus eight days.
- Weekly features use prior-only 104-report standardization and 4/13-week breadth
  changes; fit thresholds use only 2020-06 through 2022.
- 32 policies: topology `{rank odds, rank curvature}`, breadth horizon `{4,13}`
  weeks, fit tail `{q50,q70}`, mapping `{fade,follow}`, hold `{2d,4d}`.
- First 5-minute bar at/after release decides, next-open executes; 0.5x, 6 bp/side,
  split-contained exit and conservative strict MDD.
- Admission required CAGR/MDD at least 3 in fit and 2023, positive 2023 halves,
  and minimum fit/full-year/half-year trades of 25/8/3.

## Highest-ranked policy

Rank curvature, 13-week breadth change, q50 fragility, fade, two-day hold:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/Short |
|---|---:|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | +55.93% | +18.75% | 19.31% | +0.97 | 68 | 4 / 64 |
| Selection 2023 | +2.92% | +2.92% | 3.41% | +0.86 | 9 | 2 / 7 |
| 2023 H1 | +1.51% | +3.07% | 2.17% | +1.41 | 1 | 0 / 1 |
| 2023 H2 | +1.39% | +2.78% | 3.41% | +0.82 | 8 | 2 / 6 |

The strategy is overwhelmingly short, loses 8.59% in 2022 H2, and has only one
2023 H1 trade. No candidate has three trades in each 2023 half; none is statistically
adequate for admission despite the attractive aggregate fit return.

## Controls

| Variant | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -43.16% / -0.40 / 68 | -3.99% / -0.72 / 9 |
| Remove breadth multiplier | +62.03% / +1.10 / 68 | +8.96% / +1.93 / 10 |
| Swap long/short concentration fields | -43.16% / -0.40 / 68 | -3.99% / -0.72 / 9 |
| Delay release another four weeks | +34.28% / +0.54 / 70 | -5.80% / -0.74 / 7 |

The direction is specific, but breadth does not improve the result and the exact
topology is heavily regime- and side-dependent. This is not a balanced general alpha.

## Cost decomposition

| Cost per side | Fit return / CAGR / MDD / ratio | 2023 return / CAGR / MDD / ratio |
|---|---:|---:|
| 0 bp | +62.43% / +20.64% / 18.87% / +1.09 | +3.48% / +3.48% / 3.12% / +1.11 |
| 1 bp | +61.33% / +20.33% / 18.94% / +1.07 | +3.38% / +3.39% / 3.17% / +1.07 |
| 3 bp | +59.15% / +19.70% / 19.09% / +1.03 | +3.20% / +3.20% / 3.26% / +0.98 |
| 6 bp | +55.93% / +18.75% / 19.31% / +0.97 | +2.92% / +2.92% / 3.41% / +0.86 |

Costs are not the failure cause. Temporal support, side imbalance and regime drift are.

## Decision

Reject the exact concentration-tail/fade/follow/fixed-hold family before OOS. Record
the topology only as failure provenance, not a beta or alpha feature: its apparent fit
edge is concentrated in short exposure and does not maintain adequate 2023 support.
Do not tune nearby weekly tails or holds.

Artifacts:

- `training/search_cftc_concentration_topology_alpha.py`
- `results/cftc_concentration_topology_alpha_scan_2026-07-13.json`
- `tests/test_search_cftc_concentration_topology_alpha.py`
