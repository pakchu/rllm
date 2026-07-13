# Positioning-Disagreement Lifecycle Hazard Search — 2026-07-13

## Thesis

A positioning disagreement may matter less at its extreme than at its first
resolution. The experiment treats the gap between Binance top-trader positions
and all accounts as an episode:

1. prior-only 30-day disagreement z-score reaches `|1.5|`;
2. the same sign survives for at least `12h` or `36h`;
3. the first 50% contraction or zero-cross becomes a resolution hazard;
4. trade opposite the aged disagreement side as marginal inventory unwinds.

This is not another static crowding tail. It models **age, peak memory and first
passage out of disagreement**.

## Causal and data protocol

- Market and metrics sources are physically cut before `2024-01-01`.
- Binance USD-M metrics are delayed by one complete 5-minute source bar before
  use.
- Rolling means and standard deviations use values through `t-1` only.
- Any unavailable metric row resets the episode.
- The entire 2022 archive is forcibly invalidated and resets state because the
  top-trader fields contain large coverage gaps; it is neither fit nor selection.
- Fit is `2020-10-15..2021-12-31`; selection is 2023 with separate halves.
- Signal at completed bar `t`, entry at next open, `0.5x`, `6bp/side`.
- Strict MDD uses favorable-first/adverse-second OHLC high-water.
- `2024+` OOS was not opened.

## Frozen grid

Sixteen policies:

- disagreement: top-position minus global account, top-account minus global;
- minimum episode age: `12h`, `36h`;
- resolution: 50% contraction, zero-cross;
- hold: `6h`, `18h`.

Entry `|z|=1.5`, contraction fraction `0.5`, and reset `|z|<0.25` were fixed.

## Best ranked policy

Top-position minus global, age `36h`, first zero-cross, hold `18h`:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +30.96% | +24.90% | 6.76% | 3.68 | 40 | 22/18 |
| 2023 | +7.04% | +7.04% | 6.52% | 1.08 | 24 | 12/12 |
| 2023H1 | +7.00% | +14.63% | 5.82% | 2.51 | 12 | 7/5 |
| 2023H2 | +0.04% | +0.07% | 6.52% | 0.01 | 12 | 5/7 |

All three fit segments were positive with ratios `8.27`, `4.28`, and `6.03`.
The fit mean-trade test was strong (`p≈0.0019`, effect size `d≈0.49`), but 2023
was not (`p≈0.33`) and H2 was economically flat (`p≈0.98`).

## Structural controls

At standard costs:

- exact direction flip: fit `-27.93%`, 2023 `-9.71%`;
- ignore episode age: fit `+11.89%`, ratio `0.49`; 2023 `+3.53%`, ratio `0.41`;
- static `|z|=1.5` tail onset: fit `-74.35%`, 2023 `-10.00%`;
- causal 12-hour signal lag: fit `+10.30%`, ratio `0.47`; 2023 `+7.80%`, ratio
  `1.34`.

The static tail and direction controls strongly support the lifecycle object,
while the positive lag control shows timing is broad rather than sharply
identified. At zero cost, fit ratio is `4.05` and 2023 ratio is `1.34`; costs
reduce but do not create the selection weakness.

## Decision

**Not admitted as alpha.** Zero of 16 policies met CAGR/strict-MDD `>=3` in both
fit and 2023. OOS therefore stayed sealed.

The continuous episode representation is retained as **weak beta** because it
has balanced sides, positive fit segments, a losing exact flip, a catastrophic
static-tail ablation, and survives standard cost. The fixed threshold/age/hold
policy is gamma failure provenance. Do not tune nearby ages, z thresholds,
contraction fractions or holds on this sample.

Artifacts:

- `training/search_positioning_lifecycle_hazard_alpha.py`
- `tests/test_search_positioning_lifecycle_hazard_alpha.py`
- `results/positioning_lifecycle_hazard_alpha_scan_2026-07-13.json`
