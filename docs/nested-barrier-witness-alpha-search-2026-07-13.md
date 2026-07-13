# Nested Barrier-Witness Alpha — Preflight

Date: 2026-07-13

## Hypothesis

Rolling extrema are usually reduced to a price level. This experiment preserves the
**witness state** of the bar that created that level.

At every completed 5-minute bar, prior-only rolling highs and lows are located over
12-hour, 2-day and 1-week horizons. A nested barrier exists when at least two or all
three horizons point to extrema created within three bars of one another and current
price revisits those levels.

The strategy compares direction-specific taker work over the three bars ending at
the longest-horizon ancestor with work during the current approach:

- work ratio at most `0.75`: less effort reaches the same barrier, interpreted as
  depleted opposing liquidity; continue through the high/low;
- work ratio at least `1.25`: more effort fails to close through the barrier,
  interpreted as reinforced absorption; fade the touch.

This combines multi-scale price topology with origin-state retrieval. It is not a
plain rolling-max/min breakout, age counter, hit count or price-level scar.

## Causal protocol

- Physical source rows strictly before `2024-01-01`; 2024+ OOS stayed unopened.
- Every barrier is built from `[t-window, t)` and excludes decision bar `t`.
- Ties choose the most recent prior extreme; the longest touched horizon owns the
  witness work. Current completed-bar work executes only at the next 5-minute open.
- 16 policies: minimum coalescence `{2,3}`, touch width `{10,20}` bp, mapping
  `{depleted continuation, reinforced fade}`, hold `{6h,12h}`.
- Work ratios `0.75/1.25` and three-bar witness windows are structurally fixed.
- 0.5x, 6 bp/side, split-contained exits and conservative strict MDD.
- Admission required positive fit/2023 and CAGR/MDD at least 3 in both, with
  minimum 80/20/6 trades.

## Highest-ranked policy

Three-horizon coalescence, 10 bp touch, depleted continuation, 12-hour hold:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/Short |
|---|---:|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | +15.86% | +5.86% | 13.64% | +0.43 | 91 | 67 / 24 |
| Selection 2023 | +16.06% | +16.08% | 8.40% | +1.91 | 26 | 21 / 5 |
| 2023 H1 | +8.90% | +18.78% | 5.17% | +3.63 | 13 | 11 / 2 |
| 2023 H2 | +6.58% | +13.48% | 8.40% | +1.60 | 13 | 10 / 3 |

Four of five fit half-years are positive, but 2022 H2 loses 3.97% and full-fit risk
efficiency is weak. The same signals with a 6-hour hold reach 2023 ratio `3.42`
and both 2023 halves exceed `3`, yet fit return is only `+1.54%`, ratio `0.05`,
with three negative fit halves. That is 2023 regime fit, not general alpha evidence.

## Structural controls at 6 bp/side

| Variant | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -25.05% / -0.35 / 91 | -17.07% / -0.85 / 26 |
| Ignore witness work | -1.22% / -0.03 / 162 | +14.65% / +2.16 / 41 |
| Ignore cross-scale coalescence | -42.95% / -0.32 / 665 | -0.26% / -0.01 / 299 |
| Delay signal by one hour | +9.77% / +0.28 / 91 | +10.45% / +1.25 / 26 |

Removing either ancestral work or cross-scale coalescence materially damages fit;
the exact direction flip loses strongly. The representation therefore contains real
structure, but its magnitude and directional balance are insufficient.

## Cost decomposition

| Cost per side | Fit return / CAGR / MDD / ratio | 2023 return / CAGR / MDD / ratio |
|---|---:|---:|
| 0 bp | +22.36% / +8.12% / 13.64% / +0.60 | +17.89% / +17.90% / 8.40% / +2.13 |
| 1 bp | +21.25% / +7.74% / 13.64% / +0.57 | +17.58% / +17.60% / 8.40% / +2.09 |
| 3 bp | +19.07% / +6.99% / 13.64% / +0.51 | +16.97% / +16.99% / 8.40% / +2.02 |
| 6 bp | +15.86% / +5.86% / 13.64% / +0.43 | +16.06% / +16.08% / 8.40% / +1.91 |

The effect survives realistic costs. Fit instability and low CAGR/MDD—not turnover—
are the rejection reasons.

## Decision

Reject the exact 16 touch/work-ratio/fixed-hold policies as standalone alpha before
OOS. Promote only the underlying nested-barrier ancestry and witness-work ratio to
weak beta-feature status. Do not tune nearby widths, work ratios or holds on the same
sample; a later learner must treat them as continuous context with fresh-forward proof.

Artifacts:

- `training/search_nested_barrier_witness_alpha.py`
- `results/nested_barrier_witness_alpha_scan_2026-07-13.json`
- `tests/test_search_nested_barrier_witness_alpha.py`
