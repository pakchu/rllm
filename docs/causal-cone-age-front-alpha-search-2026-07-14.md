# Causal-cone breach age-front search (2026-07-14)

## Decision

**Reject the four static execution policies. Preserve the continuous age-front
state as weak beta only. Keep 2024+ sealed.**

The experiment asks a different question from static rupture mass: when price
breaches a family of expectations frozen at 168 prior hourly anchors, is the
breach spreading from younger generations into older generations? A coherent
outward wave requires all of the following on the same dominant side:

1. weighted q80 breached-anchor age moves outward;
2. weighted mean breached-anchor age moves outward; and
3. breach excess mass grows.

The score is the geometric mean of positive front velocity and positive
log-mass growth, with a nonnegative old-mass-share bonus. Geometry was frozen
at the preceding cone experiment: 2016 five-minute bars, hourly anchors,
width 2. Only 1h/6h change lags and 6h/12h holds formed the four-policy grid;
the positive-score q80 threshold was fitted on the fit split only.

## Leakage and execution protocol

- Market file was physically truncated before `2024-01-01` before feature
  generation. No 2024+ statistic was opened.
- Every anchor precedes its minute-55 decision. Each anchor's scale is a
  shifted rolling volatility ending before that anchor.
- Completed minute-55 state enters at the next minute-00 open.
- Fixed 0.5x exposure, 6 bp per side, non-overlapping fixed holds.
- Strict MDD uses favorable-first/adverse-second OHLC ordering and
  split-contained exits.
- All pre-2024 settings are exploratory; 2023 is inspected internal selection,
  not pristine OOS.

## Four-policy result

| Lag / hold | Split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---|---:|---:|---:|---:|---:|
| 6h / 12h (rank 1) | fit | +2.82% | +1.26% | 17.17% | 0.07 | 87 (38/49) |
| 6h / 12h (rank 1) | 2023 | +5.16% | +5.17% | 7.42% | 0.70 | 35 (26/9) |
| 6h / 12h (rank 1) | 2023 H1 | +2.51% | +5.14% | 7.42% | 0.69 | 19 (15/4) |
| 6h / 12h (rank 1) | 2023 H2 | +2.58% | +5.20% | 5.91% | 0.88 | 16 (11/5) |
| 6h / 6h | fit | -11.17% | -5.21% | 26.19% | -0.20 | 104 (43/61) |
| 6h / 6h | 2023 | +7.38% | +7.39% | 5.39% | 1.37 | 45 (33/12) |
| 1h / 6h | fit | -4.78% | -2.19% | 24.61% | -0.09 | 216 (96/120) |
| 1h / 6h | 2023 | +4.58% | +4.59% | 7.77% | 0.59 | 87 (61/26) |
| 1h / 12h | fit | +11.15% | +4.89% | 18.98% | 0.26 | 165 (75/90) |
| 1h / 12h | 2023 | -3.18% | -3.18% | 12.06% | -0.26 | 65 (44/21) |

Zero of four policies passed admission. The rank-1 policy lost in 2021 H1
(-9.97%, CAGR/MDD -1.46) and 2021 H2 (-2.23%, -0.50), despite both 2023 halves
being positive. Approximate fit mean-return p-value was 0.797 with effect size
`d=0.028`; the observed edge is not statistically persuasive.

## Falsification controls

| Control | Fit return / ratio | 2023 return / ratio | Interpretation |
|---|---:|---:|---|
| Exact direction flip | -14.39% / -0.36 | -9.29% / -0.77 | Direction contains information. |
| Mass growth only | +1.60% / 0.03 | +11.84% / 1.91 | 2023 benefit is not unique to age propagation. |
| Front velocity only | -20.79% / -0.34 | +2.18% / 0.22 | Age movement without mass coherence is weak. |
| Retreat then fade | -0.75% / -0.04 | -0.06% / -0.04 | Contraction reversal is unsupported and sparse. |
| Signal delayed 7 days | -5.43% / -0.17 | -0.25% / -0.03 | The limited edge is time-local, not a permanent side label. |
| Current-volatility rewrite | +5.94% / 0.17 | +4.97% / 0.72 | Frozen-anchor semantics are not uniquely supported. |
| Reversed anchor-age order | +3.26% / 0.09 | +1.47% / 0.59 | Age destruction leaves only 13/3 trades and no usable edge. |

At zero cost, rank 1 was only +8.33% / 0.24 in fit and +7.40% / 1.04 in
2023. At 10 bp per side fit already turned negative (-0.70%). This is not a
hidden gross alpha defeated solely by conservative costs.

## Conclusion

The direction flip and seven-day delay controls show that the cone ensemble
contains local directional information. However, age-front ordering does not
create stable executable risk efficiency, and the mass-only control is stronger
in 2023. Freeze the q80 mapping, lags, holds, front definition, and controls.
Retain only continuous upper/lower age-front, centroid, old-share, mass and
velocity tokens as weak beta for a materially different learner or genuinely
fresh-forward evidence.

## Artifacts

- `training/search_causal_cone_age_front_alpha.py`
- `tests/test_search_causal_cone_age_front_alpha.py`
- `results/causal_cone_age_front_alpha_scan_2026-07-14.json`
