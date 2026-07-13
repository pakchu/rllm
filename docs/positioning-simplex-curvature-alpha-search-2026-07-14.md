# Positioning simplex curvature alpha search (2026-07-14)

## Decision

**Reject the static alpha; retain only raw cohort-curvature geometry as weak
beta; keep 2024+ sealed.**

The experiment used three public Binance USD-M positioning ratios, delayed by
one complete 5-minute source row:

1. top-trader position long/short ratio;
2. global account long/short ratio;
3. taker long/short volume ratio.

Log ratios form three commensurate log-odds coordinates. At each completed hour
the model compared the latest six-hour migration vector with the preceding
six-hour migration vector. Their normalized cross product, projected onto the
cohort-consensus axis, measured non-collinear bending. The one-cell energy was:

`positive_6h_OI_change * migration_speed * abs(simplex_curvature)`.

The fit-frozen upper quintile was treated as late, leveraged crowd rotation and
the net six-hour cohort migration was faded for a fixed 12 hours. “Exhaustion”
is an economic hypothesis; only the ratios and geometry are observed.

## Causal and validation protocol

- Returned market and metrics frames are hard-filtered before `2024-01-01`.
  The shared parser may read and immediately discard later rows in a cutoff-
  crossing chunk; none enters returned frames, features, thresholds, or returns.
- Every positioning field is shifted by one complete 5-minute market row before
  completed-hour sampling. Source timestamps are asserted no later than
  decision time minus five minutes.
- One feature-only threshold: fit q80. No return-based parameter selection.
- One primary fade map, with exact continuation as the direction-flip control.
- One fixed six-hour migration scale and one fixed 12-hour hold.
- Next-open execution, 0.5x, 6 bp/side, split-contained non-overlapping trades,
  and favorable-first/adverse-second OHLC strict MDD.
- 2022 is reported only as a quarantine because the official archive has a
  large top-trader-field coverage gap. It is excluded from admission.
- 2023 is inspected internal selection. No 2024+ outcome was computed.

## Support-only preflight

All counts and the q80 threshold were frozen before returns were opened.

- Fit valid feature hours: 10,520.
- 2023 valid feature hours: 8,754.
- Fit q80 energy threshold: `0.0014066474`.

| Split | Raw (L/S) | Strict executable (L/S) |
|---|---:|---:|
| Fit | 2,104 (1,072/1,032) | 456 (226/230) |
| 2020Q4 | 248 (129/119) | 71 (31/40) |
| 2021H1 | 1,013 (522/491) | 206 (103/103) |
| 2021H2 | 843 (421/422) | 178 (92/86) |
| 2022 quarantine | 185 (106/79) | 41 (25/16) |
| 2023 | 1,097 (579/518) | 294 (149/145) |
| 2023H1 | 603 (314/289) | 153 (73/80) |
| 2023H2 | 494 (265/229) | 140 (76/64) |

Support and side balance were ample outside the disclosed 2022 gap.

## Primary results

| Split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| Fit | **-37.20%** | -31.85% | 53.28% | -0.60 | 456 (226/230) |
| 2020Q4 | +14.55% | +88.91% | 5.70% | 15.61 | 71 (31/40) |
| 2021H1 | -34.70% | -57.68% | 45.34% | -1.27 | 206 (103/103) |
| 2021H2 | -16.36% | -29.86% | 29.89% | -1.00 | 178 (92/86) |
| 2022 quarantine | -9.44% | -9.45% | 13.31% | -0.71 | 41 (25/16) |
| 2023 | **-16.96%** | -16.97% | 24.17% | -0.70 | 294 (149/145) |
| 2023H1 | -0.36% | -0.72% | 12.03% | -0.06 | 153 (73/80) |
| 2023H2 | -17.20% | -31.24% | 19.67% | -1.59 | 140 (76/64) |

The excellent short 2020Q4 result immediately failed in both 2021 halves and
both 2023 halves. It is a regime accident, not a robust alpha.

## Structural and timing controls

| Control | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact continuation flip | -17.33% / -0.40 | -17.47% / -0.72 |
| OI build × speed; no curvature | -13.53% / -0.37 | -0.27% / -0.02 |
| Curvature × speed; no OI | -30.79% / -0.74 | -13.24% / -0.62 |
| Migration speed only | -38.80% / -0.72 | -9.02% / -0.48 |
| OI build × pairwise dispersion | -32.61% / -0.72 | -9.12% / -0.43 |
| Signal delayed 5m | -40.25% / -0.63 | -16.66% / -0.69 |
| Signal delayed 1h | -33.65% / -0.56 | -21.96% / -0.81 |
| Signal delayed 24h | -31.25% / -0.56 | -12.13% / -0.76 |
| Signal delayed 7d | -18.05% / -0.37 | -5.54% / -0.35 |

Neither direction works. Curvature changes event membership—the largest event
Jaccard against structural controls was 0.483—but the energy magnitude is
dominated by simpler OI-speed geometry:

| Control score | Spearman vs primary energy |
|---|---:|
| OI build × migration speed | **0.949** |
| OI build × pairwise dispersion | **0.948** |
| Curvature × speed | 0.230 |
| Speed only | 0.051 |

The preregistered novelty gate therefore failed. The cross-product term is not
enough to identify a distinct executable energy after multiplication by OI
build and speed.

## Cost stress

| Cost/side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | -17.43% / -0.33 | -0.93% / -0.06 |
| 1 bp | -21.11% / -0.39 | -3.80% / -0.24 |
| 3 bp | -27.99% / -0.48 | -9.30% / -0.51 |
| 6 bp | -37.20% / -0.60 | -16.96% / -0.70 |
| 10 bp | -47.67% / -0.71 | -26.17% / -0.82 |
| 15 bp | -58.35% / -0.79 | -36.27% / -0.89 |

The strategy loses even at zero cost, so turnover is not the root failure.

## Conclusion

The raw three-cohort migration vectors and their curvature are causal,
observable, and can remain weak-beta tokens for a materially different learner
or RLLM. The inspected energy product, q80 tail, fade/continuation directions,
six-hour scale, 12-hour hold, and all static ablations are frozen as gamma
failure provenance. Do not rescue them with another OI multiplier, threshold,
duration, or gate on the same pre-2024 sample.

## Artifacts

- `training/search_positioning_simplex_curvature_alpha.py`
- `tests/test_search_positioning_simplex_curvature_alpha.py`
- `results/positioning_simplex_curvature_alpha_scan_2026-07-14.json`
