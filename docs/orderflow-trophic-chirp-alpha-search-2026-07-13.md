# Order-Flow Trophic Chirp Alpha — Preflight

Date: 2026-07-13

## Hypothesis

A campaign count discards the spacing between participant-succession events. This
experiment treats the event stream like a chirp signal: three clean same-direction
q95 sponsor-to-crowd successions form two interarrival gaps.

- If the new gap is at most 75% of the old gap, participation cadence is accelerating
  and the strategy continues in the event direction.
- If the new gap is at least 150% of the old gap, campaign cadence is decelerating
  and the strategy reverses for exhaustion.

The triplet is rejected if an opposite event occurred after its first event. This is
a causal temporal grammar over event intervals, not another role-tail or count sweep.

## Protocol

- Physical source rows strictly before `2024-01-01`; 2024+ OOS stayed unopened.
- Parent q95 continuation roles are frozen; only current and prior event timestamps
  are visible when a chirp is classified.
- 48 policies: six parent phase profiles, maximum pair gap `{144,288}` bars,
  branch `{acceleration continuation, deceleration reversal}`, hold `{144,288}`.
- Ratios `0.75/1.50` were fixed from the structural definition, not optimized.
- Next-open entry, 0.5x, 6 bp/side, split-contained exit and conservative strict MDD.
- Admission required fit/2023 CAGR/MDD at least 3 and minimum 50/16/5 trades.

## Highest-ranked temporal-breadth policy

Profile `(6,12,6)`, maximum gap 288 bars, acceleration continuation, hold 144 bars:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | +16.27% | +6.01% | 6.52% | +0.92 | 26 |
| Selection 2023 | +5.37% | +5.37% | 4.88% | +1.10 | 13 |
| 2023 H1 | +4.57% | +9.43% | 4.88% | +1.93 | 10 |
| 2023 H2 | +0.77% | +1.53% | 1.20% | +1.28 | 3 |

All five fit half-years and both 2023 halves were positive, but support is below the
predeclared threshold: only 26 fit trades and three 2023 H2 trades. The lowest fit
half-year ratio is 0.14, so temporal breadth does not imply sufficient efficiency.

## Structural controls

| Variant | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -16.88% / -0.38 / 26 | -6.79% / -0.81 / 13 |
| Ignore cadence; any clean triplet | +1.55% / +0.04 / 43 | +1.33% / +0.16 / 21 |
| Remove opposite-event quarantine | +6.51% / +0.23 / 29 | +5.17% / +1.06 / 15 |
| Swap sponsor/crowd phase order | -8.33% / -0.27 / 45 | +13.31% / +2.12 / 30 |
| Delay chirp by 24 bars | +11.36% / +0.85 / 26 | +2.93% / +0.55 / 13 |

Cadence compression and the exact direction matter: the chirp beats the unpaced
triplet and its flip loses. The phase-order placebo's 2023-only strength plus negative
fit warns that participant ordering remains regime-dependent.

## Cost decomposition

| Cost per side | Fit return / CAGR / MDD / ratio | 2023 return / CAGR / MDD / ratio |
|---|---:|---:|
| 0 bp | +18.10% / +6.65% / 6.52% / +1.02 | +6.19% / +6.20% / 4.88% / +1.27 |
| 1 bp | +17.80% / +6.54% / 6.52% / +1.00 | +6.06% / +6.06% / 4.88% / +1.24 |
| 3 bp | +17.18% / +6.33% / 6.52% / +0.97 | +5.78% / +5.79% / 4.88% / +1.19 |
| 6 bp | +16.27% / +6.01% / 6.52% / +0.92 | +5.37% / +5.37% / 4.88% / +1.10 |

The edge survives implementation cost; insufficient magnitude and sample support are
the limiting factors.

## Decision

Reject the exact 48 max-gap/branch/fixed-hold policies as a standalone alpha before
OOS. Retain interarrival compression and opposite-event quarantine only as weak
continuous event-grammar features. Do not tune nearby interval ratios, gap limits or
holds on the same pre-2024 sample.

Artifacts:

- `training/search_orderflow_trophic_chirp_alpha.py`
- `results/orderflow_trophic_chirp_alpha_scan_2026-07-13.json`
- `tests/test_search_orderflow_trophic_chirp_alpha.py`
