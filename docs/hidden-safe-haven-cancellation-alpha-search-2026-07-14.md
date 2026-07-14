# Hidden Safe-Haven Cancellation Alpha Search — 2026-07-14

## Decision

**Rejected as alpha.** The state is genuinely different from the existing DXY,
BTC-reversal and raw-FX controls, but its declared direction loses before costs
in both fit and 2023. The exact opposite direction also loses fit. No 2024+
outcome was opened.

## Thesis

An aggregate dollar index can look calm while JPY and CHF strengthen against
USD relative to EUR, GBP, CAD and SEK. That cross-sectional cancellation may
reveal risk aversion hidden by the broad-dollar average. If BTC has not moved
in the corresponding direction, the remaining standardized stress could be a
delayed BTC signal.

For each completed FX hour:

1. form exact continuous six-hour returns for six USD pairs;
2. orient every pair as USD strength;
3. standardize each six-hour return by `sqrt(6)` times its prior-only 30-day
   one-hour volatility, after removing six times the prior one-hour mean;
4. define risk stress as JPY/CHF strength relative to the other four pairs;
5. add BTC's standardized six-hour return to obtain unpriced stress;
6. retain only states where risk stress and unpriced stress have the same sign;
7. trade the opposite side of the fit-only q90 absolute unpriced-stress tail.

Positive unpriced stress is a short and negative unpriced stress is a long.

## Pre-outcome review and repairs

The first independent review returned `REVISE`, not `GO`. Before opening any
forward return, the implementation was changed to:

- aggregate explicit minute-59 source completion instead of using a
  left-labelled hourly resample;
- place the signal on the minute-00 BTC bar and enter only at minute-05, one
  complete 5-minute bar after the FX/BTC information boundary;
- scale six-hour returns from prior one-hour volatility with a fixed minimum of
  240 valid observations so weekend gaps do not collapse time;
- replace the proposed multiplicative score with absolute unpriced stress,
  avoiding a second loading of the safe-haven state.

The revised implementation then passed independent code review and 11 targeted
tests before outcomes were opened.

## Causal protocol

- Returned market and FX frames are strictly before `2024-01-01`.
- A cutoff-crossing parser chunk may be read and discarded; no discarded row
  enters a returned frame or computation.
- Every usable FX hour requires all six pairs, at least 55 one-minute rows per
  pair, and a minute-59 observation.
- Missing/weekend hours remain gaps. Six-hour returns require a continuous
  seven-observation path.
- Fit is `2020-06-01..2022-12-31`; 2023 is inspected internal selection with
  H1/H2 robustness. 2024+ remains sealed.
- Entry is minute-05 open, hold is fixed 12 hours, leverage is `0.5x`, and
  implementation cost is `6bp/side`.
- Trades are non-overlapping and split-contained. Strict MDD uses the
  favorable-first/adverse-second OHLC high-water convention.

## Support-only gate

Support was counted before `_future_extreme`, simulation or result writing was
allowed.

| Split | Executable trades | Long | Short |
|---|---:|---:|---:|
| Fit | 271 | 132 | 139 |
| 2023 | 128 | 61 | 67 |
| 2023 H1 | 66 | 32 | 34 |
| 2023 H2 | 61 | 28 | 33 |

There were 21,827 valid completed FX hours. All preregistered support minima
passed.

## Primary result

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| Fit | -18.79% | -7.74% | 24.92% | -0.31 | 271 | 132/139 |
| 2023 | -20.71% | -20.72% | 24.88% | -0.83 | 128 | 61/67 |
| 2023 H1 | -9.13% | -17.57% | 16.67% | -1.05 | 66 | 32/34 |
| 2023 H2 | -13.20% | -24.49% | 15.59% | -1.57 | 61 | 28/33 |

Only `2021 H2` was positive among the seven reported fit/selection half-year
segments. This is not a stable edge.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| FX safe-haven only | +5.51% / 0.13 / 308 | -16.87% / -0.89 / 107 |
| BTC reversal only | -27.25% / -0.30 / 560 | -34.37% / -0.98 / 193 |
| Broad USD only | +18.92% / 0.44 / 341 | -14.63% / -0.75 / 137 |
| Raw unstandardized contrast | +4.72% / 0.08 / 320 | -19.81% / -0.82 / 129 |
| Exact direction flip | -15.97% / -0.23 / 271 | +6.54% / 0.48 / 128 |
| Signal delay 1h | -12.94% / -0.28 / 271 | -20.78% / -0.85 / 128 |
| Signal delay 24h | -12.53% / -0.30 / 271 | -11.26% / -0.66 / 128 |
| Signal delay 7d | +7.19% / 0.16 / 272 | -20.66% / -0.86 / 125 |

The exact flip improves 2023 but remains negative in fit, so this is not a
simple sign error. Broad USD has isolated early-fit profitability and then
fails 2022 H2 and both 2023 halves.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | -4.45% / -0.10 | -14.38% / -0.75 |
| 1bp | -7.01% / -0.15 | -15.46% / -0.77 |
| 3bp | -11.91% / -0.23 | -17.60% / -0.80 |
| 6bp | -18.79% / -0.31 | -20.71% / -0.83 |
| 10bp | -27.14% / -0.36 | -24.67% / -0.87 |
| 15bp | -36.38% / -0.40 | -29.34% / -0.90 |

Zero-cost failure rejects a transaction-cost explanation.

## Novelty audit

Maximum event Jaccard against the four component controls was `0.207`:

- FX safe-haven only: `0.207`;
- BTC reversal only: `0.174`;
- broad USD only: `0.110`;
- raw contrast: `0.167`.

The score's largest inspected component Spearman correlation was `0.553` with
absolute BTC response and `0.406` with absolute safe-haven stress. The state is
therefore not a disguised copy of one component, but novelty did not create
predictive value.

## Freeze decision

- Keep the continuous safe-haven-versus-broad-USD cancellation state only as a
  weak beta feature for a materially different preregistered learner.
- Record the exact q90 same-sign fade, six-hour scale and 12-hour hold as gamma
  failure provenance.
- Do not tune nearby FX baskets, q tails, signs, response formulas, delays or
  holds on this inspected pre-2024 sample.

## Artifacts

- `training/search_hidden_safe_haven_cancellation_alpha.py`
- `tests/test_search_hidden_safe_haven_cancellation_alpha.py`
- `results/hidden_safe_haven_cancellation_alpha_scan_2026-07-14.json`
