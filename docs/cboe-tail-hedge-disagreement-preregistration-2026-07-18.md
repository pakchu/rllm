# CTHD-1: Cboe Tail-Hedge Disagreement preregistration

Status: **frozen before any exact-policy post-entry BTC outcome was inspected**.

## Hypothesis

Cboe describes SKEW as a view into tail-risk option demand and VVIX as the
expected volatility of 30-day forward VIX:

- <https://www.cboe.com/insights/posts/inside-volatility-trading-the-adventures-of-volatility-markets>
- <https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf>
- <https://cdn.cboe.com/resources/indices/documents/vvix-termstructure.pdf>

CTHD-1 looks for a state in which tail hedging and volatility uncertainty are
high **relative to the visible VIX level**. The hypothesis is deliberately
short-only: hidden convexity demand can precede broader risk repricing, while
the inverse state does not supply an equally strong long mechanism.

The signal source contains no BTC price, return, volume, funding, premium, OI,
FX, on-chain, existing-alpha, or portfolio input.

## Frozen singleton

1. Compute `log(SKEW / 100)`, `log(VVIX / VIX)`, and `log(VIX)`.
2. Strict-prior midrank each value against at most 252 earlier Cboe
   observations; require 126; append current values only after ranking.
3. Compute
   `hidden_pressure = 0.5 * (SKEW_rank + VVIX/VIX_rank) - VIX_rank`.
4. Strict-prior rank that pressure again against at most 252 earlier available
   pressure values; require 126; append only after ranking.
5. If `hidden_pressure_rank >= 0.775`, short; otherwise abstain.
6. Enter at 09:35 America/New_York on the next Cboe source date and exit at
   09:35 on the following source date. No synthetic date fill is allowed.
7. Use 0.5x exposure, 6 bp/notional/side base cost, 10 bp stress cost, exact
   funding, full-calendar CAGR, and intratrade strict MDD.

The 22.5% upper tail was chosen only from source-event support. It is the
sparsest inspected tail with at least 150 Stage-1 events, at least 30 in each
Stage-1 year, at least 140 in sealed 2023, at least 20 in each 2023 half, and no
more than 16% single-month concentration in the aggregate Stage-1 and 2023
windows. No BTC outcome informed the threshold.

| Source-only window | Events | Short |
|---|---:|---:|
| 2021 | 123 | 123 |
| 2022 | 34 | 34 |
| Stage 1 | 157 | 157 |
| 2023 H1 | 124 | 124 |
| 2023 H2 | 23 | 23 |
| sealed 2023 | 147 | 147 |

The uneven subperiod opportunity counts are disclosed, not repaired. The
outcome gate requires each subperiod to be independently positive and weekly
cluster inference prevents the dense months from being treated as independent
daily evidence.

## Sequential windows

- Stage 1: 2021–2022, with separate 2021 and 2022 gates.
- Stage 2: calendar 2023, opened only after exact Stage-1 replay and pass.
- 2024, 2025, and 2026 YTD remain sealed after Stage 2.

Both stages require positive absolute return, CAGR/strict MDD at least 3,
strict MDD at most 15%, weekly cluster sign-flip `p <= 0.10`, positive 10 bp
stress return, at least 35 bp mean gross underlying edge, sufficient trade
support, and positive subperiods.

## Controls

- SKEW rank alone
- `VVIX/VIX` rank alone
- low-VIX rank alone
- SKEW plus `VVIX/VIX` without the visible-VIX subtraction
- exact direction flip
- one Cboe-release delay
- seven Cboe-release placebo delay

The primary CAGR/MDD must exceed the best source-component control by at least
0.25. A control cannot replace the primary after outcomes are opened.

Only a performance pass opens trade-level orthogonality and marginal-portfolio
testing against the already frozen promoted/live/shadow universe.

## Version boundary

Cboe announced in 2025 that SKEW methodology modifications were being
developed. The frozen panel ends in 2023 and live promotion requires explicit
active-methodology parity:
<https://cdn.cboe.com/resources/release_notes/2025/Consultation-Results-Regarding-Proposed-Changes-to-the-Cboe-SKEW-Index-SKEW-.pdf>.

Integrity anchors:

- preregistration manifest hash:
  `4daf843d48b7fcc259c3f5a6bc533e74a3ae94ffd8a172fef49c0bb8ad8ddb91`
- preregistration JSON SHA-256:
  `d9e0e767e293d17c4845d300dad22c113b863796ef309d4d06ec8ecbe7330d0b`
- primary source-only clock SHA-256:
  `aba459bac8fd2b3ff911a596f6d99cf7f417803e74f791b93fdd1e4c88e04099`
