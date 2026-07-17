# CVTR-1: Cboe Volatility Term Rotation preregistration

Status: **frozen before any exact-policy post-entry BTC return was inspected**.

## Hypothesis

Cboe calculates VIX9D, VIX, and VIX3M from SPX options at different target
horizons. The term surface therefore measures a non-crypto risk-transfer state:
<https://www.cboe.com/tradable-products/vix/term-structure>.

CVTR-1 combines two weak, source-independent slopes:

1. `front = log(VIX9D / VIX)`
2. `broad = log(VIX / VIX3M)`

Both are strict-prior ranked. Deep contango on both segments is mapped to BTC
long; broad short-end pressure/backwardation is mapped to BTC short. This is not
a VIX-level threshold and contains no BTC state.

## Frozen singleton

- Rank each current slope against at most 252 strictly earlier Cboe observations;
  require 126 and append the current value only after ranking.
- `score = 0.5 * front_rank + 0.5 * broad_rank`.
- `score <= 0.25`: long; `score >= 0.75`: short; otherwise abstain.
- Use the completed source date only at 09:35 America/New_York on the next Cboe
  observation date.
- Exit at 09:35 on the following Cboe observation date. Source dates define
  weekends and holidays; no synthetic forward-fill is allowed.
- 0.5x exposure, 6 bp/notional/side base cost, 10 bp stress cost, exact funding,
  full-calendar CAGR, and intratrade strict MDD.

The 25% tail was selected without BTC outcomes. It is the sparsest inspected
symmetric tail that meets the preregistered source-only support floor in both
2021-2022 and sealed 2023. No performance statistic informed it.

| Source-only window | Events | Long | Short |
|---|---:|---:|---:|
| 2021 | 148 | 109 | 39 |
| 2022 | 133 | 18 | 115 |
| Stage 1 | 281 | 127 | 154 |
| 2023 H1 | 55 | 50 | 5 |
| 2023 H2 | 46 | 28 | 18 |
| sealed 2023 | 101 | 78 | 23 |

## Sequential windows

- Stage 1: 2021-2022, with separate 2021 and 2022 gates.
- Stage 2: calendar 2023, opened only after exact Stage-1 replay and pass.
- 2024, 2025, and 2026 YTD remain sealed after Stage 2.

Both stages require positive absolute return, CAGR/strict MDD at least 3,
strict MDD at most 15%, sign-flip `p <= 0.10`, positive 10 bp stress result,
adequate gross edge, trade support, both sides, and positive subperiods.

## Controls

- front slope alone
- broad slope alone
- VIX level alone
- exact direction flip
- one Cboe-release delay
- deterministic random side
- constant long on the primary clock

The primary ratio must exceed the best source-component/VIX-level control by
0.25. A control cannot replace the primary after outcomes are opened.

Only a performance pass opens trade-level orthogonality and marginal-portfolio
testing against the previously frozen promoted/live/shadow universe.

Integrity anchors:

- preregistration manifest hash:
  `c366e91f038301cf9f79eab19c58141b9d00a6ad1a1e4165d76065c1894a88bc`
- preregistration JSON SHA-256:
  `af440f63cee9fac526fe9731c2e203ab37e9ffa19847403fbd51845a25c4b7f6`
- primary source-only clock SHA-256:
  `c0250d1f40c87049f6d7639ba43f5285835441399a62968434b65c7d46ed2a93`
