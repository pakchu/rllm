# CIHM-1 preregistration — 2026-07-18

## Frozen hypothesis

**Cboe Institutional Hedge Migration (CIHM-1)** tests whether a simultaneous,
unusually large one-session increase in three option-flow proxies precedes BTC
weakness during the next Cboe source session:

1. index-option put/call relative to equity-option put/call;
2. VIX-option call volume relative to VIX-option put volume;
3. index-option share of all-product option volume.

The economic interpretation is an inference: aggregate Cboe volume does not
identify institutions, buyers versus sellers, opening versus closing trades,
or multi-leg intent.  The candidate is therefore a falsifiable migration proxy,
not a claim that every VIX call or index put is a bearish hedge.

Official inputs:

- [Cboe daily market statistics](https://www.cboe.com/us/options/market_statistics/daily/)
- [Cboe historical options data information](https://www.cboe.com/us/options/market_statistics/historical_data/)

## Causal formula

For completed source session `t`:

```text
institutional_gap_t =
  log((index_put_t + 0.5) / (index_call_t + 0.5))
  - log((equity_put_t + 0.5) / (equity_call_t + 0.5))

vix_call_pressure_t =
  log((vix_call_t + 0.5) / (vix_put_t + 0.5))

index_share_t =
  log((index_total_t + 1) / (all_products_total_t + 1))
```

Each input becomes `x_t - x_(t-1)`.  Its score is a strict-prior midrank
against at most the preceding 252 source-session changes, requiring 126 prior
changes.  The current value is appended only after its rank is fixed.

```text
score_t = mean(rank(delta institutional_gap),
               rank(delta vix_call_pressure),
               rank(delta index_share))

score_t >= 0.575 -> SHORT
otherwise        -> ABSTAIN
```

Weights are equal.  There is no BTC price, return, taker, funding, OI, premium,
FX, Kimchi, REX, implied-volatility, calendar, LLM, or regime input.

## Clock and execution

- Source: completed Cboe option-statistics session.
- Entry: next valid Cboe option-statistics date, 09:35 America/New_York.
- Exit: following valid source date, 09:35 America/New_York.
- Source no-data dates: no forward fill and no synthetic session.
- Side: short only.
- Leverage: 0.5x.
- Base cost: 6 bp/notional/side; stress cost: 10 bp/notional/side.
- Funding: exact marks satisfying `entry <= funding_time < exit`.
- CAGR: complete wall-clock split including idle cash.
- MDD: strict intratrade path with global/pre-entry HWM and all costs/funding.

## Outcome-blind support choice

No BTC market or funding row was loaded.  The threshold grid used only source
event counts:

| score threshold | 2021 | 2022 | Stage 1 | sealed 2023 |
|---:|---:|---:|---:|---:|
| 0.600 | 57 | 60 | 117 | 50 |
| 0.590 | 65 | 66 | 131 | 57 |
| 0.580 | 72 | 75 | 147 | 63 |
| **0.575** | **74** | **78** | **152** | **65** |
| 0.570 | 78 | 81 | 159 | 67 |
| 0.560 | 88 | 86 | 174 | 74 |
| 0.550 | 93 | 95 | 188 | 83 |

The preregistered rule selects the highest threshold with at least 150 Stage-1
events, 70 per Stage-1 year, 60 sealed-2023 events, 25 per sealed half, and no
more than 15% of events in one month.  Threshold 0.575 is the first pass when
scanned from high to low.  It gives 30 events in 2023 H1 and 35 in H2.

## Controls frozen before outcomes

| Control | Definition | Stage-1 clocks |
|---|---|---:|
| institutional-gap only | change rank >= 0.70 | 157 |
| VIX-call-pressure only | change rank >= 0.70 | 151 |
| index-share only | change rank >= 0.70 | 142 |
| level composite | level-rank mean >= 0.575 | 211 |
| direction flip | primary clock, long instead of short | 152 |
| one-release delay | primary state shifted one source session | 151 |
| seven-release placebo | primary state shifted seven source sessions | 151 |

The primary must beat the best component/level control CAGR/MDD by at least
0.25.  A winning control cannot replace a failed primary.

## Sequential gates

### Stage 1 — 2021–2022

- positive absolute return;
- CAGR / strict MDD >= 3.0;
- strict MDD <= 15%;
- weekly clustered sign-flip `p <= 0.10`;
- at least 150 trades, all short;
- at least 70 trades and positive absolute return in each year;
- mean gross underlying edge >= 35 bp/trade;
- positive absolute return at 10 bp/notional/side;
- mechanism margin >= 0.25.

### Stage 2 — sealed 2023, only after an exact Stage-1 pass

- same economic/risk/statistical gates;
- at least 60 trades total and 25 per half;
- positive absolute return in both halves.

Only after both stages pass may existing-sleeve correlation, trade overlap, and
marginal portfolio utility be inspected.  A failure rejects CIHM-1 without
direction, threshold, feature, clock, size, cost, or regime repair.

## Frozen identities

- Preregistration manifest hash:
  `3dd49c08c685a191f686f42cf6a27af30d057ea92529cc4008db54a4980582fe`
- Preregistration file SHA-256:
  `0709c7aff57dc1e1e7079979ec44ceb0e154c47898ea593f2bfe50d1ab4052d5`
- Outcome-blind primary clock SHA-256:
  `188196f1ea8d6ecd741306419e540b9ec9c11800d9b96d3d2ad591cc3fc94cf0`
