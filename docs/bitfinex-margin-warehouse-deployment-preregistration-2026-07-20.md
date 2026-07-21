# BFMWD-144 preregistration — 2026-07-20

## Frozen hypothesis

Bitfinex publishes separate `fUSD` and `fBTC` margin-funding balance sheets.
A financing **warehouse** forms when provided-but-unused funding rises. A later
transition in which used funding and utilization rise while that warehouse
contracts represents capital deployment into financed positions.

The frozen directional hypothesis is:

- `fUSD` warehouse deployment → **long BTC**; and
- `fBTC` warehouse deployment → **short BTC**.

This direction is fixed before any source numeric row or BTC outcome is read.
It is a hypothesis, not a claim that every Bitfinex USD/BTC funding movement is
a BTC long/short.

## Frozen source boundary

- source: official Bitfinex public funding statistics;
- symbols: exactly `fUSD`, `fBTC`;
- physical rows: 2020-01-01 through 2023-12-31 only;
- conservative availability: source hour + 15 minutes;
- warm-up: 2020;
- train: 2021–2022;
- selection: 2023;
- 2024+ remains physically sealed; and
- source construction may not load market prices, returns, Binance funding,
  labels, PnL, CAGR, or MDD.

The committed source decision contains the official API references and bounded
coverage probes:
`docs/bitfinex-margin-warehouse-deployment-source-decision-2026-07-20.md`.

## Frozen feature algebra

For each symbol and hourly anchor `t`:

```text
total_t       = funding_amount_t
used_t        = funding_amount_used_t
unused_t      = total_t - used_t
utilization_t = clip(used_t / total_t, 1e-6, 1 - 1e-6)

charge(t,w,d) = log1p(unused[t-d]) - log1p(unused[t-d-w])
deploy(t,d)   = log1p(used[t]) - log1p(used[t-d])
draw(t,d)     = log1p(unused[t-d]) - log1p(unused[t])
util(t,d)     = logit(utilization[t]) - logit(utilization[t-d])
```

Each feature is standardized by its own median and `1.4826 × MAD` over the
strictly prior 1,440 valid hourly feature observations. The current feature is
excluded; at least 1,080 prior observations are required. Zero-MAD or missing
history invalidates the anchor rather than filling it.

The current `average_period_days` must also be at or above its strictly prior
1,440-hour median. No forward/backward fill is permitted.

## Frozen four-candidate family

| candidate | warehouse `w` | deployment `d` | z floor | hold |
|---|---:|---:|---:|---:|
| `bfmwd_w12_d3_z10_h12` | 12h | 3h | 1.0 on all four features | 12h |
| `bfmwd_w24_d3_z10_h12` | 24h | 3h | 1.0 on all four features | 12h |
| `bfmwd_w12_d6_z10_h12` | 12h | 6h | 1.0 on all four features | 12h |
| `bfmwd_w24_d6_z10_h12` | 24h | 6h | 1.0 on all four features | 12h |

An anchor triggers only when all four robust z-scores meet the floor and the
tenor confirmation passes. Only a transition from non-trigger to trigger may
open an event. If `fUSD` and `fBTC` trigger together, BFMWD abstains. Events are
globally non-overlapping within each candidate.

Entry is one complete five-minute bar after conservative `available_at`; exit
is the scheduled open exactly 144 five-minute bars later. Leverage is fixed at
0.5x. There is no TP, SL, trailing exit, dynamic hold, regime gate, REX gate,
LLM side selection, or portfolio-aware veto.

## Source-only controls

These controls diagnose mechanism necessity and are never promotable:

1. remove the warehouse-charge prerequisite;
2. remove unused-funding draw confirmation;
3. remove tenor confirmation; and
4. delay all source observations by 24 hours.

## Source-support gates

Each candidate is evaluated without opening BTC outcomes. It must have:

- at least 60 train events and 30 selection events;
- at least 20 events in each train year;
- at least 12 events in each 2023 half;
- each side between 20% and 80%;
- no calendar month above 20%;
- no weekday above 25%;
- no rolling 14-day interval above 20%;
- exact-entry Jaccard no greater than 0.10 versus frozen comparator clocks; and
- bidirectional ±6-hour containment no greater than 0.35.

If no candidate passes, BFMWD is retired before market evaluation. Support
failure cannot be repaired by weakening thresholds or adding source fields.

## Frozen economic contract if support passes

Only source-supported candidates may enter the strict evaluator:

- BTCUSDT USD-M 5-minute next-open execution;
- exact realized funding;
- 6 bp/notional/side base cost and 10 bp/notional/side stress;
- full-declared-calendar absolute return and CAGR, including cash time;
- strict path MDD including pre-entry HWM, entry/exit costs and every held
  favorable-then-adverse bar extreme;
- one extra five-minute entry-delay control;
- weekly-cluster one-sided inference;
- 100,000-draw synchronized seven-day circular-block Romano–Wolf step-down
  max-t correction across all source-supported candidates; seed `20260720`.

Train and selection must independently have positive absolute return,
`CAGR/strict MDD >= 3`, `strict MDD <= 15%`, positive contained halves,
positive long and short contributions, mean gross side-adjusted move at least
30 bp, weekly-cluster adjusted `p <= 0.10`, positive 10 bp stress with ratio at
least 2.5, and a positive one-bar-delay result.

Train failure keeps 2023 sealed. Selection failure keeps every 2024+ value
sealed. No sign inversion, parameter repair, source extension, model rescue,
or LLM/RL gate may reuse the BFMWD identifier after a failed frozen stage.
