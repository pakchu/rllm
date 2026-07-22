# IVLIR-72 preregistration — intrinsic-volume latent-impact relay

## Decision

The next standalone BTC candidate is **IVLIR-72**. It replaces neither REX nor
the existing volume-clock jump sleeve. It tests a different causal object: the
first time each UTC day consumes one half of the median quote volume of the
previous 28 complete UTC days.

At that first-passage anchor, IVLIR measures cumulative taker flow from the UTC
day origin, the price response to that flow, and current location inside the
completed seven-day rolling range. A trade is admitted only when:

1. absolute cumulative flow is at least its strictly-prior event q60;
2. price has moved weakly in the same direction as flow;
3. directional price impact per unit flow is no higher than its strictly-prior
   aligned-event q70; and
4. the seven-day range still leaves symmetric headroom.

The side is fixed to the cumulative-flow sign. Entry is the next five-minute
open and exit is exactly 72 bars later. The interpretation is delayed impact:
passive liquidity temporarily buffers persistent aggressive inventory, then
quotes adjust in the same direction. Exact side-flip and component controls
must falsify that interpretation if it is wrong.

## Why this is not another prior RLLM retune

Repository review found repeated Gemma SFT/DPO failures on event side,
TAKE/SKIP, option choice, family ranking, and rationale preference. Those runs
mostly learned class or token priors and did not transfer across regimes.
IVLIR therefore creates no LLM label and trains no adapter at this stage.

The candidate clock, side, entry, and hold must first pass standalone train and
2023 selection economics. Only then may a compact LLM choose between
`TRADE_FIXED_SIDE` and `ABSTAIN`; it may never create, reverse, resize, or
retime a candidate.

## Novel causal geometry

- **Not backward volume clock:** previous work searched backward from every bar
  for a fraction of prior 24-hour volume. IVLIR accumulates forward from a
  fixed UTC-day origin and emits at most one first passage per day.
- **Not session handoff:** the anchor time is endogenous to traded notional,
  not fixed at 00:00, 08:00, or 16:00 UTC.
- **Not dual intrinsic clock:** no directional-change event-count race is used.
- **Not a raw extrema rule:** rolling max/min/current price supplies only a
  symmetric headroom constraint; it cannot create direction.

## Frozen source and split boundary

- source: official Binance USD-M `BTCUSDT` five-minute kline archive;
- physical source: `2020-01-01` through `2023-12-31`;
- train: calendar 2020–2022;
- selection: calendar 2023;
- later sealed sequence: 2024, then 2025, then 2026 YTD;
- exact source file SHA-256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`.

This repository has already inspected broad BTC history, so this is a frozen
candidate-level test, not a globally pristine holdout. At this commit the
exact IVLIR source incidence and all IVLIR post-entry returns remain unopened.

## Frozen support gates

Before any post-entry return is calculated, the source-only builder must show:

- at least 120 train events and 30 in each train year;
- at least 35 events in 2023 and 15 in each half;
- each side between 25% and 75% in all/train/selection;
- at least 30 active months;
- no month above 10%, no quarter above 20%, and no same-side run above 15.

Failure retires this exact identity. Flow threshold, impact threshold,
headroom, target fraction, side, latency, and hold may not be repaired after
incidence is opened.

## Frozen economic sequence

After a support pass, a separate hash-bound evaluator must be committed before
loading any future path. It opens train once, opens 2023 only after train
passes, and then opens 2024/2025/2026 sequentially. Every opened stage requires
positive return, CAGR/strict-MDD at least 3, strict MDD no greater than 15%,
10-bp stress-cost survival, one-extra-bar delay survival, at least 15 bp mean
gross edge, and weekly-cluster sign-flip `p <= 0.10`. Train years and 2023
halves must be individually positive.

Absolute return and CAGR use the complete calendar including idle time. Strict
MDD keeps the global pre-entry high-water mark and marks every held-bar path,
funding, entry cost, virtual adverse exit cost, and actual exit.

## Frozen artifact

Run:

```bash
.venv/bin/python -m training.preregister_intrinsic_volume_latent_impact_relay
```

The canonical artifact is
`results/intrinsic_volume_latent_impact_relay_preregistration_2026-07-23.json`.
