# IVFHR-72 preregistration — intrinsic-volume flow handoff relay

## Decision

The next standalone BTC candidate is **IVFHR-72**. It keeps IVLIR's causal
equal-notional daily clock but changes the economic event completely. A
candidate exists only when at least three consecutive eligible daily anchors
held one cumulative taker-flow sign and the current anchor switches to a
strong opposite sign while price from the UTC-day open still points against
that new flow.

The side follows the new flow sign. Entry is the next five-minute open and
exit is exactly 72 bars later. The thesis is an inventory-control handoff:
aggressive participants have changed side before price has reconciled, so the
new flow should relay into price over the next six hours.

## Source-seen boundary

This is not a pristine discovery. IVLIR-72 source incidence was already opened
and rejected because a level-triggered cumulative-flow signal produced a
26-event same-side run. No IVLIR post-entry outcome was opened. IVFHR uses that
source-only failure to define a new transition event rather than repairing the
rejected thresholds or adding a side-balance filter.

At this commit, the exact IVFHR clock, event count, timestamps, controls, and
all post-entry outcomes remain unopened.

## Frozen causal identity

1. For each UTC day, expected volume is the median total quote volume of the
   previous 28 complete UTC days, with at least 21 required.
2. Binance `date` is the UTC bar-open timestamp. A row stamped `t` covers
   `[t,t+5m)` and becomes known at `t+5m`. The anchor is the first completed
   five-minute bar that reaches 50% of expectation; anchor-open timestamps
   after 17:55 UTC are ineligible.
3. Cumulative flow is signed taker quote volume divided by cumulative quote
   volume from the UTC-day origin through the anchor.
4. The prior state is the immediately preceding **calendar-consecutive** run
   of valid daily anchors with one flow sign and must contain at least three
   anchors. A missing/incomplete day, target miss by the 17:55-open bar,
   non-finite value, non-positive cumulative volume, or exactly zero flow emits
   no anchor and resets this run; invalid days are never skipped.
5. The current sign must flip, and current absolute flow must be at least the
   q60 of the previous 180 eligible anchors, excluding current.
6. Price must lag the handoff:
   `new_sign * log(anchor_close / UTC-day_open) <= 0`.
7. For anchor timestamp `t`, decide at `t+5m`, trade the new sign at the
   `t+5m` open, and exit at `entry + 72*5m`, at 0.5x.

This is not a fixed session handoff, backward volume-clock jump, dual-clock
event-count race, moving-average crossover, or LLM-generated action.

## Frozen source-support gate

Before any post-entry price is read, the source-only builder must show:

- at least 60 train events and 12 in every train year;
- at least 18 selection events and 7 in each 2023 half;
- both sides between 25% and 75% in all/train/selection;
- at least 24 active months;
- no month above 12%, no quarter above 25%, and no event gap above 90 days;
- no same-side event run above two.

Primary gates apply to the chronologically accepted, split-contained,
non-overlapping primary clock. Windows are entry-time contained and require
the scheduled exit not to cross the window end. Controls are exact
old-side/fade, any-handoff, no-price-lag, no-flow-strength, persistence-level,
fixed-noon, and deterministic-random-side clocks. They share latency, hold,
non-overlap, containment, and clock schema. Their incidence is report-only and
cannot rescue or reject primary, although malformed control clocks fail the
build. A failed support gate retires IVFHR-72 without calculating a return.
Parameters may not be repaired after incidence is opened.

## Frozen economic and LLM sequence

If source support passes, a separate hash-bound evaluator must be committed
before opening outcomes. It opens train 2020–2022, then 2023 selection only if
train passes, then 2024, 2025, and 2026 YTD sequentially. Each opened stage
must have positive absolute return, CAGR/strict-MDD at least 3, strict MDD at
most 15%, positive base/stress-cost and one-extra-bar-delay results, at least
15 bp mean gross edge, and weekly-cluster sign-flip `p <= 0.10`.

The five-basis-point component margin compares primary mean gross underlying
bp with the maximum same-stage mean gross bp among exactly: any-handoff,
no-price-lag, no-flow-strength, persistence-level, and fixed-noon-handoff.
This statistic is measured before leverage, funding, and transaction cost.

Only after standalone train and selection pass may a compact LLM choose
`TRADE_FIXED_SIDE` or `ABSTAIN`. It may not alter the clock, side, entry, hold,
leverage, or cost model.

## Frozen artifact

Run:

```bash
.venv/bin/python -m training.preregister_intrinsic_volume_flow_handoff_relay
```

The canonical, checked-in artifact is
`results/intrinsic_volume_flow_handoff_relay_preregistration_2026-07-23.json`.
