# CATCH-12 preregistration — 2026-07-14

## Status

**Support-only and outcome-blind.** No CATCH future return, held path, win rate,
CAGR, or MDD has been opened. This document freezes the economic object,
direction, score, control family, support grid, execution timing, and later
return gate before any outcome evaluation.

- name: **CATCH-12 — Cash Auction Transfer & Catch-up Handoff**
- source SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`
- inspected feature horizon: strictly before `2024-01-01`
- direction: symmetric, `sign(Spot aggregate signed-flow fraction)`
- entry: next Binance USD-M five-minute open after the completed signal bar
- exit: fixed USD-M open 12 bars later (one hour)

The name describes an inference, not a causal claim. The measurable event is a
completed bar in which Spot activity and price appear earlier, Spot aggressor
flow is price-accepted, USD-M begins responding in the same direction, and the
close-to-close basis change still says the perpetual has not fully caught up.

## Primary event on completed bar `t`

All Spot/USD-M one-minute source rows and every forward/reverse control
denominator must be valid. A causal 24-bar post-defect quarantine applies.

Let `d = sign(spot_flow_fraction)`. Require:

1. `d != 0` and `d * spot_log_return_5m > 0`;
2. `spot_to_um_lagged_directional_alignment > 0`;
3. `lagged_directional_alignment_diff > 0`;
4. `um_minus_spot_activity_time_centroid > 0`;
5. `-d * basis_change_bp > 0`.

Condition 5 means Spot moved farther in direction `d`: because basis is
`log(USD-M / Spot)`, an upward cash lead reduces basis and a downward cash lead
increases it.

To avoid counting several algebraic forms of the same flow×response primitive,
the minute-order block is a single conservative component:

```text
directed_handoff = min(
    spot_to_um_lagged_directional_alignment,
    lagged_directional_alignment_diff,
)

score = cbrt(
    directed_handoff
    * um_minus_spot_activity_time_centroid
    * spot_flow_coherence
)
```

All three factors must be strictly positive. `flow_transfer_asymmetry`,
`return_leadership_asymmetry`, and raw venue notionals are excluded from the
primary score and retained only as controls/audit context.

The score must exceed its strictly lagged rolling percentile. The baseline is
30 days (`8,640` bars), requires seven days (`2,016` bars), reads only earlier
completed features, and never reads a return after `t`.

The action is `d` at open `t+1`. A source outage after entry cannot cancel the
already-entered fixed one-hour hold.

## Outcome-blind support selection

Test score percentiles `{0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975}` and
select the highest percentile passing every frozen floor:

- at least 2,500 non-overlapping one-hour events total;
- at least 500 in each calendar year 2020–2023;
- at least 200 in each 2023 half and 100 in each 2023 quarter;
- both long and short shares at least 35% overall and at least 150 events per
  side in every year;
- at least 42 of 48 calendar months contain at least 20 events;
- reverse-time: Jaccard `<=0.15`, primary containment `<=0.25`;
- venue-swap: Jaccard `<=0.05`, containment `<=0.10`;
- simultaneous-only: Jaccard `<=0.10`, containment `<=0.10`;
- aggregate-only: Jaccard `<=0.10`, containment `<=0.15`;
- basis-only and flow/return-asymmetry-only: Jaccard `<=0.15`, containment
  `<=0.20`;
- removing residual-basis lag retains at most `0.65` of its broader control
  clock (`primary_count / no_basis_count`) on both raw and fixed-hold clocks;
- removing activity ordering: Jaccard and containment each `<=0.40`;
- 1h stale, 24h stale, and one-bar delay: Jaccard `<=0.05`, containment
  `<=0.10`;
- scheduled CSPR, RIFT, and their union: Jaccard and primary containment each
  `<=0.01`.

If no percentile passes, reject CATCH-12 before returns. Support counts and
clock overlap may select the percentile; price outcomes may not.

## Frozen controls

1. **exact direction flip:** same primary clock, action `-d`;
2. **venue swap:** USD-M is treated as the leader and Spot as follower, with
   side `sign(um_flow_fraction)`;
3. **reverse-time placebo:** replace every forward lagged arrow by its frozen
   `later→earlier` counterpart while retaining the same cash/basis semantics;
4. **simultaneous-only:** replace lagged ordering with same-minute flow/return
   sign agreement;
5. **aggregate-only:** use only Spot aggregate flow coherence/fraction,
   price acceptance, and basis residual;
6. **basis-only:** rank the signed residual-basis magnitude after cash price
   acceptance, without minute ordering;
7. **flow/return-asymmetry-only:** use the two antisymmetric summary fields,
   without the directional-alignment or activity-timing block;
8. **no residual-basis lag:** remove condition 5;
9. **no activity ordering:** remove component/condition 4 and recompute a
   two-component score;
10. **1h stale** and **24h stale:** recompute their own lagged-percentile
   signals from input components stale by 12/288 completed bars; **one-bar
   delayed** shifts only the already-selected primary signal;
11. frozen CSPR-12 and RIFT-96 scheduled clocks for novelty overlap.

Each score-bearing control gets its own strictly lagged percentile at the same
quantile. Every novelty limit is required on both raw incidence and its own
fixed-hold non-overlapping execution schedule. Controls may falsify CATCH but
may not replace or repair it after returns are opened.

## Frozen return gate

- train: `2020-01-01 <= t < 2023-01-01`;
- selection: full 2023 and fixed H1/H2;
- 2024 test, 2025 eval, and 2026 YTD remain sealed;
- leverage `0.5x`, fee `5 bp`, slippage `1 bp` per notional side;
- multiplier `(1-0.0003)*(1+0.5r)*(1-0.0003)`;
- full-clock CAGR including idle cash;
- strict held-path MDD, favorable extreme before adverse, excluding later
  high/low from the scheduled-open exit bar;
- weekly entry-cluster Rademacher test, 100,000 draws, seed `20260714`.

CATCH advances only if train and full 2023 each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, one-sided cluster `p<0.10`,
and mean gross underlying move strictly above 12 bp. Each 2023 half must be
positive with at least 200 trades. The primary minimum train/selection ratio
must beat every frozen score-bearing control. Failure rejects v1 without
threshold, side, hold, or gate repair. Only an unchanged pass can become an
RLLM state token for abstention or sizing; an LLM may not rewrite the frozen
direction.
