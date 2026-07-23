# IVPLH-72 candidate boundary — intrinsic-volume price-lag handoff

## Selection

Select one explicitly source-seen successor candidate:
**IVPLH-72 — Intrinsic-Volume Price-Lag Handoff**.

The observable is the sign handoff between two calendar-consecutive daily
equal-notional first-passage anchors while price still points against the new
flow side. The provisional action follows the new cumulative taker-flow sign.
The intended identity is the broad `any_handoff` source-only control that the
terminal IVFHR-72 result explicitly allowed to be registered as a materially
different event.

This document selects an axis only. It does not freeze the final normalization,
state machine, execution, controls, support gates, novelty gates, or economic
evaluator. Those must be committed before any further candidate incidence is
decoded.

## Why this axis, and why rejected alternatives were not selected

URCD-72 was retired because finalized USDC mint-recipient concentration did not
produce any 2021–2022 train clocks. A new candidate therefore needs a denser,
already reproducible pre-2024 observation axis.

The initially proposed stablecoin collateral-flow conflict was rejected during
boundary review. UCBR-12's terminal result requires the next candidate to leave
stablecoin relative-price and BTC stablecoin-book transformations. Combining
those same two source families would be a renamed lineage continuation, not an
independent mechanism.

IVPLH instead uses the checksum-bound Binance USD-M BTCUSDT five-minute source
over 2020–2023. It observes an intrinsic-volume timing coordinate and a causal
flow/price disagreement. It uses no stablecoin relative price, issuance event,
funding, OI, options, FX, Kimchi premium, LLM label, or portfolio state.

## Disclosed source contamination

IVPLH is **not** candidate-incidence blind.

The frozen IVFHR-72 support report has already disclosed the broad
`any_handoff` control's aggregate source-only statistics over 2020–2023:

- 66 globally non-overlapping events;
- 29 LONG and 37 SHORT;
- 37 active months;
- maximum month share `4/66`;
- maximum quarter share `8/66`;
- maximum calendar gap about 90.70 days; and
- maximum same-side run 7.

The prior report also disclosed that this broad event removes IVFHR's
three-anchor persistence and q60 new-flow-strength requirements while retaining
the sign handoff and price-lag requirement. These observations may justify only
that the axis is operationally testable. They are not clean evidence for a
support gate, direction, hold, or profitability claim.

During this selection unit, the exact `any_handoff` rows, timestamps, yearly or
half-year counts, and comparator overlaps were not decoded. The existing clock
file's header and whole-file hash were known from the prior immutable artifact;
its data rows remain unopened for IVPLH until a separate mechanism and
preregistration are committed.

## Outcome boundary

No IVPLH-specific post-entry BTC price, return, funding cash flow, PnL,
absolute return, CAGR, strict MDD, hit rate, label, reward, or 2024+ source value
has been opened or computed.

The following order is mandatory:

1. commit this source-seen boundary;
2. commit one exact mechanism, including direction, latency, hold, controls,
   support gates, comparator cohort, and stopping rule;
3. commit and seal a preregistration builder/artifact without reading IVPLH
   clock rows;
4. commit a tested source-support/novelty evaluator before opening the existing
   `any_handoff` rows;
5. reject unchanged on any source or novelty failure; and
6. only a complete pass may authorize a separately committed strict economic
   evaluator.

## Bound predecessor evidence

- `docs/intrinsic-volume-flow-handoff-relay-support-rejection-2026-07-23.md`
- `results/intrinsic_volume_flow_handoff_relay_support_2026-07-23.json`
- `data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz`
- `docs/usdt-collateral-breadth-relay-support-rejection-2026-07-20.md`
- `docs/usdc-recipient-concentration-dislocation-support-rejection-2026-07-23.md`

The mechanism stage must hash-bind the exact predecessor artifacts it relies
on. It may not reinterpret an IVFHR control as proof of edge or use the known
global counts to tune a threshold grid.
