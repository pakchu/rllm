# TASCC-72 mechanism decision — 2026-07-23

## Decision

Preregister one new source-family-seen, outcome-blind singleton:
**TASCC-72 — Treasury Auction Settlement-Collision Carry**.

The exact TASCC basket incidence has not been derived and no TASCC BTC outcome
has been opened. The next authorized step may read only the frozen 2016–2023
Treasury auction schedule fields and hash-bound comparator clocks. Calendar
2024 onward remains sealed.

## Prior-research contamination disclosure

This is not a pristine Treasury source family.

- `TADI-1` used bid-to-cover and indirect-bidder demand changes. Its 2021–2022
  BTC Stage1 outcome was opened and failed; its 2023 outcome stayed sealed.
- `DFFB-601` used Daily Treasury Statement fiscal-flow breadth. During its
  source-only novelty test, the union of official auction and issue dates was
  materialized; DFFB stopped before BTC outcomes.
- The normalized Treasury auction panel and aggregate source counts have been
  opened repeatedly.

TASCC is not a TADI direction, rank, or hold repair. It ignores auction demand
values and uses a different economic object: simultaneous settlement of belly
and long-duration original issuance. The exact issue-date collision baskets,
their accepted execution clock, and all associated returns remain unopened at
this decision point.

## Economic hypothesis

On an issue date, successful Treasury bidders exchange cash for securities.
A settlement basket containing both belly duration (`5-Year` or `7-Year`) and
long duration (`10-Year`, `20-Year`, or `30-Year`) concentrates duration supply
and cash absorption across distinct maturity clienteles. The preregistered
hypothesis is a temporary risk-liquidity drain: **short BTC for 72 elapsed
hours** from the start of the known settlement date.

No yield, bid-to-cover, indirect share, accepted notional, Treasury price, or
crypto state enters the signal. This deliberately tests calendar geometry
rather than auction quality.

## Frozen source and event clock

1. Join the frozen normalized original-issue nominal coupon panel to the two
   frozen raw TreasuryDirect pages on exact `(auctionDate, cusip)`.
2. Materialize only `auctionDate`, `issueDate`, `cusip`, `securityType`,
   `originalSecurityTerm`, `reopening`, panel `result_available_at_utc`, and
   panel `source_complete`.
3. Require `reopening == No`, `source_complete == true`, and one of the seven
   frozen terms. Current demand/allocation fields are forbidden.
4. Group eligible rows by exact `issueDate`. A primary basket requires at least
   one distinct belly term and at least one distinct long term.
5. Define the settlement marker as `issueDate 00:00 UTC`. Every component's
   conservative result time must be no later than the marker. A basket that was
   not fully known by the marker is skipped, never backfilled.
6. Signal at the settlement marker; enter at
   `ceil_5m(signal_time) + 5 minutes`, including an exact-grid marker.
7. Exit 72 elapsed hours later, use 0.5x fixed short exposure, and reserve
   globally on `[entry, exit)`. Suppressed events are not queued.
8. Split-crossing trades are skipped. No stop, take-profit, trailing exit,
   direction override, or dynamic size is allowed.

## Frozen splits and source gates

- source warmup: 2016–2019;
- train: 2020–2022;
- selection: 2023;
- sealed: 2024 onward.

Before opening BTC rows, the primary requires at least 18 train events, 8
selection events, 6 in every train year, 3 in each 2023 half, eight active train
quarters, no month above 20%, no quarter above 40%, no gap above 90 days, and no
duplicate issue-date/CUSIP identity. Every accepted basket must contain both
required maturity groups and must be completely available before entry.

Frozen source/control clocks are:

- `belly_settlement_calendar` and `long_settlement_calendar`;
- `any_multitenor_settlement`;
- `single_tenor_settlement`;
- `auction_date_collision` using the same belly+long rule by auction date;
- `term_year_permutation`, a deterministic within-auction-year permutation of
  term labels before issue-date grouping;
- `result_time_clock`, the same primary baskets entered after their latest
  conservative result time;
- `settlement_plus_7d`, the primary marker shifted seven elapsed days.

Belly, long, any-multitenor, and result-time clocks are known component/superset
relations and are report-only at source stage; they remain mandatory economic
controls. Auction-date collision and term permutation must satisfy the frozen
specificity caps.

Clock novelty is checked against TADI primary, DFFB primary, FLCC primary,
ON-RRP primary, and current live-portfolio pure clocks. A qualifying comparator
with at least ten entries must have exact-entry Jaccard at most 0.10 and TASCC
to comparator ±12-hour containment at most 0.35.

## Economic gate and RLLM boundary

If source support passes, a separately committed evaluator must use realized
funding, 6 bp per side base cost, 10 bp stress cost, full-calendar CAGR, and
high-water/intratrade strict MDD. Train and selection each require positive
absolute return, `CAGR / strict MDD >= 3`, strict MDD at most 15%, positive
stress return, minimum trade support, positive 2023 H1/H2 return, and failure
of direction-flip, random-side, component-calendar, result-time, delayed,
auction-date, and term-permutation controls.

RLLM remains disabled until the unchanged deterministic policy passes train and
selection. A later frozen small model may choose only `TRADE_FIXED_SHORT` or
`ABSTAIN` from causal context and current position state. It may not create the
clock, reverse side, change size/hold, or repair source gates.

## Stopping rule

Any identity, provenance, causality, support, specificity, novelty, train, or
selection failure retires `TASCC-72-SOURCE-FAMILY-SEEN`. Changing maturity
groups, settlement marker, availability requirement, side, or hold requires a
new identity preregistered before access.
