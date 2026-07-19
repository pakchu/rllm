# RNCM source-quality gate — frozen before anomaly counts

## Trigger and evidence boundary

The first full 2023 centroid build stopped because at least one raw snapshot
made a cumulative bid average quote meet or exceed its paired ask average
quote.  The failing build printed no date, count, frequency, feature value,
price, return, or PnL.  A separate 365-day source scan was then started.

This document freezes the treatment **before reading that scan's counts**.  No
post-entry outcome has been opened.

## Deterministic invalid-snapshot rule

For each complete USD-M snapshot, define cumulative average quote as
`notional/depth`.  A snapshot is invalid when any of the following holds:

1. an average quote is non-finite or non-positive;
2. cumulative bid average quote increases as distance expands from 1% to 5%;
3. cumulative ask average quote decreases as distance expands from 1% to 5%;
4. any paired cumulative bid average quote is greater than or equal to ask.

An invalid snapshot is never clipped, repaired, winsorized, or replaced.  Its
entire five-minute bar is quarantined even if the other snapshots in that bar
are valid.  Quarantine is based only on contemporaneous source algebra and
cannot inspect price or later outcomes.

## Frozen source-level admission limits

The RNCM source is rejected before signal preregistration if **any** limit is
exceeded:

- invalid snapshots / verified snapshots `> 0.0001` (1 basis point);
- quarantined otherwise-timing-complete bars / timing-complete bars `> 0.001`
  (10 basis points);
- invalid snapshots on any available UTC day / that day's snapshots `> 0.01`.

The limits permit isolated archival defects while forbidding a feature whose
validity depends on broad source cleaning.  Missing official archives remain
missing and are not included in the anomaly-rate denominator.

If all three limits pass, the builder may be revised once to apply exactly the
bar quarantine above, then must be re-tested, committed, and rebuilt.  No
different tolerance or partial-snapshot salvage is allowed after counts are
read.

## Additional pre-return gates

Passing rarity alone is insufficient.  Before any RNCM return evaluation:

- ideal common price-scale changes must leave centroid skew unchanged in a
  synthetic regression test;
- a fixed absolute synthetic book observed through moving percentage bands
  must not generate the frozen RNCM event rule at an unacceptable rate;
- same-clock price-momentum and price-reversion controls must be frozen in the
  evaluator;
- the preregistration must call the quantity a **cumulative depth-weighted
  average-quote transform**, not a pure order-price centroid;
- archive publication is acknowledged as next-day.  `bar+5m` research
  availability is conditional on reconstructing the equivalent percentage
  bands from a correct live REST snapshot plus WebSocket update-ID stream;
  live parity remains a production admission gate.

