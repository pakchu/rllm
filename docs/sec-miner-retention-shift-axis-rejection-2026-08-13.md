# SEC miner-retention shift axis rejection — 2026-08-13

## Decision

Reject before preregistration and before opening any SEC filing body, filing
incidence, source value, BTC outcome, Gross9 row, execution price, or funding
value.

The screened idea used monthly public-miner operations reports filed on SEC
EDGAR, computed `retention_ratio = (BTC produced - BTC sold) / BTC produced`,
and followed the sign of its issuer-level month-over-month change during high
BTC variation.

## Existing semantic collision

This is not independent of the frozen SEC EDGAR Bitcoin Constraint Transition
Breadth (`EBCT`) family.  Its committed ontology already defines:

- a completed Bitcoin sale as `BTC_CONSTRAINT_DRAW`;
- explicit retention or accumulation as `BTC_CONSTRAINT_BUFFER`; and
- specifically, retaining all Bitcoin mined during a month and selling none as
  a positive buffer example.

EBCT then maps draw to short and buffer to long using the official SEC
`acceptanceDateTime` clock.  The proposed ratio uses the same source documents,
the same acceptance clock, the same sale-versus-retention economic polarity,
and the same directional mapping.  Replacing EBCT's discrete semantic state
with a numeric production/sales ratio or its first difference does not create
a new source object; it reparameterizes an already terminal family.

## Additional pre-source risks

Even absent the semantic collision, a fixed current miner list would introduce
survivorship risk, while an unrestricted 8-K/6-K text universe would require a
new deterministic issuer/report-period/duplicate/restatement contract before
any filing body could be opened.  Those issues do not cure the primary
collision and therefore were not explored through source incidence.

## Boundary record

- Only repository code and prior documentation were inspected.
- No SEC historical filing body or numeric production/sales observation was
  opened.
- No event count, timestamp panel, side distribution, return, or PnL was
  computed.
- No threshold, universe, clock, side, or hold was selected from incidence.
- This proposed axis is terminal and will not be repaired, inverted, or
  revived as a quantitative EBCT variant.

