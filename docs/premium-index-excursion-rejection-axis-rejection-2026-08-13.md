# Premium-index excursion-rejection axis rejection — 2026-08-13

## Decision

Reject before preregistration and before opening source incidence, BTC outcomes,
Gross9 rows, execution prices, or funding.

The proposed source object was a completed Binance BTCUSDT premium-index OHLC
path with one dominant excursion away from its prior center followed by a close
back toward that center.  The proposed trade faded the excursion direction.
This is not an independent mechanism: it is the already frozen PSR-30/6
Premium Snapback Recenter object.

It also collides directly with the later frozen HVPIWR candidate in
`training/preregister_high_volatility_premium_index_wick_rejection_relay.py`.
HVPIWR already defines daily premium-index upper/lower wicks relative to the
daily body, requires a directionally consistent rejection, applies causal
premium-magnitude and BTC-variation tails, and trades the rejection direction
for twelve hours.

## Exact collision

`training/preregister_premium_snapback_recenter.py` already:

- reads the premium-index `open`, `high`, `low`, and `close` fields;
- computes rolling upper and lower excursions from a strict-prior center;
- requires a large one-sided excursion and a small terminal deviation from the
  center; and
- maps a lower-only excursion to long and an upper-only excursion to short.

The same implementation also freezes a `simple_level` control containing the
one-sided excursion without the recenter/path requirements.  Consequently,
using a candle close-location, wick ratio, range asymmetry, different path
length, funding-boundary subset, or nearby threshold would only repair or
reparameterize PSR or HVPIWR rather than create a new source object.

## Boundary record

- Repository code and prior documentation only were inspected.
- No database values were queried.
- No source rows, event counts, timestamps, sides, returns, or PnL were opened.
- No formula, threshold, clock, side, hold, or universe was selected after
  seeing incidence or outcomes.
- This axis is terminal and will not be repaired or inverted.
