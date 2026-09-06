# CVVH-432 cross-venue volatility-shape handoff mechanism decision

## Decision

`CVVH-432` is the only successor mechanism opened after the terminal ESDI
source replay failure.  It uses already committed, completed-hour Binance
BTCBVOL and Deribit DVOL OHLC envelopes.  This decision was made without
decoding either source file and without opening candidate incidence, BTC
execution prices, funding, returns, PnL, Gross9 paths, or future outcomes.

The candidate tests whether a directional and wider Binance volatility-index
hour can lead an opposite, weaker Deribit volatility-index body.  A rising
dominant Binance volatility envelope maps to `SHORT`; the exact falling mirror
maps to `LONG`.  The mapping is fixed and is not a causal claim.

## Exact hourly state

For each venue candle:

```text
body  = (close - open) / open
range = (high - low) / open
```

All OHLC tokens are parsed as exact decimal strings and converted to their
exact integer-coefficient rational values.  All four values must be finite and
strictly positive, with `high >= max(open, close)` and
`low <= min(open, close)`.  The implementation uses unrounded
positive-denominator cross multiplication rather than binary-float or
fixed-precision division.  To make hostile compact exponents fail before
integer expansion, each token is limited to 128 coefficient digits and a base
10 exponent in `[-128, 128]`; the committed source schema lies inside this
source-independent envelope.

At completed joined hour `T`, the primary state is:

- `SHORT` when the Binance body is positive, the Deribit body is negative,
  `abs(Binance body) > abs(Deribit body)`, and the normalized Binance range is
  strictly greater;
- `LONG` for the exact sign mirror; and
- `NONE` otherwise, including every equality.

Both joined rows must be source-valid, exact one-to-one UTC hours.  An event
requires the immediately previous joined hour to be valid and exactly one hour
earlier, the current state to be `LONG` or `SHORT`, and the previous state to
differ.  Thus a continued state does not emit, an opposite-side transition may
emit, and the first valid row after any gap or invalid row cannot emit.

## Causal schedule

`T` is the completed-hour clock, not candle-open time.  Joint availability is
the later source availability.  Entry is:

```text
ceil_to_5m(joint availability) + 5 elapsed minutes
```

An already aligned hour still waits five minutes.  The position exits exactly
432 five-minute bars later (36 hours), uses fixed 0.5x leverage, and later
economics charge 6 bp per notional side at base cost and 10 bp at stress.

Candidates sort by `(entry, T, canonical id, side)`.  One global position is
reserved on `[entry, exit)`; a candidate is accepted only when its entry is not
earlier than the previous accepted exit.  Suppressed candidates are never
queued.

Every own-clock id is the exact UTF-8 string
`CVVH-432|<control>|T=<RFC3339 whole-second UTC Z>`.  The deterministic-random
control hashes the exact UTF-8 preimage
`CVVH-432|<primary-id>|RANDOM_SIDE` with SHA-256; first digest byte `<128` maps
to `LONG`, otherwise `SHORT`.  Parent-set control ids replace `<control>` while
preserving the primary `T`.

## Frozen controls

Four controls build and reserve independent clocks with the same onset rule:

1. `deribit_led`: swap venue leadership; direction is opposite the dominant
   Deribit body;
2. `body_lead_only`: omit Binance range leadership;
3. `range_lead_only`: omit Binance body-magnitude leadership; and
4. `stale_deribit`: at `t` use Binance `t` with Deribit `t-1`; its previous
   state uses Binance `t-1` with Deribit `t-2`, requires all three consecutive
   valid hours, and retains current-`T` availability.

The accepted primary parent set also produces direction-flip, deterministic
SHA-256 side, constant-long, constant-short, and one-five-minute-bar delayed
controls without rerunning reservation.  Controls can never replace or repair
the primary.

## Boundaries and stop rule

The pure implementation is
`training/cross_venue_volatility_shape_handoff.py`.  It has no filesystem,
network, pandas, market, funding, outcome, or portfolio dependency.  Source
hashes, support floors, novelty comparators, Gross9 authority, economic gates,
write-once claims, and future-veto sequencing are frozen in the next
write-once preregistration stage before either source is decoded.

Any later source-support, novelty, or economic failure retires this exact
identity.  There is no threshold repair, hold/latency grid, polarity inversion,
rank-2 substitution, or endpoint replacement.
