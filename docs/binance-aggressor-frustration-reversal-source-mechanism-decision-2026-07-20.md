# BAFR-24F source and mechanism decision — 2026-07-20

## Decision

**Proceed only with an outcome-blind source build and support/novelty audit.**
`BAFR-24F` (Binance Aggressor Frustration Reversal) is a repairable new
candidate, not a profitability claim. No BAFR feature incidence, event count,
post-entry price, funding cash flow, return, PnL, absolute return, CAGR, strict
MDD, or hit rate has been opened at this commit.

The candidate uses the ordered interaction between public aggressor side and
trade-price direction inside Binance USD-M `BTCUSDT` aggregate trades. It does
not claim to observe passive-book replenishment, hidden liquidity, participant
identity, or a parent order.

Official source references reviewed on 2026-07-20:

- [Binance public-data repository, archive schema, checksums, and MIT licence](https://github.com/binance/binance-public-data)
- [Binance USD-M compressed aggregate-trade REST schema](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#compressed-aggregate-trades-list)
- [Binance USD-M aggregate-trade WebSocket stream](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Aggregate-Trade-Streams)

## Why this is not the old five-minute rejection signal

Existing CBFR, MFIC, terminal-absorption, impact-debt, WFRS, and related
families compare aggregate aggressive flow with a bar return or a multi-bar
price response. BAFR instead classifies every ordered aggregate-trade event by
the immediately observable trade-price tick state before summing a completed
five-minute bucket.

This distinction is conditional, not assumed. The BitMEX zero-tick frustration
sketch recorded the same broad economic intuition but was never implemented or
evaluated because its source failed the production-entitlement gate. BAFR uses
the already verified and live-compatible Binance source. It must still pass a
frozen clock-overlap gate against prior Binance absorption/rejection families;
otherwise it is retired as a hidden re-expression of them.

## Frozen raw-event state

The parser must preserve the official archive order and fail closed rather
than sorting malformed input. Aggregate-trade IDs and transaction timestamps
must be monotonic; IDs must be unique.

For event `i`, let `p_i` be price and `m_i` the buyer-maker flag:

- aggressor side `a_i = -1` when `m_i=true` (buyer passive, seller aggressive);
- aggressor side `a_i = +1` when `m_i=false` (buyer aggressive);
- tick `u_i=+1` when `p_i > p_(i-1)`;
- tick `u_i=-1` when `p_i < p_(i-1)`; and
- when prices are equal, carry the most recent nonzero tick.

Before the first nonzero tick, tick state is unavailable. State carries across
five-minute buckets, normal UTC day boundaries, and normal month boundaries.
It resets whenever the current aggregate-trade ID is not exactly the previous
ID plus one. After a reset, the first event becomes a new price anchor and an
equal-price sequence remains unavailable until a nonzero price change occurs.

Every independently reproducible monthly artifact must therefore bind both its
own daily archive hashes and the prior-day warmup archive/hash used to establish
the initial state. The first requested source day may start unavailable if no
valid warmup archive exists.

## Frozen five-minute observable

For completed UTC five-minute bucket `t`:

```text
buy_frustrated_t  = sum(price * quantity where a=+1 and u=-1)
sell_frustrated_t = sum(price * quantity where a=-1 and u=+1)
quote_notional_t  = sum(price * quantity over all events)
score_t = (sell_frustrated_t - buy_frustrated_t) / quote_notional_t
```

Thus positive score is price resilience against aggressive sellers and maps
long; negative score is price resilience against aggressive buyers and maps
short. The neutral/unknown case does not trade. The object is named
**trade-price frustration**, not proven quote absorption.

The source artifact may also expose component counts and notionals needed for
integrity and zero/nonzero-tick nulls, but no alternative component may replace
the frozen score after an outcome is opened.

## Frozen policy

There is one policy and no threshold, side, entry, or holding-period grid:

1. Exclude every known source-gap UTC day, every missing five-minute bucket,
   and the following 24 buckets. Do not fill them.
2. On clean observations, compute the rolling 90th percentile of
   `abs(score)` over the prior 8,640 clean five-minute observations. Shift the
   baseline by one complete bar and require at least 2,016 prior observations.
3. Signal only when `abs(score_t)` is at or above that strictly prior threshold.
4. Set side to `sign(score_t)`.
5. Enter at the next five-minute open after `t` completes.
6. Exit at the open after exactly 24 held five-minute bars (two hours).
7. Select signals chronologically without overlapping positions. A new entry
   may occur at the same open as the preceding scheduled exit.

No stop, take-profit, side filter, regime gate, extra confirmation, threshold
repair, or holding-period repair is allowed under the `BAFR-24F` name.

## Outcome-blind admission gates

The source and support stage may read only event fields, timestamps, source
quality metadata, the completed score, and the official kline timestamp grid.
It may not parse market OHLC values or funding.

The non-overlapping clock must have:

- at least 250 events over 2020–2023;
- at least 40 events in every calendar year;
- at least 20 events in each 2023 half-year;
- long and short shares each between 25% and 75%;
- no calendar month containing more than 20% of all events; and
- no more than 2% globally quarantined/unavailable five-minute rows.

Failure retires BAFR before any outcome is opened. Thresholds and support floors
must not be weakened.

## Outcome-blind novelty gate

Using timestamps and sides only, compare the admitted clock with CBFR-72,
MFIC, NETF, WFRS, and order-flow campaign terminal absorption. Use a
deterministic one-to-one match with a tolerance of 12 five-minute bars. Every
comparison must have:

- Jaccard overlap no greater than `0.20`; and
- BAFR clock containment no greater than `0.30`.

Failure against any family retires BAFR as non-independent before returns.

## Required source and mechanism tests

The next implementation commit must prove at least:

- equal-price events inherit the last nonzero tick across bar/day/month
  boundaries when aggregate IDs are contiguous;
- an aggregate-ID discontinuity resets price and tick state;
- no equal-price event after reset is classified before a new nonzero tick;
- buyer-maker false on a down tick contributes only to buy frustration;
- buyer-maker true on an up tick contributes only to sell frustration;
- multiplying every price by one positive constant leaves score and side
  unchanged;
- mutating an event in a later bucket cannot alter an earlier bucket;
- archive replay and a live-message replay produce identical classified events
  and five-minute scores; and
- missing, duplicate, reversed, or overflowing event streams fail closed.

The later evaluator must also freeze exact-side-flip, aggressor-flow-only,
trade-price-direction-only, strict-nonzero-tick, carried-zero-tick,
completed-bar flow/rejection, one-hour stale, and 24-hour stale controls before
opening outcomes. Controls may falsify the mechanism but may not replace it.
