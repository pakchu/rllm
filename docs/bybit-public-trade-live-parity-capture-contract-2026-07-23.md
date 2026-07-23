# Bybit public-trade live-parity capture contract — 2026-07-23

## Authorization and purpose

The corrected source-only audit passed and is bound here by:

- audit document SHA-256:
  `fe324cccfb0c3f66963c142b9a6c0237489420313750de873622cadb10e8c112`
- v2 result SHA-256:
  `916a55f7cd957eff39e84b2ac383c2b49cb342e2012a0f8bc15c3af98b3b3cb0`
- v2 manifest hash:
  `c36f46c8399692b62d202a7331c9215fc3a5684cc3b2d57ca04d7fc7c83a5f84`

This contract authorizes one prospective source-parity capture for provisional
**BSEA-24**.  It does not authorize a BSEA feature, event clock, direction,
market outcome, or profitability evaluation.

## Official live semantics reviewed before capture

Current Bybit V5 documentation states:

- REST `GET /v5/market/recent-trade` accepts `category=linear`,
  `symbol=BTCUSDT`, and a non-spot limit up to 1,000.  It returns execution ID,
  symbol, price, size, taker side, execution time in milliseconds, block/RPI
  flags, and cross sequence.  It does **not** document response-list order:
  <https://bybit-exchange.github.io/docs/v5/market/recent-trade>.
- WebSocket `publicTrade.BTCUSDT` exposes trade time, symbol, taker side, size,
  price, price-change direction, trade ID, block/RPI flags, and cross sequence.
  Items inside one message are match-time ascending; one message may hold up
  to 1,024 trades; multiple messages may share one `seq`:
  <https://bybit-exchange.github.io/docs/v5/websocket/public/trade>.
- Public topics do not require authentication, and Bybit recommends a ping
  every 20 seconds.  No replay cursor is documented:
  <https://bybit-exchange.github.io/docs/v5/ws/connect>.
- The published generic HTTP IP limit is 600 requests per five seconds.  No
  endpoint-specific recent-trade limit was found in the reviewed table:
  <https://bybit-exchange.github.io/docs/v5/rate-limit>.
- Bybit advertises downloadable historical market/trade CSV data, but the
  reviewed official pages do not specify archive publication cadence or SLA:
  <https://bybit-exchange.github.io/docs/>,
  <https://www.bybit.com/en/derivative-activity/history-data/>.

The following are therefore empirical questions, not assumptions: REST list
ordering, archive publication time, archive/live ID equivalence, and global
ordering across separate WebSocket messages.

## Frozen transport

- REST endpoint:
  `https://api.bybit.com/v5/market/recent-trade`
- REST query:
  `category=linear&symbol=BTCUSDT&limit=1000`
- WebSocket endpoint:
  `wss://stream.bybit.com/v5/public/linear`
- WebSocket topic: `publicTrade.BTCUSDT`
- Later archive endpoint:
  `https://public.bybit.com/trading/BTCUSDT/BTCUSDT<UTC-DAY>.csv.gz`

Redirects, alternate hosts, authentication, endpoint fallback, and testnet are
forbidden.  A transport or regional-access failure rejects this capture; it is
not repaired by another venue or generic OHLCV.

## Frozen prospective capture

The first production-shaped run after code and tests are committed must:

1. start and finish within one UTC calendar day;
2. abort before network access at repository-filesystem use of 300 GiB or more;
3. open exactly one WebSocket session, receive a successful subscription
   acknowledgement, then receive at least one valid trade message;
4. run for exactly 600 monotonic seconds after the first valid trade message;
5. issue one REST snapshot immediately after WebSocket readiness, poll once per
   monotonic second with `limit=1000`, and issue one final snapshot after the
   WebSocket capture closes;
6. reject on disconnect, reconnect, malformed payload, queue/write overflow,
   response error, clock reversal, or crossing UTC midnight; and
7. write only to one new immutable ignored directory under `data/`, never
   overwrite a capture, and never commit raw public-trade payloads.

Each raw WebSocket message and REST response is stored as compressed NDJSON
with local UTC receipt/request timestamps, monotonic timestamps, request or
message ordinal, and SHA-256 of the exact received JSON bytes.  A separately
hashed manifest may be committed only after validation.  No account key,
private endpoint, position, order, Binance value, return, or PnL may be read.

## Normalization and identity

REST keeps the raw fields `execId`, `symbol`, `price`, `size`, `side`, `time`,
`isBlockTrade`, `isRPITrade`, and `seq`.  WebSocket keeps message `ts` and raw
trade fields `T`, `s`, `S`, `v`, `p`, `L`, `i`, `BT`, `RPI`, and `seq`.

- Price and size remain exact decimal strings and compare by finite positive
  `Decimal` value, not binary float or text formatting.
- Times and sequences must be nonnegative integers; taker side must be exactly
  `Buy` or `Sell`; symbol must be exactly `BTCUSDT`; flags must be booleans.
- REST response order is discarded.  WebSocket within-message order is
  checked for nondecreasing `T` and otherwise retained.
- `seq` is never a unique key, parent-order proxy, or required `+1` sequence.
- REST `execId`, WebSocket `i`, and archive `trdMatchID` remain distinct source
  labels until exact equality is demonstrated.  A composite-key fallback may
  diagnose a failure but cannot pass parity.
- Duplicate source IDs with identical normalized fields are counted; a
  duplicate ID with conflicting fields is an immediate rejection.

## REST/WebSocket parity gate

The capture passes this interim gate only if all conditions hold:

1. at least 1,000 unique valid WebSocket trades and at least 10 REST snapshots;
2. zero malformed messages, reconnects, conflicting duplicate IDs, or UTC-day
   crossings;
3. every adjacent REST snapshot shares at least one execution ID, proving that
   the 1,000-row rolling windows overlap rather than silently leap;
4. the exact REST `execId` and WebSocket `i` intersection contains at least
   1,000 IDs;
5. for every common ID, symbol, taker side, price, size, match time, sequence,
   block flag, and RPI flag agree exactly after the frozen normalization;
6. every WebSocket trade received after the first REST response completed and
   before the final REST request began occurs in the union of REST IDs; and
7. every REST trade strictly inside the first/last WebSocket match-time range
   occurs in the WebSocket ID set.

Local-time boundary rows are excluded only by rules 6 and 7; no percentage
tolerance, sampled comparison, order inference, or post-run boundary adjustment
is allowed.  Failure keeps BSEA source parity unproven.

## Archive parity gate

The capture remains `PENDING_ARCHIVE` until the exact UTC-day archive appears.
Archive cadence is not inferred from a prior day.  When available, one verifier
must stream the gzip once, hash it, retain only rows whose `trdMatchID` occurs
in the committed live-capture manifest, and discard the raw archive.

For every captured live ID, archive `trdMatchID` must equal both REST `execId`
and WebSocket `i`; symbol, side, exact decimal price/size, and millisecond match
time must agree.  `Decimal(timestamp) * 1000` must be an exact integer; rounding
is forbidden.  Every live ID inside the archive-covered capture envelope must
exist exactly once.  Relative order for records with different millisecond
timestamps must agree; tie order within one millisecond is not treated as
documented semantics.

The archive lacks historical `seq` and block fields, so those are verified only
between REST and WebSocket.  Recent `RPI` is a parity control only and remains
forbidden from the primary BSEA feature.

## Stop and next step

Any failed REST/WebSocket or archive gate rejects BSEA without feature repair.
Only a fully passing three-surface audit may authorize one separately committed
venue-relative sequence-disagreement mechanism and null battery.  Candidate
incidence, Binance comparator values, direction, and outcomes remain sealed
until then.
