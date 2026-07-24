# Deribit options trade-ledger source boundary — 2026-07-24

## Decision

Open one new source identity:
**DOTL-v1 — Deribit BTC Options Trade Ledger**.

DOTL asks whether Deribit's official historical public-trade service and live
public market-data surfaces form one causal, reproducible ledger for executed
BTC inverse-option trades. This stage does not define an alpha, candidate
timestamp, side, hold, threshold, model, reward, or profitability claim.

No historical or live trade response body, option-flow value, candidate
incidence, BTC return, funding value, PnL, CAGR, or drawdown was opened while
writing this boundary. Only official documentation and HTTP response metadata
were inspected.

## Why this is a new source axis

Prior Deribit research in this repository opened:

- hourly DVOL levels and changes;
- BTC perpetual funding;
- option delivery/expiry events and strike-wall snapshots; and
- one DVOL/Binance-premium relay.

It did not build a hash-bound ledger of individual public option executions.
DOTL therefore measures executed risk transfer rather than an implied-volatility
index, funding payment, expiry calendar, delivery settlement, or static strike
wall.

Prior family records remain immutable:

```text
docs/deribit-dvol-alpha-2026-07-13.md
SHA256 21e056c535a72d28b6b3799c489fba008e63272cac28eb6a93d8b68aa1c237a3

docs/options-perpetual-demand-relay-support-rejection-2026-07-19.md
SHA256 062715a7a5a8a0d39d14cdf1995e5b360660e9978726510c51231e5636099b91

docs/deribit-expiry-hedge-release-support-rejection-2026-07-20.md
SHA256 c8b0c18743057e9b217c932e742784f2f810241a4723869cbe9c912db88a25c2

docs/deribit-expiry-wall-handoff-support-rejection-2026-07-21.md
SHA256 7382308356b7f185889ea1b5dbc634530b823a723283184bcad8e5ba83fcba11

docs/deribit-funding-impulse-absorption-source-rejection-2026-07-20.md
SHA256 bfca5eb028839e80920438fb5a71e95ef35606551b4191e1138afea775489559
```

A DOTL pass cannot revive, repair, waive, or retroactively pass any of those
retired candidates.

## Official surfaces

### Historical public trades

```text
https://history.deribit.com/api/v2/public/get_last_trades_by_currency
```

Deribit's institutional setup material identifies the `history.deribit.com`
service as the historical public trade/instrument surface. The exact endpoint
returned HTTP `400` to a body-free `HEAD` request on 2026-07-24, confirming that
the host and route were active without opening a market-data response body.

The historical host does not have a complete current method reference of its
own. DOTL therefore freezes the current documented production contract for
`public/get_last_trades_by_currency` as the only admissible contract on the
historical host:

```text
request method: GET
path: /api/v2/public/get_last_trades_by_currency
query parameters, lexicographically encoded:
  currency: BTC
  kind: option
  start_timestamp: integer Unix millisecond, first page only
  end_timestamp: integer Unix millisecond
  start_id: decimal BTC trade-ID string, continuation pages only
  count: 1000
  sorting: asc
response:
  jsonrpc: "2.0"
  result.trades: array
  result.has_more: boolean
```

Current production documentation defines `start_id` as the first included BTC
trade ID, `end_timestamp` as the most recent timestamp, `count` maximum 1,000,
and `sorting=asc` as ascending trade-ID order. DOTL does not assume that the
history host honors this contract: the first response body is the empirical
gate. A different request grammar, envelope, cursor meaning, sort order, or
field contract is a terminal source failure, not an invitation to discover a
replacement API.

### Recent public REST

```text
https://www.deribit.com/api/v2/public/get_last_trades_by_currency
```

Current Deribit API documentation states that this public endpoint returns
only the last 24 hours. It permits `currency=BTC`, `kind=option`, exact
millisecond start/end bounds, decimal BTC trade-ID cursors, up to 1,000 rows,
and deterministic ascending or descending trade-ID sorting. DOTL uses the same
frozen request/envelope/pagination contract on the recent and historical hosts.

### Live public WebSocket

```text
wss://www.deribit.com/ws/api/v2
channel: trades.option.BTC.100ms
```

Deribit's current subscription documentation defines
`trades.(kind).(currency).(interval)` as the consolidated public trade stream
and explicitly documents the `100ms` aggregation interval. Raw public trade
channels require authentication; DOTL uses the unauthenticated `100ms` channel
and must compare trade records, not message boundaries or arrival latency.

### Instrument semantics

The frozen universe is inverse BTC options whose symbol is exactly:

```text
BTC-DDMMMYY-STRIKE-C
BTC-DDMMMYY-STRIKE-P
```

Deribit's inverse-option specification defines this symbol grammar, European
cash settlement, one-BTC contract multiplier, and 08:00 UTC expiration. Linear
`BTC_USDC` options, option combos, futures, perpetuals, spot instruments,
settlement rows, quotes, and order-book updates are outside DOTL-v1.

## Frozen canonical record

DOTL may retain only:

```text
trade_id
trade_seq
instrument_name
timestamp
direction
tick_direction
index_price
price
amount
mark_price
iv
block_trade_id
block_trade_leg_count
combo_id
combo_trade_id
block_rfq_id
```

Hard identity fields:

```text
trade_id, trade_seq, instrument_name, timestamp, direction,
price, amount
```

Auxiliary execution-time fields:

```text
tick_direction, index_price, mark_price, iv,
block_trade_id, block_trade_leg_count,
combo_id, combo_trade_id, block_rfq_id
```

Auxiliary fields must remain null when absent. They may support provenance
controls later but may not be imputed. Their presence and cross-surface
agreement are reported separately from hard ledger identity. The first DOTL
successor is forbidden from using auxiliary fields, so a source can be
identity-valid without claiming that every auxiliary value is immutable across
historical and live systems.

Canonical validity:

- `trade_id` is a nonempty string and unique per currency;
- `trade_seq` and `timestamp` are exact integers;
- `timestamp` is an exact Unix millisecond;
- `instrument_name` matches the frozen inverse-BTC option grammar;
- `direction` is exactly `buy` or `sell` and is the aggressor/taker direction;
- `price` and `amount` are finite positive arbitrary-precision decimals;
- when present, `tick_direction` is exactly `0`, `1`, `2`, or `3`;
- when present, `index_price` and `mark_price` are finite positive
  arbitrary-precision decimals;
- when present, `iv` is a finite nonnegative arbitrary-precision decimal; and
- no float round-trip, timestamp rounding, symbol repair, or ID synthesis is
  permitted.

`contracts` is excluded because current documentation says it may be absent in
historical trades. `amount` is the canonical size field.

## Source-parity gate

The first DOTL implementation may open only a prospectively fixed recent
parity interval and one prospectively fixed old schema interval. It must be
committed and HEAD-clean before either response body is read.

### Recent two-surface interval

The wall-clock interval is fixed before the verifier and before any trade body:

```text
start_ms = 2026-07-24T13:00:00.000Z
end_ms   = 2026-07-24T13:20:00.000Z
```

The verifier must be running and subscribed before `start_ms`, discard any
earlier notification, retain exactly the server-timestamp half-open interval
`[start_ms, end_ms)`, continue receiving until
`2026-07-24T13:20:05.000Z` to admit the final documented 100 ms batch, and then
send one `public/unsubscribe` for the exact channel. It must continue reading
while retaining only server timestamps in `[start_ms, end_ms)` until the
matching unsubscribe response arrives. The response must arrive by
`2026-07-24T13:20:10.000Z` and confirm the exact channel; otherwise the capture
fails. Only after that acknowledgement may the socket close and the recent
REST query begin. Starting after `start_ms`, disconnecting, reconnecting,
resubscribing, or changing the interval is a terminal failure.

Before subscription and after the five-second close grace, the verifier must
call Deribit's public server-time method and bracket each request with host
`CLOCK_REALTIME` plus `CLOCK_MONOTONIC_RAW`. The returned server millisecond
must fall within the host request interval expanded by 1,000 ms, host UTC may
not reverse, monotonic time must strictly advance, and each round trip must be
at most 2,000 ms. Failure retires the capture. Server-time responses are source
integrity metadata, not candidate inputs.

Recent REST pages use the same frozen algorithm as the old historical scan:
page 1 uses `start_timestamp=start_ms` and `end_timestamp=end_ms-1`; later
pages use the exact prior last `start_id` inclusively and the unchanged end
timestamp. Every request fixes `currency=BTC`, `kind=option`, `count=1000`, and
`sorting=asc`. All other pagination and overlap invariants below apply
unchanged; `has_more=false` is required for finite completion.

Recent WS↔REST parity passes only if:

1. the two hard trade-ID sets are exactly equal;
2. every hard identity field agrees exactly for every ID;
3. auxiliary-field presence and equality are reported field by field, without
   being promoted to hard identity;
4. within each instrument, sorting by `trade_seq` produces unique sequence
   numbers and nondecreasing timestamps;
5. no duplicate ID or conflicting sequence record exists; and
6. the interval contains at least 100 unique option trades and at least four
   distinct option instruments.

The 100-trade/four-instrument floor is an operational transport floor, not
alpha-support evidence.

### Old historical schema interval

The exact historical request interval is fixed now:

```text
[2021-01-04T00:00:00.000Z, 2021-01-05T00:00:00.000Z)
currency=BTC
kind=option
sorting=asc
count=1000
```

This interval was selected as the first complete UTC day beginning on the first
Monday of 2021, not from trade incidence. Its sole purpose is to prove that the
historical service can return a complete, deterministic, frozen-schema ledger
from a materially old period. It must contain at least 50 valid inverse-BTC
option trades across at least eight instruments. No BTC outcome or candidate
feature may be joined.

Historical pagination is frozen:

```text
fixed_start_ms = 2021-01-04T00:00:00.000Z
fixed_end_ms   = 2021-01-05T00:00:00.000Z
request_end_ms = fixed_end_ms - 1

page 1 params:
  currency=BTC, kind=option
  start_timestamp=fixed_start_ms
  end_timestamp=request_end_ms
  count=1000, sorting=asc

page n>1 params:
  currency=BTC, kind=option
  start_id=previous_last_trade_id
  end_timestamp=request_end_ms
  count=1000, sorting=asc
```

For every page:

1. the JSON-RPC envelope must match the frozen contract, `error` must be
   absent, and any response `id` is recorded but cannot affect parsing;
2. IDs must be decimal BTC trade-ID strings and strictly increasing within
   each raw page;
3. every timestamp must satisfy
   `fixed_start_ms <= timestamp < fixed_end_ms`;
4. a continuation page must begin with the exact prior last ID and an
   exact-equal hard record; that one boundary overlap is required, verified,
   and discarded before global uniqueness checks;
5. every remaining continuation record must have an ID greater than the prior
   last ID, and accepted IDs must be globally unique;
6. `has_more=false` terminates acquisition, including when the final page has
   exactly 1,000 rows;
7. any row outside the fixed interval, cursor reversal, duplicate/conflicting
   ID, empty `has_more=true` page, or nontermination retires DOTL; and
8. at most 10,000 pages are permitted.

No timestamp tie is advanced by timestamp. Continuation is exclusively
the known prior `trade_id`; synthesizing an unobserved numeric cursor is
forbidden. The required one-record overlap avoids loss when more than 1,000
trades share a millisecond. Gaps between accepted numeric IDs are allowed
because the currency-wide ID sequence also contains non-option trades.

After the complete old-day scan, the verifier must replay every page using its
exact original request in the same order. Each canonical raw-page hard-record
hash, `has_more`, first ID, and last ID must match the first read exactly. The
required continuation overlap is verified again. Full-day raw responses are
streamed and discarded.

### Terminal failures

Any of the following retires DOTL-v1 unchanged:

- authentication is required for a frozen public surface;
- a redirect leaves an exact Deribit host;
- the old interval is empty or violates the frozen schema;
- recent hard ID sets or hard canonical fields differ;
- pagination is incomplete, ambiguous, reversing, or nonreplayable;
- rate limiting prevents exact completion under the frozen retry budget;
- a response is truncated or malformed after body transfer begins; or
- source acquisition would push filesystem use to 300 GiB or more.

There is no sampled parity, percentage tolerance, missing-trade allowance,
composite-key fallback, historical/live field substitution, alternate old
day, or post-result retry with a different interval.

The verifier may retry an identical request at most twice, after fixed waits of
five and fifteen seconds, only for a connection failure before any response
body byte or for HTTP `429`. A partial body, JSON/schema failure, application
error, or third identical failure is terminal. Historical page requests are
paced at no more than four requests per elapsed second.

## Acquisition contract after a pass

Only a complete DOTL parity pass may authorize a historical aggregate build.
The build must:

- stream pages oldest-first and never persist raw trade responses;
- begin at the first supported complete UTC day no later than 2021-01-01;
- end before 2024-01-01 for initial candidate construction;
- hash every canonical page and record request/response metadata;
- deduplicate only exact byte-equivalent records with the same `trade_id`;
- fail on conflicting duplicates, unexplained page/time gaps, reversing IDs,
  or schema drift; numeric ID gaps caused by other BTC product kinds remain
  valid;
- preserve exact 24/7 UTC time without exchange-session filling;
- record missing intervals as unavailable rather than zero flow; and
- abort before filesystem use reaches 300 GiB.

2024 and later historical trades remain sealed until one source-supported,
pre-outcome-novel mechanism passes its train and 2023 selection gates.

## Authorized successor design space

After DOTL source passage, one separately committed mechanism may use only
causal relations among:

- call versus put aggressor direction;
- near, middle, and far expiry horizons;
- own-history ordinal flow states; and
- current source validity/age.

The preferred object is a **cross-horizon convexity-transfer grammar**:
structural agreement across option horizons versus short-horizon demand that
is contradicted by longer horizons. Exact bins, aggregation, side, transition,
hold, cooldown, controls, and support gates are not defined here and must be
committed before their incidence is computed.

Forbidden successors:

- raw call/put ratio, total option volume, total trade count, average ticket,
  standalone IV, standalone skew, or a fitted threshold on one of them;
- static expiry wall, delivery clock, DVOL tail, funding impulse, or OPDR
  repair;
- strike, strike-distance, moneyness, IV, mark/index price, tick direction,
  block, combo, or RFQ fields in the first successor;
- inferring investor identity from public trade direction;
- treating block taker direction as informed direction without an explicit
  falsification control;
- generic price momentum, lead/lag, return prediction, or outcome-selected
  window search;
- analyzer/trader two-model architecture; or
- an LLM that creates timestamps, sides, leverage, or holds.

The first successor must hash-bind the exact DEWH-144, DEHR-72, OPDR-24, and
prior DVOL clocks before its own incidence is computed. It must predeclare
exact-entry overlap, tolerant daily overlap, same-side reproduction, Jaccard,
and signed occupied-exposure correlation gates. A failed or undefined novelty
metric retires the candidate before outcomes. Expiry horizon may describe only
the maturity of each executed trade; it may not use prior strike walls,
delivery rows, expiry-event clocks, or their candidate states.

## RLLM boundary

Any later deterministic composer owns candidate time, fixed side, hold,
leverage, and cost. A small Gemma policy may receive only symbolic causal
relations plus current position/risk state and choose:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

Raw prices, sizes, IV values, timestamps, dates, IDs, split labels, future
paths, rewards, PnL, performance summaries, and historical rank numbers are
forbidden from the model input.

## Official references

- Deribit public recent trades and trade-ID/timestamp pagination contract:
  <https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency>
- Deribit trade subscription by kind/currency:
  <https://docs.deribit.com/subscriptions/trades/tradeskindcurrencyinterval>
- Deribit public server clock:
  <https://docs.deribit.com/api-reference/supporting/public-get_time>
- Deribit market-data collection:
  <https://support.deribit.com/hc/en-us/articles/29592500256669-Market-Data-Collection-Best-Practices>
- Deribit inverse option contract:
  <https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options>
- Deribit block-trade publication:
  <https://support.deribit.com/hc/en-us/articles/25944688627229-Block-Trading>
- Deribit account/API setup:
  <https://support.deribit.com/hc/en-us/articles/29636577934365-Account-Setup-Guide>

These references were current on 2026-07-24. The historical host is less
fully described in the current public API reference than the recent/live
surfaces, so DOTL deliberately treats historical availability and parity as a
hard empirical source gate rather than an assumption.

## Mandatory sequence

1. independently review this corrected boundary for hidden prior-family repair,
   mutable intervals, and unverifiable parity;
2. commit this source boundary only after P0/P1 findings are closed;
3. implement and test the one-shot DOTL source-parity verifier;
4. commit the verifier from a HEAD-clean worktree;
5. execute the parity audit once and retire unchanged on any terminal failure;
6. on a pass, commit one exact mechanism and preregistration before historical
   aggregation or candidate incidence;
7. run source support and prior-family novelty before BTC outcomes;
8. open train, selection, and sealed extensions sequentially, stopping at the
   first failed gate.
