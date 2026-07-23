# Bybit public-trade sequence source-axis decision — 2026-07-23

## Decision

The next BTC alpha search will inspect the official Bybit `BTCUSDT` linear-
perpetual **individual public-trade sequence**.  The provisional mechanism is
**BSEA-24** (Bybit Sequence-Entropy Absorption, two-hour hold), but this file
freezes only the source axis and access boundary.  It does not freeze a signal,
open event incidence, or make a profitability claim.

The source is selected because this repository has already opened Bybit
funding and premium-index history, but not Bybit's historical public-trade
archive.  Repository search before this decision found no
`public.bybit.com/trading` consumer and no Bybit trade-sequence feature.  The
existing Bybit work is limited to the auxiliary funding/premium panel and the
cross-venue funding-consensus experiment.

## Official evidence available before source-value access

The following official pages were reviewed without opening a historical trade
record:

- Bybit's API home explicitly offers public OHLCV and trade-history CSVs for
  backtesting, research, and quantitative model development:
  <https://bybit-exchange.github.io/docs/>.
- The official `BTCUSDT` trade-archive directory lists daily files from
  `2020-03-25` through `2026-07-22` at the decision time:
  <https://public.bybit.com/trading/BTCUSDT/>.
- The public recent-trade REST response documents execution ID, price, size,
  taker side, execution time, block/RPI flags, and cross sequence:
  <https://bybit-exchange.github.io/docs/v5/market/recent-trade>.
- The public WebSocket trade channel is real-time, orders each message's trades
  by match time, and documents match time, taker side, size, price, tick
  direction, trade ID, block/RPI flags, and cross sequence:
  <https://bybit-exchange.github.io/docs/v5/websocket/public/trade>.
- Bybit describes the API as supporting automated trading and historical
  market-data download:
  <https://www.bybit.com/future-activity/developer>.
- Current API use remains subject to Bybit's API terms:
  <https://www.bybit.com/en/help-center/article/API-Terms>.

Only directory/file-name metadata was opened.  No archive header, historical
trade row, price, size, side, trade ID, feature, candidate clock, BTC outcome,
funding, return, PnL, CAGR, or strict MDD was read for this decision.

## Why this is not permission to repeat prior flow searches

Generic trade volume, trade count, average ticket, signed notional, aggregate
imbalance, price impact, zero-tick frustration, fill dispersion, same-
millisecond cascades, flow campaigns, and Spot/USD-M temporal torsion have
already been inspected in this repository.  A new venue alone does not make a
new alpha.

BSEA may advance only if the bulk archive exposes a stable, ordered individual-
execution representation that maps conservatively to the current REST and
WebSocket feeds.  Its mandatory new object would be the **ordering of taker-
side states**, aggregated first into fixed causal micro-buckets so exchange-
specific match fragmentation cannot masquerade as persistence.  The tentative
economic question is whether unusually persistent Bybit taker-side states,
when Binance flow and price fail to confirm them, identify follower crowding
that is being absorbed.

The following are forbidden as the primary candidate under this decision:

- Bybit-only price or volume momentum;
- another threshold on aggregate taker imbalance;
- individual-fill HHI or ticket-size dispersion;
- a generic Bybit-versus-Binance lead/lag or transfer-entropy rule;
- trade-ID gaps, UUID bytes, or archive defects treated as economics; and
- post-source switching among continuation and reversal directions.

## Frozen bounded feasibility probe

After this document is committed, a source-only probe may:

1. issue HTTP metadata requests for exactly
   `2020-03-25`, `2023-01-01`, and `2026-07-22`;
2. decompress only enough of each exact file to obtain its CSV header and at
   most one first data record for type/schema validation;
3. record response metadata, compressed bytes consumed, header fields, field
   count, and hashes of the consumed raw prefixes; and
4. inspect the archive directory names to verify complete calendar coverage
   for 2023, without downloading those files.

The probe must not aggregate trades, count candidate states, load Binance
comparators, or open any post-entry market value.  It passes only if every
boundary exposes timestamp, symbol, taker side, size, price, and a stable
execution identifier with documented live counterparts.  Schema drift must be
explicitly mapped; silently dropping fields is forbidden.  Absence of taker
side, ordering, or execution identity rejects this source axis without a
fallback to OHLCV.

## Historical/live parity and disk contract

Before an alpha clock can be built:

- a recent archive day, REST sample, and WebSocket capture must reconcile on
  timestamp, side, price, size, execution identity, ordering, duplicates, and
  omissions over an overlap window;
- WebSocket message boundaries and shared `seq` values must not be interpreted
  as parent orders;
- block/RPI fields are optional only as controls and cannot enter the primary
  unless the same semantics exist through the full frozen historical window;
- source records must be assigned only to completed UTC micro-buckets, and a
  five-minute feature becomes actionable no earlier than the following
  five-minute open;
- every daily archive must be streamed, hashed, reduced, and discarded before
  the next day; raw archives may not accumulate; and
- the builder must abort before a download whenever filesystem usage is at or
  above **300 GiB**.  Decision-time usage was approximately 288 GiB.

Bybit's public material supports backtesting and automated API trading, but the
project still must obey the current API terms and rate limits.  This document
does not grant redistribution rights and does not authorize storing account
credentials or private data.

## Next immutable sequence

1. Commit this source-axis decision before opening any archive record.
2. Run and commit the exact bounded feasibility probe above.
3. Reject the source immediately on schema, coverage, terms, disk, or live-
   parity infeasibility; do not repair BSEA with generic volume features.
4. Only a passing source audit may receive one exact mechanism decision and a
   machine-readable preregistration with one side rule and one hold.
5. Build an outcome-blind source/support/null evaluator.  It must include
   Bybit-only, Binance-only, aggregate-imbalance, venue-swap, time-reversal,
   stale, direction-flip, and prior-clock containment controls.
6. Only support, novelty, and live-parity passes may authorize a separately
   committed strict economic evaluator.  Open pre-2024 outcomes first and
   later calendars sequentially, stopping at the first failed gate.

Passing a source or support gate is only permission for the next stage.  It is
not evidence of an alpha.
