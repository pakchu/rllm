# Kraken urgency-to-constraint reversal source-feasibility rejection — 2026-07-20

## Decision

**Do not preregister or evaluate KUCR now.** The proposed Kraken
urgency-to-constraint reversal is rejected at the production-source gate,
before feature incidence, thresholds, BTC outcomes, funding, returns, PnL,
absolute return, CAGR, or strict MDD are opened.

KUCR would have required an early same-side burst of Kraken market-taker
executions, a late burst of opposite-side taker limit-order executions, and a
rejection of the early price excursion. The tentative action was to trade the
opposite side at the next completed Binance 5-minute-bar open. The mechanism
is not disproven; its required historical source and live-equivalent contract
are not yet production-ready.

Official references reviewed on 2026-07-20:

- [Kraken downloadable historical time and sales](https://support.kraken.com/articles/360047543791-downloadable-historical-market-data-time-and-sales-)
- [Kraken historical-data guide](https://docs.kraken.com/exchange/guides/general/historical-data)
- [Kraken Spot WebSocket v2 trade channel](https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/trade)
- [Kraken Global Terms of Service](https://www.kraken.com/legal/global-terms)

## Bounded source-only findings

A bounded HTTP range probe inspected only the central directory and the first
compressed prefix of the official all-pair archive. It established the
following without downloading the complete archive or calculating an outcome:

- the complete ZIP was 12,554,214,086 bytes when probed;
- `TimeAndSales_Combined/XBTUSD.csv` was present;
- its compressed and uncompressed sizes were 603,212,773 and 2,702,663,341
  bytes respectively;
- the CSV had no header and exactly three fields:
  `unix_seconds, price, volume`; and
- its first observed timestamp was `1381095255`, establishing long calendar
  coverage but not enriched trade semantics.

The temporary probe bytes were deleted. No complete Kraken source prefix was
retained.

The downloadable archive therefore cannot reconstruct KUCR. In particular it
does not expose taker side, taker order type, or trade ID. Kraken's REST
`Trades` response does expose `[price, volume, time, buy/sell, market/limit,
miscellaneous, trade_id]`, but it is capped at 1,000 trades per request and
must be paginated using the nanosecond `last` cursor. Kraken's own guide lists
a 1–2 second safe delay for bulk trade collection, while another sentence on
the same page says 100–200 milliseconds. That internal guidance mismatch must
not be resolved by silently selecting the faster rate.

## Why this is not a production alpha source yet

1. **The usable history is not the bulk history.** The practical bulk archive
   lacks the two fields that define the mechanism. A multi-year enriched
   history would require deep REST pagination rather than a checksum-audited
   official dump.
2. **Archive/live parity is unproven.** REST and WebSocket v2 expose broadly
   corresponding trade fields, but the project has not replayed an overlap
   window to prove equality of side, price, volume/quantity, order type,
   timestamp, trade ID, ordering, and de-duplication. A WebSocket message batch
   must not be treated as one parent taker order.
3. **The semantic claim needs narrowing.** Kraken's `limit` field identifies a
   taker limit-order execution. It does not reveal passive liquidity, a hidden
   order, or guaranteed constraint-induced absorption. Calling it
   "marketable-limit pressure" would overstate the observable.
4. **The causal direction is not isolated.** A Kraken reversal may merely lag
   Binance perpetual or global BTC flow. Any future design would need frozen
   Binance-prior/current price and taker-flow controls before making a Kraken
   leadership claim.
5. **The mechanism is only conditionally novel.** `ord_type` is a new source
   field in this repository, but the early urgency, rejected excursion, late
   opposition, and two-hour fade shape is close to existing REX,
   cash-leadership, CLASP/CSPR, RIFT, and terminal-absorption families. Order
   type would have to remain mandatory and survive explicit ablations.
6. **Production-use permission is not unambiguous.** Kraken's Global Terms
   permit use of its content only for the user's own benefit, while also
   restricting data extraction, automation, third-party applications, and
   commercial exploitation except as expressly permitted. Kraken's API guide
   simultaneously documents automated pagination. The repository has no
   written permission or API-specific licence resolving that boundary for a
   derived, profit-seeking production signal. This is an operational gate, not
   a legal conclusion.

## Reopening boundary

KUCR may be reconsidered only after all of the following are satisfied without
opening post-entry outcomes:

- Kraken confirms the intended historical collection and profit-seeking live
  use are permitted;
- an enriched REST prefix is collected with request, cursor, gap, duplicate,
  and raw-content-hash manifests;
- a contemporaneous REST/WebSocket overlap proves field-level live parity;
- the exact early/late windows, strictly-prior threshold clock, rejection
  fraction, side, and 24-bar hold are committed in advance;
- an outcome-blind support gate passes by year, half-year, month, and side;
  and
- no-order-type, ZIP-only wick, Binance-flow, side-flip, time-reversal, stale,
  and prior-family overlap nulls are frozen.

No parameter repair or outcome evaluation is allowed under the KUCR name
before those conditions pass. The next alpha search should use an official
source that already supports both historical reconstruction and live parity.
