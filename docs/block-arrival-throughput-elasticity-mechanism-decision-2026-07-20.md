# Block-arrival throughput elasticity mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **BATE-288 — Block-Arrival Throughput
Elasticity, 24-hour hold**. It will measure transaction and block-weight
throughput over a short packet of completed Bitcoin blocks, then test whether a
broad settlement-demand burst or drought persists in BTC after a conservative
confirmation and availability delay.

This decision opens no complete 2020–2023 block prefix, candidate incidence,
market value, funding mark, post-decision return, PnL, or 2024+ source row. The
bounded source probes used only three ten-block pages and two exact calendar
boundary searches. No candidate threshold or outcome was calculated from
those probes.

## Why this is a new data-generating mechanism

Existing repository candidates already use price action, derivatives basis,
funding, open interest, liquidations, daily aggregate fees, hash-price stress,
and difficulty-adjustment clocks. The existing blockspace candidate uses
daily `FeeTotNtv`, `IssTotNtv`, `BlkCnt`, and `TxCnt`; it does not use individual
block weight, transaction count, or observed inter-block elapsed time. The
existing difficulty candidate uses adjustment-height events rather than
blockspace throughput.

BATE instead asks whether demand is consuming blockspace faster than the chain
is supplying it:

1. a packet ending at height `h` contains completed blocks `h-5` through `h`;
2. the elapsed supply interval starts at the timestamp of block `h-6` and ends
   at the timestamp of block `h`;
3. `weight_throughput = sum(weight[h-5:h]) / elapsed_seconds`;
4. `tx_throughput = sum(tx_count[h-5:h]) / elapsed_seconds`; and
5. concordantly high values represent broad settlement pressure, while
   concordantly low values represent a demand drought.

Using both channels is deliberate. Block weight alone can be elevated by
witness-heavy, low-transaction payloads; transaction count alone ignores how
much scarce blockspace those transactions consume. Concordance is intended to
identify broad throughput rather than one payload format. The exact transform,
strictly-prior reference, thresholds, transition rule, and support minima will
be frozen in a separate preregistration before the complete source prefix is
downloaded.

The tentative economic action is long after a high-throughput onset and short
after a low-throughput onset, with a fixed 288 five-minute-bar hold. This is a
hypothesis, not a profitability claim.

## Official source contract

Blockstream Esplora documents:

- `GET /block/:hash` with `id`, `height`, `timestamp`, `mediantime`,
  `tx_count`, `size`, `weight`, and `previousblockhash`;
- `GET /blocks/:start_height` as a ten-block descending page; and
- `GET /block/:hash/status` with `in_best_chain` and `next_best`.

Official references:

- [Esplora HTTP API](https://github.com/Blockstream/esplora/blob/master/API.md)
- [Esplora MIT license](https://github.com/Blockstream/esplora/blob/master/LICENSE)
- [Bitcoin Core 30.0 `getblock`](https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblock/)
- [Bitcoin Core 30.0 `getblockchaininfo`](https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockchaininfo/)
- [Bitcoin Core MIT license](https://github.com/bitcoin/bitcoin/blob/master/COPYING)
- [BIP 141 block-weight limit](https://github.com/bitcoin/bips/blob/master/bip-0141.mediawiki)
- [Bitcoin block-header timestamp rules](https://developer.bitcoin.org/reference/block_chain.html)

A bounded Esplora boundary probe found:

- first block with header timestamp at or after `2020-01-01T00:00:00Z`:
  height `610691`, timestamp `2020-01-01T00:03:05Z`;
- its predecessor: `2019-12-31T23:41:32Z`;
- first block with header timestamp at or after `2024-01-01T00:00:00Z`:
  height `823786`, timestamp `2024-01-01T00:29:38Z`; and
- its predecessor: `2023-12-31T23:46:12Z`.

The frozen source interval is therefore heights `610691` through `823785`
inclusive: 213,095 blocks physically excluding the first 2024 block. A packet
may signal only when its six-confirmation block also lies inside that prefix,
so the latest eligible packet end is height `823779`.

## Availability and leakage boundary

Bitcoin header timestamps are miner-reported event fields, not historical
first-seen timestamps. Consensus constrains them relative to median time past
and network-adjusted time, but does not make them a publication log. Historical
research will therefore never assume that block `h` became usable exactly at
its header timestamp.

For a packet ending at `h`:

1. wait through block `h+6`;
2. take the maximum header timestamp observed through `h+6`;
3. add a conservative two-hour historical publication embargo;
4. round up to the next five-minute boundary; and
5. consume one additional complete five-minute latency bar before entry.

Live production must be stricter: record each block's actual local first-seen
time, fail closed until `h+6` is observed on the active chain, and use the
later of actual availability or the historical synthetic timestamp. A reorg
invalidates unconfirmed packets. Historical/live parity must be audited before
promotion.

Source-only support code may load block metadata but no BTC market, funding,
premium, OI, liquidation, order-book, FX, or post-entry field. Complete
2020–2023 source rows remain unopened until the downloader, integrity tests,
and exact support preregistration are committed.

## Production path and data-use boundary

The public Esplora endpoint is suitable for a bounded, reproducible private
research backfill, but no hosted-service SLA is assumed. Production will use a
self-hosted pruned Bitcoin Core node, poll or wait for new blocks, call
`getblock` for `weight`, `nTx`, `time`, and chain linkage, and persist local
first-seen timestamps. Pruning keeps the live source independent of a full
historical transaction index and compatible with the repository's disk cap.

Raw public responses remain ignored and local. Committed artifacts may contain
only source URLs, source range and continuity audits, hashes, aggregate support
counts, and derived research outputs allowed by the upstream license.

## Why Binance mark/index klines are not selected next

Binance exposes official mark-price and index-price kline endpoints and public
monthly archives, so this alternative is operationally easy:

- [mark-price klines](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data)
- [index-price klines](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Index-Price-Kline-Candlestick-Data)
- [Binance public data repository](https://github.com/binance/binance-public-data)

However, perp-minus-mark and mark-minus-index geometry is algebraically close
to premium/basis and to the already rejected spot/perp wick family. It remains
a valid future execution/liquidation-anchor input, but it is less independent
than block-arrival throughput and is not the next candidate.

## Frozen research sequence

1. Commit this mechanism decision without opening the full source prefix.
2. Implement and test one resumable, source-only Esplora block-summary loader.
3. Commit one exact BATE support preregistration before running that loader over
   the complete 2020–2023 interval.
4. Reject without repair if continuity, clock causality, side balance, calendar
   dispersion, or source-only support fails. Do not alter packet size,
   confirmation count, thresholds, side, or hold after incidence is visible.
5. If support passes, hash-freeze one strict market evaluator before loading
   any post-entry path.
6. Evaluate 2021–2022 train and 2023 selection first. Both must pass absolute
   return, full-calendar CAGR/strict-MDD, stress-cost, delayed-entry, and
   statistical gates before any 2024+ market outcome is opened.
7. Open 2024, 2025, and 2026 YTD sequentially. Stop at the first failed sealed
   year and do not repair the policy.

The branch is globally contaminated by prior BTC research. This sequence can
support only a candidate-level frozen claim, never a pristine global human
holdout claim.
