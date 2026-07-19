# Binance COIN-M ETH liquidation-snapshot source design — 2026-07-19

## Purpose

Build a checksum-verified, causal, price-free five-minute source from Binance's
official `ETHUSD_PERP` force-order snapshots. This source is for an
outcome-blind ETH-to-BTC relay preregistration; it opens no BTC execution price,
return, funding, label, or PnL.

The existing BTC source builder and frozen BTC artifact are not modified. The
ETH source has a separate schema and hash boundary.

## Official source semantics

- Binance describes the COIN-M force-order feed as a liquidation-order
  **snapshot** stream that emits at most one latest snapshot per symbol per
  1,000ms interval and emits nothing when there is no liquidation. It is a
  censored stress indicator, not a complete liquidation-fill tape:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-coin-m-futures/api/ws-streams/~/>.
- Daily ZIP files and adjacent `.CHECKSUM` files are published under the
  official Binance Vision COIN-M tree:
  <https://data.binance.vision/?prefix=data%2Ffutures%2Fcm%2Fdaily%2FliquidationSnapshot%2FETHUSD_PERP%2F>.
- `BUY` is a forced purchase closing a short; `SELL` is a forced sale closing a
  long. The archive's order side is retained only through these directional
  aggregates.

## Retained causal fields

Each five-minute row retains only:

- UTC bar-open time;
- feature availability at bar end plus one second;
- source-valid flag;
- total, forced-buy, and forced-sell snapshot counts;
- forced-buy, forced-sell, total, and signed accumulated contract quantities;
- directional liquidation imbalance.

No price, USD notional, return, funding, open interest, label, signal, or PnL
field is written. The later ETH/BTC relay must normalize each symbol against
its own strictly-prior distribution. It may not compare raw ETH and BTC
contract quantities. Because the signal uses within-symbol quantile ratios, a
constant contract multiplier would cancel and is deliberately unnecessary.

## Integrity and latency contract

- Verify each archive against its adjacent official checksum before parsing.
- Remove exact whole-row duplicates only; preserve distinct snapshots that
  share a millisecond.
- Treat a missing archive or checksum as an invalid day, never as zero
  liquidation.
- Require positive integral contract quantities and internally consistent IOC
  filled-order fields.
- Keep raw ZIP payloads only in memory.
- A completed five-minute row is available at bar end plus one second.
- Any executable clock must enter no earlier than the next five-minute open.

## Physical range

The source is frozen to `2023-06-25` inclusive through `2024-10-15`
exclusive, matching the existing BTC COIN-M source boundary. The build opens no
market outcomes; archive support and missing dates are source metadata only.

## Frozen outcome-blind archive audit

- archive days requested: 478;
- checksum-verified available days: 474;
- missing archive/checksum dates: `2023-08-05`, `2023-09-09`,
  `2023-09-23`, `2023-09-25`;
- raw rows: 56,186;
- unique snapshots after exact deduplication: 28,092;
- exact duplicates removed: 28,094;
- five-minute rows: 137,664, of which 136,512 are source-valid and 11,637
  contain at least one snapshot;
- panel SHA-256:
  `8d17ab3d5f9592f5254fef2e649065233be1777b8976983b4af38c77a8cc5bff`;
- manifest SHA-256:
  `c515731a9029d1786c8650f5106923d4cfbe8c35ed7a947f5420a16154601f5d`.

All available days except `2023-09-21` contain exactly two copies of each
snapshot row. On `2023-09-21`, 76 raw rows collapse to 37 unique rows. The
builder applies the same exact-whole-row deduplication rule globally and does
not infer why the source duplicated records.

## Stop conditions

Reject the source if checksums, exact-duplicate semantics, UTC-day bounds,
integer quantities, missing-day invalidation, deterministic five-minute grid,
or price-free output cannot be enforced. Do not repair the frozen BTC builder
to accommodate ETH.
