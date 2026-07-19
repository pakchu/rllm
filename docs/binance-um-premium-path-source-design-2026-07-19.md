# Binance USD-M premium path source design — 2026-07-19

## Purpose

Build a source-only `BTCUSDT` premium-index path panel for a new pre-price
alpha family. This artifact may establish clocks and support counts, but it may
not expose BTC execution prices, returns, funding or PnL.

## Frozen source contract

- Official source: Binance Vision monthly USD-M `premiumIndexKlines` archives.
- Symbol / interval: `BTCUSDT` / `1m`.
- Requested range: `[2020-01-01, 2026-07-01)` UTC.
- Every published ZIP must match its adjacent `.CHECKSUM` file.
- Both legacy millisecond timestamps and the later microsecond archive format
  are normalized to milliseconds only after rejecting mixed units.
- A source bar opened at `t` closes at `t + 1 minute - 1 millisecond`; its
  conservative feature availability is `t + 1 minute + 1 second`.
- Missing archives or rows remain on the complete one-minute grid with
  `source_valid=false` and missing premium OHLC. They are never zero-filled.
- Raw archives are held only in memory and are not retained.

Retained fields are limited to source timestamps, validity and premium-index
OHLC. Names that resemble BTC execution price, return, funding or PnL are
forbidden from the output schema.

Official archive root:
<https://data.binance.vision/?prefix=data/futures/um/monthly/premiumIndexKlines/>

Official endpoint semantics:
<https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data>

## Intended candidate boundary

The source is being built for one outcome-blind multi-bar path candidate. It
does not authorize a threshold grid, direction repair, hold search, price gate,
regime gate, stop, take-profit or model selection. Candidate rules and strict
evaluator controls must be frozen separately before any BTC outcome is opened.
