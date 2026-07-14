# Checksummed Binance kline reference

## Why this dataset exists

The aggTrade audit initially used the existing wave-market cache as its kline
reference. Direct comparison with Binance's current official archive found a
material source-version example at `2023-11-10 15:05 UTC`: the official daily
kline (`high=37168.7`, `close=37145.6`, `volume=1252.754`) agrees with the
daily aggTrade reconstruction, while both the existing cache and official
monthly kline archive contain older values
(`high=37118.7`, `close=37118.4`, `volume=206.313`). The reference side of the
audit therefore uses the current daily archive family on both sides rather
than mixing daily aggTrades with monthly klines.

## Frozen build contract

- Source: official Binance USD-M daily 5m kline archives:
  https://data.binance.vision/?prefix=data/futures/um/daily/klines/
- Range: `[2020-01-01, 2024-01-01)`; no later outcome rows are read.
- Every daily `.CHECKSUM` is verified before parsing and rechecked on resume.
- Headerless older archives and current headered archives are both supported.
- Every requested month must have exact, unique, monotonic 5m coverage.
- Gzip output fixes `mtime=0`; monthly and combined SHA-256 values are recorded.
- CSV byte hashes are deterministic; `build_manifest.json` intentionally
  records a changing `as_of` audit timestamp and is not byte-deterministic.
- Emitted fields are only UTC date, OHLC, base/quote volume, trade count and
  taker-buy base/quote volume. No future label or return is computed.

The resulting frame is the audit and preregistration price source. The older
wave cache remains useful for external features, but its OHLCV columns are not
authoritative when current Binance archives disagree.
