# Binance BTCBVOLUSDT hourly source design — 2026-07-19

## Purpose

Build an outcome-blind hourly volatility-index source from Binance's official
`BTCBVOLUSDT` one-second daily archives. The intended research use is a
cross-venue volatility disagreement with causally closed Deribit DVOL candles.
No directional BTC return is opened by this source stage.

## Contract

- official archive root:
  <https://data.binance.vision/?prefix=data/option/daily/BVOLIndex/BTCBVOLUSDT/>;
- every ZIP is verified against its adjacent `.CHECKSUM` file;
- rows must identify `BTCBVOLUSDT`, `BTCBVOL`, and `USDT` exactly;
- calculation timestamps must be unique and increasing; Binance's small
  sub-second calculation jitter is floored to its UTC second, and duplicate
  floored seconds fail closed;
- an hourly candle is valid only with all 3,600 one-second observations;
- feature availability and earliest trade time are the next hour boundary;
- incomplete hours are retained on the grid but all OHLC values are quarantined;
- raw ZIPs are never persisted;
- 2024+ construction fails closed unless `--open-oos` is explicitly supplied
  after a pre-2024 policy freeze.

The archive field is treated only as the published Binance index value. The
research must not infer unavailable option skew, trader identity, or individual
contract flow from this scalar series.
