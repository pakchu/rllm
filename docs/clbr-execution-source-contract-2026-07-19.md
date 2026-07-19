# CLBR-24 execution-source contract — 2026-07-19

This stage freezes executable market and funding inputs without loading the
CLBR-24 event clock or calculating a strategy return.

## Official inputs

- Binance USD-M `BTCUSDT` five-minute trade-price klines from daily Binance
  Vision archives.  Every ZIP is verified against its adjacent `.CHECKSUM`:
  <https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fdaily%2Fklines%2FBTCUSDT%2F5m%2F>.
- Exact USD-M funding history from the bounded official `fundingRate` endpoint:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history>.
- Official 8h mark-price kline opens provide settlement marks when historical
  funding rows omit `markPrice`:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#mark-price-klinecandlestick-data>.

## Physical boundary

- source interval: `2023-06-25 00:00:00` inclusive through
  `2024-10-15 00:00:00` exclusive;
- every market day must contain exactly 288 unique, ordered 5m bars;
- any missing archive, checksum mismatch, duplicate, or OHLC-envelope violation
  aborts the build;
- funding and mark API calls include an explicit end time before the exclusive
  boundary;
- output is physically split into train, test, and eval files so staged
  evaluators need not load unopened windows.

All timestamps are UTC represented as timezone-naive values.  Funding uses the
exact returned timestamp.  A missing historical funding mark maps to the open
of the official 8h mark-price kline at `floor(fundingTime, 8h)`; overlap rows
must keep the implied funding-cash discrepancy below 0.1bp of notional.

## Frozen artifacts

The build verified 478 daily market archives and produced 137,664 complete 5m
bars plus 1,434 funding events.  Of those funding rows, 1,049 include the
recorded mark directly and 385 use the uniform 8h mark-open proxy.  The maximum
implied funding-cash discrepancy on overlap rows is only
`0.0013484319911147846` bp of notional.

| split | market SHA-256 | funding SHA-256 |
|---|---|---|
| train | `fa78e344e576ed3d1e911325613bce1465bfc76c259c0a3733cb350e1cdac2e4` | `b94daae411b41d447e52dd0490a269ffd28eaf9316ddbea0da8a6a293d7d44ce` |
| test | `3cbc1198ee32b5d77cdfa468bdaf9ed34af346a962b7026c70c70f3ff0ba7af7` | `4b16e60417d30592679d41eeac2d08231c0bd37d337a73dbe9b8c0e43d285414` |
| eval | `212a441e2e8213eda528e2cd586853515785f51ee4291ef8bf8f05ae0d6e52f4` | `07dc50bbdff43f6704d819bea0ef0e32c5ff93d7072cdb8252753c122bec8fbd` |

Manifest SHA-256:
`50b86d6ab896a1c913ee83311f416f67392f29f6fb5a143f59f3abc08448d0c6`.
