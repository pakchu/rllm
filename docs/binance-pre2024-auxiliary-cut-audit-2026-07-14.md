# Binance BTCUSDT pre-2024 auxiliary cut audit — 2026-07-14

## Result

The existing official Binance USD-M BTCUSDT funding and premium sources were
physically cut to `[2021-01-01, 2024-01-01)` before cross-venue research.
Future rows in the original 2020–2026 cache are absent from the new files.

| source | range | rows | SHA256 |
|---|---|---:|---|
| funding | 2021-01-01 00:00 through 2023-12-31 16:00 UTC | 3,285 | `654c668e3aea344d5906465cbbd090f2e4ff0c47e9d4bd8cf3856c24549cfc97` |
| premium index 1h | 2021-01-01 00:00 through 2023-12-31 23:00 UTC | 26,280 | `ed2626c14591cf77f927f71559b81f3c2d0be1d1d5085af4abf7884578f4f972` |

Manifest:
`results/binance_um_aux_btc_2021_2023_manifest.json`.

## Integrity policy

- Both original gzip hashes are pinned before decoding.
- Funding's exchange timestamps carry up to 47 ms of observed API jitter. They
  are normalized to the nearest UTC 8-hour boundary only after asserting a
  maximum allowed jitter of 1,000 ms.
- The normalized funding series is a complete 8-hour grid.
- Premium candles form a complete, duplicate-free hourly grid and pass finite
  OHLC invariants.
- Output gzip files use deterministic headers.
- The builder rejects an output boundary after 2024-01-01.

Official endpoint semantics:

- funding history:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History>
- premium index kline:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data>

No return or outcome was computed. These files only establish a symmetric
Binance/Bybit causal feature range for the next support-only experiment.
