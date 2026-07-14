# Bybit BTCUSDT funding/premium source audit — 2026-07-14

## Result

A physically pre-2024 Bybit linear-perpetual auxiliary panel was built for a
new cross-venue alpha family. No return, post-2023 market row, or trading
outcome was requested or opened.

| source | range | rows | SHA256 |
|---|---|---:|---|
| BTCUSDT funding | 2021-01-01 00:00 through 2023-12-31 16:00 UTC | 3,285 | `d7e019f34120d84d7c574a361a670b104d8f0c17f9b155d2dd01f1dc74913204` |
| BTCUSDT premium index 1h | 2021-01-01 00:00 through 2023-12-31 23:00 UTC | 26,280 | `ebfed8281a9e9e9780bbe542c04d00bf52d2dcebe175caa3c8aa3a94f361482b` |

The hash manifest is
`results/bybit_linear_aux_btc_2021_2023_manifest.json`. Raw API data remains
under ignored `data/` storage and is not added to the repository.

## Official source contract

The builder uses two public Bybit V5 endpoints:

- funding-rate history:
  <https://bybit-exchange.github.io/docs/v5/market/history-fund-rate>
- premium-index kline history:
  <https://bybit-exchange.github.io/docs/v5/market/premium-index-kline>

The funding documentation defines `fundingRate` and
`fundingRateTimestamp`, supports linear `BTCUSDT`, and allows up to 200 rows.
The premium documentation defines reverse-ordered candles with start, open,
high, low, and close, and allows up to 1,000 rows. The builder explicitly
requests `[2021-01-01, 2024-01-01)` in bounded chunks and drops any row outside
that interval before validation.

Live endpoint probes on 2026-07-14 found no BTCUSDT funding rows in the first
days of 2020 and complete rows from 2021-01-01 onward. The new cross-venue
protocol therefore begins in 2021 rather than filling unavailable history.

## Integrity checks

- funding timestamps are unique, increasing, and exactly eight hours apart;
- the first funding timestamp is 2021-01-01 00:00 UTC and the final one is
  2023-12-31 16:00 UTC;
- premium timestamps form an exact, duplicate-free hourly grid with 26,280
  rows;
- all premium OHLC fields are finite and every high is at or above its low;
- deterministic gzip (`mtime=0`, empty embedded filename) makes data hashes
  reproducible;
- the builder refuses any end boundary after 2024-01-01.

The historical premium endpoint sometimes carries the prior close as the next
row's `open` while high/low reflect only updates inside the hour. Consequently
the audit does not impose the usual `low <= open/close <= high` invariant. The
future strategy is frozen to the completed **close** value; it does not infer
an executable Bybit price from premium OHLC.

## Reproduction

```bash
PYTHONPATH=. python training/build_bybit_linear_aux_btc_2021_2023.py
PYTHONPATH=. .venv/bin/pytest -q tests/test_build_bybit_linear_aux_btc.py
```

The source is suitable for support-only cross-venue feature research. It is
not itself evidence that a funding-dispersion strategy is profitable.
