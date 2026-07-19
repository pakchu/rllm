# Binance stablecoin-quote BTC flow source audit — 2026-07-19

## Decision

Freeze a new **source-only** hourly panel for a BTC alpha family based on the
propagation of spot taker flow across three stablecoin quote books:

- `BTCUSDT` — reference stablecoin spot book;
- `BTCUSDC` — independent live alternative quote book;
- `BTCFDUSD` — independent live alternative quote book after launch.

The panel contains no retained price, return, label, perpetual outcome,
funding, REX, rank-7, portfolio, or forward field. Source completeness is not
profitability evidence. A strategy must be preregistered separately before any
BTCUSDT perpetual outcome is opened.

## Official source and live parity

- Binance public-data schema and publication timing:
  <https://github.com/binance/binance-public-data>
- Official monthly Spot archive root:
  <https://data.binance.vision/?prefix=data/spot/monthly/klines/>
- Frozen archive pattern:
  `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1h/{SYMBOL}-1h-{YYYY-MM}.zip`
- Published checksum pattern: archive URL plus `.CHECKSUM`
- Live public exchange metadata:
  <https://data-api.binance.vision/api/v3/exchangeInfo>

The upstream schema defines twelve Spot kline fields, including base volume,
trade count, and taker-buy base volume. It also documents the timestamp change
from milliseconds to microseconds on `2025-01-01`. The builder validates both
formats and normalizes persisted open/close timestamps to microseconds.

The live metadata check on 2026-07-19 returned `TRADING`, `baseAsset=BTC`, the
expected quote asset, and `isSpotTradingAllowed=true` for all three symbols.
This proves current public-source reproducibility, not access from every legal
jurisdiction and not alpha quality.

## Frozen source contract

| Item | Value |
|---|---|
| Interval | completed UTC `1h` Spot klines |
| Historical window | `[2023-07-01, 2026-07-01)` |
| Symbols | `BTCUSDT`, `BTCUSDC`, `BTCFDUSD` |
| USDT/USDC activation | `2023-07-01 00:00 UTC` |
| FDUSD activation | first official row, `2023-08-04 08:00 UTC` |
| Missing-data rule | missing rows allowed only before frozen activation; no fill or stale carry |
| Persisted observables | BTC base volume, trade count, taker-buy BTC, taker-sell BTC, signed taker flow |
| Discarded fields | OHLC prices, quote notional, taker-buy quote notional, ignore field |
| Integrity | every official ZIP checked against its published SHA-256 |
| Availability | row usable only after its exact `close_time_us` |

The FDUSD launch is part of the source contract, not a value imputation. A
downstream alternative-quote aggregate may use only the books whose completed
rows exist at a decision time. It must not backfill FDUSD before launch.

## Integrity result

Builder:
`training/build_binance_stablecoin_quote_flow.py`

Artifact:
`data/binance_stablecoin_quote_flow_btc_2023_2026/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-06-30T23.csv.gz`

Manifest:
`data/binance_stablecoin_quote_flow_btc_2023_2026/build_manifest.json`

| Check | Result |
|---|---:|
| Verified monthly archives | 107 |
| Expected active rows | 78,088 |
| Observed rows | 78,088 |
| Complete rows | 78,088 |
| First row | `2023-07-01 00:00 UTC` |
| Last row | `2026-06-30 23:00 UTC` |
| Panel SHA-256 | `064d1c88d5a72efe43bb05b360b1e6b62d75366d52e8bd9fafe963a9e2f9862b` |
| Manifest SHA-256 | `9e6a82b9747df5c0ba1c9278e436551de03ef6136c0ad3aeb05f0a451ed12134` |

Two complete network builds were byte-identical.

### Row coverage

| Symbol | 2023 H2 | 2024 | 2025 | 2026 H1 |
|---|---:|---:|---:|---:|
| BTCUSDT | 4,416 | 8,784 | 8,760 | 4,344 |
| BTCUSDC | 4,416 | 8,784 | 8,760 | 4,344 |
| BTCFDUSD | 3,592 | 8,784 | 8,760 | 4,344 |

FDUSD base volume dominates some historical intervals and contracts sharply in
2026 H1, while USDC participation grows. That non-stationarity is exactly why a
downstream rule must normalize from strictly prior history and use breadth or
shares rather than a fixed raw-volume threshold. It is not permission to pick
thresholds after viewing returns.

## Leakage boundary

For a source hour starting at `t`, all aggregate fields are unavailable until
the exact end of `[t,t+1h)`. Any downstream strategy must:

1. use only completed rows;
2. shift every expanding/rolling threshold by one completed source hour;
3. make a decision no earlier than `t+1h`;
4. enter the BTCUSDT perpetual no earlier than the following 5-minute open;
5. fail closed if an expected active book is absent or malformed;
6. keep source-support selection isolated from perpetual OHLC and funding.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pakchu/rllm/.venv/bin/python \
  -m training.build_binance_stablecoin_quote_flow --workers 12

/home/pakchu/rllm/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_build_binance_stablecoin_quote_flow.py
```

## Next work unit

Preregister exactly one stablecoin-quote flow mechanism, including side,
normalization, event threshold, hold, costs, support gates, controls, split
calendar, and stop-on-first-failure rule. Only then may a support builder create
an outcome-blind event clock.
