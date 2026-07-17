# Binance BTC cross-collateral positioning metrics source audit — 2026-07-17

## Decision

The official Binance Vision five-minute `metrics` archives are suitable for a
**source-only** USD-M/COIN-M positioning experiment from 2021-07-08 through
2023-12-31. No executable price, future return, label, strategy PnL, CAGR, or
drawdown was loaded while building or auditing this panel.

- USD-M instrument: `BTCUSDT`
- COIN-M instrument: `BTCUSD_PERP`
- range: `[2021-07-08, 2024-01-01)` UTC
- full five-minute grid: **261,216** rows
- joint source-complete rows: **220,370**
- panel SHA-256:
  `ab9f18ba7745f21b17ac1124c45bb755245d404d66100c595bb77631f4bc1757`
- manifest:
  `results/binance_cross_collateral_metrics_btc_2021_2023_manifest.json`

## Official source contract

Binance publishes public daily archives under Binance Vision and publishes a
sidecar checksum for each ZIP. The builder downloaded and checksum-verified
every available USD-M and COIN-M day rather than relying on an unofficial data
vendor.

- public-data project: <https://github.com/binance/binance-public-data>
- USD-M archive prefix:
  <https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/>
- COIN-M archive prefix:
  <https://data.binance.vision/?prefix=data/futures/cm/daily/metrics/BTCUSD_PERP/>
- USD-M open-interest statistics:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics>
- USD-M taker buy/sell volume:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume>
- COIN-M open-interest statistics:
  <https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/rest-api/Open-Interest-Statistics>
- COIN-M taker buy/sell volume:
  <https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/rest-api/Taker-Buy-Sell-Volume>

Each archive has the fixed schema:

1. `create_time`
2. `symbol`
3. `sum_open_interest`
4. `sum_open_interest_value`
5. `count_toptrader_long_short_ratio`
6. `sum_toptrader_long_short_ratio`
7. `count_long_short_ratio`
8. `sum_taker_long_short_vol_ratio`

Only open interest and taker ratio are common enough for the proposed
cross-collateral experiment. All three COIN-M long/short-ratio fields are empty
throughout this pre-2024 source and are explicitly excluded from the candidate.

## Unit-safe comparison

The two products do not share raw units. The successor alpha must not subtract
their levels.

- USD-M `sum_open_interest_value` is used only through a log change.
- COIN-M `sum_open_interest` is a contract count and is used only through a log
  change; for BTCUSD perpetual contracts this avoids treating the BTC-valued
  `sum_open_interest_value` as directly comparable with USDT notional.
- cross-venue features compare dimensionless changes and log taker ratios, not
  USD-M and COIN-M levels.

## Integrity and fail-closed availability

The builder examined **1,814** venue-days, verified **1,803** published
archive/checksum pairs, and declared the exact **11** missing days. It accepts
only the exact schema and symbol, five-minute-aligned UTC
timestamps, finite nonnegative ratios, and nonnegative open-interest fields.
Exact duplicate rows may be collapsed; conflicting duplicates fail the build.

Observed source defects are preserved as unavailable rows, never forward-filled:

| defect | USD-M | COIN-M |
|---|---:|---:|
| missing archive days | 0 | 11 |
| missing five-minute rows inside published days | 102 | 22 |
| zero/missing open-interest rows | 145 | 0 |
| missing/zero taker-ratio rows | 37,252 | 344 |

Joint-complete coverage by calendar year:

| year | complete rows | total rows | coverage |
|---:|---:|---:|---:|
| 2021 partial | 49,893 | 50,976 | 97.88% |
| 2022 | 67,120 | 105,120 | 63.85% |
| 2023 | 103,357 | 105,120 | 98.32% |

The large 2022 reduction comes primarily from missing USD-M taker-ratio values,
not from interpolation or a strategy filter. Any later feature lookback,
signal, confirmation, entry delay, held path, and exit must fail closed across
these gaps and a frozen post-gap quarantine.

## Causality and live-parity boundary

`create_time` is treated as the completed observation timestamp, not as proof
that the row was tradable at that exact millisecond. A later strategy must use
at least one full five-minute availability delay and enter no earlier than the
following open. Live operation must fetch the corresponding USD-M and COIN-M
metrics before that deadline and enforce freshness; stale data cannot be
silently carried forward.

Historical archives are backfilled files, so this audit proves timestamped
source integrity and deterministic replay, not exchange-publication latency at
the historical instant. That limitation is explicit and must be checked in
forward shadow before live promotion.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m training.build_binance_cross_collateral_metrics --workers 16
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_build_binance_cross_collateral_metrics.py
```
