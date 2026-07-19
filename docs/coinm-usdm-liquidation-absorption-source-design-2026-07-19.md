# COIN-M / USD-M Liquidation Absorption Activity Source Design (2026-07-19)

## Purpose

`training/build_coinm_usdm_liquidation_absorption_source.py` builds an outcome-blind Binance USD-M BTCUSDT 5-minute activity source for the COIN-M / USD-M liquidation absorption research setup. It is a source artifact only: it does not compute returns, PnL, labels, trading signals, or strategy outcomes.

## Official provenance

- Venue/product: Binance USD-M Futures.
- Symbol/interval: `BTCUSDT` `5m` klines.
- Archive root: `https://data.binance.vision/data/futures/um/daily/klines`.
- Daily archive URL shape: `.../BTCUSDT/5m/BTCUSDT-5m-YYYY-MM-DD.zip`.
- Checksum sidecar URL shape: the same archive URL plus `.CHECKSUM`.
- Default date window: `2023-06-25` inclusive through `2024-10-15` exclusive.

Each daily zip is fetched from `data.binance.vision`, verified against the official published SHA-256 checksum sidecar, parsed in memory, and then discarded. Raw zip files are not persisted.

## Retained columns

The output keeps only activity-time fields:

1. `date` — 5-minute bar open time, UTC timestamp represented as tz-naive like the existing project Binance kline references.
2. `feature_available_time` — bar end plus one second (`date + 5 minutes + 1 second`).
3. `quote_asset_volume` — Binance kline quote volume.
4. `taker_buy_quote` — Binance kline taker buy quote volume.
5. `taker_sell_quote` — `quote_asset_volume - taker_buy_quote`.
6. `taker_imbalance` — `(taker_buy_quote - taker_sell_quote) / quote_asset_volume`, set to `0` when quote volume is zero.
7. `number_of_trades` — Binance kline trade count.

OHLC prices and base volume are parsed only as part of the official kline schema and are not retained in the emitted source.

## Validation

The builder fails closed if any of the following checks fail:

- Official checksum sidecar is missing, malformed, or does not match the zip payload.
- A daily zip does not contain exactly one CSV with the expected kline schema.
- Daily open times are not strictly increasing and unique.
- Daily close times are not exactly 5-minute bar close times.
- A day or the combined output is missing any 5-minute grid slot.
- Activity values are negative or non-finite.
- `taker_buy_quote` or `taker_sell_quote` exceeds `quote_asset_volume`.
- `taker_buy_quote + taker_sell_quote` does not reconstruct `quote_asset_volume` within numerical tolerance.
- `taker_imbalance` is outside `[-1, 1]`.
- `feature_available_time` is not exactly bar end plus one second.

## Default artifacts

- Data output directory: `data/binance_um_activity_5m_2023_2024`.
- Manifest: `results/binance_um_activity_5m_2023_2024_manifest.json`.

The manifest records the official archive root, checksum verification status, raw archive non-retention, retained columns, validation summary, output file SHA-256, per-day archive SHA-256 values, and builder SHA-256.

## Limits

This source is intentionally narrow. It is not a liquidation feed, does not inspect COIN-M liquidation outcomes, and does not evaluate any strategy. It supplies only checksum-verified USD-M BTCUSDT 5-minute activity context that can be joined causally by `feature_available_time` in later preregistered research.
