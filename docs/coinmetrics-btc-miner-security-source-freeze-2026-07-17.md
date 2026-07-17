# Coin Metrics BTC miner-security source freeze

This work unit freezes a new, price-independent source axis for later alpha
research. It does **not** inspect any post-entry BTC return.

## Frozen source

- interval: 2019-01-01 through 2023-12-31 observation dates
- rows: 1,826 daily observations
- columns: `HashRate`, `IssTotNtv`, `FeeTotNtv`, `BlkCnt`
- availability: `AssetEODCompletionTime`, with a mandatory one-day minimum lag
- gzip SHA-256: `448a101834df33f69abaeafe9aadfccd8ce9c3d6ad7816c1c2448189a12b8379`

`HashRate` is Coin Metrics' mean estimated network hash-solving rate. Issuance,
fees, and block count are native-chain quantities. Price, market-cap, and
exchange-address-tag fields are deliberately excluded.

## Causality boundary

The early history contains a large 2021 backfill. Such rows may only enter a
reference set after their recorded `available_at`; they may never emit a signal
as if they had been timely in 2019 or 2020. The 2023-12-31 observation was not
available until 2024-01-01 and therefore cannot create a pre-2024 entry.

The file hash freezes the downloaded value vintage, but this is not a complete
archive of every historical Coin Metrics revision. Any future live promotion
must pass a forward-vintage parity check.

## Sealed boundary

No source observation after 2023-12-31 and no 2024-or-later BTC outcome was
loaded by this work unit.

Official catalog:
https://community-api.coinmetrics.io/v4/catalog/metrics?metrics=HashRate%2CIssTotNtv%2CFeeTotNtv%2CBlkCnt%2CAssetEODCompletionTime
