# Binance BTCUSDT aux feature coverage — 2026-06-23

## Why this was added

The previous full-data coverage audit showed a structural hole in BTC single-asset derivative features:

- `funding_rate`, `funding_zscore`, `oi_change`, `oi_zscore` were all constant zero/unusable.
- Local `data/binance_um_aux_2023_2026/` contained funding/premium files for other symbols, but not BTCUSDT.
- Running alpha/RLLM searches with these columns silently made the model believe derivative-state inputs existed when they did not.

## Change

Downloaded public Binance USD-M BTCUSDT aux data and added a leak-safe join path:

- Funding: `BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz`, 7,029 rows.
- Premium index klines: `BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz`, 56,232 rows.
- Join rule: backward-as-of only.
- Premium close rule: use `close_time` as the availability timestamp, not the kline open time, so hourly premium closes are not visible before they complete.

Generated data and audit outputs are ignored and not committed.

## Coverage result

Command:

```bash
.venv/bin/python -m training.feature_coverage_audit \
  --input-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/feature_coverage_audit_2020_2026_wave_binance_aux_2026-06-23.json \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --external-tolerance 30min \
  --binance-funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --binance-premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz
```

Summary:

- Rows: 674,785, `2019-12-31 15:00:00` → `2026-05-31 15:00:00`.
- Feature count: 117.
- Derivatives aux coverage improved from `0/4 usable` to `7/9 usable`.
- Newly usable:
  - `mkt__funding_rate`: nonzero 0.99984, std 0.00021338.
  - `mkt__funding_zscore`: nonzero 0.35850, std 1.04811.
  - `mkt__funding_available`: nonzero 0.99984.
  - `mkt__premium_index`: nonzero 0.99374, std 0.00063579.
  - `mkt__premium_index_zscore`: nonzero 0.99952, std 1.18480.
  - `mkt__premium_index_change`: nonzero 0.99881, std 0.00047775.
  - `mkt__premium_available`: nonzero 0.99982.
- Still unusable:
  - `mkt__oi_change`: constant zero.
  - `mkt__oi_zscore`: constant zero.

## Implication

The next alpha/RLLM iteration should use funding and premium state, but must still either acquire real BTC open-interest history or exclude OI-derived columns from candidate pools.  Treat any result depending on OI columns as invalid until that hole is fixed.
