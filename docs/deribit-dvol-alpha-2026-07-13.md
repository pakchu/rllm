# Deribit BTC DVOL standalone alpha (2026-07-13)

## Verdict

Deribit DVOL contains a real 24-48h directional relation in 2021-2023, but the
frozen standalone tail rules fail in 2025.  No alpha/live candidate was
promoted.

## Data and causality

- Official endpoint: [Deribit volatility-index history](https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data)
- BTC 1h candles: 45,505 complete hourly rows
- Available range: 2021-03-24 through 2026-06-02
- Each candle joins on `close_time = date + 1h`; the opening timestamp is never
  exposed as a completed value.

## Protocol

- Fit thresholds: 2021-04-15 through 2022-12-31
- Selection: 2023 full year plus H1/H2 robustness
- Physical pre-future manifest SHA-256:
  `d83e5eedbad1917d0311135f46db5ef3a35c69217953ad14a34c292516392882`
- Future: 2024 Test, 2025 Eval, 2026 YTD
- Costs: 0.5x, 5 bp fee + 1 bp slippage per side
- MDD: worst-order favorable-to-adverse OHLC high-water path drawdown

## Best evidence

| Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| DVOL 90d z upper 15%, long, 12h | 2023 Select | +37.46% | 37.49% | 6.51% | 5.76 | 171 |
| same | 2024 Test | +21.94% | 21.89% | 9.03% | 2.42 | 254 |
| same | 2025 Eval | -4.88% | -4.89% | 25.88% | -0.19 | 159 |
| same | 2026 YTD | -5.57% | -12.86% | 11.60% | -1.11 | 89 |
| DVOL 90d change upper 25%, long, 48h | 2023 Select | +34.07% | 34.10% | 9.68% | 3.52 | 63 |
| same | 2024 Test | +30.13% | 30.06% | 12.91% | 2.33 | 110 |
| same | 2025 Eval | -6.76% | -6.77% | 19.93% | -0.34 | 57 |
| same | 2026 YTD | +4.29% | 10.63% | 11.85% | 0.90 | 44 |

## Interpretation

High/rising long-horizon implied volatility acted as a rebound context through
2024, then became a poor unconditional long trigger in 2025.  This rules out a
standalone DVOL tail alpha.  DVOL remains suitable as an independent state input
to the continual positioning critic, where the model can condition its meaning
on trend, OI, and crowding rather than assume a fixed direction.

## Artifacts

- Downloader: `training/download_deribit_volatility_index.py`
- Search: `training/search_deribit_dvol_alpha.py`
- Tests: `tests/test_search_deribit_dvol_alpha.py`
- Manifest: `results/deribit_dvol_top10_manifest_2026-07-13.json`
- Result: `results/deribit_dvol_alpha_scan_2026-07-13.json`
