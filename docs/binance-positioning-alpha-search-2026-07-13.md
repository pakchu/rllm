# Binance positioning-disagreement alpha search (2026-07-13)

## Verdict

The new data family is real and live-obtainable, but the first standalone rule
family did **not** pass the promotion target.  The 2023-selected Top-10 contained
zero candidates with `CAGR / strict MDD >= 3` in both 2024 and 2025.  Nothing was
added to the alpha pool.

## New independent data

- Official source: [Binance public-data repository](https://github.com/binance/binance-public-data)
- Archive pattern: `data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip`
- Downloaded range: `2020-09-01` through `2026-06-01`
- Rows: 604,166 five-minute observations
- Nominal coverage: 99.895%
- Fields:
  - open interest and notional open interest
  - top-trader account long/short ratio
  - top-trader position long/short ratio
  - global account long/short ratio
  - taker buy/sell volume ratio

The field meanings match Binance's official USD-M [market-data API catalog](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).
Every archive row is delayed by one complete five-minute source bar before
feature use.  This is conservative for the taker ratio because Binance's archive
row groups the preceding taker-flow bucket with the current positioning
snapshot.

### Data-quality quarantine

The archive's top-trader fields are only about 12.7% populated during 2022.
Therefore:

- threshold fit: `2020-10-15` through `2021-12-31`
- 2022: excluded from fit and selection
- selection: 2023 full year plus H1/H2 robustness
- sealed future reporting: 2024 Test, 2025 Eval, 2026 YTD

## Feature thesis

The experiment tested only economically interpretable positioning differences:

- top-trader position ratio minus top-trader account ratio (`smart_size`)
- top-trader position ratio minus global account ratio (`smart_retail`)
- top-trader account ratio minus global account ratio
- global/top-trader crowding extremes
- positioning versus taker-flow absorption

All transforms are past-only rolling z-scores or changes.  The 2023 Top-10
manifest was frozen as SHA-256
`5b33c65106a4a2326a330b909130eba5ccf88ca8d344ffb68c0dd3b8cc45e73b`
before future statistics were computed.

## Selected evidence

Default execution uses 0.5x, 5 bp fee + 1 bp slippage per side, next-bar-open
entry, fixed hold, full-window CAGR, and strict intraposition adverse-excursion
MDD.

| Candidate | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| global account z30d, tail 10%, 18h hold | 2023 Select | +39.80% | 39.83% | 3.50% | 11.37 | 138 |
| same | 2024 Test | +13.13% | 13.10% | 8.73% | 1.50 | 126 |
| same | 2025 Eval | -14.50% | -14.51% | 16.38% | -0.89 | 127 |
| same | 2026 YTD | +4.01% | 9.90% | 3.47% | 2.86 | 59 |
| top-account z30d, tail 10%, 12h hold | 2023 Select | +30.44% | 30.47% | 5.55% | 5.49 | 209 |
| same | 2024 Test | -2.96% | -2.96% | 14.73% | -0.20 | 193 |
| same | 2025 Eval | -19.42% | -19.44% | 20.69% | -0.94 | 210 |
| same | 2026 YTD | -3.62% | -8.49% | 9.41% | -0.90 | 89 |

## Failure diagnosis

The positioning relation was exceptionally strong in 2020-2023 but changed
afterward.  A pre-registered follow-up interaction that required positioning
and 30-day price trend to agree reduced the 2025 loss, but still produced only
weak ratios (best selected family: 2024 `0.13`, 2025 `0.53`, 2026 `0.74`).
This is a relation-shift problem, not a threshold-resolution problem.  Further
gate sweeps on the same rule family are not justified.

The next experiment should use these positioning variables as inputs to a
risk-aware nonlinear model that can learn trend/crowding interactions, while
keeping the same delayed-source and frozen-manifest protocol.

## Artifacts

- Downloader: `training/download_binance_um_metrics.py`
- Search: `training/search_positioning_disagreement_alpha.py`
- Tests:
  - `tests/test_download_binance_um_metrics.py`
  - `tests/test_search_positioning_disagreement_alpha.py`
- Result: `results/positioning_disagreement_alpha_scan_2026-07-13.json`

## Other official data conclusions

- Binance market-wide liquidations are exposed as the official
  [`!forceOrder@arr` WebSocket stream](https://developers.binance.com/legacy-docs/derivatives/usds-margined-futures/websocket-market-streams/All-Market-Liquidation-Order-Streams),
  not as a public historical REST feed.  Account `forceOrders` is signed and
  user-specific.
- Deribit provides public [volatility-index history](https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data)
  and limited [option mark-price history](https://docs.deribit.com/api-reference/market-data/public-get_mark_price_history),
  but no directly published historical skew series was found in the official
  API.
