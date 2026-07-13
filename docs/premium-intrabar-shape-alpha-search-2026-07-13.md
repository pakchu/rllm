# Premium-index intrabar shape alpha search (2026-07-13)

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

| rank | policy | 2024 | 2025 | 2026 | combined | +union combined | alpha |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | `{'window': 8640, 'range_z': 3.0, 'shape_threshold': 0.75, 'mode': 'close_location', 'direction': 'follow', 'hold': 96}` | -11.80/-11.77/19.91/-0.59/142 | -7.61/-7.62/14.79/-0.52/87 | 2.34/5.71/6.92/0.82/63 | -16.61/-7.24/23.71/-0.31/292 | 140.57/43.78/13.20/3.32/451 | no |
| 2 | `{'window': 2016, 'range_z': 3.0, 'shape_threshold': 0.75, 'mode': 'close_location', 'direction': 'follow', 'hold': 96}` | -16.45/-16.42/21.59/-0.76/163 | -10.26/-10.26/16.47/-0.62/108 | 6.02/15.08/6.78/2.22/64 | -20.50/-9.06/30.39/-0.30/335 | 92.17/31.02/11.99/2.59/488 | no |

## Interpretation

- Alpha-pool qualifiers: 0; baseline union `180.12/53.12/9.45/5.62/291`.
- Pre-2024 selection admitted 2 unique paths from 192 fixed policies.
- Only complete five-row premium-index intervals are used; range normalization ends at t-1 and execution begins at t+1 open.
- Standalone strength is insufficient without positive marginal contribution to the fixed six-sleeve union.
- 2024-2026 are replay evidence, not fresh future data for live promotion.
- Rank 1 failed with `-16.61/-7.24/23.71/-0.31/292`; direction flip also failed at `-17.82/-7.80/22.16/-0.35/292`, and 10bp/side stress fell to `-25.80/-11.61/30.72/-0.38/292`.
- Adding rank 1 reduced the union from `180.12/53.12/9.45/5.62/291` to `140.57/43.78/13.20/3.32/451`. Signal disjointness therefore did not translate into marginal alpha.
- Preserve continuous wick/close-location/range fields only as beta research context; reject this exact extreme-range-onset plus fixed-direction/fixed-hold mapping as gamma.

## Reproduction

```bash
python -m training.search_premium_intrabar_shape_alpha --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz --spot-csv data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz --manifest-output results/premium_intrabar_shape_top10_manifest_2026-07-13.json --output results/premium_intrabar_shape_alpha_scan_2026-07-13.json --docs-output docs/premium-intrabar-shape-alpha-search-2026-07-13.md
```
