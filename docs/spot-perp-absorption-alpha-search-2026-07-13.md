# Spot–perpetual residual absorption alpha search (2026-07-13)

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

## Frozen Top-10 replay

| rank | policy | 2024 | 2025 | 2026 | combined | +union combined | alpha |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | `{'window': 8640, 'z_entry': 2.0, 'contraction_delta': 0.25, 'mode': 'basis_reversion', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 144}` | -0.92/-0.91/18.70/-0.05/488 | -4.13/-4.13/17.40/-0.24/454 | -20.40/-42.21/21.41/-1.97/211 | -24.41/-10.93/37.00/-0.30/1154 | 26.83/10.33/22.73/0.45/1097 | no |
| 2 | `{'window': 8640, 'z_entry': 2.0, 'contraction_delta': 0.0, 'mode': 'basis_reversion', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 144}` | -45.09/-45.02/47.19/-0.95/537 | -37.23/-37.25/41.99/-0.89/526 | -22.79/-46.29/25.25/-1.83/237 | -73.38/-42.16/74.79/-0.56/1302 | -61.86/-32.89/64.07/-0.51/1254 | no |
| 3 | `{'window': 2016, 'z_entry': 3.0, 'contraction_delta': 0.0, 'mode': 'flow_fade', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 96}` | -17.41/-17.37/22.93/-0.76/216 | -19.80/-19.81/23.54/-0.84/207 | -8.14/-18.46/10.04/-1.84/100 | -39.25/-18.63/42.15/-0.44/525 | 46.74/17.19/24.22/0.71/603 | no |
| 4 | `{'window': 2016, 'z_entry': 3.0, 'contraction_delta': 0.25, 'mode': 'flow_fade', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 48}` | -12.57/-12.54/19.42/-0.65/190 | -10.77/-10.78/15.25/-0.71/184 | -5.74/-13.25/8.50/-1.56/77 | -26.47/-11.94/30.26/-0.39/451 | 101.10/33.51/15.43/2.17/562 | no |
| 5 | `{'window': 8640, 'z_entry': 2.0, 'contraction_delta': 0.25, 'mode': 'lead_residual', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 96}` | -14.01/-13.99/25.14/-0.56/567 | -12.72/-12.73/29.57/-0.43/543 | -15.49/-33.25/17.45/-1.91/267 | -36.55/-17.15/37.76/-0.45/1378 | -2.03/-0.85/27.17/-0.03/1261 | no |
| 6 | `{'window': 8640, 'z_entry': 2.0, 'contraction_delta': 0.0, 'mode': 'lead_residual', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 96}` | -36.00/-35.95/41.43/-0.87/672 | -19.61/-19.62/33.73/-0.58/661 | -16.22/-34.65/17.95/-1.93/316 | -56.92/-29.41/57.49/-0.51/1650 | -28.24/-12.83/36.14/-0.35/1529 | no |
| 7 | `{'window': 2016, 'z_entry': 3.0, 'contraction_delta': 0.0, 'mode': 'flow_fade', 'direction': 'contra', 'phase': 'expansion_onset', 'max_hold': 48}` | -23.36/-23.31/27.30/-0.85/264 | -18.08/-18.09/20.85/-0.87/251 | -9.89/-22.13/10.55/-2.10/115 | -43.50/-21.03/44.00/-0.48/631 | 64.35/22.81/16.45/1.39/684 | no |

## Required comparator

Deterministic six-sleeve union combined: `180.12/53.12/9.45/5.62/291`.

## Interpretation

- Alpha-pool qualifiers: 0; live-grade: 0 by protocol.
- Direct perp-spot basis is explicitly residualized against the completed premium index before event construction; current values never enter rolling fit statistics.
- A standalone pass is insufficient: the seventh stream must improve the existing union on both absolute return and CAGR/MDD.
- All passes remain shadow-only because 2024-2026 are not fresh calendar windows for the broader programme.

## Reproduction

```bash
python -m training.search_spot_perp_absorption_alpha --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz --spot-csv data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz --manifest-output results/spot_perp_absorption_top10_manifest_2026-07-13.json --output results/spot_perp_absorption_alpha_scan_2026-07-13.json --docs-output docs/spot-perp-absorption-alpha-search-2026-07-13.md
```
