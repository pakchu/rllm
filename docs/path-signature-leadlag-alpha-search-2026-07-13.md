# Price/order-flow path-signature alpha preflight (2026-07-13)

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

| policy | fit | 2023 | 2023H1 | 2023H2 | flipped 2023 | flow-only 2023 |
|---|---:|---:|---:|---:|---:|---:|
| `flow_led_continuation w24` | -78.96/-45.29/79.95/-0.57/1909 | -43.18/-43.20/44.17/-0.98/897 | -17.95/-32.92/19.59/-1.68/409 | -30.75/-51.78/31.08/-1.67/488 | -40.67/-40.69/41.73/-0.98/897 | -69.39/-69.42/69.87/-0.99/2180 |
| `price_led_crowding_fade w24` | -60.93/-30.48/61.35/-0.50/1837 | -37.99/-38.01/38.29/-0.99/626 | -21.22/-38.20/21.95/-1.74/331 | -21.29/-37.83/21.50/-1.76/295 | -24.55/-24.56/26.91/-0.91/626 | -76.76/-76.79/76.81/-1.00/2180 |
| `flow_led_continuation w72` | -69.82/-37.09/73.19/-0.51/1164 | -20.52/-20.53/28.29/-0.73/488 | 0.02/0.05/10.24/0.00/207 | -20.54/-36.64/22.53/-1.63/281 | -31.04/-31.06/34.08/-0.91/488 | -37.21/-37.23/43.29/-0.86/1073 |
| `price_led_crowding_fade w72` | -58.80/-29.04/60.16/-0.48/1197 | -40.06/-40.08/42.02/-0.95/407 | -29.78/-51.00/32.08/-1.59/214 | -14.64/-26.96/16.64/-1.62/193 | 0.84/0.84/10.54/0.08/407 | -57.76/-57.78/60.90/-0.95/1073 |
| `flow_led_continuation w144` | -19.90/-8.23/52.47/-0.16/766 | -4.40/-4.40/24.88/-0.18/316 | -12.76/-24.09/24.88/-0.97/148 | 9.59/19.94/7.99/2.50/168 | -30.17/-30.19/35.11/-0.86/316 | -31.04/-31.06/33.98/-0.91/617 |
| `price_led_crowding_fade w144` | -1.38/-0.54/22.51/-0.02/779 | -32.42/-32.43/36.63/-0.89/289 | -16.33/-30.21/22.88/-1.32/155 | -19.23/-34.55/21.01/-1.64/134 | 2.62/2.63/11.14/0.24/289 | -33.82/-33.84/48.38/-0.70/617 |

## Verdict

- Eligible policies: 0 of 6; OOS opened: **no**.
- Every original policy lost in fit or 2023/H1/H2. Direction flips also lacked two-half stability, so this is not a simple sign error.
- Signed-area filtering usually reduced the loss versus flow-only, but did not create a positive executable edge after 6bp/side.
- Reject these exact fixed quantile/direction/hold mappings without spending the 2024-2026 replay budget.
- Continuous signed-area and flow-direction fields remain research context only; they are not an alpha.

## Reproduction

```bash
python -m training.search_path_signature_leadlag_alpha --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz --manifest-output results/path_signature_leadlag_top6_manifest_2026-07-13.json --output results/path_signature_leadlag_alpha_scan_2026-07-13.json --docs-output docs/path-signature-leadlag-alpha-search-2026-07-13.md
```
