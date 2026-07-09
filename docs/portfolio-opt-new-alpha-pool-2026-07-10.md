# Portfolio opt with new alpha pool (2026-07-10)

Protocol: Weights ranked on test2024 only. eval2025/ytd2026 are report-only validation to avoid selecting on eval.

Evaluated weight sets: 2,699; gross cap=3.0.

Metric cell format: `abs_return/CAGR/strict_MDD/CAGR_MDD/trades`.

## Top selected by 2024 test only

| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |
|---:|---:|---|---:|---:|---:|
| 1 | 3.00 | `{'long_funding_compression_premium': 1.375665, 'long_range_funding_premium': 0.548118, 'short_premium_kimchi_union': 1.076217}` | 161.90/161.38/8.10/19.93/124 | 65.53/65.58/8.15/8.05/157 | 44.35/141.60/6.77/20.92/93 |
| 2 | 3.00 | `{'long_funding_compression_premium': 1.26823, 'long_range_funding_premium': 0.50906, 'short_premium_kimchi_union': 1.22271}` | 153.15/152.67/7.74/19.72/124 | 65.49/65.55/7.85/8.35/157 | 43.79/139.36/5.99/23.27/93 |
| 3 | 3.00 | `{'long_funding_compression_premium': 1.467534, 'long_range_funding_premium': 0.884218, 'short_premium_kimchi_union': 0.648248}` | 185.60/184.99/9.83/18.82/124 | 66.80/66.86/9.45/7.07/157 | 44.09/140.55/9.37/15.00/93 |
| 4 | 2.50 | `{'long_funding_compression_premium': 1.071669, 'long_range_funding_premium': 0.619745, 'short_premium_kimchi_union': 0.808587}` | 127.09/126.71/6.76/18.75/124 | 53.50/53.55/7.09/7.55/157 | 35.51/107.54/6.20/17.35/93 |
| 5 | 3.00 | `{'long_funding_compression_premium': 1.070973, 'long_range_funding_premium': 0.498466, 'short_premium_kimchi_union': 1.430561}` | 140.56/140.12/7.48/18.74/124 | 65.62/65.67/8.20/8.00/157 | 42.59/134.57/5.36/25.10/93 |
| 6 | 3.00 | `{'long_funding_compression_premium': 1.514788, 'long_range_funding_premium': 0.262456, 'short_fx_stress': 1.222756}` | 150.05/149.58/8.12/18.43/103 | 62.26/62.31/8.13/7.67/103 | 35.01/105.71/5.49/19.24/79 |
| 7 | 3.00 | `{'long_funding_compression_premium': 0.788434, 'long_range_funding_premium': 0.76312, 'long_minimal_funding_premium': 0.1564, 'short_premium_kimchi_union': 1.292047}` | 140.20/139.77/7.65/18.26/153 | 67.48/67.54/7.90/8.55/183 | 40.77/127.44/5.78/22.05/122 |
| 8 | 3.00 | `{'long_funding_compression_premium': 1.313088, 'long_range_funding_premium': 1.260939, 'short_premium_kimchi_union': 0.425974}` | 195.40/194.74/10.83/17.98/124 | 68.67/68.73/10.20/6.74/157 | 42.35/133.62/11.07/12.07/93 |
| 9 | 3.00 | `{'long_funding_compression_premium': 1.176473, 'long_range_funding_premium': 1.26346, 'short_premium_kimchi_union': 0.560067}` | 186.38/185.76/10.36/17.92/124 | 69.17/69.23/9.72/7.12/157 | 41.80/131.47/10.03/13.11/93 |
| 10 | 3.00 | `{'long_funding_compression_premium': 0.591827, 'long_range_funding_premium': 1.270079, 'short_premium_kimchi_union': 1.130412, 'short_fx_stress': 0.007682}` | 149.39/148.93/8.34/17.85/156 | 70.51/70.57/7.95/8.88/182 | 38.90/120.23/6.39/18.83/111 |
| 11 | 3.00 | `{'long_funding_compression_premium': 1.516842, 'short_premium_kimchi_union': 1.483158}` | 142.51/142.07/7.97/17.83/88 | 62.23/62.28/8.85/7.04/130 | 45.78/147.37/5.20/28.32/64 |
| 12 | 2.50 | `{'long_funding_compression_premium': 1.175095, 'long_range_funding_premium': 0.214097, 'short_premium_kimchi_union': 0.88011, 'short_fx_stress': 0.230698}` | 113.75/113.42/6.37/17.81/156 | 51.15/51.19/6.43/7.97/182 | 34.62/104.29/4.63/22.52/111 |
| 13 | 3.00 | `{'long_funding_compression_premium': 0.773521, 'long_range_funding_premium': 0.971999, 'long_minimal_funding_premium': 0.430814, 'short_premium_kimchi_union': 0.823666}` | 155.54/155.05/8.72/17.78/153 | 68.46/68.52/8.92/7.68/183 | 40.40/126.01/8.31/15.17/122 |
| 14 | 3.00 | `{'long_funding_compression_premium': 1.5, 'short_premium_kimchi_union': 1.5}` | 141.49/141.05/7.94/17.77/88 | 62.24/62.30/8.85/7.04/130 | 45.67/146.91/5.15/28.51/64 |
| 15 | 2.50 | `{'long_funding_compression_premium': 0.725448, 'long_range_funding_premium': 0.729986, 'short_premium_kimchi_union': 0.943404, 'short_premium_panic': 0.101162}` | 111.05/110.73/6.28/17.64/154 | 54.78/54.83/6.27/8.74/225 | 32.76/97.57/4.93/19.79/118 |
| 16 | 2.50 | `{'long_funding_compression_premium': 0.79138, 'long_range_funding_premium': 0.563085, 'short_premium_kimchi_union': 0.951412, 'short_premium_panic': 0.194122}` | 105.46/105.16/5.98/17.58/154 | 54.12/54.17/6.38/8.50/225 | 32.63/97.09/4.62/21.00/118 |
| 17 | 2.50 | `{'long_funding_compression_premium': 0.809634, 'long_range_funding_premium': 0.884746, 'short_premium_kimchi_union': 0.80562}` | 124.50/124.13/7.06/17.57/124 | 55.15/55.20/7.10/7.78/157 | 33.76/101.17/6.22/16.28/93 |
| 18 | 3.00 | `{'long_funding_compression_premium': 1.184637, 'long_range_funding_premium': 1.283997, 'short_premium_kimchi_union': 0.443416, 'short_fx_stress': 0.08795}` | 187.53/186.91/10.67/17.52/156 | 69.10/69.16/9.70/7.13/182 | 41.01/128.39/10.29/12.48/111 |
| 19 | 3.00 | `{'long_funding_compression_premium': 1.386218, 'long_range_funding_premium': 0.815844, 'short_fx_stress': 0.797939}` | 172.37/171.81/9.85/17.45/103 | 65.62/65.67/8.24/7.97/103 | 36.95/112.91/8.14/13.87/79 |
| 20 | 3.00 | `{'long_funding_compression_premium': 1.742198, 'short_premium_kimchi_union': 1.257802}` | 156.43/155.94/9.03/17.27/88 | 61.98/62.03/8.88/6.99/130 | 47.21/153.24/5.88/26.05/64 |

## All-window diagnostic only

| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |
|---:|---:|---|---:|---:|---:|
| 1 | 3.00 | `{'long_range_funding_premium': 0.407897, 'long_minimal_funding_premium': 0.891006, 'short_fx_stress': 1.321754, 'short_premium_panic': 0.379343}` | 85.06/84.82/6.77/12.53/127 | 66.11/66.17/5.38/12.30/146 | 23.87/67.27/4.97/13.55/101 |
| 2 | 2.50 | `{'long_range_funding_premium': 1.194932, 'short_premium_kimchi_union': 0.524201, 'short_fx_stress': 0.414736, 'short_premium_panic': 0.366131}` | 86.89/86.66/7.00/12.37/151 | 58.20/58.25/4.87/11.96/199 | 23.35/65.59/4.39/14.95/104 |
| 3 | 3.00 | `{'long_range_funding_premium': 1.105787, 'long_minimal_funding_premium': 0.48999, 'short_fx_stress': 1.128572, 'short_premium_panic': 0.27565}` | 108.06/107.75/9.15/11.77/127 | 69.88/69.95/5.49/12.75/146 | 25.21/71.65/5.14/13.95/101 |
| 4 | 3.00 | `{'long_range_funding_premium': 1.789971, 'short_premium_kimchi_union': 0.088477, 'short_fx_stress': 0.864752, 'short_premium_panic': 0.2568}` | 128.90/128.51/11.02/11.66/151 | 73.93/74.00/6.18/11.97/199 | 26.41/75.63/6.18/12.24/104 |
| 5 | 3.00 | `{'long_range_funding_premium': 1.05412, 'long_minimal_funding_premium': 0.580295, 'short_premium_kimchi_union': 0.404426, 'short_fx_stress': 0.96116}` | 114.26/113.92/8.70/13.10/150 | 68.89/68.95/5.98/11.53/157 | 28.44/82.46/5.60/14.71/108 |
| 6 | 2.00 | `{'long_range_funding_premium': 0.838451, 'short_premium_kimchi_union': 0.33905, 'short_fx_stress': 0.392748, 'short_premium_panic': 0.429752}` | 58.91/58.76/5.10/11.51/151 | 44.23/44.27/3.70/11.95/199 | 17.15/46.28/3.28/14.11/104 |
| 7 | 2.50 | `{'long_range_funding_premium': 1.296861, 'short_premium_kimchi_union': 0.263159, 'short_fx_stress': 0.881611, 'short_premium_panic': 0.058369}` | 95.02/94.76/8.28/11.44/151 | 57.24/57.29/4.92/11.64/199 | 21.74/60.44/4.43/13.65/104 |
| 8 | 3.00 | `{'long_range_funding_premium': 0.718808, 'long_minimal_funding_premium': 0.763576, 'short_fx_stress': 1.3834, 'short_premium_panic': 0.134215}` | 99.36/99.07/8.17/12.13/127 | 66.89/66.94/5.86/11.42/146 | 24.65/69.79/4.83/14.44/101 |
| 9 | 3.00 | `{'long_funding_compression_premium': 0.799323, 'short_premium_kimchi_union': 0.396991, 'short_fx_stress': 1.276683, 'short_premium_panic': 0.527003}` | 87.62/87.38/6.79/12.86/150 | 63.28/63.33/5.62/11.26/223 | 26.84/77.07/6.71/11.49/107 |
| 10 | 3.00 | `{'long_range_funding_premium': 1.31838, 'short_premium_kimchi_union': 0.337075, 'short_fx_stress': 1.344545}` | 112.09/111.76/8.90/12.55/121 | 69.79/69.85/6.28/11.12/131 | 24.75/70.15/4.92/14.24/79 |
| 11 | 2.50 | `{'long_range_funding_premium': 1.252988, 'short_premium_kimchi_union': 0.331426, 'short_fx_stress': 0.915585}` | 94.33/94.06/8.04/11.70/121 | 56.78/56.83/5.11/11.12/131 | 21.88/60.87/4.36/13.95/79 |
| 12 | 2.50 | `{'long_range_funding_premium': 0.762942, 'long_minimal_funding_premium': 0.246988, 'short_fx_stress': 0.762167, 'short_premium_panic': 0.727903}` | 67.91/67.73/6.10/11.10/127 | 56.65/56.70/4.12/13.75/146 | 18.94/51.71/3.97/13.03/101 |
| 13 | 3.00 | `{'long_range_funding_premium': 1.485481, 'short_fx_stress': 1.514519}` | 118.45/118.10/10.21/11.57/68 | 70.40/70.46/6.37/11.06/52 | 23.31/65.44/5.13/12.77/47 |
| 14 | 3.00 | `{'long_funding_compression_premium': 0.866417, 'short_fx_stress': 1.289369, 'short_premium_panic': 0.844214}` | 84.87/84.64/6.71/12.62/97 | 64.32/64.38/5.85/11.00/144 | 25.15/71.43/6.48/11.02/75 |
| 15 | 2.50 | `{'long_range_funding_premium': 0.88786, 'long_minimal_funding_premium': 0.234805, 'short_fx_stress': 0.863914, 'short_premium_panic': 0.51342}` | 75.50/75.30/6.88/10.94/127 | 56.57/56.62/4.13/13.70/146 | 19.48/53.36/3.89/13.71/101 |
| 16 | 3.00 | `{'long_range_funding_premium': 1.100934, 'long_minimal_funding_premium': 0.110298, 'short_premium_kimchi_union': 0.193913, 'short_premium_panic': 1.594855}` | 78.87/78.65/6.69/11.76/148 | 75.08/75.15/6.88/10.93/200 | 24.87/70.51/5.00/14.10/115 |
| 17 | 3.00 | `{'long_funding_compression_premium': 0.912489, 'long_minimal_funding_premium': 0.015039, 'short_fx_stress': 1.142225, 'short_premium_panic': 0.930247}` | 86.28/86.04/6.77/12.72/126 | 64.78/64.84/5.94/10.91/170 | 26.06/74.45/6.07/12.26/104 |
| 18 | 2.50 | `{'long_range_funding_premium': 0.858698, 'short_fx_stress': 0.679535, 'short_premium_panic': 0.961767}` | 63.38/63.22/5.78/10.93/98 | 57.81/57.86/4.19/13.80/120 | 17.73/48.03/4.41/10.89/72 |
| 19 | 3.00 | `{'long_range_funding_premium': 1.235192, 'short_premium_kimchi_union': 0.598455, 'short_fx_stress': 1.166353}` | 109.29/108.97/8.08/13.48/121 | 69.54/69.60/6.41/10.86/131 | 26.17/74.83/4.99/14.99/79 |
| 20 | 2.50 | `{'long_funding_compression_premium': 0.523171, 'long_range_funding_premium': 0.944088, 'short_premium_kimchi_union': 0.312121, 'short_fx_stress': 0.72062}` | 108.92/108.61/7.69/14.12/156 | 55.02/55.07/5.10/10.81/182 | 26.45/75.77/4.66/16.25/111 |

## Event counts

```json
{
  "test2024": {
    "long_funding_compression_premium": 35,
    "long_range_funding_premium": 36,
    "long_minimal_funding_premium": 29,
    "short_premium_kimchi_union": 53,
    "short_fx_stress": 32,
    "short_premium_panic": 30
  },
  "eval2025": {
    "long_funding_compression_premium": 51,
    "long_range_funding_premium": 27,
    "long_minimal_funding_premium": 26,
    "short_premium_kimchi_union": 79,
    "short_fx_stress": 25,
    "short_premium_panic": 68
  },
  "ytd2026": {
    "long_funding_compression_premium": 32,
    "long_range_funding_premium": 29,
    "long_minimal_funding_premium": 29,
    "short_premium_kimchi_union": 32,
    "short_fx_stress": 18,
    "short_premium_panic": 25
  }
}
```

## Interpretation

- If a 2024-selected top row fails 2025/2026, the alpha mix is not robust enough for live sizing despite good in-sample/test selection.
- All-window diagnostic is useful for research direction only; do not treat it as clean validation.
