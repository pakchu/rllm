# Dynamic exit all-candidates scan (2026-07-08)

Expanded dynamic-exit scan over existing, wave/volume, orthogonal, alt/OI/CVD entry candidates. Fixed holding period removed except optional max_bars safety cap; exit thresholds train<2024 only; split-end forced close; 5bp per side; strict in-position MDD.

## Entry counts
```json
{
  "train": {
    "pb30_base": 517,
    "pb30_addon": 993,
    "nonpb30_taker": 258,
    "oi_raw": 1950,
    "oi_wave_lowpos144": 917,
    "oi_alt_ratio72": 1843,
    "oi_upbit_like_altlow": 1862,
    "weekend_reversal_long": 1307,
    "cvd_flow_cont_long": 888,
    "cvd_bear_div_long": 1941,
    "cvd_bull_div_long": 1378,
    "alt_rotation_long": 2943,
    "alt_rotation_short": 2935,
    "kimchi_dxy_short": 1217,
    "rex_rule": 829
  },
  "test2024": {
    "pb30_base": 42,
    "pb30_addon": 89,
    "nonpb30_taker": 40,
    "oi_raw": 365,
    "oi_wave_lowpos144": 156,
    "oi_alt_ratio72": 153,
    "oi_upbit_like_altlow": 212,
    "weekend_reversal_long": 402,
    "cvd_flow_cont_long": 267,
    "cvd_bear_div_long": 557,
    "cvd_bull_div_long": 402,
    "alt_rotation_long": 632,
    "alt_rotation_short": 571,
    "kimchi_dxy_short": 362,
    "rex_rule": 131
  },
  "eval2025": {
    "pb30_base": 51,
    "pb30_addon": 158,
    "nonpb30_taker": 35,
    "oi_raw": 232,
    "oi_wave_lowpos144": 132,
    "oi_alt_ratio72": 88,
    "oi_upbit_like_altlow": 144,
    "weekend_reversal_long": 362,
    "cvd_flow_cont_long": 269,
    "cvd_bear_div_long": 549,
    "cvd_bull_div_long": 432,
    "alt_rotation_long": 688,
    "alt_rotation_short": 688,
    "kimchi_dxy_short": 292,
    "rex_rule": 51
  },
  "ytd2026": {
    "pb30_base": 35,
    "pb30_addon": 110,
    "nonpb30_taker": 19,
    "oi_raw": 94,
    "oi_wave_lowpos144": 48,
    "oi_alt_ratio72": 35,
    "oi_upbit_like_altlow": 59,
    "weekend_reversal_long": 149,
    "cvd_flow_cont_long": 81,
    "cvd_bear_div_long": 237,
    "cvd_bull_div_long": 222,
    "alt_rotation_long": 153,
    "alt_rotation_short": 126,
    "kimchi_dxy_short": 142,
    "rex_rule": 50
  }
}
```

| rank | entry | side | exit | min/max | 2024 ret/CAGR/MDD/ratio/trades | 2025 ret/CAGR/MDD/ratio/trades | 2026 ret/CAGR/MDD/ratio/trades |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | rex_rule | long | cvd_bear | 1/0 | 22.31/22.26/9.32/2.39/74 | 3.08/3.08/1.78/1.73/7 | 1.15/2.76/1.08/2.55/2 |
| 2 | rex_rule | long | cvd_bear | 1/288 | 20.40/20.35/9.32/2.18/71 | 3.08/3.08/1.78/1.73/7 | 1.15/2.76/1.08/2.55/2 |
| 3 | rex_rule | long | cvd_bear | 6/0 | 27.57/27.51/10.39/2.65/72 | 3.02/3.02/1.78/1.70/7 | 1.26/3.02/1.08/2.80/2 |
| 4 | rex_rule | long | cvd_bear | 6/288 | 25.58/25.52/10.39/2.46/69 | 3.02/3.02/1.78/1.70/7 | 1.26/3.02/1.08/2.80/2 |
| 5 | oi_alt_ratio72 | long | vwap_overheat | 48/0 | 26.79/26.73/10.23/2.61/103 | 13.73/13.74/8.52/1.61/52 | 6.31/15.73/8.06/1.95/23 |
| 6 | oi_alt_ratio72 | long | vwap_overheat | 48/288 | 26.79/26.73/10.23/2.61/103 | 13.73/13.74/8.52/1.61/52 | 6.31/15.73/8.06/1.95/23 |
| 7 | rex_rule | long | oi_unwind_or_funding_hot | 6/0 | 23.02/22.97/10.05/2.29/80 | 2.12/2.12/1.61/1.32/7 | 0.63/1.51/1.08/1.40/2 |
| 8 | rex_rule | short | wave_lower_or_cvd_bull | 48/0 | 4.35/4.35/3.45/1.26/16 | 12.13/12.14/3.79/3.21/31 | 8.86/22.47/4.19/5.36/28 |
| 9 | rex_rule | short | wave_lower_or_cvd_bull | 48/288 | 4.35/4.35/3.45/1.26/16 | 12.13/12.14/3.79/3.21/31 | 8.86/22.47/4.19/5.36/28 |
| 10 | oi_alt_ratio72 | long | wave_upper_or_cvd_bear | 12/0 | 31.43/31.36/7.53/4.17/118 | 10.97/10.97/8.96/1.22/61 | 1.40/3.38/6.49/0.52/30 |
| 11 | oi_alt_ratio72 | long | alt_riskoff | 24/72 | 6.55/6.54/5.39/1.21/56 | 14.42/14.43/7.54/1.91/62 | 0.90/2.15/4.91/0.44/20 |
| 12 | oi_alt_ratio72 | long | vwap_overheat | 24/0 | 18.33/18.29/7.06/2.59/108 | 11.11/11.11/9.31/1.19/56 | 9.90/25.28/6.26/4.04/25 |
| 13 | oi_alt_ratio72 | long | vwap_overheat | 24/288 | 18.33/18.29/7.06/2.59/108 | 11.11/11.11/9.31/1.19/56 | 9.90/25.28/6.26/4.04/25 |
| 14 | oi_alt_ratio72 | long | oi_unwind_or_funding_hot | 12/288 | 16.02/15.98/8.22/1.94/35 | 9.52/9.53/8.14/1.17/37 | 2.81/6.83/5.62/1.21/27 |
| 15 | rex_rule | long | cvd_bear | 48/0 | 16.96/16.92/11.48/1.47/52 | 3.82/3.82/3.34/1.15/7 | 1.19/2.85/1.08/2.64/1 |
| 16 | oi_alt_ratio72 | long | vwap_overheat | 12/0 | 14.94/14.91/7.02/2.12/116 | 9.98/9.99/9.52/1.05/63 | 6.33/15.77/6.26/2.52/29 |
| 17 | oi_alt_ratio72 | long | vwap_overheat | 12/288 | 14.94/14.91/7.02/2.12/116 | 9.98/9.99/9.52/1.05/63 | 6.33/15.77/6.26/2.52/29 |
| 18 | oi_alt_ratio72 | long | alt_riskoff | 48/288 | 11.49/11.47/10.39/1.10/109 | 8.61/8.62/8.44/1.02/53 | 5.37/13.29/6.26/2.12/27 |
| 19 | oi_alt_ratio72 | long | wave_upper_or_cvd_bear | 1/0 | 29.23/29.16/7.12/4.10/122 | 8.27/8.28/8.43/0.98/68 | 0.14/0.33/4.44/0.07/31 |
| 20 | oi_alt_ratio72 | long | wave_upper_or_cvd_bear | 1/288 | 29.23/29.16/7.12/4.10/122 | 8.27/8.28/8.43/0.98/68 | 0.14/0.33/4.44/0.07/31 |
| 21 | rex_rule | long | oi_unwind_or_funding_hot | 48/0 | 17.04/17.00/12.28/1.38/58 | 2.45/2.46/2.57/0.95/7 | 0.56/1.34/1.08/1.24/1 |
| 22 | rex_rule | short | cvd_bull | 48/0 | 4.12/4.11/4.56/0.90/17 | 12.84/12.85/3.85/3.34/29 | 6.56/16.38/4.72/3.47/27 |
| 23 | rex_rule | short | cvd_bull | 48/288 | 4.12/4.11/4.56/0.90/17 | 12.84/12.85/3.85/3.34/29 | 6.56/16.38/4.72/3.47/27 |
| 24 | rex_rule | long | wave_upper_or_cvd_bear | 6/0 | 9.49/9.47/10.75/0.88/76 | 3.33/3.33/1.78/1.88/7 | 0.39/0.94/1.08/0.87/2 |
| 25 | rex_rule | long | wave_upper_or_cvd_bear | 6/288 | 9.49/9.47/10.75/0.88/76 | 3.33/3.33/1.78/1.88/7 | 0.39/0.94/1.08/0.87/2 |
| 26 | rex_rule | long | oi_unwind_or_funding_hot | 6/288 | 8.72/8.70/10.05/0.87/41 | 2.12/2.12/1.61/1.32/7 | 0.63/1.51/1.08/1.40/2 |
| 27 | rex_rule | long | cvd_bear | 48/288 | 9.84/9.82/11.48/0.86/17 | 3.82/3.82/3.34/1.15/7 | 1.19/2.85/1.08/2.64/1 |
| 28 | rex_rule | long | oi_unwind_or_funding_hot | 12/0 | 21.35/21.30/10.81/1.97/77 | 1.40/1.40/1.69/0.83/7 | 0.47/1.13/1.08/1.04/2 |
| 29 | rex_rule | long | oi_unwind_or_funding_hot | 12/288 | 9.14/9.13/10.81/0.84/40 | 1.40/1.40/1.69/0.83/7 | 0.47/1.13/1.08/1.04/2 |
| 30 | rex_rule | long | wave_upper_or_cvd_bear | 48/0 | 22.08/22.03/10.70/2.06/59 | 2.96/2.96/3.64/0.81/7 | 0.54/1.30/1.08/1.21/1 |
| 31 | rex_rule | long | wave_upper_or_cvd_bear | 48/288 | 22.08/22.03/10.70/2.06/59 | 2.96/2.96/3.64/0.81/7 | 0.54/1.30/1.08/1.21/1 |
| 32 | rex_rule | long | cvd_bear | 12/0 | 24.35/24.29/9.42/2.58/71 | 1.98/1.98/2.46/0.80/7 | 1.19/2.85/1.08/2.64/1 |
| 33 | rex_rule | long | cvd_bear | 12/288 | 22.40/22.35/9.42/2.37/68 | 1.98/1.98/2.46/0.80/7 | 1.19/2.85/1.08/2.64/1 |
| 34 | rex_rule | long | alt_riskoff | 48/72 | 9.89/9.87/9.66/1.02/28 | 2.04/2.04/2.68/0.76/7 | 0.05/0.11/0.46/0.24/1 |
| 35 | oi_alt_ratio72 | long | oi_unwind_or_funding_hot | 24/288 | 29.15/29.08/8.84/3.29/34 | 6.60/6.61/8.85/0.75/34 | 1.88/4.56/5.01/0.91/26 |
| 36 | oi_alt_ratio72 | long | oi_unwind_or_funding_hot | 12/0 | 30.02/29.95/13.81/2.17/104 | 6.56/6.56/8.95/0.73/64 | 2.81/6.83/5.62/1.21/27 |
| 37 | oi_alt_ratio72 | long | alt_riskoff | 48/0 | 11.49/11.47/10.39/1.10/109 | 6.02/6.02/8.44/0.71/56 | 5.37/13.29/6.26/2.12/27 |
| 38 | oi_alt_ratio72 | long | alt_riskoff | 48/144 | 6.53/6.51/9.19/0.71/84 | 10.14/10.15/8.44/1.20/42 | 5.37/13.29/6.26/2.12/27 |
| 39 | oi_alt_ratio72 | long | vwap_overheat | 6/0 | 11.72/11.69/7.02/1.67/118 | 6.43/6.43/9.18/0.70/68 | 7.01/17.56/2.25/7.79/29 |
| 40 | oi_alt_ratio72 | long | vwap_overheat | 6/288 | 11.72/11.69/7.02/1.67/118 | 6.43/6.43/9.18/0.70/68 | 7.01/17.56/2.25/7.79/29 |
| 41 | rex_rule | long | wave_upper_or_cvd_bear | 12/0 | 8.07/8.05/11.74/0.69/73 | 2.06/2.06/2.99/0.69/7 | 0.54/1.30/1.08/1.21/1 |
| 42 | rex_rule | long | wave_upper_or_cvd_bear | 12/288 | 8.07/8.05/11.74/0.69/73 | 2.06/2.06/2.99/0.69/7 | 0.54/1.30/1.08/1.21/1 |
| 43 | oi_wave_lowpos144 | long | vwap_overheat | 48/144 | 12.25/12.23/4.79/2.55/24 | 5.67/5.68/8.92/0.64/22 | 4.07/9.98/1.11/8.98/2 |
| 44 | rex_rule | long | oi_unwind_or_funding_hot | 48/288 | 7.68/7.67/12.28/0.62/28 | 2.45/2.46/2.57/0.95/7 | 0.56/1.34/1.08/1.24/1 |
| 45 | rex_rule | long | wave_upper_or_cvd_bear | 24/0 | 15.51/15.47/11.41/1.36/66 | 2.23/2.23/3.64/0.61/7 | 0.54/1.30/1.08/1.21/1 |
| 46 | rex_rule | long | wave_upper_or_cvd_bear | 24/288 | 15.51/15.47/11.41/1.36/66 | 2.23/2.23/3.64/0.61/7 | 0.54/1.30/1.08/1.21/1 |
| 47 | rex_rule | long | oi_unwind_or_funding_hot | 24/0 | 17.87/17.83/12.08/1.48/67 | 1.45/1.45/2.57/0.56/7 | 0.56/1.34/1.08/1.24/1 |
| 48 | rex_rule | long | oi_unwind_or_funding_hot | 24/288 | 9.06/9.04/12.08/0.75/34 | 1.45/1.45/2.57/0.56/7 | 0.56/1.34/1.08/1.24/1 |
| 49 | rex_rule | long | cvd_bear | 24/0 | 19.62/19.57/9.90/1.98/61 | 1.87/1.87/3.34/0.56/7 | 1.19/2.85/1.08/2.64/1 |
| 50 | rex_rule | long | cvd_bear | 24/288 | 17.75/17.71/9.90/1.79/58 | 1.87/1.87/3.34/0.56/7 | 1.19/2.85/1.08/2.64/1 |
| 51 | oi_alt_ratio72 | long | alt_riskoff | 24/144 | 4.98/4.97/9.01/0.55/118 | 15.35/15.36/7.54/2.04/62 | 2.76/6.70/4.91/1.36/29 |
| 52 | oi_alt_ratio72 | long | alt_riskoff | 24/288 | 4.98/4.97/9.01/0.55/118 | 14.62/14.63/7.54/1.94/62 | 2.76/6.70/4.91/1.36/29 |
| 53 | oi_alt_ratio72 | long | alt_riskoff | 24/0 | 4.98/4.97/9.01/0.55/118 | 11.25/11.26/7.54/1.49/65 | 2.76/6.70/4.91/1.36/29 |
| 54 | oi_wave_lowpos144 | long | oi_unwind_or_funding_hot | 48/0 | 13.26/13.23/13.76/0.96/73 | 6.45/6.46/12.51/0.52/57 | 4.27/10.49/9.32/1.13/25 |
| 55 | oi_wave_lowpos144 | long | oi_unwind_or_funding_hot | 48/288 | 7.17/7.15/10.97/0.65/38 | 6.45/6.46/12.51/0.52/57 | 4.27/10.49/9.32/1.13/25 |
| 56 | rex_rule | long | vwap_overheat | 48/0 | 31.92/31.84/9.34/3.41/62 | 1.42/1.42/2.76/0.52/7 | 0.38/0.90/0.46/1.97/1 |
| 57 | rex_rule | long | vwap_overheat | 48/288 | 31.92/31.84/9.34/3.41/62 | 1.42/1.42/2.76/0.52/7 | 0.38/0.90/0.46/1.97/1 |
| 58 | rex_rule | long | alt_riskoff | 12/72 | 3.45/3.45/7.11/0.48/109 | 0.61/0.61/1.11/0.55/7 | 0.05/0.11/0.46/0.24/1 |
| 59 | rex_rule | long | wave_upper_or_cvd_bear | 1/0 | 5.60/5.59/11.60/0.48/77 | 3.48/3.49/1.65/2.11/7 | 0.50/1.21/1.08/1.12/2 |
| 60 | rex_rule | long | wave_upper_or_cvd_bear | 1/288 | 5.60/5.59/11.60/0.48/77 | 3.48/3.49/1.65/2.11/7 | 0.50/1.21/1.08/1.12/2 |
