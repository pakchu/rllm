# Combined REX/OI + new alpha portfolio opt (2026-07-10)

This reruns the portfolio search after restoring the legacy REX/OI-heavy sleeves that were omitted from the new-alpha-only scan.

Protocol: Weights ranked on test2024 only; eval2025 and ytd2026 are report-only. Robust diagnostic is explicitly eval-influenced research, not clean selection. Nonzero weights are discretized to min 0.25 and 0.05 step.
Gross cap=7.0; cost each side=0.0600%; new alpha unit leverage=0.5; nonzero weight min=0.25, step=0.05.
Metric cell format: `abs_return/CAGR/strict_MDD/CAGR_MDD/trades`.

## Top selected by 2024 test only

| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |
|---:|---:|---|---:|---:|---:|
| 1 | 6.90 | `{'nonpb30_taker': 0.55, 'oi_raw': 0.25, 'rex_rule': 1.05, 'short_premium_panic': 1.05, 'new_long_funding_compression_premium': 1.7, 'new_long_range_funding_premium': 1.0, 'new_short_premium_kimchi_union': 1.3}` | 722.46/718.92/11.32/63.50/444 | 230.97/231.24/11.70/19.76/372 | 97.92/410.24/9.12/44.98/190 |
| 2 | 6.90 | `{'nonpb30_taker': 0.45, 'oi_raw': 0.25, 'rex_rule': 0.95, 'short_premium_panic': 0.95, 'new_long_funding_compression_premium': 1.95, 'new_long_range_funding_premium': 0.8, 'new_short_premium_kimchi_union': 1.55}` | 715.62/712.12/12.14/58.66/444 | 224.81/225.07/11.65/19.32/372 | 101.03/429.63/9.17/46.83/190 |
| 3 | 6.80 | `{'nonpb30_taker': 0.7, 'oi_raw': 0.35, 'rex_rule': 1.35, 'short_premium_panic': 1.35, 'new_long_funding_compression_premium': 1.5, 'new_long_range_funding_premium': 0.9, 'new_short_premium_kimchi_union': 0.65}` | 736.87/733.23/12.61/58.16/444 | 234.60/234.87/12.14/19.34/372 | 89.70/361.10/9.01/40.07/190 |
| 4 | 6.90 | `{'nonpb30_taker': 0.45, 'rex_rule': 1.15, 'oi_high_sel': 1.45, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.6, 'new_long_range_funding_premium': 0.95, 'new_short_premium_kimchi_union': 0.7}` | 819.63/815.46/14.09/57.86/427 | 320.09/320.51/11.54/27.78/372 | 75.90/285.07/13.85/20.58/202 |
| 5 | 6.90 | `{'nonpb30_taker': 0.35, 'rex_rule': 0.85, 'oi_high_sel': 1.1, 'rex_dyn_short_exit': 0.45, 'new_long_funding_compression_premium': 1.75, 'new_long_range_funding_premium': 1.05, 'new_short_premium_kimchi_union': 1.35}` | 769.01/765.16/13.26/57.70/427 | 288.79/289.16/10.80/26.76/372 | 84.88/333.64/12.96/25.75/202 |
| 6 | 6.85 | `{'nonpb30_taker': 0.45, 'oi_raw': 0.25, 'rex_rule': 0.95, 'short_premium_panic': 0.95, 'new_long_funding_compression_premium': 2.1, 'new_long_range_funding_premium': 1.25, 'new_short_premium_kimchi_union': 0.9}` | 809.97/805.86/14.12/57.09/444 | 225.30/225.57/14.11/15.99/372 | 98.71/415.16/10.29/40.33/190 |
| 7 | 6.85 | `{'nonpb30_taker': 0.45, 'oi_raw': 0.25, 'rex_rule': 0.95, 'short_premium_panic': 0.95, 'new_long_funding_compression_premium': 1.8, 'new_long_range_funding_premium': 0.7, 'new_short_premium_kimchi_union': 1.75}` | 665.73/662.55/11.68/56.74/444 | 221.80/222.06/11.62/19.11/372 | 99.38/419.28/9.30/45.06/190 |
| 8 | 6.85 | `{'nonpb30_taker': 0.3, 'rex_rule': 0.75, 'oi_high_sel': 0.95, 'rex_dyn_short_exit': 0.4, 'new_long_funding_compression_premium': 2.05, 'new_long_range_funding_premium': 0.8, 'new_short_premium_kimchi_union': 1.6}` | 739.41/735.76/13.46/54.66/427 | 269.10/269.43/11.10/24.27/372 | 90.08/363.34/12.37/29.36/202 |
| 9 | 6.95 | `{'nonpb30_taker': 0.8, 'oi_raw': 0.8, 'rex_rule': 1.65, 'new_long_funding_compression_premium': 1.8, 'new_long_range_funding_premium': 1.1, 'new_short_premium_kimchi_union': 0.8}` | 894.26/889.60/16.40/54.25/417 | 352.86/353.33/11.22/31.49/349 | 85.33/336.18/12.76/26.34/176 |
| 10 | 6.85 | `{'nonpb30_taker': 0.3, 'rex_rule': 0.75, 'oi_high_sel': 0.95, 'rex_dyn_short_exit': 0.4, 'new_long_funding_compression_premium': 1.9, 'new_long_range_funding_premium': 0.75, 'new_short_premium_kimchi_union': 1.8}` | 702.23/698.81/13.01/53.70/427 | 269.13/269.46/10.50/25.67/372 | 89.33/359.00/12.02/29.86/202 |
| 11 | 6.95 | `{'nonpb30_taker': 0.55, 'oi_raw': 0.55, 'rex_rule': 1.05, 'new_long_funding_compression_premium': 2.35, 'new_long_range_funding_premium': 1.4, 'new_short_premium_kimchi_union': 1.05}` | 919.48/914.64/17.30/52.86/417 | 293.89/294.26/14.33/20.54/349 | 95.19/393.60/13.32/29.55/176 |
| 12 | 6.95 | `{'nonpb30_taker': 0.6, 'oi_raw': 0.6, 'rex_rule': 1.2, 'new_long_funding_compression_premium': 1.95, 'new_long_range_funding_premium': 1.15, 'new_short_premium_kimchi_union': 1.45}` | 823.67/819.47/15.59/52.55/417 | 308.13/308.52/12.12/25.45/349 | 93.08/380.97/12.05/31.61/176 |
| 13 | 6.80 | `{'nonpb30_taker': 0.45, 'oi_raw': 0.35, 'rex_rule': 1.3, 'oi_upbit_ratio288_low': 1.5, 'oi_alt_ratio72_dyn_exit': 0.25, 'new_long_funding_compression_premium': 1.45, 'new_long_range_funding_premium': 0.85, 'new_short_premium_kimchi_union': 0.65}` | 724.28/720.72/13.75/52.40/599 | 327.38/327.81/10.30/31.83/439 | 68.62/248.10/11.40/21.76/213 |
| 14 | 6.80 | `{'nonpb30_taker': 0.7, 'oi_raw': 0.35, 'rex_rule': 1.35, 'short_premium_panic': 1.35, 'new_long_funding_compression_premium': 1.4, 'new_long_range_funding_premium': 0.55, 'new_short_premium_kimchi_union': 1.1}` | 663.42/660.24/12.61/52.37/444 | 231.32/231.60/9.92/23.34/372 | 90.03/363.03/9.40/38.64/190 |
| 15 | 7.00 | `{'nonpb30_taker': 0.55, 'oi_raw': 0.55, 'rex_rule': 1.05, 'new_long_funding_compression_premium': 2.2, 'new_long_range_funding_premium': 0.9, 'new_short_premium_kimchi_union': 1.75}` | 807.59/803.50/15.35/52.34/417 | 293.23/293.59/12.42/23.64/349 | 98.15/411.66/11.89/34.63/176 |
| 16 | 6.75 | `{'nonpb30_taker': 0.45, 'oi_raw': 0.25, 'rex_rule': 0.95, 'short_premium_panic': 0.95, 'new_long_range_funding_premium': 2.55, 'new_short_fx_stress': 1.25, 'new_short_premium_panic': 0.35}` | 558.45/555.91/10.64/52.26/418 | 245.02/245.31/8.48/28.91/335 | 64.64/228.81/9.27/24.68/169 |
| 17 | 6.90 | `{'nonpb30_taker': 0.45, 'rex_rule': 1.15, 'oi_high_sel': 1.45, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.5, 'new_long_range_funding_premium': 0.6, 'new_short_premium_kimchi_union': 1.15}` | 739.45/735.80/14.09/52.21/427 | 316.13/316.54/10.09/31.38/372 | 76.33/287.31/12.94/22.20/202 |
| 18 | 6.85 | `{'nonpb30_taker': 0.3, 'rex_rule': 0.75, 'oi_high_sel': 0.95, 'rex_dyn_short_exit': 0.4, 'new_long_funding_compression_premium': 1.6, 'new_long_range_funding_premium': 0.75, 'new_short_premium_kimchi_union': 2.1}` | 645.81/642.74/12.33/52.14/427 | 269.84/270.17/9.92/27.23/372 | 87.27/347.13/11.79/29.45/202 |
| 19 | 6.80 | `{'nonpb30_taker': 0.3, 'rex_rule': 0.75, 'oi_high_sel': 0.95, 'rex_dyn_short_exit': 0.4, 'new_long_funding_compression_premium': 2.15, 'new_long_range_funding_premium': 1.3, 'new_short_premium_kimchi_union': 0.95}` | 833.98/829.71/15.94/52.06/427 | 270.45/270.79/12.98/20.86/372 | 87.33/347.50/13.69/25.39/202 |
| 20 | 6.65 | `{'nonpb30_taker': 0.3, 'oi_raw': 0.25, 'rex_rule': 0.9, 'oi_upbit_ratio288_low': 1.05, 'new_long_funding_compression_premium': 1.75, 'new_long_range_funding_premium': 0.7, 'new_short_premium_kimchi_union': 1.7}` | 630.66/627.69/12.13/51.73/496 | 273.26/273.60/11.00/24.86/387 | 80.64/310.29/9.47/32.75/190 |

## Robust diagnostic only (eval-influenced)

| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |
|---:|---:|---|---:|---:|---:|
| 1 | 6.70 | `{'nonpb30_taker': 1.2, 'rex_rule': 1.5, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.1, 'new_long_range_funding_premium': 0.5, 'new_short_premium_kimchi_union': 1.5}` | 486.57/484.45/13.84/35.00/427 | 293.70/294.07/8.27/35.56/372 | 91.84/373.63/10.95/34.11/202 |
| 2 | 6.75 | `{'nonpb30_taker': 1.2, 'rex_rule': 1.5, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.3, 'new_long_range_funding_premium': 0.55, 'new_short_premium_kimchi_union': 1.3}` | 527.84/525.48/13.84/37.96/427 | 297.26/297.63/8.73/34.10/372 | 94.36/388.62/10.95/35.48/202 |
| 3 | 6.70 | `{'nonpb30_taker': 1.2, 'rex_rule': 1.5, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.45, 'new_long_range_funding_premium': 0.55, 'new_short_premium_kimchi_union': 1.1}` | 546.18/543.71/13.84/39.28/427 | 293.82/294.19/8.99/32.74/372 | 94.55/389.75/10.95/35.58/202 |
| 4 | 6.85 | `{'nonpb30_taker': 1.2, 'oi_raw': 0.45, 'rex_rule': 1.5, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.45, 'new_long_range_funding_premium': 0.55, 'new_short_premium_kimchi_union': 1.1}` | 588.09/585.38/14.10/41.51/432 | 310.04/310.44/9.30/33.38/376 | 91.09/369.21/11.52/32.05/203 |
| 5 | 6.90 | `{'nonpb30_taker': 1.2, 'oi_raw': 0.45, 'rex_rule': 1.5, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.3, 'new_long_range_funding_premium': 0.55, 'new_short_premium_kimchi_union': 1.3}` | 568.57/565.97/14.10/40.14/432 | 313.62/314.02/9.04/34.73/376 | 90.90/368.12/11.52/31.95/203 |
| 6 | 6.90 | `{'nonpb30_taker': 1.2, 'oi_raw': 0.45, 'rex_rule': 1.5, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.55, 'new_long_range_funding_premium': 0.9, 'new_short_premium_kimchi_union': 0.7}` | 660.26/657.11/14.10/46.60/432 | 317.28/317.68/10.13/31.37/376 | 91.66/372.55/11.52/32.34/203 |
| 7 | 6.65 | `{'nonpb30_taker': 1.3, 'rex_rule': 1.6, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.65, 'new_long_funding_compression_premium': 1.2, 'new_long_range_funding_premium': 0.7, 'new_short_premium_kimchi_union': 0.9}` | 527.98/525.62/14.73/35.68/427 | 300.70/301.08/9.06/33.25/372 | 90.75/367.23/11.74/31.28/202 |
| 8 | 7.00 | `{'nonpb30_taker': 1.2, 'rex_rule': 1.45, 'oi_wave_lowpos144': 0.3, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.15, 'new_long_range_funding_premium': 0.5, 'new_short_premium_kimchi_union': 1.5}` | 537.97/535.56/13.79/38.83/511 | 325.57/325.99/8.72/37.37/446 | 91.50/371.66/12.09/30.75/221 |
| 9 | 6.85 | `{'nonpb30_taker': 1.2, 'oi_raw': 0.45, 'rex_rule': 1.5, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.1, 'new_long_range_funding_premium': 0.5, 'new_short_premium_kimchi_union': 1.5}` | 524.61/522.27/14.10/37.04/432 | 309.92/310.31/8.59/36.14/376 | 88.42/353.75/11.52/30.71/203 |
| 10 | 6.75 | `{'nonpb30_taker': 1.2, 'rex_rule': 1.5, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.55, 'new_long_range_funding_premium': 0.9, 'new_short_premium_kimchi_union': 0.7}` | 613.93/611.06/13.84/44.14/427 | 300.77/301.15/9.83/30.62/372 | 95.12/393.23/10.95/35.90/202 |
| 11 | 6.90 | `{'nonpb30_taker': 0.45, 'rex_rule': 0.7, 'oi_low': 0.25, 'oi_high_sel': 0.45, 'bear_rex_short': 0.6, 'new_long_funding_compression_premium': 1.6, 'new_long_range_funding_premium': 0.75, 'new_short_premium_kimchi_union': 2.1}` | 560.98/558.42/12.26/45.55/490 | 297.65/298.03/9.96/29.92/411 | 93.50/383.50/10.57/36.29/215 |
| 12 | 6.95 | `{'nonpb30_taker': 0.8, 'oi_raw': 0.8, 'rex_rule': 1.65, 'new_long_funding_compression_premium': 1.55, 'new_long_range_funding_premium': 0.65, 'new_short_premium_kimchi_union': 1.5}` | 760.04/756.26/16.06/47.08/417 | 348.20/348.66/9.71/35.90/349 | 85.36/336.36/11.30/29.75/176 |
| 13 | 6.90 | `{'nonpb30_taker': 0.8, 'oi_raw': 0.8, 'rex_rule': 1.65, 'new_long_funding_compression_premium': 1.3, 'new_long_range_funding_premium': 0.6, 'new_short_premium_kimchi_union': 1.75}` | 694.22/690.85/16.06/43.01/417 | 344.47/344.93/9.17/37.62/349 | 82.71/321.61/11.01/29.22/176 |
| 14 | 6.75 | `{'nonpb30_taker': 1.2, 'rex_rule': 1.5, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.6, 'new_long_funding_compression_premium': 1.6, 'new_long_range_funding_premium': 0.25, 'new_short_fx_stress': 1.3}` | 522.47/520.15/13.84/37.58/406 | 290.99/291.35/7.44/39.15/318 | 82.41/319.94/10.95/29.21/188 |
| 15 | 6.90 | `{'nonpb30_taker': 0.8, 'oi_raw': 0.8, 'rex_rule': 1.65, 'new_long_funding_compression_premium': 1.7, 'new_long_range_funding_premium': 0.65, 'new_short_premium_kimchi_union': 1.3}` | 784.78/780.84/16.06/48.61/417 | 344.21/344.66/9.97/34.58/349 | 85.46/336.89/11.54/29.19/176 |
| 16 | 6.90 | `{'nonpb30_taker': 1.25, 'rex_rule': 1.6, 'oi_wave_lowpos144': 0.3, 'oi_high_sel': 0.3, 'rex_dyn_short_exit': 0.65, 'new_long_funding_compression_premium': 1.2, 'new_long_range_funding_premium': 0.7, 'new_short_premium_kimchi_union': 0.9}` | 571.05/568.44/15.01/37.88/511 | 332.28/332.72/9.36/35.53/446 | 89.19/358.20/12.48/28.70/221 |
| 17 | 6.70 | `{'nonpb30_taker': 1.4, 'rex_rule': 1.75, 'oi_high_sel': 0.35, 'rex_dyn_short_exit': 0.7, 'new_long_funding_compression_premium': 1.15, 'new_long_range_funding_premium': 0.45, 'new_short_premium_kimchi_union': 0.9}` | 512.19/509.92/16.11/31.66/427 | 312.63/313.03/8.89/35.21/372 | 89.79/361.63/12.71/28.46/202 |
| 18 | 6.80 | `{'nonpb30_taker': 0.5, 'rex_rule': 0.8, 'oi_low': 0.25, 'oi_high_sel': 0.5, 'bear_rex_short': 0.65, 'new_long_funding_compression_premium': 1.75, 'new_long_range_funding_premium': 1.0, 'new_short_premium_kimchi_union': 1.35}` | 625.16/622.22/13.11/47.45/490 | 306.91/307.30/10.81/28.43/411 | 90.53/365.95/11.32/32.33/215 |
| 19 | 6.70 | `{'nonpb30_taker': 0.35, 'oi_raw': 0.3, 'rex_rule': 1.05, 'oi_upbit_ratio288_low': 1.15, 'new_long_funding_compression_premium': 1.65, 'new_long_range_funding_premium': 0.95, 'new_short_premium_kimchi_union': 1.25}` | 690.74/687.40/13.37/51.40/496 | 295.25/295.62/10.46/28.27/387 | 77.07/291.17/10.17/28.63/190 |
| 20 | 6.65 | `{'nonpb30_taker': 1.4, 'rex_rule': 1.75, 'oi_high_sel': 0.35, 'rex_dyn_short_exit': 0.7, 'new_long_funding_compression_premium': 1.2, 'new_long_range_funding_premium': 0.7, 'new_short_premium_kimchi_union': 0.55}` | 546.05/543.59/16.11/33.75/427 | 312.78/313.18/9.42/33.24/372 | 88.71/355.40/12.71/27.97/202 |

## Event counts

```json
{
  "train": {
    "oi_alt_ratio72_dyn_exit": 813,
    "oi_high_sel": 966,
    "oi_raw": 1012,
    "oi_upbit_ratio288_low": 344,
    "oi_vol_alt_ratio72": 958,
    "oi_vol_volmom288": 447,
    "oi_wave_slope288_low": 335,
    "oi_low": 313,
    "oi_wave_lowpos144": 475,
    "pb30_addon": 347,
    "pb30_base": 189,
    "nonpb30_taker": 242,
    "short_premium_panic": 135,
    "short_premium_wave_lowvol": 61,
    "short_kimchi3d": 175,
    "rex_rule": 356,
    "rex_wave_vol144_high": 146,
    "rex_wave_pricez288_low": 90,
    "bear_rex_short": 183,
    "rex_dyn_short_exit": 218,
    "oi_vol_alt_ratio288": 39
  },
  "test2024": {
    "oi_alt_ratio72_dyn_exit": 103,
    "oi_low": 64,
    "oi_raw": 197,
    "oi_vol_alt_ratio72": 92,
    "oi_vol_volmom288": 116,
    "new_long_range_funding_premium": 36,
    "oi_high_sel": 192,
    "oi_upbit_ratio288_low": 79,
    "oi_wave_slope288_low": 65,
    "nonpb30_taker": 34,
    "oi_vol_alt_ratio288": 69,
    "oi_wave_lowpos144": 84,
    "short_premium_panic": 27,
    "short_premium_wave_lowvol": 11,
    "new_short_premium_kimchi_union": 53,
    "new_short_premium_panic": 30,
    "new_short_fx_stress": 32,
    "pb30_base": 20,
    "short_kimchi3d": 37,
    "pb30_addon": 38,
    "new_long_funding_compression_premium": 35,
    "new_long_minimal_funding_premium": 29,
    "rex_rule": 62,
    "rex_wave_pricez288_low": 32,
    "rex_wave_vol144_high": 24,
    "bear_rex_short": 14,
    "rex_dyn_short_exit": 15
  },
  "eval2025": {
    "new_short_premium_kimchi_union": 79,
    "new_short_premium_panic": 68,
    "short_kimchi3d": 29,
    "new_short_fx_stress": 25,
    "oi_high_sel": 120,
    "oi_raw": 124,
    "oi_vol_volmom288": 66,
    "oi_wave_lowpos144": 74,
    "oi_wave_slope288_low": 69,
    "oi_low": 40,
    "oi_vol_alt_ratio288": 41,
    "oi_vol_alt_ratio72": 53,
    "oi_upbit_ratio288_low": 38,
    "pb30_addon": 50,
    "short_premium_panic": 23,
    "short_premium_wave_lowvol": 11,
    "oi_alt_ratio72_dyn_exit": 52,
    "nonpb30_taker": 35,
    "rex_rule": 33,
    "new_long_funding_compression_premium": 51,
    "pb30_base": 20,
    "new_long_minimal_funding_premium": 26,
    "new_long_range_funding_premium": 27,
    "bear_rex_short": 26,
    "rex_dyn_short_exit": 27,
    "rex_wave_pricez288_low": 6,
    "rex_wave_vol144_high": 6
  },
  "ytd2026": {
    "new_long_funding_compression_premium": 32,
    "new_long_minimal_funding_premium": 29,
    "new_long_range_funding_premium": 29,
    "new_short_premium_kimchi_union": 32,
    "oi_high_sel": 44,
    "oi_raw": 45,
    "short_premium_panic": 14,
    "short_kimchi3d": 14,
    "new_short_fx_stress": 18,
    "nonpb30_taker": 14,
    "pb30_addon": 46,
    "pb30_base": 17,
    "oi_vol_alt_ratio288": 14,
    "oi_vol_alt_ratio72": 21,
    "oi_wave_lowpos144": 19,
    "oi_wave_slope288_low": 19,
    "oi_low": 17,
    "oi_alt_ratio72_dyn_exit": 23,
    "oi_vol_volmom288": 20,
    "short_premium_wave_lowvol": 5,
    "oi_upbit_ratio288_low": 14,
    "bear_rex_short": 23,
    "rex_dyn_short_exit": 27,
    "rex_rule": 24,
    "rex_wave_vol144_high": 6,
    "new_short_premium_panic": 25,
    "rex_wave_pricez288_low": 1
  }
}
```

## Interpretation

- The previous new-alpha-only portfolio was not an apples-to-apples replacement for the gross 5.75/6.10 REX/OI portfolios because it omitted `rex_rule`, `bear_rex_short`, dynamic REX exits, and OI sleeves.
- Use the 2024-selected table as cleaner than the robust diagnostic, but do not call 2026 pristine: the candidate universe itself includes prior research artifacts that may have been influenced by later-period analysis.
- Legacy and new sleeve leverage semantics differ; do not deploy a combined row without a live-size normalization pass.
