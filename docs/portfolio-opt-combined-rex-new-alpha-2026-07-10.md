# Combined REX/OI + new alpha portfolio opt (2026-07-10)

This reruns the portfolio search after restoring the legacy REX/OI-heavy sleeves that were omitted from the new-alpha-only scan.

Protocol: Weights ranked on test2024 only; eval2025 and ytd2026 are report-only. Robust diagnostic is explicitly eval-influenced research, not clean selection.
Gross cap=7.0; cost each side=0.0600%; new alpha unit leverage=0.5.
Metric cell format: `abs_return/CAGR/strict_MDD/CAGR_MDD/trades`.

## Top selected by 2024 test only

| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |
|---:|---:|---|---:|---:|---:|
| 1 | 6.97 | `{'pb30_base': 0.13156, 'nonpb30_taker': 0.526238, 'oi_raw': 0.263119, 'rex_rule': 1.052476, 'short_premium_panic': 1.052476, 'new_long_funding_compression_premium': 1.691859, 'new_long_range_funding_premium': 0.9784, 'new_short_premium_kimchi_union': 1.276528}` | 747.08/743.38/11.26/66.04/464 | 242.15/242.44/10.85/22.35/392 | 99.46/419.79/9.27/45.30/207 |
| 2 | 6.97 | `{'pb30_base': 0.170065, 'nonpb30_taker': 0.680259, 'oi_raw': 0.34013, 'rex_rule': 1.360518, 'short_premium_panic': 1.360518, 'new_long_funding_compression_premium': 1.497455, 'new_long_range_funding_premium': 0.902246, 'new_short_premium_kimchi_union': 0.661465}` | 782.14/778.21/12.68/61.38/464 | 252.92/253.22/11.30/22.40/392 | 93.75/384.98/9.64/39.95/207 |
| 3 | 6.97 | `{'pb30_base': 0.118181, 'nonpb30_taker': 0.472722, 'oi_raw': 0.236361, 'rex_rule': 0.945445, 'short_premium_panic': 0.945445, 'new_long_funding_compression_premium': 1.950923, 'new_long_range_funding_premium': 0.777323, 'new_short_premium_kimchi_union': 1.526256}` | 734.89/731.26/12.10/60.45/464 | 234.71/234.99/11.57/20.32/392 | 103.30/443.99/9.10/48.80/207 |
| 4 | 6.97 | `{'pb30_base': 0.118181, 'nonpb30_taker': 0.472722, 'oi_raw': 0.236361, 'rex_rule': 0.945445, 'short_premium_panic': 0.945445, 'new_long_funding_compression_premium': 2.081209, 'new_long_range_funding_premium': 1.253969, 'new_short_premium_kimchi_union': 0.919324}` | 839.45/835.15/14.00/59.64/464 | 238.21/238.49/13.51/17.65/392 | 101.76/434.20/10.39/41.80/207 |
| 5 | 6.97 | `{'pb30_base': 0.118181, 'nonpb30_taker': 0.472722, 'oi_raw': 0.236361, 'rex_rule': 0.945445, 'short_premium_panic': 0.945445, 'new_long_funding_compression_premium': 1.798562, 'new_long_range_funding_premium': 0.721932, 'new_short_premium_kimchi_union': 1.734007}` | 696.45/693.07/11.62/59.64/464 | 234.65/234.92/11.59/20.26/392 | 102.45/438.61/9.28/47.28/207 |
| 6 | 6.97 | `{'nonpb30_taker': 0.460625, 'rex_rule': 1.128892, 'oi_high_sel': 1.426516, 'rex_dyn_short_exit': 0.583918, 'oi_alt_ratio72_dyn_exit': 0.114912, 'new_long_funding_compression_premium': 1.593641, 'new_long_range_funding_premium': 0.9602, 'new_short_premium_kimchi_union': 0.703953}` | 830.79/826.54/13.88/59.56/530 | 321.46/321.87/12.16/26.46/424 | 76.77/289.60/13.75/21.06/225 |
| 7 | 6.97 | `{'nonpb30_taker': 0.351224, 'rex_rule': 0.860775, 'oi_high_sel': 1.087712, 'rex_dyn_short_exit': 0.445235, 'oi_alt_ratio72_dyn_exit': 0.08762, 'new_long_funding_compression_premium': 1.774722, 'new_long_range_funding_premium': 1.02632, 'new_short_premium_kimchi_union': 1.339049}` | 783.57/779.64/13.42/58.09/530 | 291.88/292.25/11.27/25.92/424 | 86.03/340.11/12.84/26.49/225 |
| 8 | 6.97 | `{'pb30_base': 0.118181, 'nonpb30_taker': 0.472722, 'oi_raw': 0.236361, 'rex_rule': 0.945445, 'short_premium_panic': 0.945445, 'new_long_range_funding_premium': 2.538478, 'new_short_premium_kimchi_union': 0.125475, 'new_short_fx_stress': 1.226363, 'new_short_premium_panic': 0.364185}` | 591.24/588.50/10.31/57.06/491 | 264.16/264.48/8.47/31.21/434 | 69.08/250.35/9.44/26.52/218 |
| 9 | 6.97 | `{'nonpb30_taker': 0.313943, 'rex_rule': 0.769406, 'oi_high_sel': 0.972255, 'rex_dyn_short_exit': 0.397974, 'oi_alt_ratio72_dyn_exit': 0.078319, 'new_long_funding_compression_premium': 2.036332, 'new_long_range_funding_premium': 0.811353, 'new_short_premium_kimchi_union': 1.593073}` | 767.20/763.37/13.57/56.25/530 | 277.69/278.03/11.43/24.32/424 | 91.11/369.33/12.46/29.63/225 |
| 10 | 6.97 | `{'pb30_base': 0.170065, 'nonpb30_taker': 0.680259, 'oi_raw': 0.34013, 'rex_rule': 1.360518, 'short_premium_panic': 1.360518, 'new_long_funding_compression_premium': 1.403713, 'new_long_range_funding_premium': 0.559293, 'new_short_premium_kimchi_union': 1.09816}` | 707.05/703.61/12.68/55.50/464 | 249.52/249.82/9.80/25.50/392 | 94.12/387.16/9.45/40.97/207 |
| 11 | 6.97 | `{'nonpb30_taker': 0.300147, 'oi_raw': 0.253971, 'rex_rule': 0.923531, 'oi_upbit_ratio288_low': 1.038972, 'bear_rex_short': 0.13853, 'oi_alt_ratio72_dyn_exit': 0.161618, 'new_long_funding_compression_premium': 1.756874, 'new_long_range_funding_premium': 0.705199, 'new_short_premium_kimchi_union': 1.693815}` | 662.94/659.77/11.94/55.27/613 | 297.04/297.42/10.56/28.16/465 | 85.13/335.02/9.49/35.30/236 |
| 12 | 6.97 | `{'nonpb30_taker': 0.820312, 'oi_raw': 0.820312, 'rex_rule': 1.640625, 'new_long_funding_compression_premium': 1.805755, 'new_long_range_funding_premium': 1.088003, 'new_short_premium_kimchi_union': 0.797649}` | 902.33/897.61/16.25/55.25/417 | 355.00/355.47/11.20/31.74/349 | 85.19/335.38/12.88/26.04/176 |
| 13 | 6.97 | `{'nonpb30_taker': 0.313943, 'rex_rule': 0.769406, 'oi_high_sel': 0.972255, 'rex_dyn_short_exit': 0.397974, 'oi_alt_ratio72_dyn_exit': 0.078319, 'new_long_funding_compression_premium': 1.877301, 'new_long_range_funding_premium': 0.753538, 'new_short_premium_kimchi_union': 1.80992}` | 725.73/722.16/13.09/55.19/530 | 277.65/277.99/10.77/25.81/424 | 90.32/364.72/12.08/30.18/225 |
| 14 | 6.97 | `{'nonpb30_taker': 0.313943, 'rex_rule': 0.769406, 'oi_high_sel': 0.972255, 'rex_dyn_short_exit': 0.397974, 'oi_alt_ratio72_dyn_exit': 0.078319, 'new_long_funding_compression_premium': 2.172321, 'new_long_range_funding_premium': 1.308866, 'new_short_premium_kimchi_union': 0.959571}` | 880.10/875.53/16.06/54.51/530 | 281.65/282.00/13.46/20.95/424 | 89.44/359.63/13.86/25.95/225 |
| 15 | 6.97 | `{'nonpb30_taker': 0.460625, 'rex_rule': 1.128892, 'oi_high_sel': 1.426516, 'rex_dyn_short_exit': 0.583918, 'oi_alt_ratio72_dyn_exit': 0.114912, 'new_long_funding_compression_premium': 1.493878, 'new_long_range_funding_premium': 0.595219, 'new_short_premium_kimchi_union': 1.168698}` | 747.18/743.48/13.88/53.57/530 | 317.26/317.67/10.67/29.78/424 | 77.23/292.04/12.82/22.78/225 |
| 16 | 6.97 | `{'nonpb30_taker': 0.313943, 'rex_rule': 0.769406, 'oi_high_sel': 0.972255, 'rex_dyn_short_exit': 0.397974, 'oi_alt_ratio72_dyn_exit': 0.078319, 'new_long_funding_compression_premium': 1.585311, 'new_long_range_funding_premium': 0.737856, 'new_short_premium_kimchi_union': 2.117592}` | 666.65/663.46/12.39/53.56/530 | 278.08/278.42/10.25/27.15/424 | 88.31/353.09/11.85/29.79/225 |
| 17 | 6.97 | `{'pb30_base': 0.118181, 'nonpb30_taker': 0.472722, 'oi_raw': 0.236361, 'rex_rule': 0.945445, 'short_premium_panic': 0.945445, 'new_long_funding_compression_premium': 1.518819, 'new_long_range_funding_premium': 0.706908, 'new_short_premium_kimchi_union': 2.028775}` | 641.63/638.59/11.94/53.49/464 | 234.98/235.26/12.10/19.44/392 | 100.36/425.41/9.53/44.63/207 |
| 18 | 6.97 | `{'pb30_base': 0.118181, 'nonpb30_taker': 0.472722, 'oi_raw': 0.236361, 'rex_rule': 0.945445, 'short_premium_panic': 0.945445, 'new_long_funding_compression_premium': 2.148223, 'new_long_range_funding_premium': 0.372207, 'new_short_fx_stress': 1.734073}` | 686.26/682.94/12.80/53.37/443 | 228.45/228.72/10.80/21.18/338 | 85.45/336.82/9.53/35.33/193 |
| 19 | 6.97 | `{'pb30_base': 0.170065, 'nonpb30_taker': 0.680259, 'oi_raw': 0.34013, 'rex_rule': 1.360518, 'short_premium_panic': 1.360518, 'new_long_funding_compression_premium': 1.294088, 'new_long_range_funding_premium': 0.519439, 'new_short_premium_kimchi_union': 1.247639}` | 679.37/676.10/12.68/53.33/464 | 249.28/249.58/9.26/26.94/392 | 93.35/382.60/9.67/39.55/207 |
| 20 | 6.97 | `{'pb30_base': 0.13156, 'nonpb30_taker': 0.526238, 'oi_raw': 0.263119, 'rex_rule': 1.052476, 'short_premium_panic': 1.052476, 'new_long_range_funding_premium': 1.886457, 'new_short_premium_kimchi_union': 0.827564, 'new_short_fx_stress': 0.65475, 'new_short_premium_panic': 0.578016}` | 526.88/524.53/9.87/53.15/491 | 262.17/262.49/8.02/32.74/434 | 73.23/271.23/9.84/27.57/218 |

## Robust diagnostic only (eval-influenced)

| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |
|---:|---:|---|---:|---:|---:|
| 1 | 6.97 | `{'nonpb30_taker': 1.175774, 'rex_rule': 1.469718, 'oi_wave_lowpos144': 0.293944, 'oi_high_sel': 0.293944, 'rex_dyn_short_exit': 0.587887, 'new_long_funding_compression_premium': 1.445086, 'new_long_range_funding_premium': 0.575778, 'new_short_premium_kimchi_union': 1.130527}` | 594.01/591.26/13.93/42.45/511 | 324.88/325.30/9.34/34.82/446 | 93.54/383.72/11.85/32.37/221 |
| 2 | 6.97 | `{'nonpb30_taker': 1.175774, 'rex_rule': 1.469718, 'oi_wave_lowpos144': 0.293944, 'oi_high_sel': 0.293944, 'rex_dyn_short_exit': 0.587887, 'new_long_funding_compression_premium': 1.541591, 'new_long_range_funding_premium': 0.928839, 'new_short_premium_kimchi_union': 0.680961}` | 660.34/657.19/13.93/47.18/511 | 329.09/329.52/10.16/32.43/446 | 93.10/381.12/11.85/32.16/221 |
| 3 | 6.97 | `{'nonpb30_taker': 1.175774, 'rex_rule': 1.469718, 'oi_wave_lowpos144': 0.293944, 'oi_high_sel': 0.293944, 'rex_dyn_short_exit': 0.587887, 'new_long_funding_compression_premium': 1.332229, 'new_long_range_funding_premium': 0.534749, 'new_short_premium_kimchi_union': 1.284412}` | 569.57/566.97/13.93/40.70/511 | 324.59/325.01/9.07/35.84/446 | 92.77/379.16/11.85/31.99/221 |
| 4 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.147751, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'oi_high_sel': 0.295503, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.435791, 'new_long_range_funding_premium': 0.572075, 'new_short_premium_kimchi_union': 1.123255}` | 610.55/607.70/13.89/43.74/708 | 324.33/324.75/9.37/34.66/570 | 92.32/376.46/11.91/31.60/266 |
| 5 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.147751, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'oi_high_sel': 0.295503, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.531676, 'new_long_range_funding_premium': 0.922865, 'new_short_premium_kimchi_union': 0.676581}` | 678.06/674.80/13.89/48.57/708 | 328.52/328.95/10.18/32.32/570 | 91.90/374.00/11.91/31.39/266 |
| 6 | 6.97 | `{'nonpb30_taker': 0.465501, 'rex_rule': 0.698252, 'oi_wave_lowpos144': 0.116375, 'oi_low': 0.232751, 'oi_high_sel': 0.465501, 'bear_rex_short': 0.581876, 'new_long_funding_compression_premium': 1.575187, 'new_long_range_funding_premium': 0.733144, 'new_short_premium_kimchi_union': 2.10407}` | 573.19/570.56/11.91/47.90/574 | 307.20/307.60/9.82/31.34/485 | 92.17/375.60/11.00/34.14/234 |
| 7 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.147751, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'oi_high_sel': 0.295503, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.323661, 'new_long_range_funding_premium': 0.53131, 'new_short_premium_kimchi_union': 1.276151}` | 585.67/582.97/13.89/41.96/708 | 324.04/324.46/9.09/35.67/570 | 91.56/371.98/11.91/31.22/266 |
| 8 | 6.97 | `{'nonpb30_taker': 1.175774, 'rex_rule': 1.469718, 'oi_wave_lowpos144': 0.293944, 'oi_high_sel': 0.293944, 'rex_dyn_short_exit': 0.587887, 'new_long_funding_compression_premium': 1.125018, 'new_long_range_funding_premium': 0.52362, 'new_short_premium_kimchi_union': 1.502752}` | 534.47/532.07/13.93/38.20/511 | 324.68/325.10/8.68/37.45/446 | 91.10/369.31/11.85/31.16/221 |
| 9 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.443254, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.435791, 'new_long_range_funding_premium': 0.572075, 'new_short_premium_kimchi_union': 1.123255}` | 609.44/606.60/13.89/43.66/516 | 324.45/324.87/9.45/34.38/450 | 90.37/365.04/11.91/30.64/222 |
| 10 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.443254, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.531676, 'new_long_range_funding_premium': 0.922865, 'new_short_premium_kimchi_union': 0.676581}` | 676.85/673.60/13.89/48.48/516 | 328.64/329.06/10.26/32.07/450 | 89.96/362.64/11.91/30.44/222 |
| 11 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.147751, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'oi_high_sel': 0.295503, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.117782, 'new_long_range_funding_premium': 0.520252, 'new_short_premium_kimchi_union': 1.493087}` | 549.93/547.44/13.89/39.40/708 | 324.13/324.55/8.71/37.26/570 | 89.91/362.32/11.91/30.41/266 |
| 12 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.443254, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.323661, 'new_long_range_funding_premium': 0.53131, 'new_short_premium_kimchi_union': 1.276151}` | 584.60/581.91/13.89/41.88/516 | 324.15/324.57/9.18/35.37/450 | 89.62/360.67/11.91/30.27/222 |
| 13 | 6.97 | `{'nonpb30_taker': 0.300147, 'oi_raw': 0.253971, 'rex_rule': 0.923531, 'oi_upbit_ratio288_low': 1.038972, 'bear_rex_short': 0.13853, 'oi_alt_ratio72_dyn_exit': 0.161618, 'new_long_funding_compression_premium': 1.483615, 'new_long_range_funding_premium': 0.690523, 'new_short_premium_kimchi_union': 1.98175}` | 611.48/608.62/11.66/52.19/613 | 297.41/297.79/9.95/29.91/465 | 83.24/324.50/8.73/37.16/236 |
| 14 | 6.97 | `{'nonpb30_taker': 0.520385, 'rex_rule': 0.780578, 'oi_wave_lowpos144': 0.130096, 'oi_low': 0.260193, 'oi_high_sel': 0.520385, 'bear_rex_short': 0.650482, 'new_long_funding_compression_premium': 1.762053, 'new_long_range_funding_premium': 1.018993, 'new_short_premium_kimchi_union': 1.32949}` | 664.30/661.13/12.88/51.33/574 | 325.79/326.22/10.92/29.88/485 | 90.27/364.42/12.05/30.24/234 |
| 15 | 6.97 | `{'pb30_base': 0.199219, 'nonpb30_taker': 0.796875, 'oi_raw': 0.398438, 'rex_rule': 1.59375, 'short_premium_panic': 1.59375, 'new_long_funding_compression_premium': 0.853432, 'new_long_range_funding_premium': 0.397215, 'new_short_premium_kimchi_union': 1.139978}` | 632.95/629.96/14.81/42.54/464 | 256.25/256.56/8.67/29.60/392 | 86.33/341.81/10.73/31.86/207 |
| 16 | 6.97 | `{'nonpb30_taker': 0.81571, 'oi_raw': 0.101964, 'rex_rule': 1.019638, 'oi_wave_lowpos144': 0.101964, 'oi_high_sel': 0.203928, 'rex_dyn_short_exit': 0.407855, 'new_long_funding_compression_premium': 1.542772, 'new_long_range_funding_premium': 0.718057, 'new_short_premium_kimchi_union': 2.06077}` | 578.92/576.26/12.16/47.40/708 | 282.99/283.34/9.58/29.59/570 | 99.14/417.80/10.15/41.18/266 |
| 17 | 6.97 | `{'nonpb30_taker': 0.809782, 'rex_rule': 1.012227, 'oi_wave_lowpos144': 0.202445, 'oi_high_sel': 0.202445, 'rex_dyn_short_exit': 0.404891, 'new_long_funding_compression_premium': 1.54965, 'new_long_range_funding_premium': 0.721258, 'new_short_premium_kimchi_union': 2.069958}` | 567.52/564.93/12.33/45.80/511 | 283.06/283.41/9.60/29.51/446 | 100.03/423.34/10.08/41.98/221 |
| 18 | 6.97 | `{'nonpb30_taker': 1.182011, 'oi_raw': 0.443254, 'rex_rule': 1.477513, 'oi_wave_lowpos144': 0.147751, 'rex_dyn_short_exit': 0.591005, 'new_long_funding_compression_premium': 1.117782, 'new_long_range_funding_premium': 0.520252, 'new_short_premium_kimchi_union': 1.493087}` | 548.92/546.44/13.89/39.33/516 | 324.24/324.66/8.79/36.93/450 | 87.99/351.24/11.91/29.48/222 |
| 19 | 6.97 | `{'nonpb30_taker': 0.820312, 'oi_raw': 0.820312, 'rex_rule': 1.640625, 'new_long_funding_compression_premium': 1.560517, 'new_long_range_funding_premium': 0.626382, 'new_short_premium_kimchi_union': 1.504506}` | 765.95/762.13/16.02/47.57/417 | 350.09/350.56/9.75/35.96/349 | 85.27/335.83/11.41/29.44/176 |
| 20 | 6.97 | `{'nonpb30_taker': 0.81571, 'oi_raw': 0.305891, 'rex_rule': 1.019638, 'oi_wave_lowpos144': 0.101964, 'rex_dyn_short_exit': 0.407855, 'new_long_funding_compression_premium': 1.542772, 'new_long_range_funding_premium': 0.718057, 'new_short_premium_kimchi_union': 2.06077}` | 578.20/575.55/12.16/47.34/516 | 283.06/283.42/9.63/29.43/450 | 97.75/409.23/10.15/40.33/222 |

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
- Use the 2024-selected table as the clean selection protocol. Treat the robust table as research direction only because it ranks using 2025/YTD outcomes.
- Legacy and new sleeve leverage semantics differ; do not deploy a combined row without a live-size normalization pass.
