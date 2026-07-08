# Gross<=20 step 0.05 cost 6bp MDD20 portfolio scan with dynamic sleeves (2026-07-08)

{"gross_cap": 20.0, "cost_each_side": 0.0006, "condition": "test2024/eval2025/ytd2026 strict_mdd_pct<=20 and CAGR/MDD>=5; nonzero weights >=0.10 and multiples of 0.05", "rank": "among qualified: OOS split return sum, then min ratio/ytd ratio", "selection_scope": "quantized seeded/local/random scan over pre-wave robust/live sleeves + wave/volume sleeves + dynamic-exit sleeves; gross<=20"}

evaluated_unique=5865, qualified_count=542


## Reference: e39 robust 5.75 train-CAGR combo

Weights: `nonpb30_taker=0.5`, `oi_high_sel=0.5`, `bear_rex_short=1.0`, gross `2.0`.

| split | ret | CAGR | MDD | ratio | trades | sharpe |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 43.25% | 43.14% | 7.79% | 5.53 | 240 | 2.75 |
| 2025 | 67.84% | 67.90% | 4.07% | 16.67 | 181 | 4.03 |
| 2026 | 14.66% | 38.62% | 6.10% | 6.33 | 81 | 1.25 |

The top quantized candidates below use 6bp/side and enforce nonzero weight >=0.10 in 0.05 increments.

## Top qualified
| rank | gross | weights | 2024 ret/CAGR/MDD/ratio | 2025 ret/CAGR/MDD/ratio | 2026 ret/CAGR/MDD/ratio | trades 24/25/26 |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6.10 | `{'nonpb30_taker': 0.65, 'oi_raw': 0.55, 'rex_rule': 2.0, 'oi_upbit_ratio288_low': 2.25, 'bear_rex_short': 0.3, 'oi_alt_ratio72_dyn_exit': 0.35}` | 390.71/389.11/19.71/19.74 | 355.20/355.68/13.19/26.96 | 30.96/90.40/16.58/5.45 | 489/308/143 |
| 2 | 5.60 | `{'nonpb30_taker': 0.8, 'rex_rule': 1.9, 'oi_upbit_ratio288_low': 2.9}` | 330.51/329.23/19.69/16.72 | 304.53/304.91/14.16/21.53 | 29.78/86.33/17.10/5.05 | 175/106/52 |
| 3 | 5.55 | `{'pb30_base': 0.6, 'nonpb30_taker': 1.0, 'oi_raw': 0.2, 'rex_rule': 1.8, 'oi_upbit_ratio288_low': 1.2, 'bear_rex_short': 0.75}` | 280.65/279.60/19.79/14.13 | 335.27/335.70/11.01/30.48 | 44.97/142.68/19.68/7.25 | 406/276/137 |
| 4 | 4.80 | `{'pb30_base': 0.4, 'nonpb30_taker': 1.05, 'rex_rule': 2.0, 'oi_wave_lowpos144': 0.15, 'oi_high_sel': 1.2}` | 348.61/347.23/19.78/17.56 | 278.23/278.58/9.57/29.10 | 33.89/100.73/19.39/5.19 | 392/282/118 |
| 5 | 6.85 | `{'nonpb30_taker': 1.3, 'rex_rule': 0.85, 'oi_upbit_ratio288_low': 0.75, 'oi_high_sel': 1.55, 'rex_dyn_short_exit': 2.4}` | 332.32/331.02/19.87/16.66 | 282.59/282.94/13.93/20.32 | 45.83/146.10/19.01/7.69 | 382/253/123 |
| 6 | 5.80 | `{'nonpb30_taker': 1.05, 'rex_rule': 1.4, 'oi_high_sel': 1.35, 'bear_rex_short': 1.0, 'oi_alt_ratio72_dyn_exit': 1.0}` | 314.21/313.00/19.87/15.75 | 293.61/293.98/14.48/20.30 | 39.15/120.04/18.15/6.61 | 405/266/128 |
| 7 | 5.15 | `{'nonpb30_taker': 1.6, 'rex_rule': 1.5, 'oi_high_sel': 1.3, 'bear_rex_short': 0.1, 'oi_alt_ratio72_dyn_exit': 0.65}` | 345.37/344.01/18.31/18.79 | 245.75/246.04/11.42/21.55 | 33.72/100.10/19.59/5.11 | 405/266/128 |
| 8 | 5.35 | `{'nonpb30_taker': 1.7, 'rex_rule': 1.9, 'oi_upbit_ratio288_low': 1.75}` | 300.68/299.54/18.24/16.42 | 280.49/280.83/11.78/23.84 | 38.72/118.42/19.67/6.02 | 175/106/52 |
| 9 | 5.35 | `{'pb30_base': 0.1, 'nonpb30_taker': 1.65, 'rex_rule': 2.0, 'oi_upbit_ratio288_low': 1.05, 'oi_alt_ratio72_dyn_exit': 0.55}` | 296.14/295.02/18.62/15.84 | 267.78/268.10/9.17/29.24 | 44.84/142.16/19.59/7.26 | 298/178/92 |
| 10 | 5.50 | `{'nonpb30_taker': 1.7, 'oi_raw': 0.65, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.2, 'rex_dyn_short_exit': 0.85}` | 285.60/284.54/19.62/14.50 | 276.52/276.86/10.85/25.51 | 45.20/143.60/19.57/7.34 | 392/293/129 |
| 11 | 5.50 | `{'pb30_base': 0.2, 'nonpb30_taker': 0.65, 'rex_rule': 1.25, 'oi_high_sel': 1.95, 'rex_dyn_short_exit': 1.45}` | 324.64/323.39/19.70/16.42 | 245.25/245.54/12.19/20.14 | 33.52/99.40/15.26/6.51 | 323/235/126 |
| 12 | 5.65 | `{'pb30_base': 0.65, 'rex_rule': 1.55, 'oi_low': 0.8, 'oi_high_sel': 0.75, 'rex_dyn_short_exit': 1.9}` | 275.51/274.50/18.98/14.46 | 276.70/277.04/12.62/21.95 | 45.57/145.06/19.49/7.44 | 353/240/129 |
| 13 | 5.45 | `{'nonpb30_taker': 1.7, 'oi_raw': 0.2, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.2, 'oi_high_sel': 0.4, 'rex_dyn_short_exit': 0.85}` | 278.58/277.55/19.53/14.21 | 271.57/271.90/10.94/24.85 | 47.62/153.40/17.56/8.73 | 584/413/173 |
| 14 | 5.40 | `{'rex_rule': 1.75, 'oi_upbit_ratio288_low': 1.6, 'oi_low': 1.15, 'rex_dyn_short_exit': 0.9}` | 284.75/283.69/19.80/14.33 | 282.83/283.18/11.89/23.82 | 29.96/86.92/16.51/5.26 | 220/138/82 |
| 15 | 5.70 | `{'pb30_base': 0.65, 'rex_rule': 1.1, 'oi_upbit_ratio288_low': 1.0, 'oi_high_sel': 0.95, 'bear_rex_short': 1.55, 'rex_dyn_short_exit': 0.45}` | 231.09/230.27/19.60/11.75 | 327.39/327.81/11.67/28.10 | 39.00/119.51/16.03/7.45 | 382/264/149 |
| 16 | 5.15 | `{'nonpb30_taker': 1.85, 'rex_rule': 1.9, 'oi_wave_lowpos144': 0.5, 'oi_upbit_ratio288_low': 0.9}` | 281.63/280.59/17.61/15.94 | 275.82/276.16/9.34/29.58 | 38.58/117.90/19.72/5.98 | 259/180/71 |
| 17 | 6.90 | `{'pb30_base': 0.75, 'rex_rule': 0.4, 'oi_wave_lowpos144': 0.8, 'oi_upbit_ratio288_low': 2.15, 'bear_rex_short': 0.75, 'rex_dyn_short_exit': 2.05}` | 216.67/215.93/19.57/11.04 | 329.44/329.87/15.94/20.70 | 44.10/139.20/18.41/7.56 | 274/218/124 |
| 18 | 5.45 | `{'pb30_base': 0.65, 'nonpb30_taker': 0.15, 'rex_rule': 1.45, 'oi_high_sel': 1.55, 'rex_dyn_short_exit': 1.65}` | 295.76/294.64/19.82/14.86 | 253.78/254.09/12.29/20.68 | 40.50/125.18/16.93/7.40 | 323/235/126 |
| 19 | 5.45 | `{'nonpb30_taker': 1.7, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.4, 'oi_high_sel': 0.4, 'rex_dyn_short_exit': 0.85}` | 266.20/265.23/19.70/13.46 | 273.18/273.51/10.94/24.99 | 48.70/157.85/16.86/9.36 | 387/289/128 |
| 20 | 5.20 | `{'nonpb30_taker': 0.95, 'rex_rule': 1.4, 'oi_wave_lowpos144': 0.25, 'oi_low': 0.45, 'oi_high_sel': 0.95, 'bear_rex_short': 1.2}` | 251.75/250.85/19.78/12.68 | 302.54/302.92/10.51/28.81 | 33.55/99.51/16.43/6.06 | 450/328/141 |

## Top diagnostic if no/limited qualified
| rank | pass | gross | weights | 2024 ret/CAGR/MDD/ratio | 2025 ret/CAGR/MDD/ratio | 2026 ret/CAGR/MDD/ratio |
|---:|---:|---:|---|---:|---:|---:|
| 1 | True | 6.10 | `{'nonpb30_taker': 0.65, 'oi_raw': 0.55, 'rex_rule': 2.0, 'oi_upbit_ratio288_low': 2.25, 'bear_rex_short': 0.3, 'oi_alt_ratio72_dyn_exit': 0.35}` | 390.71/389.11/19.71/19.74 | 355.20/355.68/13.19/26.96 | 30.96/90.40/16.58/5.45 |
| 2 | True | 5.60 | `{'nonpb30_taker': 0.8, 'rex_rule': 1.9, 'oi_upbit_ratio288_low': 2.9}` | 330.51/329.23/19.69/16.72 | 304.53/304.91/14.16/21.53 | 29.78/86.33/17.10/5.05 |
| 3 | True | 5.55 | `{'pb30_base': 0.6, 'nonpb30_taker': 1.0, 'oi_raw': 0.2, 'rex_rule': 1.8, 'oi_upbit_ratio288_low': 1.2, 'bear_rex_short': 0.75}` | 280.65/279.60/19.79/14.13 | 335.27/335.70/11.01/30.48 | 44.97/142.68/19.68/7.25 |
| 4 | True | 4.80 | `{'pb30_base': 0.4, 'nonpb30_taker': 1.05, 'rex_rule': 2.0, 'oi_wave_lowpos144': 0.15, 'oi_high_sel': 1.2}` | 348.61/347.23/19.78/17.56 | 278.23/278.58/9.57/29.10 | 33.89/100.73/19.39/5.19 |
| 5 | True | 6.85 | `{'nonpb30_taker': 1.3, 'rex_rule': 0.85, 'oi_upbit_ratio288_low': 0.75, 'oi_high_sel': 1.55, 'rex_dyn_short_exit': 2.4}` | 332.32/331.02/19.87/16.66 | 282.59/282.94/13.93/20.32 | 45.83/146.10/19.01/7.69 |
| 6 | True | 5.80 | `{'nonpb30_taker': 1.05, 'rex_rule': 1.4, 'oi_high_sel': 1.35, 'bear_rex_short': 1.0, 'oi_alt_ratio72_dyn_exit': 1.0}` | 314.21/313.00/19.87/15.75 | 293.61/293.98/14.48/20.30 | 39.15/120.04/18.15/6.61 |
| 7 | True | 5.15 | `{'nonpb30_taker': 1.6, 'rex_rule': 1.5, 'oi_high_sel': 1.3, 'bear_rex_short': 0.1, 'oi_alt_ratio72_dyn_exit': 0.65}` | 345.37/344.01/18.31/18.79 | 245.75/246.04/11.42/21.55 | 33.72/100.10/19.59/5.11 |
| 8 | True | 5.35 | `{'nonpb30_taker': 1.7, 'rex_rule': 1.9, 'oi_upbit_ratio288_low': 1.75}` | 300.68/299.54/18.24/16.42 | 280.49/280.83/11.78/23.84 | 38.72/118.42/19.67/6.02 |
| 9 | True | 5.35 | `{'pb30_base': 0.1, 'nonpb30_taker': 1.65, 'rex_rule': 2.0, 'oi_upbit_ratio288_low': 1.05, 'oi_alt_ratio72_dyn_exit': 0.55}` | 296.14/295.02/18.62/15.84 | 267.78/268.10/9.17/29.24 | 44.84/142.16/19.59/7.26 |
| 10 | True | 5.50 | `{'nonpb30_taker': 1.7, 'oi_raw': 0.65, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.2, 'rex_dyn_short_exit': 0.85}` | 285.60/284.54/19.62/14.50 | 276.52/276.86/10.85/25.51 | 45.20/143.60/19.57/7.34 |
| 11 | True | 5.50 | `{'pb30_base': 0.2, 'nonpb30_taker': 0.65, 'rex_rule': 1.25, 'oi_high_sel': 1.95, 'rex_dyn_short_exit': 1.45}` | 324.64/323.39/19.70/16.42 | 245.25/245.54/12.19/20.14 | 33.52/99.40/15.26/6.51 |
| 12 | True | 5.65 | `{'pb30_base': 0.65, 'rex_rule': 1.55, 'oi_low': 0.8, 'oi_high_sel': 0.75, 'rex_dyn_short_exit': 1.9}` | 275.51/274.50/18.98/14.46 | 276.70/277.04/12.62/21.95 | 45.57/145.06/19.49/7.44 |
| 13 | True | 5.45 | `{'nonpb30_taker': 1.7, 'oi_raw': 0.2, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.2, 'oi_high_sel': 0.4, 'rex_dyn_short_exit': 0.85}` | 278.58/277.55/19.53/14.21 | 271.57/271.90/10.94/24.85 | 47.62/153.40/17.56/8.73 |
| 14 | True | 5.40 | `{'rex_rule': 1.75, 'oi_upbit_ratio288_low': 1.6, 'oi_low': 1.15, 'rex_dyn_short_exit': 0.9}` | 284.75/283.69/19.80/14.33 | 282.83/283.18/11.89/23.82 | 29.96/86.92/16.51/5.26 |
| 15 | True | 5.70 | `{'pb30_base': 0.65, 'rex_rule': 1.1, 'oi_upbit_ratio288_low': 1.0, 'oi_high_sel': 0.95, 'bear_rex_short': 1.55, 'rex_dyn_short_exit': 0.45}` | 231.09/230.27/19.60/11.75 | 327.39/327.81/11.67/28.10 | 39.00/119.51/16.03/7.45 |
| 16 | True | 5.15 | `{'nonpb30_taker': 1.85, 'rex_rule': 1.9, 'oi_wave_lowpos144': 0.5, 'oi_upbit_ratio288_low': 0.9}` | 281.63/280.59/17.61/15.94 | 275.82/276.16/9.34/29.58 | 38.58/117.90/19.72/5.98 |
| 17 | True | 6.90 | `{'pb30_base': 0.75, 'rex_rule': 0.4, 'oi_wave_lowpos144': 0.8, 'oi_upbit_ratio288_low': 2.15, 'bear_rex_short': 0.75, 'rex_dyn_short_exit': 2.05}` | 216.67/215.93/19.57/11.04 | 329.44/329.87/15.94/20.70 | 44.10/139.20/18.41/7.56 |
| 18 | True | 5.45 | `{'pb30_base': 0.65, 'nonpb30_taker': 0.15, 'rex_rule': 1.45, 'oi_high_sel': 1.55, 'rex_dyn_short_exit': 1.65}` | 295.76/294.64/19.82/14.86 | 253.78/254.09/12.29/20.68 | 40.50/125.18/16.93/7.40 |
| 19 | True | 5.45 | `{'nonpb30_taker': 1.7, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.4, 'oi_high_sel': 0.4, 'rex_dyn_short_exit': 0.85}` | 266.20/265.23/19.70/13.46 | 273.18/273.51/10.94/24.99 | 48.70/157.85/16.86/9.36 |
| 20 | True | 5.20 | `{'nonpb30_taker': 0.95, 'rex_rule': 1.4, 'oi_wave_lowpos144': 0.25, 'oi_low': 0.45, 'oi_high_sel': 0.95, 'bear_rex_short': 1.2}` | 251.75/250.85/19.78/12.68 | 302.54/302.92/10.51/28.81 | 33.55/99.51/16.43/6.06 |
