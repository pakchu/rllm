# VPIN + formulaic alpha scan (2026-07-09)

VPIN/order-flow-toxicity and 101 Formulaic Alphas inspired scan. Thresholds fit train<2024 only. Cost 6bp/side. Strict MDD includes intra-position adverse excursion. Diagnostic, not live-promoted.

## Source ideas
- 101 Formulaic Alphas: price/volume/VWAP/range transforms and short holding periods.
- VPIN/order-flow toxicity: rolling volume imbalance over volume, approximated from Binance taker buy quote volume.

| rank | name | side | hold/stride | train ratio/trades | 2024 ret/CAGR/MDD/ratio/trades | 2025 ret/CAGR/MDD/ratio/trades | 2026 ret/CAGR/MDD/ratio/trades | terms |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | vpin_high_sell_pressure_flip | long | 144/24 | -0.21/428 | 21.99/21.94/11.29/1.94/114 | 3.19/3.19/13.27/0.24/119 | -15.60/-28.35/23.80/-1.19/60 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 2 | vpin_high_sell_pressure_flip | long | 144/12 | -0.22/538 | 19.51/19.47/12.68/1.54/134 | -0.09/-0.09/17.20/-0.01/136 | -26.87/-45.94/33.66/-1.36/68 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 3 | vpin_high_sell_pressure_flip | long | 72/12 | -0.11/592 | 15.26/15.23/10.30/1.48/156 | -0.56/-0.56/11.35/-0.05/149 | -14.40/-26.34/17.94/-1.47/76 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 4 | vpin_high_sell_pressure_flip | long | 96/12 | -0.11/571 | 16.16/16.13/11.20/1.44/144 | -3.98/-3.98/14.13/-0.28/142 | -17.44/-31.40/22.41/-1.40/73 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 5 | volume_delta_pressure_short_flip | long | 144/24 | -0.13/845 | -10.19/-10.17/32.01/-0.32/222 | 5.58/5.58/21.56/0.26/219 | -30.14/-50.60/35.69/-1.42/115 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank <= 0.4757` |
| 6 | vpin_high_sell_pressure_flip | long | 72/6 | -0.25/705 | 0.28/0.28/12.94/0.02/183 | -4.80/-4.80/14.90/-0.32/166 | -7.90/-14.93/13.40/-1.11/89 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 7 | vpin_high_sell_pressure_flip | long | 72/24 | -0.14/462 | 15.13/15.10/8.59/1.76/124 | -5.79/-5.80/12.98/-0.45/124 | -6.99/-13.28/12.73/-1.04/63 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 8 | ret_vol_corr_reversal_long_flip | short | 144/6 | -0.68/1244 | -13.02/-12.99/28.32/-0.46/278 | -10.86/-10.86/26.95/-0.40/257 | -2.52/-4.89/16.81/-0.29/149 | `fq_ret_vol_corr_72 <= -0.3005; vp_ret_rank_72 <= 0.3165` |
| 9 | vpin_high_sell_pressure_flip | long | 144/6 | -0.29/616 | -10.06/-10.04/22.67/-0.44/155 | -12.43/-12.44/25.16/-0.49/144 | -20.17/-35.78/30.56/-1.17/79 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 10 | ret_vol_corr_reversal_short_flip | long | 72/6 | -0.44/1460 | -16.65/-16.62/33.12/-0.50/326 | -7.30/-7.31/25.16/-0.29/286 | -16.84/-30.40/24.57/-1.24/187 | `fq_ret_vol_corr_72 >= 0.3013; vp_ret_rank_72 >= 0.68` |
| 11 | vpin_toxic_rally_cont | long | 96/24 | -0.07/1232 | -14.31/-14.29/27.01/-0.53/314 | -5.92/-5.93/27.22/-0.22/295 | -27.16/-46.37/30.72/-1.51/143 | `vx_toxic_rally >= 2.925` |
| 12 | vpin_toxic_rally_revert_flip | long | 96/24 | -0.07/1232 | -14.31/-14.29/27.01/-0.53/314 | -5.92/-5.93/27.22/-0.22/295 | -27.16/-46.37/30.72/-1.51/143 | `vx_toxic_rally >= 2.925` |
| 13 | volume_delta_pressure_short_flip | long | 96/24 | -0.24/953 | -11.69/-11.67/23.76/-0.49/245 | -15.01/-15.02/28.22/-0.53/247 | -25.89/-44.51/31.36/-1.42/126 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank <= 0.4757` |
| 14 | ret_vol_corr_reversal_long | long | 96/24 | 0.16/975 | -24.06/-24.02/44.92/-0.53/217 | -8.25/-8.25/17.83/-0.46/198 | -8.54/-16.10/20.29/-0.79/122 | `fq_ret_vol_corr_72 <= -0.3005; vp_ret_rank_72 <= 0.3165` |
| 15 | vwap_revert_short_flip | long | 144/24 | -0.34/1159 | 48.51/48.39/24.28/1.99/311 | -23.38/-23.39/42.64/-0.55/308 | -15.57/-28.31/21.94/-1.29/149 | `fq_vwap_revert_z <= -0.8658; vp_vpin_z_144 <= 0.7063` |
| 16 | vpin_high_sell_pressure_flip | long | 96/6 | -0.22/662 | 17.32/17.29/14.15/1.22/172 | -11.25/-11.26/19.77/-0.57/153 | -14.10/-25.83/19.92/-1.30/84 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 17 | volume_delta_pressure_short_flip | long | 144/12 | -0.21/1387 | 7.46/7.45/37.97/0.20/329 | -19.97/-19.99/35.07/-0.57/339 | -46.63/-70.90/52.04/-1.36/161 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank <= 0.4757` |
| 18 | ret_vol_corr_reversal_short_flip | long | 72/24 | -0.38/1011 | -12.94/-12.91/27.34/-0.47/231 | -12.64/-12.64/21.85/-0.58/196 | -7.28/-13.81/12.11/-1.14/131 | `fq_ret_vol_corr_72 >= 0.3013; vp_ret_rank_72 >= 0.68` |
| 19 | vpin_high_buy_pressure_flip | short | 96/12 | -0.27/530 | -9.12/-9.10/14.83/-0.61/134 | -7.30/-7.31/17.67/-0.41/118 | 3.15/6.29/9.71/0.65/62 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 >= 1.03` |
| 20 | vpin_high_sell_pressure_flip | long | 36/24 | -0.31/565 | -6.23/-6.22/10.30/-0.60/156 | -10.10/-10.10/15.77/-0.64/154 | -9.49/-17.79/13.41/-1.33/75 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 <= -1.034` |
| 21 | vpin_high_buy_pressure_flip | short | 96/6 | -0.34/638 | -17.09/-17.05/26.03/-0.66/157 | -7.08/-7.08/23.98/-0.30/136 | -9.20/-17.28/15.27/-1.13/73 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 >= 1.03` |
| 22 | vpin_high_buy_pressure_flip | short | 144/6 | -0.46/574 | -20.21/-20.17/30.67/-0.66/142 | 2.41/2.41/22.32/0.11/128 | -4.80/-9.22/11.67/-0.79/67 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 >= 1.03` |
| 23 | volume_delta_pressure_long_flip | short | 96/24 | -0.42/963 | -19.04/-19.00/28.85/-0.66/238 | -13.80/-13.81/22.87/-0.60/240 | -38.89/-62.02/39.21/-1.58/115 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank >= 0.527` |
| 24 | vwap_revert_short_flip | long | 96/24 | -0.37/1355 | 8.80/8.78/30.15/0.29/362 | -24.93/-24.94/36.38/-0.69/357 | -35.48/-57.74/41.84/-1.38/172 | `fq_vwap_revert_z <= -0.8658; vp_vpin_z_144 <= 0.7063` |
| 25 | volume_delta_pressure_short_flip | long | 144/6 | -0.09/1733 | 27.84/27.78/41.65/0.67/431 | -30.32/-30.34/44.12/-0.69/442 | -44.15/-68.19/51.07/-1.34/213 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank <= 0.4757` |
| 26 | volume_delta_pressure_long_flip | short | 144/24 | -0.39/856 | -28.08/-28.03/40.35/-0.69/214 | -13.99/-14.00/26.03/-0.54/216 | -27.29/-46.55/32.04/-1.45/105 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank >= 0.527` |
| 27 | volume_delta_pressure_long | long | 144/12 | -0.27/1392 | -18.02/-17.99/33.97/-0.53/330 | -28.38/-28.40/40.54/-0.70/334 | 0.01/0.02/21.94/0.00/159 | `fq_volume_delta_rank >= 0.6089; fq_signed_vol_pressure_rank >= 0.527` |
| 28 | lowtox_formulaic_momo_short_flip | long | 144/6 | -0.37/1559 | -7.97/-7.95/50.64/-0.16/393 | -32.67/-32.69/46.25/-0.71/416 | -26.53/-45.46/36.49/-1.25/207 | `vx_lowtox_momo_short >= 2.193` |
| 29 | vpin_high_buy_pressure_flip | short | 144/12 | -0.45/495 | -19.66/-19.62/27.62/-0.71/124 | 3.68/3.68/15.86/0.23/114 | 0.50/0.98/10.69/0.09/56 | `vp_vpin_z_144 >= 1.137; vp_imb_z_72 >= 1.03` |
| 30 | ret_vol_corr_reversal_short_flip | long | 96/24 | -0.30/975 | -4.24/-4.23/24.03/-0.18/219 | -19.31/-19.32/26.93/-0.72/190 | -1.75/-3.42/11.51/-0.30/126 | `fq_ret_vol_corr_72 >= 0.3013; vp_ret_rank_72 >= 0.68` |
