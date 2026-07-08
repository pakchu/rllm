# Orthogonal alpha search (2026-07-08)

orthogonal alpha scan; thresholds fit on train<2024 only; test=2024/eval=2025/ytd2026 reported; 5bp per side; strict in-position MDD; candidate families avoid direct wave/volume sleeve reuse except generic vol regime; diagnostic not live-promoted

## Nonzero-trade candidates

| rank | name | side | hold/stride | 2024 ret/CAGR/MDD/ratio/trades | 2025 ret/CAGR/MDD/ratio/trades | 2026 ret/CAGR/MDD/ratio/trades | terms |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | weekend_reversal_long | long | 24/24 | 2.04/2.03/4.56/0.45/20 | 3.16/3.16/4.52/0.70/21 | -0.57/-1.35/2.93/-0.46/18 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 2 | weekend_reversal_long | long | 48/24 | 1.52/1.52/7.69/0.20/17 | 5.72/5.72/4.52/1.27/19 | 3.31/8.09/2.11/3.83/14 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 3 | weekend_reversal_long | long | 144/24 | 1.24/1.24/17.14/0.07/16 | 0.28/0.28/9.09/0.03/17 | 4.41/10.86/3.62/3.00/12 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 4 | kimchi_dxy_short | short | 72/24 | -0.43/-0.43/6.93/-0.06/26 | -0.77/-0.77/9.14/-0.08/34 | 5.61/13.92/2.73/5.10/10 | `x_kimchi_dxy_spread >= 2.17; dxy_momentum >= 0.0003078` |
| 5 | weekend_reversal_long | long | 72/24 | -1.28/-1.28/11.48/-0.11/17 | -0.95/-0.95/9.23/-0.10/19 | 5.46/13.53/2.52/5.36/14 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 6 | weekend_reversal_long | long | 72/6 | -0.89/-0.89/7.70/-0.12/25 | 1.81/1.81/8.61/0.21/27 | 3.61/8.84/5.32/1.66/18 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 7 | weekend_reversal_short | short | 24/24 | -0.55/-0.55/3.49/-0.16/21 | 1.74/1.74/5.36/0.32/15 | -5.00/-11.52/5.91/-1.95/10 | `x_weekend >= 0.5; x_ret_72 >= 0.009121; x_rvol_z_144 >= 0.6007` |
| 8 | vol_compress_break_long | long | 144/24 | 13.85/13.82/9.91/1.39/95 | -1.84/-1.84/10.86/-0.17/85 | -30.56/-58.13/31.73/-1.83/41 | `x_rvol_z_288 <= -1.19; x_ret_24 >= 0.002721` |
| 9 | weekend_reversal_long | long | 96/24 | -3.32/-3.31/13.18/-0.25/16 | -0.12/-0.12/8.39/-0.01/17 | 3.44/8.41/2.83/2.97/13 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 10 | weekend_reversal_long | long | 48/12 | -2.43/-2.43/8.48/-0.29/23 | 6.46/6.47/5.29/1.22/25 | 2.98/7.26/4.76/1.53/21 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 11 | kimchi_dxy_long | long | 72/12 | -2.36/-2.36/11.07/-0.21/35 | -2.84/-2.84/9.22/-0.31/39 | 10.55/27.06/3.91/6.92/21 | `x_kimchi_dxy_spread <= -2.204; dxy_momentum <= -0.0003049` |
| 12 | kimchi_dxy_short | short | 96/24 | -2.59/-2.58/8.60/-0.30/26 | -4.03/-4.03/12.27/-0.33/34 | 6.51/16.26/3.24/5.02/10 | `x_kimchi_dxy_spread >= 2.17; dxy_momentum >= 0.0003078` |
| 13 | weekend_reversal_long | long | 48/6 | -1.39/-1.39/7.29/-0.19/27 | -3.75/-3.75/11.42/-0.33/31 | -1.52/-3.59/5.90/-0.61/23 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 14 | weekend_reversal_long | long | 144/12 | -6.18/-6.17/17.94/-0.34/20 | -3.57/-3.57/12.02/-0.30/22 | 8.24/20.82/4.76/4.37/16 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 15 | kimchi_dxy_short | short | 48/24 | -0.58/-0.58/4.72/-0.12/26 | -4.09/-4.09/10.99/-0.37/34 | 4.88/12.06/2.68/4.50/10 | `x_kimchi_dxy_spread >= 2.17; dxy_momentum >= 0.0003078` |
| 16 | kimchi_dxy_short | short | 144/24 | -5.47/-5.46/13.37/-0.41/26 | -6.36/-6.37/16.00/-0.40/33 | 12.04/31.19/2.42/12.88/10 | `x_kimchi_dxy_spread >= 2.17; dxy_momentum >= 0.0003078` |
| 17 | session_asia_fade_short | short | 144/24 | -6.91/-6.90/18.50/-0.37/75 | -11.73/-11.74/28.21/-0.42/79 | -1.89/-4.45/13.09/-0.34/25 | `x_asia >= 0.5; x_ret_24 >= 0.00496; taker_imbalance <= -0.06087` |
| 18 | funding_premium_revert_short | short | 96/6 | -2.52/-2.52/7.21/-0.35/4 | -1.69/-1.69/4.02/-0.42/7 | 0.43/1.03/2.91/0.35/6 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 19 | weekend_reversal_long | long | 72/12 | -5.56/-5.55/12.95/-0.43/21 | 1.92/1.92/8.77/0.22/24 | 3.27/7.99/4.81/1.66/18 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 20 | funding_premium_revert_short | short | 144/6 | -3.52/-3.52/8.15/-0.43/4 | -0.36/-0.37/3.48/-0.10/7 | 2.16/5.24/2.42/2.16/6 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 21 | down_absorb_long | long | 96/24 | -12.94/-12.92/36.54/-0.35/327 | -12.02/-12.03/25.00/-0.48/307 | -15.01/-32.18/29.41/-1.09/148 | `x_down_vol_absorb >= 9.823e-05; x_ret_72 <= -0.004249` |
| 22 | weekend_reversal_long | long | 96/12 | -6.59/-6.58/13.56/-0.49/21 | -5.28/-5.28/11.59/-0.46/23 | 0.94/2.26/4.86/0.46/17 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 23 | vol_compress_break_long | long | 144/12 | -3.06/-3.05/17.60/-0.17/118 | -5.13/-5.14/10.58/-0.49/107 | -22.75/-45.99/24.90/-1.85/47 | `x_rvol_z_288 <= -1.19; x_ret_24 >= 0.002721` |
| 24 | weekend_reversal_short | short | 24/12 | -2.80/-2.79/6.20/-0.45/27 | -4.89/-4.89/10.00/-0.49/24 | -5.88/-13.46/7.11/-1.89/13 | `x_weekend >= 0.5; x_ret_72 >= 0.009121; x_rvol_z_144 >= 0.6007` |
| 25 | kimchi_dxy_short | short | 24/24 | -1.79/-1.78/4.36/-0.41/27 | -3.93/-3.93/8.03/-0.49/34 | 2.59/6.29/2.66/2.36/10 | `x_kimchi_dxy_spread >= 2.17; dxy_momentum >= 0.0003078` |
