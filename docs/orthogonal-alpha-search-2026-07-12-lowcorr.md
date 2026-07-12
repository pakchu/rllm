# Orthogonal alpha search (2026-07-12-lowcorr)

orthogonal alpha scan; thresholds fit on train<2024 only; test=2024/eval=2025/ytd2026 reported; 5bp per side; strict in-position MDD; candidate families avoid direct wave/volume sleeve reuse except generic vol regime; diagnostic not live-promoted

| rank | name | side | hold/stride | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades | terms |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | weekend_reversal_long | long | 24/24 | 0.34/20 | 0.60/21 | -0.71/18 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 2 | weekend_reversal_long | long | 48/24 | 0.15/17 | 1.17/19 | 3.43/14 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 3 | funding_premium_revert_long | long | 24/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 4 | funding_premium_revert_long | long | 24/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 5 | funding_premium_revert_long | long | 48/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 6 | funding_premium_revert_long | long | 48/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 7 | funding_premium_revert_long | long | 72/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 8 | funding_premium_revert_long | long | 72/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 9 | funding_premium_revert_long | long | 96/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 10 | funding_premium_revert_long | long | 96/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 11 | funding_premium_revert_long | long | 144/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 12 | funding_premium_revert_long | long | 144/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread <= -1.824; premium_index_zscore <= -0.9768` |
| 13 | funding_premium_revert_short | short | 24/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 14 | funding_premium_revert_short | short | 24/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 15 | funding_premium_revert_short | short | 48/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 16 | funding_premium_revert_short | short | 48/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 17 | funding_premium_revert_short | short | 72/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 18 | funding_premium_revert_short | short | 72/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 19 | funding_premium_revert_short | short | 96/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 20 | funding_premium_revert_short | short | 96/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 21 | funding_premium_revert_short | short | 144/12 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 22 | funding_premium_revert_short | short | 144/24 | 0.00/0 | 0.00/0 | 0.00/0 | `x_funding_premium_spread >= 1.803; premium_index_zscore >= 0.9797` |
| 23 | weekend_reversal_long | long | 144/24 | 0.05/16 | -0.01/17 | 2.80/12 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 24 | weekend_reversal_long | long | 72/24 | -0.14/17 | -0.14/19 | 4.96/14 | `x_weekend >= 0.5; x_ret_72 <= -0.008328; x_rvol_z_144 >= 0.6007` |
| 25 | kimchi_dxy_short | short | 72/24 | -0.13/26 | -0.15/34 | 4.83/10 | `x_kimchi_dxy_spread >= 2.17; dxy_momentum >= 0.0003078` |
