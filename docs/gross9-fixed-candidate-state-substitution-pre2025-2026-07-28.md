# Gross9 fixed-candidate and state-substitution pre-2025 battery

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

- evaluated cells: 32
- passing cells: 0
- decision: **reject_battery**
- 2025, 2026, and July metrics are absent and cannot rerank this artifact.

| candidate | mode | changed weight | pass | train | 2024 | min ratio improvement | max accepted entry Jaccard |
|---|---|---:|:---:|---:|---:|---:|---:|
| semimarkov_top10_strict_majority_long | state_substitution | 0.25 | N | 2407.67/163.00/36.44/4.47/976 | 212.03/211.30/16.48/12.82/221 | -0.254 | 0.0095 |
| nonpb30_taker | addition | 0.25 | N | 2474.07/165.07/37.88/4.36/1103 | 231.89/231.08/16.93/13.65/237 | -0.287 | 0.0103 |
| oi_divergence_highfreq | addition | 0.25 | N | 2384.94/162.28/37.92/4.28/1873 | 240.89/240.03/15.83/15.16/400 | -0.366 | 0.0158 |
| bocpd_top10_strict_majority_long | state_substitution | 0.25 | N | 2729.43/172.70/35.11/4.92/971 | 210.16/209.44/16.55/12.65/221 | -0.425 | 0.0072 |
| kalman_top10_strict_majority_long | state_substitution | 0.25 | N | 2668.28/170.92/36.38/4.70/943 | 206.43/205.73/16.27/12.64/212 | -0.437 | 0.0051 |
| nonpb30_taker | addition | 0.50 | N | 2390.41/162.45/39.79/4.08/1103 | 241.53/240.67/16.90/14.24/237 | -0.629 | 0.0103 |
| semimarkov_top10_strict_majority_long | state_substitution | 0.50 | N | 2263.60/158.37/36.30/4.36/976 | 201.83/201.15/16.23/12.39/221 | -0.688 | 0.0095 |
| oi_divergence_highfreq | addition | 0.50 | N | 2214.88/156.76/39.25/3.99/1873 | 260.13/259.19/14.81/17.50/400 | -0.718 | 0.0158 |
| bocpd_top10_strict_majority_long | state_substitution | 0.50 | N | 2904.87/177.67/33.63/5.28/971 | 198.17/197.50/16.15/12.23/221 | -0.852 | 0.0072 |
| kalman_top10_strict_majority_long | state_substitution | 0.50 | N | 2777.54/174.09/36.19/4.81/943 | 191.05/190.41/15.59/12.21/212 | -0.867 | 0.0051 |
| nonpb30_taker | addition | 0.75 | N | 2302.82/159.65/41.96/3.80/1103 | 251.38/250.47/16.88/14.84/237 | -0.974 | 0.0103 |
| oi_divergence_highfreq | addition | 0.75 | N | 2044.89/150.95/40.56/3.72/1873 | 280.23/279.19/13.89/20.10/400 | -1.057 | 0.0158 |
| semimarkov_top10_strict_majority_long | state_substitution | 0.75 | N | 2121.68/153.61/36.38/4.22/976 | 191.88/191.24/16.06/11.91/221 | -1.173 | 0.0095 |
| nonpb30_taker | addition | 1.00 | N | 2211.90/156.66/44.07/3.55/1103 | 261.45/260.50/16.85/15.46/237 | -1.292 | 0.0103 |
| kalman_top10_strict_majority_long | state_substitution | 0.75 | N | 2879.91/176.98/36.01/4.91/943 | 176.31/175.73/14.91/11.79/212 | -1.293 | 0.0051 |
| oi_divergence_highfreq | addition | 1.00 | N | 1876.73/144.87/42.04/3.45/1873 | 301.21/300.07/14.48/20.72/400 | -1.400 | 0.0158 |
| bocpd_top10_strict_majority_long | state_substitution | 0.75 | N | 3077.94/182.38/34.21/5.33/971 | 186.51/185.89/16.26/11.43/221 | -1.647 | 0.0072 |
| semimarkov_top10_strict_majority_long | state_substitution | 1.00 | N | 1982.53/148.74/37.48/3.97/976 | 182.18/181.58/15.89/11.42/221 | -1.654 | 0.0095 |
| kalman_top10_strict_majority_long | state_substitution | 1.00 | N | 2974.38/179.58/36.72/4.89/943 | 162.19/161.68/14.28/11.32/212 | -1.754 | 0.0051 |
| semimarkov_top10_strict_majority_long | state_substitution | 1.25 | N | 1846.72/143.75/38.59/3.72/976 | 172.72/172.16/15.73/10.95/221 | -2.132 | 0.0095 |
| semimarkov_top10_strict_majority_long | state_substitution | 1.50 | N | 1714.75/138.67/39.71/3.49/976 | 163.51/162.99/15.56/10.47/221 | -2.605 | 0.0095 |
| kalman_top10_strict_majority_long | state_substitution | 1.25 | N | 3059.97/181.90/37.98/4.79/943 | 148.69/148.22/14.28/10.38/212 | -2.696 | 0.0051 |
| bocpd_top10_strict_majority_long | state_substitution | 1.00 | N | 3247.05/186.80/35.17/5.31/971 | 175.17/174.60/16.82/10.38/221 | -2.700 | 0.0072 |
| semimarkov_top10_strict_majority_long | state_substitution | 1.75 | N | 1587.07/133.50/40.83/3.27/976 | 154.53/154.05/15.40/10.01/221 | -3.073 | 0.0095 |
| semimarkov_top10_strict_majority_long | state_substitution | 2.00 | N | 1464.05/128.26/41.95/3.06/833 | 145.79/145.34/15.23/9.54/199 | -3.538 | 0.0095 |
| kalman_top10_strict_majority_long | state_substitution | 1.50 | N | 3135.80/183.91/39.17/4.69/943 | 135.77/135.35/14.28/9.48/212 | -3.598 | 0.0051 |
| bocpd_top10_strict_majority_long | state_substitution | 1.25 | N | 3410.54/190.94/36.08/5.29/971 | 164.16/163.63/17.40/9.41/221 | -3.672 | 0.0072 |
| kalman_top10_strict_majority_long | state_substitution | 1.75 | N | 3201.07/185.62/40.31/4.60/943 | 123.41/123.05/14.28/8.62/212 | -4.460 | 0.0051 |
| bocpd_top10_strict_majority_long | state_substitution | 1.50 | N | 3566.74/194.77/36.94/5.27/971 | 153.46/152.98/17.98/8.51/221 | -4.568 | 0.0072 |
| kalman_top10_strict_majority_long | state_substitution | 2.00 | N | 3255.07/187.01/41.40/4.52/800 | 111.61/111.29/14.28/7.80/190 | -5.284 | 0.0051 |
| bocpd_top10_strict_majority_long | state_substitution | 1.75 | N | 3714.02/198.27/37.76/5.25/971 | 143.09/142.65/18.56/7.69/221 | -5.393 | 0.0072 |
| bocpd_top10_strict_majority_long | state_substitution | 2.00 | N | 3850.77/201.44/38.79/5.19/828 | 133.03/132.63/19.15/6.92/199 | -6.154 | 0.0072 |

## Frozen top 1

- No cell passed; future veto remains closed.

## Boundaries

- Addition cells beat a same-gross pro-rata leverage counterfactual; that comparator is diagnostic and non-deployable.
- State cells keep Gross9 total gross and funding/premium family gross fixed by replacing Markov weight.
- State acceptance Jaccard excludes Markov by contract, but exact Markov overlap is still reported.
- Every candidate uses frozen signals, next-open execution, 0.5x unit leverage, and 6bp per side.
- All source alphas and later windows are research-exposed; a survivor is forward-shadow only.
