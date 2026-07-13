# Causal online expert alpha search (2026-07-13)

This experiment replaces static future-selected gates with a selector that learns only from expert trades whose exits are already observable.

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

## Frozen Top-10 replay

| rank | selector | 2024 test | 2025 eval | 2026 holdout | combined | pool | strong |
|---:|---|---:|---:|---:|---:|:---:|:---:|
| 1 | `{'method': 'adverse_utility', 'mae_penalty': 1.0, 'lookback': 10, 'top_k': 1, 'threshold': 0.1}` | 38.98/38.88/5.46/7.12/30 | 19.98/20.00/4.97/4.02/32 | 1.77/4.31/3.71/1.16/24 | 69.70/24.45/5.99/4.08/86 | no | no |
| 2 | `{'method': 'adverse_utility', 'mae_penalty': 1.0, 'lookback': 10, 'top_k': 1, 'threshold': 0.0}` | 41.49/41.38/5.46/7.58/31 | 16.98/16.99/6.80/2.50/37 | 3.53/8.69/3.71/2.34/26 | 71.35/24.95/6.80/3.67/94 | no | no |
| 3 | `{'method': 'adverse_utility', 'mae_penalty': 0.5, 'lookback': 10, 'top_k': 1, 'threshold': 0.1}` | 40.41/40.32/5.46/7.39/30 | 17.78/17.79/5.54/3.21/35 | 3.17/7.80/4.96/1.57/29 | 70.63/24.74/7.49/3.30/94 | no | no |
| 4 | `{'method': 'adverse_utility', 'mae_penalty': 0.5, 'lookback': 10, 'top_k': 2, 'threshold': 0.1}` | 48.73/48.61/7.16/6.79/39 | 20.01/20.03/8.40/2.38/45 | 3.55/8.76/6.69/1.31/36 | 84.84/28.93/8.69/3.33/120 | no | no |
| 5 | `{'method': 'adverse_utility', 'mae_penalty': 1.0, 'lookback': 10, 'top_k': 2, 'threshold': 0.1}` | 46.16/46.04/7.16/6.43/36 | 25.15/25.17/4.76/5.29/37 | 0.44/1.07/4.27/0.25/25 | 83.73/28.61/7.16/4.00/98 | no | no |
| 6 | `{'method': 'adverse_utility', 'mae_penalty': 0.5, 'lookback': 10, 'top_k': 1, 'threshold': 0.0}` | 40.41/40.32/5.46/7.39/30 | 17.78/17.79/5.54/3.21/35 | 9.34/23.93/3.73/6.41/32 | 80.83/27.77/6.30/4.41/97 | no | no |
| 7 | `{'method': 'adverse_utility', 'mae_penalty': 1.0, 'lookback': 10, 'top_k': 2, 'threshold': 0.0}` | 48.79/48.67/7.16/6.80/37 | 22.02/22.04/6.54/3.37/42 | -1.69/-4.01/6.11/-0.66/28 | 78.49/27.08/8.15/3.32/107 | no | no |
| 8 | `{'method': 'adverse_utility', 'mae_penalty': 0.5, 'lookback': 10, 'top_k': 2, 'threshold': 0.0}` | 48.73/48.61/7.16/6.79/39 | 22.40/22.42/6.58/3.41/46 | 14.08/37.24/4.72/7.89/41 | 107.68/35.30/7.16/4.93/126 | no | no |
| 9 | `{'method': 'normalized_mean', 'mae_penalty': 0.0, 'lookback': 20, 'top_k': 1, 'threshold': 0.0}` | 44.51/44.40/5.46/8.14/32 | 20.87/20.89/4.80/4.35/25 | 0.79/1.90/4.65/0.41/23 | 76.04/26.36/5.46/4.83/80 | no | no |
| 10 | `{'method': 'normalized_mean', 'mae_penalty': 0.0, 'lookback': 20, 'top_k': 2, 'threshold': 0.1}` | 55.14/55.00/5.75/9.56/40 | 22.17/22.19/4.80/4.62/30 | 2.34/5.71/6.97/0.82/35 | 93.97/31.53/6.97/4.52/105 | no | no |

## Fixed-policy comparators

| policy | 2024 test | 2025 eval | 2026 holdout | combined |
|---|---:|---:|---:|---:|
| `long_funding_compression_premium` | 52.31/52.17/5.88/8.87/35 | 17.00/17.01/5.01/3.40/51 | 16.02/42.92/4.59/9.35/32 | 106.75/35.05/5.88/5.96/118 |
| `long_range_funding_premium` | 45.49/45.38/5.75/7.89/36 | 22.01/22.03/4.80/4.59/27 | 10.45/26.97/5.26/5.13/29 | 96.06/32.11/5.75/5.58/92 |
| `long_minimal_funding_premium` | 30.66/30.59/5.88/5.20/29 | 18.02/18.04/5.01/3.60/26 | 11.80/30.73/4.59/6.70/29 | 72.40/25.27/5.88/4.30/84 |
| `short_premium_kimchi_union` | 12.34/12.32/7.48/1.65/67 | 18.44/18.45/5.93/3.11/96 | 13.07/34.33/6.23/5.51/40 | 50.45/18.41/7.49/2.46/203 |
| `short_fx_stress` | 12.37/12.34/5.53/2.23/39 | 14.35/14.36/3.67/3.91/34 | 8.63/22.01/3.73/5.90/22 | 40.19/15.00/5.91/2.54/96 |
| `short_premium_panic` | 5.03/5.02/5.96/0.84/41 | 16.82/16.83/4.69/3.59/78 | 4.76/11.81/5.46/2.16/30 | 28.53/10.94/6.19/1.77/149 |
| `deterministic_all_expert_union` | 68.03/67.85/9.45/7.18/104 | 38.59/38.63/7.31/5.28/129 | 19.75/54.22/6.62/8.19/57 | 180.12/53.12/9.45/5.62/291 |

## Interpretation

- Standalone metric qualifiers: 7; incremental alpha-pool qualifiers: 0; strong-shadow qualifiers: 0; live-grade: 0.
- The deterministic no-learning union is the required marginal-value comparator. No selector beat it on both combined return and CAGR/MDD, so this online selection usage is rejected as a new alpha.
- The six experts are fixed templates. The selector may adapt only after a counterfactual expert trade has fully exited.
- 2024+ rows did not influence selector hyperparameters, Top-10 ranking, or path de-duplication in this run.
- Important provenance limit: the six sleeve definitions were committed on 2026-07-10 after this research programme had repeatedly inspected 2024-2026. These are mechanically frozen historical replays, not fresh-data OOS claims.
- The rank-8 strong row was discovered only after replay as one member of the frozen Top-10; its reported p-value is Bonferroni-adjusted for all ten members.
- Even a passing row remains shadow research because this research programme has repeatedly inspected the same later windows.

## Reproduction

```bash
python -m training.search_causal_online_expert_alpha --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz --manifest-output results/causal_online_expert_top10_manifest_2026-07-13.json --output results/causal_online_expert_alpha_scan_2026-07-13.json --docs-output docs/causal-online-expert-alpha-search-2026-07-13.md
```
