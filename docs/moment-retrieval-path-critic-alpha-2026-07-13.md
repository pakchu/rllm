# MOMENT delayed dense-retrieval path critic (2026-07-13)

## Verdict

MOMENT PCA32의 현재 상태와 최근 8-anchor 평균 상태로 과거 analog를 검색하고,
48h 지연 label만 memory에 추가하는 retrieval critic을 평가했다. 정식
`alpha_pool`/`live_grade` 통과자는 **0개**였지만, 최초로 2024/2025/2026
raw score IC가 모두 양수인 조합을 확인했다.

## Best evidence

| Rank | Retrieval / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 5 | current+mean8, k128, median λ0.5, long, 180d q70 | 2023 select | +10.46% | 10.47% | 6.73% | 1.56 | 17 |
| 5 | same | 2024 Test | +19.61% | 19.57% | 7.07% | 2.77 | 33 |
| 5 | same | 2025 Eval | +12.01% | 12.02% | 3.12% | 3.85 | 21 |
| 5 | same | 2026 YTD | +0.00% | 0.01% | 3.50% | 0.00 | 5 |
| 7 | same retrieval, both | 2024 Test | +15.98% | 15.94% | 7.07% | 2.25 | 34 |
| 7 | same | 2025 Eval | +10.45% | 10.45% | 3.12% | 3.35 | 24 |
| 7 | same | 2026 YTD | +1.99% | 4.85% | 3.50% | 1.39 | 7 |

`current_mean8/k128/median_lambda0.5` signed utility의 Spearman은
2024/2025/2026에서 `0.0499 / 0.0260 / 0.0493`이었다. Parametric model과
달리 2025 relation flip을 피했지만, rolling percentile gate가 이미 비용과 MAE를
반영한 positive utility를 다시 희소화해 2026 거래 수가 부족했다.

## Strict protocol

- query: current PCA32 또는 current + causal mean8 PCA32
- fit-only 표준화 후 cosine kNN, k=64/128
- empirical q25/q50/q75 return/long-MAE/short-MAE
- memory에는 `signal + 1 + 576 <= current signal`인 label만 추가
- phase1 target/embedding/inference는 2024 이전만 수행
- fit-prefix score NaN, 최소 200 prior score threshold
- 2023 actual executable path Top-10 manifest 선기록
- phase2 causal retrieval 완전 재실행 후 2023 hash 검증

## Sources and artifacts

- Retrieval-augmented forecasting reference: <https://paperswithcode.com/paper/retrieval-augmented-time-series-forecasting>
- Search: `training/search_moment_retrieval_path_critic_alpha.py`
- Tests: `tests/test_moment_retrieval_path_critic_alpha.py`
- Frozen manifest: `results/moment_retrieval_path_critic_top10_manifest_2026-07-13.json`
- Result: `results/moment_retrieval_path_critic_alpha_scan_2026-07-13.json`

