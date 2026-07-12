# MOMENT retrieval recency-memory validation (2026-07-13)

## Verdict

Frozen direct-utility Top family에서 base retrieval spec을 파생하고 memory를
최근 365일/730일/all로 제한했다. 2023 Top-10 고정 후
`alpha_pool`/`live_grade` 통과자는 **0개**였다.

| Rank | Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | current+mean8, k128, 730d memory, median λ0.5, both | 2023 select | +15.06% | 15.07% | 1.62% | 9.31 | 10 |
| 1 | same | 2024 Test | +20.51% | 20.47% | 5.54% | 3.70 | 28 |
| 1 | same | 2025 Eval | +0.06% | 0.06% | 7.40% | 0.01 | 31 |
| 1 | same | 2026 YTD | -6.59% | -15.11% | 11.11% | -1.36 | 8 |

2년 memory는 2023/2024를 개선했지만 2025와 2026에서 붕괴했다. 따라서 오래된
analog의 희석만이 relation flip 원인은 아니다. 현재 입력 데이터에서 model class,
online update, risk target, retrieval window를 바꾸는 것만으로 목표 alpha는 확인되지
않았다.

## Leakage controls

- base family는 OOS metric이 없는 frozen direct-utility manifest에서만 파생
- family/source/data/PCA hash 일치 강제
- 48h label maturity와 memory age를 동시에 만족한 row만 검색
- phase1은 2024 이전 target/embedding/retrieval만 수행
- 2023 path manifest 선기록 후 phase2 완전 재실행/hash 검증

## Artifacts

- Evaluator: `training/evaluate_moment_retrieval_recency_alpha.py`
- Tests: `tests/test_moment_retrieval_recency_alpha.py`
- Frozen manifest: `results/moment_retrieval_recency_top10_manifest_2026-07-13.json`
- Result: `results/moment_retrieval_recency_alpha_scan_2026-07-13.json`

