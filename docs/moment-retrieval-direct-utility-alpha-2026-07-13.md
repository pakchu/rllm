# MOMENT retrieval direct-utility execution (2026-07-13)

## Verdict

이미 return, MAE, 12bp 비용을 반영한 signed utility에 rolling percentile gate를
추가하지 않고 양수면 long, 음수면 short로 직접 실행했다. 2023 Top-10 고정 후
`alpha_pool`/`live_grade` 통과자는 **0개**였다.

| Rank | Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | current+mean8, k128, median λ0.5, long | 2024 Test | +19.61% | 19.57% | 7.07% | 2.77 | 33 |
| 4 | same | 2025 Eval | +12.01% | 12.02% | 3.12% | 3.85 | 21 |
| 4 | same | 2026 YTD | +0.00% | 0.01% | 3.50% | 0.00 | 5 |
| 5 | same retrieval, both | 2024 Test | +15.98% | 15.94% | 7.07% | 2.25 | 34 |
| 5 | same | 2025 Eval | +10.45% | 10.45% | 3.12% | 3.35 | 24 |
| 5 | same | 2026 YTD | +1.99% | 4.85% | 3.50% | 1.39 | 7 |

Gate 제거는 거래 수를 의미 있게 늘리지 못했다. 병목은 threshold가 아니라 오래된
analog가 현재 memory에 계속 남아 utility 분포를 희석하는 구조일 가능성이 있다.

## Artifacts

- Evaluator: `training/evaluate_moment_retrieval_direct_utility_alpha.py`
- Tests: `tests/test_moment_retrieval_direct_utility_alpha.py`
- Frozen manifest: `results/moment_retrieval_direct_utility_top10_manifest_2026-07-13.json`
- Result: `results/moment_retrieval_direct_utility_alpha_scan_2026-07-13.json`

