# Chronos-2 embedding probe alpha validation (2026-07-13)

## Verdict

Chronos-2의 zero-shot forecast head 대신 encoder representation을 사용했다.
2020-2022에만 PCA와 저용량 ERM/V-REx/Group-DRO probe를 학습하고, 2023
executed-path Top-10을 고정한 뒤 2024/2025/2026을 평가했다. 정식
`alpha_pool`/`live_grade` 통과자는 **0개**였다.

## Best evidence

| Rank | Probe / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | PCA32 linear V-REx, long, 180d q90 | 2023 select | +51.36% | 51.40% | 3.30% | 15.55 | 42 |
| 1 | same | 2024 Test | +17.64% | 17.60% | 12.84% | 1.37 | 62 |
| 1 | same | 2025 Eval | -16.15% | -16.16% | 18.63% | -0.87 | 36 |
| 1 | same | 2026 YTD | -5.98% | -13.78% | 9.19% | -1.50 | 18 |
| 3 | PCA32 linear Group-DRO, long, 180d q95 | 2024 Test | +30.89% | 30.82% | 9.06% | 3.40 | 41 |
| 3 | same | 2025 Eval | -8.48% | -8.49% | 12.46% | -0.68 | 24 |
| 3 | same | 2026 YTD | +1.87% | 4.56% | 3.93% | 1.16 | 9 |

가장 안정적인 raw score는 PCA64 MLP V-REx probe였고 Spearman은
2024/2025/2026 각각 `0.0416 / 0.0435 / 0.0286`이었다. 약한 일반화는
존재하지만 48h long-tail 분류와 rolling threshold를 거치면 경제적 alpha로
유지되지 않았다. 특히 2023 long selection 성능이 과도하고 2025가 음수로
반전해 selection-regime 과적합으로 판단한다.

## Causal protocol

- 각 anchor 이전 720개 완료된 1h bin만 encoder에 입력
- Chronos `embed()`는 `prediction_length=0`, `future_target=None` 경로 사용
- target + 8개 공변량의 encoder token을 3,072차원으로 요약
- PCA/whitening, tail label, probe 학습은 2020-2022만 사용
- 미래 score 진단은 frozen manifest를 기록한 이후에만 계산
- 2023 actual executed-path hash로 중복 제거한 Top-10 고정
- next-bar 5m open, 48h hold, 0.5x, 편도 6bp
- full-window CAGR, strict intratrade MDD

## Artifacts

- Search: `training/search_chronos2_embedding_probe_alpha.py`
- Tests: `tests/test_chronos2_embedding_probe_alpha.py`
- Frozen manifest: `results/chronos2_embedding_probe_top10_manifest_2026-07-13.json`
- Result: `results/chronos2_embedding_probe_alpha_scan_2026-07-13.json`

