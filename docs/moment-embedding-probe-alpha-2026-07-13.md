# MOMENT-1-small embedding alpha validation (2026-07-13)

## Verdict

공개·MIT 라이선스인 MOMENT-1-small(약 35.3M parameters)의 frozen encoder
representation을 사용했다. 2020-2022에만 PCA와 저용량
ERM/V-REx/Group-DRO probe를 학습하고 2023 executed-path Top-10을 고정한 뒤
2024/2025/2026을 평가했다. `alpha_pool`/`live_grade` 통과자는 **0개**였다.

## Best evidence

| Rank | Probe / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | PCA16 linear Group-DRO, long, 365d q90 | 2023 select | +66.64% | 66.70% | 3.25% | 20.53 | 64 |
| 1 | same | 2024 Test | -2.13% | -2.13% | 13.08% | -0.16 | 47 |
| 1 | same | 2025 Eval | -8.05% | -8.06% | 12.66% | -0.64 | 33 |
| 1 | same | 2026 YTD | +0.77% | 1.85% | 6.10% | 0.30 | 16 |
| 8 | PCA32 MLP V-REx10, long, 180d q90 | 2024 Test | +32.25% | 32.18% | 13.31% | 2.42 | 67 |
| 8 | same | 2025 Eval | -6.67% | -6.68% | 16.08% | -0.42 | 79 |
| 8 | same | 2026 YTD | +0.11% | 0.27% | 7.76% | 0.03 | 16 |

PCA32 MLP V-REx10 raw score의 Spearman은 2024/2025/2026에서
`0.0591 / 0.0095 / 0.0658`로 모두 양수였다. 따라서 encoder에는 약한 OOS
정보가 있지만, fit accuracy가 높고 2023 성능이 비정상적으로 커서 고정 probe와
절대 tail threshold가 regime 변화에 과적합됐다. 다음 실험은 gate 추가가 아니라
**48시간 지연 label을 사용하는 causal continual probe**로 이 결함을 직접 다룬다.

## Causal protocol

- 각 anchor 이전 512개 완료된 1h bin만 입력
- 9개 variate별 mean-patch와 final-patch embedding을 보존
- label/PCA/whitening/probe 학습은 2020-2022 fit row만 사용
- 미래 target 진단은 frozen manifest 기록 이후에만 계산
- 2023 actual executed-path hash로 중복 제거한 Top-10 고정
- next-bar 5m open, 48h hold, 0.5x, 편도 6bp
- full-window CAGR, strict intratrade MDD

## Sources and artifacts

- Model card: <https://huggingface.co/AutonLab/MOMENT-1-small>
- Official code: <https://github.com/moment-timeseries-foundation-model/moment>
- Search: `training/search_moment_embedding_probe_alpha.py`
- Tests: `tests/test_moment_embedding_probe_alpha.py`
- Frozen manifest: `results/moment_embedding_probe_top10_manifest_2026-07-13.json`
- Result: `results/moment_embedding_probe_alpha_scan_2026-07-13.json`

