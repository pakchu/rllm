# MOMENT delayed continual-probe alpha validation (2026-07-13)

## Verdict

Frozen MOMENT PCA representation 위에 48시간 지연 label만 사용하는 continual
classifier/regressor를 붙였다. 2023 이전 단계에서는 2024+ target을 읽지 않았고,
2023 executed-path Top-10 manifest를 기록한 뒤 선택된 알고리즘만 처음부터
결정론적으로 재실행했다. `alpha_pool`/`live_grade` 통과자는 **0개**였다.

## Best evidence

| Rank | Probe / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | PCA16 slow continual MLP classifier, long, 365d q80 | 2023 select | +24.93% | 24.95% | 5.70% | 4.38 | 61 |
| 1 | same | 2024 Test | +7.25% | 7.23% | 17.19% | 0.42 | 83 |
| 1 | same | 2025 Eval | -2.66% | -2.66% | 12.55% | -0.21 | 58 |
| 1 | same | 2026 YTD | +2.02% | 4.91% | 5.40% | 0.91 | 23 |
| 2 | same, 180d q80 | 2024 Test | +22.20% | 22.15% | 15.29% | 1.45 | 78 |
| 2 | same | 2025 Eval | -1.33% | -1.33% | 12.24% | -0.11 | 64 |
| 2 | same | 2026 YTD | +4.48% | 11.10% | 4.35% | 2.55 | 28 |
| 10 | PCA16 fast continual MLP classifier, short, 365d q90 | 2024 Test | -10.92% | -10.90% | 14.17% | -0.77 | 37 |
| 10 | same | 2025 Eval | +13.06% | 13.07% | 6.28% | 2.08 | 42 |
| 10 | same | 2026 YTD | +4.03% | 9.96% | 3.95% | 2.52 | 20 |

Continual MLP raw score의 Spearman도 2024에는 음수, 2025에는 양수로 뒤집혔다.
예를 들어 PCA16 slow model은 `-0.0668 / +0.0532 / +0.0431`이었다.
즉 단순 recent replay는 변화를 빨리 학습하는 것이 아니라 뒤늦게 이전 regime을
추종했고, foundation representation의 relation flip을 해결하지 못했다.

## Leakage guards

- initial standardizer/calibration/weights는 2020-2022 fit row만 사용
- current sample은 예측 후 index/ready-position만 queue
- target은 `signal + 1 + 576 bars`가 현재 signal보다 과거가 된 뒤에만 읽음
- phase 1은 signal date가 2024 이전인 구간에서만 실행
- 2023 actual executable path Top-10을 먼저 파일로 고정
- phase 2는 선택된 stream만 처음부터 재실행하고 2023 path hash 일치 강제
- source data/PCA/model revision hash 일치 강제
- full-window CAGR, strict intratrade MDD, 편도 6bp

## Artifacts

- Search: `training/search_moment_continual_probe_alpha.py`
- Tests: `tests/test_moment_continual_probe_alpha.py`
- Frozen manifest: `results/moment_continual_probe_top10_manifest_2026-07-13.json`
- Result: `results/moment_continual_probe_alpha_scan_2026-07-13.json`

