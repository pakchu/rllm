# River contextual action-utility alpha search (2026-07-13)

## Verdict

Raw 48시간 수익률 회귀를 대체해 LONG/SHORT 실행 보상과 방향별 MAE를
각각 학습하는 delayed multi-head contextual policy를 평가했다. 2023에서
사전 고정한 distinct Top-10 가운데 `alpha_pool`과 `live_grade`는 모두
**0개**였다.

즉, reward target을 경로 위험까지 확장해도 현재 feature -> 개별 48시간
outcome의 지도학습 관계는 연도 간 안정적이지 않았다.

## Protocol

- causal feature: compact 52개 또는 full 100개
- online model: River 0.25.0 ARF+ADWIN / Hoeffding Adaptive Tree
- heads:
  - long net return
  - short net return
  - long leveraged MAE loss
  - short leveraged MAE loss
- utility: `predicted_net_return - mae_penalty * predicted_mae_loss`
- flat utility: 0
- side policy: both / long-only / short-only
- gate: utility > 0 또는 현재 점수를 제외한 과거 180/365일 rolling q70/q80/q90/q95
- label 공개: 다음 5분봉 진입 후 48시간 path가 종료된 뒤
- update order: 완료 라벨 학습 -> 현재 예측 -> 현재 outcome pending 등록
- 선택: 2023 strict full-window CAGR/MDD 기준 distinct Top-10
- 중복 제거: 2023의 **실제 비중첩 체결 경로**만 hash
- 평가: 2024 Test / 2025 Eval / 2026 YTD prequential OOS
- 편도 비용 6bp, 0.5x, full-window CAGR, intratrade adverse excursion strict MDD

총 324개 사전 policy specification 중 2023 양수·8회 이상 거래 후보는
117개였다. Manifest는 2024+ metric 계산 전에 기록됐고
`later_metrics_included=false`다.

## Top-10 evidence

| Rank | Model / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | HATR full, long, MAE 1.0, 180d q90 | 2023 select | +24.74% | 24.76% | 3.60% | 6.87 | 44 |
| 1 | same | 2024 Test | +0.28% | 0.28% | 10.15% | 0.03 | 26 |
| 1 | same | 2025 Eval | -3.86% | -3.86% | 11.62% | -0.33 | 27 |
| 1 | same | 2026 YTD | +5.52% | 13.78% | 2.21% | 6.23 | 7 |
| 3 | HATR full, long, MAE 0.5, 365d q90 | 2024 Test | +8.21% | 8.19% | 8.15% | 1.00 | 29 |
| 3 | same | 2025 Eval | +1.82% | 1.83% | 9.31% | 0.20 | 34 |
| 3 | same | 2026 YTD | +2.81% | 6.89% | 10.85% | 0.64 | 13 |

Top-10은 전부 long-only였다. 2026에서 양수인 후보도 있었지만 2024/2025
일반화 기준을 통과하지 못했다. 이는 post-hoc 2026 후보 채택 사유가 될 수
없다.

## Learnability finding

Outcome head의 OOS Spearman은 대부분 절댓값 0.1 이내였다.

- HATR full long net: 2023 -0.030 / 2024 -0.024 / 2025 -0.039 / 2026 +0.052
- HATR full short net: 2023 -0.028 / 2024 -0.024 / 2025 -0.033 / 2026 -0.138
- ARF full long net: 2023 -0.106 / 2024 -0.040 / 2025 -0.087 / 2026 -0.098
- MAE head 역시 방향과 연도에 따라 부호가 변했다.

따라서 실패 원인은 단순 gate threshold가 아니라 **샘플별 path label과 현재
feature 사이의 불변 관계가 약한 것**이다. 다음 실험은 개별 연도 평균 성능이
아니라 여러 train environment에서 동시에 방향이 유지되는 feature/state만
선택하는 invariant/group-DRO branch로 전환한다.

## Validity fixes applied before the final run

1. long-only/short-only는 허용되지 않은 반대 방향 점수에 veto되지 않는다.
2. rolling threshold도 각 허용 side의 utility history로 별도 보정한다.
3. raw anchor 신호가 아니라 비중첩 실제 체결 경로로 후보를 de-duplicate한다.
4. 2024+ 경로는 selection hash와 manifest에 포함하지 않는다.

## Artifacts

- Search: `training/search_river_contextual_utility_alpha.py`
- Tests: `tests/test_river_contextual_utility_alpha.py`
- Frozen manifest: `results/river_contextual_utility_top10_manifest_2026-07-13.json`
- Result: `results/river_contextual_utility_alpha_scan_2026-07-13.json`
- River source: <https://github.com/online-ml/river>
