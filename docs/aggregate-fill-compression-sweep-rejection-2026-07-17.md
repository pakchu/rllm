# AFCS-144 train 탈락 — 2023 비개봉

## 판정

**AFCS-144를 폐기한다. 2023 및 2024 이후 outcome은 열지 않았다.** evaluator와
8개 event/control clock은 성과 개봉 전에 hash로 봉인됐고, stage-1 parser는
2023 첫 행의 수치를 변환하기 전에 중단했다.

| 구간/비용 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래수 |
|---|---:|---:|---:|---:|---:|
| 2020–2022, 6bp/side | -4.0336% | -1.3627% | 21.8358% | -0.0624 | 421 |
| 2020–2022, 10bp/side | -18.9079% | -6.7462% | 29.6073% | -0.2279 | 421 |
| 2020, 6bp/side | +9.9672% | +9.9458% | 9.8131% | 1.0135 | 123 |
| 2021, 6bp/side | -0.5674% | -0.5678% | 11.3580% | -0.0500 | 61 |
| 2022, 6bp/side | -12.2338% | -12.2417% | 21.6163% | -0.5663 | 237 |

롱 217회, 숏 204회로 방향 편향 때문은 아니다. 그러나 2022 H2만
절대수익 -11.9781%, strict MDD 18.8012%였고, 2020의 양수 성과도 CAGR/MDD
1.0135에 불과해 목표 3에 미달했다.

## 실패 원인

고정 신호의 평균 비용 전 기초자산 움직임은 **11.1991bp**였다. 전체 고정수량
ledger에서 가격 PnL은 초기자본 대비 +22.0945%, funding은 +0.3007%였지만,
실제 진입·청산 비용이 -26.4289%를 소모해 최종 절대수익이 음수가 됐다.
10bp stress에서는 절대수익이 -18.9079%로 더 악화됐다.

이는 단순 비용 문제만도 아니다.

- strict MDD 21.8358%로 사전등록 상한 15% 초과
- 주간 cluster sign-flip p=0.5229로 통계 gate 실패
- 2021과 2022 모두 음수, 특히 2022 H2 비용 전 평균 움직임도 -8.2248bp
- `no_aligned_response` control의 CAGR/MDD -0.0398이 primary -0.0624보다 높아
  component-removal mechanism gate 실패
- 1시간 지연, 1일 shift, random-side placebo는 모두 탈락했지만 primary 자체가
  9개 gate 중 거래수와 placebo gate 2개만 통과

방향 반전도 절대수익 -40.4683%여서 단순 sign flip으로 수리할 수 없다. 관측된
2020 성과를 기준으로 threshold·기간·방향을 다시 고르는 것은 no-repair 계약과
충돌하므로 허용하지 않는다.

## 봉인 유지와 다음 탐색

- 2023 execution OHLC/PnL: **봉인 유지**
- 2024, 2025, 2026: **봉인 유지**
- 기존 portfolio와의 realized orthogonality: standalone 실패로 미개봉

다음 후보는 aggTrade fill compression의 threshold 변형이 아니라, 다른 데이터
생성 메커니즘과 다른 holding clock을 사용하는 새 가설이어야 한다.
