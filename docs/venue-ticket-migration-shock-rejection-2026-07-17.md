# VTMS-288 train 기각 — 2023 미개봉

## 결론

**VTMS-288은 2020–2022 train에서 기각했다. 2023과 2024 이후 결과는 열지
않았다.**

| 기간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래수 |
|---|---:|---:|---:|---:|---:|
| 2020–2022 | **+8.68%** | **2.81%** | **24.34%** | **0.12** | **336** |
| 10bp/side stress | **-5.00%** | **-1.69%** | **27.28%** | **-0.06** | **336** |

- long 176 / short 160
- 평균 gross underlying move: +20.67bp
- weekly cluster sign-flip p-value: 0.3312
- price PnL: +28.78% initial equity
- transaction cost: -19.45%
- funding: -0.65%

## 실패 gate

| gate | 기준 | 결과 |
|---|---:|---:|
| CAGR/strict MDD | 3 이상 | **0.12 — FAIL** |
| strict MDD | 15% 이하 | **24.34% — FAIL** |
| weekly cluster p | 0.10 이하 | **0.3312 — FAIL** |
| 10bp/side stress 절대수익 | 양수 | **-5.00% — FAIL** |

절대수익 양수, 거래수 250 이상, 평균 gross move 20bp 초과, 모든 component
removal ratio 상회, rejection placebo 비통과 조건은 만족했다. 그러나 네 개의
핵심 경제·통계 gate 실패만으로 사전등록 rejection contract가 발동한다.

## 연도별 안정성

| 연도 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래수 |
|---|---:|---:|---:|---:|---:|
| 2020 | -16.15% | -16.12% | 23.57% | -0.68 | 105 |
| 2021 | +19.87% | +19.88% | 17.50% | 1.14 | 117 |
| 2022 | +8.13% | +8.14% | 16.76% | 0.49 | 114 |

2020 H1의 절대수익은 -13.44%였고 평균 gross move도 -37.93bp였다. 2021과
2022의 양수 결과가 이 초기 손실을 상쇄했을 뿐, 여러 regime에 걸친 안정적인
알파로 볼 수 없다.

## 메커니즘 분해

고정된 branch-only 진단은 가설의 비대칭을 드러냈다.

| 고정 branch | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래수 |
|---|---:|---:|---:|---:|---:|
| Spot dominant | +64.99% | +18.16% | 24.48% | 0.74 | 176 |
| USD-M dominant | -34.13% | -12.99% | 39.46% | -0.33 | 160 |

Spot branch가 전체 양의 gross edge를 만들었지만, 단독으로도 목표 비율 3과
MDD 15%를 크게 못 미친다. USD-M branch는 방향 가설 자체와 반대로 움직였다.
이 관찰을 이용해 VTMS의 sign/threshold/branch를 사후 수정하지 않는다. 필요하면
별도 후보로 다시 사전등록해야 한다.

component removal은 모두 primary보다 낮았지만, 1시간 지연 대조군이
+15.03% 절대수익과 0.19 ratio로 primary의 +8.68%, 0.12보다 높았다. 완전한
placebo gate를 통과하지는 않았으나, 신호 시점의 정밀성이 강하지 않다는 추가
증거다.

## 누수·봉인 확인

- evaluator source와 10개 clock을 결과 개봉 전에 커밋·해시 동결했다.
- Stage 1 parser는 2023 첫 행을 수치 변환 전에 중단했다.
- 마지막 market timestamp: `2022-12-31 23:55:00`
- 마지막 funding timestamp: `2022-12-31 16:00:00`
- Stage 2 실행은 `stage 1 did not authorize 2023`으로 중단됐으며 2023 결과
  artifact는 생성되지 않았다.
- 고정 Stage 1 artifact SHA-256:
  `90a4a05e5a422a2641e2026a5cf68750709d62bac0fa41ff9a91ab40f9b709af`

VTMS-288은 폐기한다. 후속 탐색은 이 후보를 조정하지 않고 다른 독립 축을 새로
사전등록해야 한다.
