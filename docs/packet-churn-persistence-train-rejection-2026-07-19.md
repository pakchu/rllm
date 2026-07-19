# PCP-6 train rejection — 2026-07-19

PCP-6는 `70e3393`에서 evaluator를 먼저 봉인한 뒤 2020–2022 outcome만 열었다.
2023 selection과 2024+ forward는 열지 않았다.

## Primary 결과

| 기간 | 절대수익률 | CAGR | strict MDD | CAGR/MDD | 거래 | L/S | cluster p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020–2022 | -8.22% | -2.82% | 20.58% | -0.14 | 147 | 78/69 | 0.7300 |
| 2020 | -10.87% | -10.85% | 13.84% | -0.78 | 48 | 28/20 | - |
| 2021 | -0.36% | -0.36% | 14.17% | -0.03 | 54 | 28/26 | - |
| 2022 | +3.35% | +3.35% | 6.61% | 0.51 | 45 | 22/23 | - |

편도 10bp stress에서는 절대수익률 `-13.47%`, CAGR `-4.71%`, strict MDD
`22.70%`였다. 기본 비용에서도 거래당 평균 gross는 `+1.95bp`뿐이고 평균 net은
`-5.13bp`여서 비용을 넘지 못했다.

Gate 여섯 개 중 거래 수 `>=100`만 통과했다. 절대수익, 위험조정 비율, MDD,
stress, weekly-cluster 유의성이 모두 실패했다.

## 고정 control

| control | 절대수익률 | CAGR | strict MDD | CAGR/MDD |
|---|---:|---:|---:|---:|
| side flip | -10.63% | -3.67% | 21.09% | -0.17 |
| immediate entry | -8.56% | -2.94% | 21.51% | -0.14 |
| extra latency | -11.23% | -3.89% | 23.52% | -0.17 |

방향 반전과 ±1봉 지연 control도 모두 음수였다. 따라서 실패 원인은 한 방향의
부호나 한 봉의 실행 지연이 아니라, packet-churn persistence 사건의 평균 edge가
비용보다 작고 2020–2022에 안정적이지 않다는 것이다.

## 결정

- PCP-6 primary와 세 control을 alpha 후보에서 영구 기각한다.
- 결과를 본 뒤 threshold, 방향, 확인 길이, hold를 수리하지 않는다.
- `selection --stage`는 `PermissionError`로 거부되는 것을 확인했다.
- 2023, 2024, 2025, 2026 PCP 수익은 봉인 상태로 유지한다.

근거 artifact:
`results/packet_churn_persistence_train_2020_2022_2026-07-19.json`
