# COIN-M roll-migration pre-2024 rejection — 2026-07-19

## 결론

contract-specific front/next 롤 이동은 사건 지원도는 충분했지만 방향성 alpha가
아니다. repaired v2 source에서 support를 통과한 next-led 한 후보만 동결된
evaluator로 열었고, 모든 핵심 수익·위험·통계 gate에 실패했다. 2024 이후는
열지 않는다.

## 기본 비용 결과 — 0.5x, 6bp/side

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래 | mean gross | mean net | weekly p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit 2020-07-15~2022 | -79.11% | -47.03% | 79.36% | -0.59 | 2,586 | +0.11bp | -5.94bp | 1.000 |
| 2023 selection | -47.91% | -47.94% | 48.21% | -0.99 | 890 | -2.59bp | -7.30bp | 1.000 |
| 2023 H1 | -30.78% | -52.40% | 31.08% | -1.69 | 426 | -5.19bp | -8.59bp | — |
| 2023 H2 | -24.75% | -43.14% | 25.26% | -1.71 | 464 | -0.21bp | -6.11bp | — |

fit의 다섯 반기가 모두 음수다. 거래 수 부족이나 한쪽 방향 편향이 아니라,
고정 60분 뒤 gross edge 자체가 0에 가깝다는 실패다. round trip 비용이 equity
기준 약 6bp가 되므로 작은 음의 순수익이 2,586회 누적됐다.

## 대조군

- 10bp/side stress: fit -92.58%, 2023 -63.53%
- 방향 반전: fit -79.70%, 2023 -34.38%
- 1시간 지연: fit -77.22%, 2023 -36.71%
- 24시간 지연: fit -73.21%, 2023 -46.76%

원 방향과 반전 방향이 모두 손실인 것은 안정적인 방향 alpha가 없다는 뜻이다.
또 원 신호가 1시간·24시간 지연보다 일관되게 우월하지 않아 사건 시점 정보도
입증되지 않았다.

## 데이터·누수 상태

- 일별 primary + checksum 월별 absent-key fallback source SHA `d2126e5...`
- v1 일별 유효 행은 v2에서도 전부 동일
- v1 결측 경로 실행은 statistic artifact 생성 전에 중단
- v2 후보/임계치/hold/support gate는 v1 동결 코드를 그대로 재사용
- next-bar open 진입, 고정 exit open, held 5분봉 high/low strict MDD
- full-calendar CAGR, 6bp/side base, 10bp/side stress
- 2024 test, 2025 eval, 2026 holdout 미개봉

## 폐기 범위와 다음 축

다음 항목은 폐기한다.

- 단일 5분봉의 next volume-share jump
- `imbalance * sqrt(contract volume)` 압력
- 같은 봉 가격 수용만으로 만든 60분 continuation
- front-local rejected-flow 30분 fade는 support concentration 단계에서 이미 탈락

임계치 완화, 방향 반전, hold 변경으로 이 family를 수리하지 않는다. 다음 탐색은
거래비용보다 훨씬 큰 조건부 gross edge가 가능한 독립 축이어야 하며, 특히
단일봉 microstructure가 아니라 강제 포지션 이전의 **다중봉 상태 전이와 가격
충격 비대칭**을 outcome-blind하게 먼저 고정해야 한다.
