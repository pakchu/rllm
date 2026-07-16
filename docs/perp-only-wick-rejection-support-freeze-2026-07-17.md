# POWR-12 outcome-blind 지원도 동결

## 판정

**PASS — 사전등록한 지원도 게이트를 모두 통과했으며 post-signal outcome은 아직 열지 않았다.**

이 단계는 완성된 signal bar의 Binance Spot/Perp OHLC만 읽어 이벤트 시계를
계산했다. 진입 이후 가격 경로, 거래 수익률, 승률, CAGR, MDD 및 funding은
읽거나 계산하지 않았다. 정책의 q95, 6bp wick floor, Spot/Perp 비율 0.5,
진입 지연 3봉 및 12봉 hold는 결과 확인 전에 고정됐고 수정하지 않았다.

## 동결된 이벤트 수

| 구간 | 비중첩 이벤트 수 |
|---|---:|
| 2020 | 305 |
| 2021 | 188 |
| 2022 | 101 |
| 2023 | 43 |
| 2023 H1 | 29 |
| 2023 H2 | 14 |
| **전체** | **637** |

- raw event: `726`
- long / short: `51.3344% / 48.6656%`
- lower / upper rejection branch: `51.3344% / 48.6656%`
- 단일 월 최대 집중도: `9.7331%`
- 사전등록 기준: train 500+, 각 train 연도 80+, 2023 40+,
  2023 H1/H2 20+/10+, 각 방향 35–65%, 각 branch 20%+, 월 집중 12% 이하

2020년 305건에서 2023년 43건으로 이벤트가 감소한 점은 outcome 이전에
공개하는 구조적 감쇠 경고다. 향후 2023 성능 및 반기 게이트를 완화하는
근거로 사용하지 않는다.

## 소스와 결측 처리

- Perp 1m: `2,103,840`행, SHA256
  `0b55bb0c3b845a90da738e746c769b19c1de4ac230ca8f1fccb6c361c4a9a41f`
- Spot 1m: `2,101,493`행, SHA256
  `bc6e0fd6b773ab6458a5de88fb9589161d1adf4ac1d0e7024f252515909f4a54`
- 5분 grid: `420,768`행
- Perp incomplete 5분 봉: `0`
- Spot incomplete 5분 봉: `471`
- joint complete 5분 봉: `420,297`

Signal 및 그 다음 latency bucket은 joint complete여야 한다. 미래 hold 구간의
Spot 누락으로 이미 발생한 거래를 사후 삭제하지 않는다. DB snapshot은
point-in-time snapshot이 아니므로 이 결과는 해시로 고정한 historical backfill
재생이며, live 승격 전 forward parity가 별도로 필요하다.

## 동결된 클록과 구현

- 사전등록 commit: `e168d6ff560d429138f586e625ba54e1c9710c97`
- outcome-blind clock builder commit: `c19390feb43632477481936fe90e9188901ae520`
- builder SHA256:
  `4adf1812ea75170d5360f3ab40ecec8403682401659451515b0b3c83f0e8f583`
- support JSON:
  `results/perp_only_wick_rejection_support_2026-07-17.json`
- support JSON SHA256:
  `6e753d4dbd525f5c6e45882d7df6b1f5fe6e614727f939f7853c7c3c857d347d`
- primary clock:
  `results/perp_only_wick_rejection_clock_2026-07-17.csv`
- primary clock 행 수: `637`
- primary clock SHA256:
  `7ecd567bf182fd7f92a8a1583b8f82c409ea5530d2e0eef25174880d52502619`

클록에는 위치, 시각, 방향, branch, 진입 지연 및 고정 hold만 있고 가격이나
수익률 열은 없다. 독립 code-review는 causality, `t+15m` 진입, strict-prior
threshold, 결측 정책, int64 위치 계산 및 support gate에 blocker가 없다고
판정했다.

## 사전 대조군 지원도

| 대조군 | 비중첩 수 | primary 진입 Jaccard |
|---|---:|---:|
| direction flip | 637 | 1.000000 |
| Spot-only wick | 8,609 | 0.000649 |
| common wick | 8,462 | 0.000000 |
| basis-free Perp wick | 8,476 | 0.002861 |
| 진입 1봉 추가 지연 | 637 | 0.000000 |
| stale Spot 1h | 9,124 | 0.002259 |
| stale Spot 1d | 8,810 | 0.002760 |

대조군은 support만 계산됐고 성능 outcome은 열지 않았다. 평가기 동결 단계에서
모든 대조군의 정확한 clock hash를 결과 열람 전에 별도로 봉인한다.

## 다음 게이트

strict MDD, full-wall-clock CAGR, 기본/8bp stress 비용, realized funding 및
weekly-cluster sign-flip을 구현한 평가기를 먼저 커밋하고 해시로 봉인한다.
그 다음에만 2020–2022 train과 2023 selection/H1/H2 outcome을 한 번 연다.
어느 게이트든 실패하면 정책을 수리하지 않고 POWR-12를 기각하며 2024 이후는
계속 봉인한다.
