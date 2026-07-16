# LVRT-R0 outcome-blind 지원도 동결

## 판정

**PASS — 가격 outcome을 열기 전 이벤트 클록 지원도 게이트를 모두 통과했다.**

이 단계에서는 Binance kline 파일의 SHA256만 검증하고 `date` 열만 읽었다. `open`, `high`, `low`, `close`, volume, 미래 수익률, CAGR, MDD는 읽거나 계산하지 않았다.

## 동결된 이벤트 수

| 구간 | 비중첩 이벤트 수 |
|---|---:|
| 2020 | 427 |
| 2021 | 368 |
| 2022 | 317 |
| 2023 | 147 |
| 2023 H1 | 74 |
| 2023 H2 | 73 |
| 전체 | **1,259** |

- long 비중: `48.8483%`
- short 비중: `51.1517%`
- 단일 월 최대 집중도: `8.9754%`
- raw setup: `3,572`
- raw two-bar confirmation: `1,440`
- 비중첩 후: `1,259`

사전 게이트인 전체 250개, 연도별 40개, 2023 반기별 20개, 방향별 25–75%, 단일 월 20% 이하를 모두 만족한다.

초기 탐색 수치 1,244개와 정식 동결 수치 1,259개의 차이는 정식 구현이 기준선을 **직전 8,640개 clean 관측**으로 계산하고 모든 소스 격리 경로를 코드로 재생한 결과다. 수익률을 보아 조정한 차이가 아니다.

## 소스 무결성

- aggTrade feature SHA256: `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- market SHA256: `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- source audit SHA256: `5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6`
- source audit: PASS
- 완전한 5분 시장 타임스탬프: 420,768개
- aggTrade feature 행: 420,732개
- 누락 feature bar: 36개
- 격리 bar: 1,682개
- source-ID-gap UTC 일자: 5개

setup, confirmation, next-open entry, 12개 보유 봉, 예정 exit 중 격리 구간이나 split 경계를 만난 이벤트는 제외했다.

## 동결된 클록

- 경로: `results/liquidity_vacuum_replenishment_clock_2026-07-17.csv`
- 행 수: `1,259`
- SHA256: `ed9dd6391df2118ac09d147a4e57c3cb3f6e105a13f6c0d973ee424cfedd54d2`
- 지원 결과 SHA256: `bbce868ab2ca861bb1e56d49d4be228d20fe7e63f4dbaf66ff7b0eb1f8a3fbc6`

클록에는 위치, 시각, 방향, 고정 hold만 있다. 가격이나 사후 결과는 없다.

## 대조군 클록

| 대조군 | 비중첩 수 | primary 진입 Jaccard |
|---|---:|---:|
| direction flip | 1,259 | 1.0000 |
| reversal 확인 제거 | 2,756 | 0.0000 |
| 진입 1봉 추가 지연 | 1,259 | 0.0000 |
| setup 1일 이동 | 1,235 | 0.0061 |
| confirmation 부호 순열 | 899 | 0.4272 |

방향 반전은 의도적으로 동일 시계를 사용한다. 부호 순열 Jaccard가 높다는 사실은 outcome 이전에 공개하며, 향후 placebo가 primary 성능 게이트를 통과하면 사전등록대로 LVRT-R0를 기각한다.

## 다음 단계

이제 strict evaluator의 소스와 테스트를 먼저 커밋하고 해시로 동결한다. 그 다음에만 2020–2022 train과 2023 selection 가격 outcome을 한 번 연다. 실패 시 q80, 방향, 확인 봉, 12봉 hold, 비용 또는 게이트를 수정하지 않고 LVRT-R0를 기각한다. 2024 이후는 계속 봉인한다.
