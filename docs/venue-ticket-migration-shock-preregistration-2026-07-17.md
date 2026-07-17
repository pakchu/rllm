# VTMS-288 사전등록 — Venue Ticket Migration Shock

## 목적과 직교성

VTMS-288은 현재 portfolio의 OI·funding/premium·김프/FX·REX·Markov 상태를
사용하지 않는다. 이전 cross-venue 연구가 수익률/활동량 lead-lag, flow-to-price
timing, cash/perp rejection을 다뤘다면 이 후보의 핵심은 **Spot과 USD-M 사이에서
평균 개별 체결 ticket 크기가 한 시간 안에 어느 쪽으로 급격히 이동했는가**다.

결과를 보지 않은 count-only 점검에서 고정 규칙은 2020–2023 총 442개의
비중첩 사건을 만들었다. train 2020/2021/2022는 106/118/116개, 2023은
102개(H1/H2 47/55), 롱/숏 233/209, Spot/USD-M branch 224/218개였다. 이는
수익성 증거가 아니라 한 개의 가설을 평가할 지원도가 충분하다는 근거다.

## 경제적 가설

완료된 5분봉 `t`에서 다음 값을 만든다.

- Spot ticket: Spot quote notional / Spot trade count
- USD-M ticket: USD-M quote notional / underlying trade-ID count
- `r[t] = log(Spot ticket / USD-M ticket)`
- `dr[t] = r[t] - r[t-12]`

Spot branch는 `r`이 직전 30일 q95 이상이고 `dr`이 q97.5 이상일 때, USD-M
branch는 각각 q5/q2.5 이하일 때 활성화한다. 지배 venue의 flow coherence와
signed price response도 각각 strictly-prior q75 이상이어야 한다. 방향은 지배
venue의 signed aggressive flow와 같다. 이 조건은 큰 ticket이 곧 정보라는
주장이 아니라, **급격한 venue migration과 실제 가격 수용이 함께 나타난 경우**만
검증하는 proxy다.

## 누수·집행 계약

- 모든 rolling 기준은 현재 봉을 제외한 과거 8,640 clean bar만 사용
- 최소 과거 clean 관측 2,016개
- 어느 branch든 `t-1` 비활성 → `t` 활성인 onset만 신호
- source gap/incomplete bucket과 이후 24봉 격리, 보간 없음
- `t` 종료 후 한 계산 봉을 비우고 `t+2` open 진입
- 288봉(24시간) 고정 보유, 비중첩, 0.5x
- 기본 6bp/notional/side, stress 10bp
- funding은 `[entry, exit)` 실제 settlement rate
- global/pre-entry HWM 및 favorable-before-adverse held OHLC strict MDD
- warm-up과 미거래 현금을 포함한 full-clock CAGR

## 순차 검증

1. 이 문서·코드·manifest를 커밋한다.
2. outcome 없이 primary와 모든 control clock을 동결한다.
3. evaluator를 별도 커밋·동결한다.
4. 2020–2022를 먼저 열고 모든 gate 통과 시에만 2023을 연다.
5. train/2023 각각 절대수익 양수, CAGR/strict MDD 3 이상, MDD 15% 이하,
   주간 cluster p 0.10 이하, 평균 gross move 20bp 초과, 10bp stress 양수 요구.
6. 2023 H1/H2도 각각 양수이며 35거래 이상이어야 한다.
7. standalone 통과 후에만 기존 portfolio와 executed orthogonality를 계산한다.

방향·q95/q97.5/q75·`t+2`·288봉·비용·gate는 결과를 본 뒤 바꾸지 않는다.
