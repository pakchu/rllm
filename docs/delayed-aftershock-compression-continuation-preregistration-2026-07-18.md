# DACC-48 사전등록 — Delayed Aftershock Compression Continuation

## 가설

기존 점프 알파는 큰 변동과 흐름을 같은 시점에 읽고 즉시 추종했다. DACC-48은
그 거래와 시간적으로 분리한다. 예외적인 단일 5분 충격 뒤 30분간 가격과 흐름이
압축되고, 그 뒤 별도의 15분 구간에서 같은 방향 flow와 range가 다시 팽창하며
압축 상자를 돌파할 때만 진입한다.

즉, 단순 점프가 아니라 **충격 → 조용한 압축 → 늦은 재가속**의 순서를 검증한다.
REX, OI, funding/premium, 김프/FX/DXY, alt rotation, Markov, ExtraTrees, LLM은
신호에 사용하지 않는다.

## 고정 정책

5분 종가 수익률을 `r[t] = log(close[t]/close[t-1])`, 방향을
`d = sign(r[j])`로 둔다.

### 충격 `j`

- `abs(r[j]) >= max(40bp, strict-prior q99.5(abs(r)))`
- q99.5 기준선은 직전 8,640개 clean 관측, 최소 2,016개를 사용
- 직전 72봉 bipower scale 대비 `abs(r[j])/sigma72_pre[j] >= 4`
- `d * taker_imbalance[j] >= 0.15`
- 충격 봉 quote volume이 strict-prior 8,640봉 중앙값 이상

### 압축 `j+1..j+6`

- `box_width = log(max(high)/min(low)) <= 0.55 * abs(r[j])`
- `abs(log(close[j+6]/close[j])) <= 0.20 * abs(r[j])`
- 충격 종가 대비 불리한 box excursion `<= 0.30 * abs(r[j])`
- 6봉 signed quote flow의 절대값이 strict-prior 6봉-flow 중앙값 이하

### 재가속 `j+7..j+9`

- 방향 수익률 `>= max(15bp, 0.30 * box_width)`
- `close[j+9]`가 압축 상자 방향 edge를 5bp 이상 돌파
- 방향 3봉 flow `>= max(0.10, strict-prior q70(abs(3봉 flow)))`
- 방향 보정 `(재가속 flow - 압축 flow) >= 0.10`
- 재가속 3봉 평균 true range가 압축 6봉 평균의 1.50배 이상

판단은 `j+9` 종가 뒤, 진입은 다음 시가 `j+10`이다. 48봉(240분) 고정 보유,
stop/TP 없음, 비중첩, 0.5x, 6bp/notional/side를 적용한다. 비용 스트레스는
10bp/notional/side이고 funding은 `[entry, exit)`의 실현 settlement만 반영한다.

모든 임계값과 창 길이는 outcome 전에 단 하나로 고정한다. 결과를 보고 q99.5,
압축/재가속 길이, flow/range 기준, 진입 지연 또는 hold를 고치지 않는다.

## 데이터·누수 경계

- Binance USD-M BTCUSDT 5분 kline: 2020-01-01 ~ 2023-12-31
- 420,768개 완전한 5분 봉, SHA256
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- 실현 funding 4,383건, SHA256
  `c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7`
- signal feature는 완료된 `j+9`까지만 사용하고 `j+10` 시가에 집행한다.
- rolling quantile/median과 shock scale은 현재 관측을 제외한다.
- 결측 execution 봉은 보간하지 않고 fail-closed 한다.

이 레포는 다른 정책으로 pre-2024 수익률을 이미 본 적이 있으므로 완전히 깨끗한
시장 홀드아웃이라고 주장하지 않는다. 다만 정확한 DACC-48 결과는 아직 열지
않았고, 2024 이후는 계속 봉인한다.

## outcome 이전 지원도·직교성 게이트

- 2020–2022 비중첩 이벤트 150개 이상, 각 연도 30개 이상
- 2023 40개 이상, H1/H2 각각 15개 이상
- long/short 각각 30–70%
- 단일 월 최대 15%, 단일 UTC 주 최대 8%
- 기존 jump / jump-volume-clock / efficient-recovery clock 대비
  - exact entry Jaccard `<= 0.02`
  - 기존 entry ±6시간 안에 있는 DACC entry 비율 `<= 0.25`
  - position-time Jaccard `<= 0.15`

하나라도 실패하면 post-entry 가격·funding·수익률을 열지 않고 정책을 기각한다.

## 성능 게이트

Train은 2020–2022, selection은 2023으로 고정한다. 두 구간 각각:

- 절대수익 양수
- CAGR/strict MDD `>= 3`
- strict MDD `<= 15%`
- 주간 cluster sign-flip `p <= 0.10`
- 비용 전 평균 기초자산 움직임 `> 12bp`
- 10bp/notional/side 스트레스에서도 절대수익 양수

2023 H1/H2도 각각 양수여야 하고 진입을 한 봉 더 늦춘 `j+11` 대조군도
train/2023에서 양수여야 한다. CAGR는 미거래 현금 구간을 포함하며 strict MDD는
global/pre-entry HWM, 보유 중 favorable-before-adverse OHLC, funding 및
진입/청산/가상청산 비용을 포함한다.

## 대조군

- 동일 clock 방향 반전
- 충격 직후 `j+1` 즉시 진입
- 압축 조건 제거
- flow 조건 제거
- breakout/range 조건 제거
- 진입 한 봉 추가 지연
- 전체 사건 geometry를 정확히 하루 앞당긴 placebo

no-compression, no-flow, no-range 또는 하루 이동 대조군 중 하나가 primary 전체
게이트를 독립적으로 통과하면 DACC 메커니즘을 기각한다. pre-2024 성능을 통과한
경우에만 기존 live/shadow alpha와 일별 PnL 절대 Pearson `<=0.30` 및 포트폴리오
한계 개선을 추가 검사한다.

2024·2025·2026 YTD는 이 순차 게이트를 통과하기 전까지 열지 않는다.
