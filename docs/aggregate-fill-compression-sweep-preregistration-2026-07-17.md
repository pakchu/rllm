# AFCS-144 사전등록 — Aggregate-Fill Compression Sweep

## 목적과 직교성

AFCS-144는 기존 포트폴리오의 OI·funding/premium·김프/FX·REX·Markov 상태를
사용하지 않는다. 기존 aggTrades 연구가 주로 이벤트 명목 꼬리, 도착 버스트,
flow/price 반전, breadth-vs-capital 불일치, impact curvature를 사용한 것과 달리,
이 정책의 핵심 상태는 **하나의 공개 aggTrade가 포괄하는 underlying trade-ID
span의 평균 크기**다.

2020–2022에서 성과를 보지 않고 피처 분포와 비중첩 사건 수만 점검했다. 고정
규칙은 421개 사건, 연도별 124/61/236개, 롱/숏 217/204개를 만들었다. 이 수치는
수익성 증거가 아니라 단일 가설을 평가할 지원도가 충분하다는 근거다.

## 경제적 가설

완료된 5분봉 `t`에서 다음을 모두 요구한다.

- `underlying_trades_per_agg_event`: 직전 30일 clean q97.5 이상
- `flow_coherence`: 직전 30일 clean q90 이상
- `signed_price_response`: 직전 30일 clean q80 이상이면서 양수
- `quote_notional`: 직전 30일 clean 중앙값 이상
- aggTrade 사건 수 64개 이상, signed flow 방향이 0이 아님
- 직전 봉은 동일 상태가 아니어야 함

모든 기준선은 현재 봉을 제외하고 최소 2,016개의 과거 clean 관측치만 쓴다.
신호 방향은 signed aggressive notional과 같으며, 정보가 체결 압축 스윕 이후
천천히 확산된다는 가설로 12시간 지속성을 검증한다. `first_trade_id`와
`last_trade_id`의 span은 체결 압축 프록시일 뿐, 하나의 parent order나 실제
호가 깊이를 직접 관측했다고 주장하지 않는다.

## 실행과 누수 방지

- `t` 종료 후 한 개의 완전한 계산 봉을 비우고 `t+2` open 진입
- 144개 5분봉 보유 후 예정 open 청산, 포지션 비중첩
- 0.5x, 기본 6bp/notional/side, 스트레스 10bp/notional/side
- funding은 `[entry, exit)`의 실제 settlement rate를 정확히 반영
- source-ID-gap 일자, 누락 feature bar와 이후 24봉은 보간 없이 격리
- strict MDD는 global/pre-entry HWM, 진입 비용, favorable-before-adverse held
  OHLC, funding, 가상 청산 비용과 실제 청산 비용을 포함
- CAGR는 warm-up과 미거래 현금을 포함한 전체 split 달력으로 계산

## 순차 검증

1. 이 문서·코드·manifest를 커밋한다.
2. 가격·미래 OHLC·funding PnL을 읽지 않고 2020–2023 event clock과 대조군
   clock을 동결한다.
3. evaluator 소스와 모든 clock hash를 별도 커밋/동결한다.
4. train 2020–2022를 먼저 열고, 통과한 경우에만 2023을 연다.
5. train과 2023 각각 절대수익 양수, CAGR/strict MDD 3 이상, MDD 15% 이하,
   주간 cluster p 0.10 이하, 비용 전 평균 기초자산 움직임 20bp 초과,
   10bp 비용 stress 양수를 요구한다. 2023 H1/H2도 각각 양수여야 한다.
6. 통과 시에만 2024를 열고, 이후 2025·2026을 같은 규칙으로 순차 개봉한다.
7. standalone 통과 후에만 기존 포트폴리오와 entry/position/daily-PnL 직교성 및
   한계 기여를 계산한다.

방향, q97.5/q90/q80/median 기준, `t+2` 진입, 144봉 hold, 비용 또는 gate는
결과를 본 뒤 수정하지 않는다. 정확한 권위 원본은
`training/preregister_aggregate_fill_compression_sweep.py`다.
