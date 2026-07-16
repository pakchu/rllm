# LVRT-R0 사전등록 — Liquidity Vacuum Replenishment Transition

## 결론부터

LVRT-R0는 기존 포트폴리오의 OI·펀딩·김프·REX·Markov 피쳐를 사용하지 않는다. Binance USD-M BTCUSDT aggTrade에서 관측한 **버스트성**, **이벤트 명목 집중도**, **공격적 매수/매도 흐름**, **그 다음 5분 봉의 반대 흐름과 가격 반전**만으로 한 시간짜리 역방향 포지션을 만든다.

이 문서는 수익률을 열기 전에 단 하나의 정책을 고정한다. 2020–2023 시장 수익률은 다른 연구에서 이미 본 적이 있으므로 완전히 깨끗한 시장 홀드아웃이라고 주장하지 않는다. 다만 아래 정확한 조합의 결과는 아직 열지 않았으며, 2024 이후는 계속 봉인한다.

## 경제적 가설과 한계

1. `t0`에서 거래 도착이 버스트하고 명목이 소수 이벤트에 집중된다.
2. 같은 방향 공격적 체결이 실제 가격을 움직인다.
3. 바로 다음 완료 봉 `t1`에서 공격적 흐름과 가격이 함께 반전된다.
4. 이를 직접 호가 잔량이 아닌 **refill transition의 체결 기반 프록시**로 해석하고 `t0` 흐름의 반대 방향을 한 시간 보유한다.

aggTrade에는 L2 호가가 없으므로 수동 주문의 실제 재충전, 큐 위치, hidden order는 주장하지 않는다.

## 고정 피쳐와 시계

- `s0 = sign(signed_quote_notional[t0])`, `s0 != 0`
- `signed_price_response[t0] > 0`
- `interarrival_burstiness[t0] >=` 직전 clean 관측 8,640개의 q80
- `event_notional_hhi[t0] >=` 직전 clean 관측 8,640개의 q80
- 두 기준선 모두 최소 2,016개 clean 관측 필요
- `t0`, `t1` 모두 `agg_trade_count >= 64`
- `sign(signed_quote_notional[t1]) == -s0`
- `s0 * micro_log_return[t1] < 0`
- 판단은 `t1` 종가 후, 진입은 다음 5분 시가 `t2`
- 포지션 방향 `-s0`, 12봉(60분) 고정 보유 후 예정 시가 청산
- 0.5x, 기본 비용 6bp/notional/side, 스트레스 8bp/notional/side
- stop/TP/동적 청산 없음, 포지션 비중첩

기준선은 반드시 한 봉 shift한 과거 clean 관측만 사용한다. setup, confirmation, entry, 전체 보유 경로, exit 중 하나라도 소스 gap 격리 구간에 걸리거나 split을 넘으면 거래를 버린다.

## 데이터 계약

- aggTrade feature SHA256: `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- 5분 kline SHA256: `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- source audit SHA256: `5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6`
- audit는 PASS 상태여야 한다.
- 검증된 source-ID-gap UTC 일자, 누락 5분 슬롯, 이후 24봉은 보간하지 않고 격리한다.

## outcome 이전 지원도 게이트

- 2020–2023 비중첩 거래 250개 이상
- 각 연도 40개 이상
- 2023 H1/H2 각 20개 이상
- long/short 각 25–75%
- 단일 월 집중도 20% 이하

하나라도 실패하면 가격 수익률을 열지 않는다.

## train/selection 게이트

- train: 2020-01-01 ~ 2023-01-01
- selection: 2023-01-01 ~ 2024-01-01
- 2024·2025·2026 YTD: 봉인
- train과 2023 각각 절대수익 > 0, CAGR/strict MDD >= 3, strict MDD <= 15%, 주간 cluster sign-flip p <= 0.10
- train 거래 120개 이상, 2023 거래 80개 이상
- 2023 H1/H2 각각 절대수익 > 0, 거래 20개 이상
- train과 2023 각각 평균 비용 전 기초자산 움직임 > 12bp
- 8bp/notional/side 비용 스트레스에서도 train과 2023 절대수익 > 0

CAGR는 거래하지 않은 현금 기간도 포함한 전체 달력으로 계산한다. strict MDD는 진입 전 global HWM, 진입 비용, 보유 중 유리한 극값 후 불리한 극값, 가상 청산 비용, 실제 청산 비용을 포함한다.

## 위약·대조군

- 동일 시계의 방향 반전
- 반전 확인을 제거한 setup-only 정적 신호
- 진입을 한 봉 더 지연
- setup을 정확히 하루 전으로 이동
- seed `20260717`로 confirmation 부호를 순열화하는 기각 대조군

대조군은 정책 수리나 대체 후보가 아니다. 하루 이동 또는 부호 순열 placebo가 primary의 전체 성능 게이트를 독립적으로 통과하면 LVRT-R0를 기각한다.

## 직교성 판정

pre-2024 성능을 통과한 경우에만 재현 가능한 기존 live/shadow 진입 클록과 비교한다. exact entry Jaccard <= 0.05, position-time Jaccard <= 0.15, 일별 PnL 절대 Pearson <= 0.30을 요구하고, 기존 포트폴리오에 추가했을 때 한계 성능 개선도 확인한다.

정책이 실패하면 q80, 확인 방식, 방향, 12봉 보유, 비용 또는 게이트를 바꾸지 않는다. 실패한 LVRT-R0를 그대로 기각하고 별도 아이디어를 새로 사전등록한다.
