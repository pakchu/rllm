# Binance COIN-M quarterly front/next strip source design — 2026-07-19

## 목적과 경계

기존 연구는 perpetual–current-quarter carry, USD-M–COIN-M 동일만기 wedge,
spot–perpetual basis를 이미 검증했다. 이번 소스는 그 임계치를 수리하지
않고, **동일 COIN-M 담보 안에서 실제 전월물과 차월물 사이의 roll
migration**을 관찰하기 위한 별도 축이다.

이 단계는 전략 수익률, 미래 OHLC 경로, CAGR, MDD, 2024년 이후 행을
열지 않는다. Binance Vision의 contract-specific 일별 5분봉만 사용한다.
월별 ZIP은 실측상 일부 달에서 마지막 일요일까지만 포함하므로
roll-calendar 편향을 막기 위해 사용하지 않는다.

공식 아카이브:
<https://data.binance.vision/?prefix=data/futures/cm/daily/klines/>

## 고정 계약 선택

- signal-bar 범위: `2020-07-01` 이상, `2023-12-31 23:55` 미만
- feature/trade availability는 항상 `2024-01-01` 미만
- 계약: `BTCUSD_200925`부터 `BTCUSD_240628`까지 명시적 분기물
- delivery: 심볼의 `YYMMDD` 08:00 UTC
- 각 완료시각마다 delivery가 아직 지나지 않은 가장 가까운 두 계약을
  달력으로 먼저 결정한다.
- front 행이 없다고 next를 front로 승격하지 않는다.
- `feature_available_time = close_time + 1ms`, 정상 5분봉에서는 다음
  5분 경계와 같다.
- feature가 완료된 바로 그 다음 시가보다 일찍 거래할 수 없다.
- delivery 시각에 완료되는 봉은 사용할 수 없고 모든 전략 exit는
  delivery 전에 끝나야 한다.

## 무결성

- 각 ZIP은 같은 경로의 `.CHECKSUM` SHA-256으로 검증
- 정확히 한 CSV member, 고정 12-column schema
- timestamp 중복/역순 금지
- `close_time = open_time + 299,999ms`
- archive header의 `quote_volume`은 COIN-M API 경제단위상 BTC
  `base_asset_volume`, `taker_buy_quote_volume`은 BTC
  `taker_buy_base_asset_volume`으로 명시적으로 이름을 바꾼다.
- 유한 양수 OHLC와 envelope, 비음수 volume/trade count, taker volume
  상한 검증
- 전체 출력은 완전한 5분 UTC grid이며 결측은 invalid로 남기고
  forward-fill하지 않는다.

## 다음 가설 경계

지원도 단계에서만 다음 두 사건을 고려한다.

1. **next-led migration continuation**: 차월물 거래량 점유율이 높고
   차월물 taker pressure가 자기 가격에 수용되며 전월물도 반대로
   거부하지 않을 때 차월물을 같은 방향으로 짧게 거래한다.
2. **front rejected-flow fade**: 만기 45일 안에서 전월물에만 몰린 taker
   pressure가 자기 가격과 차월물에서 확인되지 않을 때 전월물을 반대로
   짧게 거래한다.

이는 단순 basis tail, funding carry, 또는 두 만기 spread 평균회귀가
아니다. 실제 만기별 aggressor flow와 거래량 migration의 정보수용/거부를
사용한다. 정확한 lookback, threshold, hold, 비용, inverse COIN-M ledger는
소스 지원도를 확인한 뒤 수익률을 보기 전에 별도 사전등록한다.

## 재현

```bash
PYTHONPATH=. /home/pakchu/rllm/.venv/bin/python \
  -m training.build_binance_coinm_quarterly_strip
PYTHONPATH=. /home/pakchu/rllm/.venv/bin/python -m pytest -q \
  tests/test_build_binance_coinm_quarterly_strip.py
```
