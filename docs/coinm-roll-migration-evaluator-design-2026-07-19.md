# COIN-M roll-migration strict evaluator design — 2026-07-19

## 목적과 동결 순서

지원도를 통과한 두 사건의 2020-07-15~2022 fit과 2023 selection 성과만
평가한다. 평가기 코드·테스트·비용·게이트를 먼저 커밋하고 evaluator freeze를
별도 exclusive artifact로 만든 뒤에만 진입 후 OHLC를 읽는다. 2024 이후는 계속
물리적으로 봉인한다.

## 역선형 COIN-M 원장

BTCUSD 분기물 한 계약의 액면은 USD 100으로 고정한다. 계약 수 `N`, 진입가
`P0`, 평가가 `Pt`, 방향 `s`일 때 BTC 손익은 다음과 같다.

```text
coin_pnl = s * N * 100 * (1/P0 - 1/Pt)
usd_pnl  = coin_pnl * Pt
```

따라서 고정 USD 액면 대비 평가 손익률은 정확히
`s * (Pt/P0 - 1)`이다. 연구 원장은 매 거래 직전 USD equity의 0.5배에 해당하는
fractional contract를 사용한다. 이는 알파 비교용이며 실제 정수 계약 반올림과
BTC 담보 자체의 가격 베타는 모델링하지 않는다. 따라서 통과하더라도 바로
production 승격할 수 없다.

분기물에는 funding이 없고, 기본 비용은 진입/청산 각각 6bp, stress는 각각
10bp다. 비용은 고정 USD 액면에 가산적으로 부과한다.

## 엄격 MDD

- 전체 달력 기간을 CAGR 분모로 사용해 무포지션 기간도 포함
- 이전 거래와 진입 전 equity의 global HWM 유지
- 진입 비용을 즉시 반영
- 보유 5분봉의 long `high/low`, short `low/high`로 favorable/adverse mark 계산
- 봉 내부 순서를 알 수 없으므로 favorable HWM을 먼저 인정한 뒤 adverse를 적용
- adverse mark에는 이미 낸 진입 비용과 가상 즉시 청산 비용을 모두 적용
- 고정 청산에도 양쪽 비용 적용
- 거래 경로에 결측 OHLC나 계약 전환이 있으면 거래를 건너뛰지 않고 평가 전체를
  중단

## 고정 통과 기준

- fit과 2023 모두 절대수익 양수
- fit과 2023 모두 `CAGR / strict MDD >= 3`
- fit과 2023 모두 strict MDD `<=15%`
- fit 400회, 2023 100회 이상
- fit과 2023 mean net bps 양수
- 주별 cluster sign-flip one-sided `p<0.10` 양쪽 구간 모두
- fit 5개 반기 중 4개 이상 양수, 각 50회 이상
- 2023 H1/H2 모두 양수, 각 40회 이상
- 10bp/side stress에서도 fit·2023 모두 양수
- 방향 반전은 fit·2023 모두 음수
- 원 신호가 1시간·24시간 지연 대조군보다 fit·2023 모두 우월

지연 대조군은 raw event clock을 다시 non-overlap 처리하지 않는다. 이미 동결된
base schedule의 동일 거래만 일괄 이동하고, 구간/동일 계약/만기 안전성 때문에
실행 불가능해진 행만 제거한다. 따라서 지연 과정에서 원래 건너뛴 사건이 새로
편입되지 않으며 대조군 schedule hash도 결과에 기록한다.

한 항목이라도 실패하면 2024를 열지 않는다. 결과를 본 뒤 방향, hold,
임계치, 비용, 게이트를 바꾸지 않는다.

## 재현 순서

```bash
PYTHONPATH=. uv run --with pytest==8.4.1 python -m pytest -q \
  tests/test_evaluate_coinm_roll_migration_pre2024.py

# 평가기 코드 커밋 뒤
PYTHONPATH=. uv run python -m training.evaluate_coinm_roll_migration_pre2024 \
  --freeze-only

# freeze artifact 커밋 뒤에만 실행
PYTHONPATH=. uv run python -m training.evaluate_coinm_roll_migration_pre2024
```
