# COIN-M quarterly roll-migration preregistration — 2026-07-19

## 목적

기존 BTCUSDT 가격·펀딩·김프 계열과 다른 축인 **실제 인도물 계약 간 주문흐름
이동**을 사용한다. Binance COIN-M front/next 분기물 5분봉에서 완료된 현재
봉까지만 읽고, 2024년 이후와 진입 후 가격은 열지 않은 채 사건 빈도와 시간
분산만 고정한다.

## 공통 피처와 시계

- `volume`은 두 계약 모두 USD 100 액면의 **계약 수**다. front/next 비중은
  `log1p`가 아니라 원 계약 수 `V_N/(V_F+V_N)`로 계산한다.
- 체결 압력은 `(2*taker_buy_volume-volume)/volume * sqrt(volume)`이다.
- front/next 계약쌍마다 초기화한 직전 7일 rolling median/IQR로 robust z-score를
  계산한다. 현재 봉은 통계량에서 `shift(1)`로 제외하고 7일 창의 80%인 1,613개
  유효 이력이 필요하다.
- 현재 두 계약의 합산 계약 수가 같은 계약쌍의 직전 7일 25% 분위수 이상이어야
  한다. 희박한 차기물의 비율 급등을 롤 이동으로 오인하지 않는다.
- 현재 완료봉의 `log(close/open)`은 미래 성과가 아니라 체결 압력이 가격에
  수용됐는지를 판정하는 신호 입력이다.
- 모든 피처는 봉 마감 후인 다음 5분 경계에만 알려지며, 진입도 그보다 빠르지
  않다.
- front 만기 45일 이내에서만 작동하고, 고정 청산 뒤에도 front와 next 모두
  만기까지 12시간 이상 남아야 한다.

## 고정 후보 2개

### 1. Next-led roll migration continuation

- next 계약 비중 `>=10%`
- `z(next share) >= 1.0`
- `z(abs(next pressure)) >= 2.0`
- next 압력 방향의 현재 next 봉 수익률 `>=5bp`
- 같은 방향의 현재 front 봉 수익률 `>=-1bp`
- next 계약을 압력 방향으로 진입, 60분 고정 보유

해석: 유동성과 공격 주문이 차기물로 이동했고 차기물 가격이 이를 수용했으며,
근월물이 정면으로 거부하지 않은 경우의 짧은 지속을 노린다.

### 2. Front-local rejected-flow fade

- front 계약 비중 `>=50%`
- `z(front share) >= 0.75`
- `z(abs(front pressure)) >= 1.25`
- front 압력 방향의 현재 front 봉 수익률 `<=-2bp`
- `z(abs(next pressure)) <= 0.75`
- front 압력 방향의 현재 next 봉 수익률 `<=0`
- front 계약을 압력 반대 방향으로 진입, 30분 고정 보유

해석: 근월물에만 몰린 공격 주문을 근월물 가격과 차기물이 모두 거부하면,
국소적인 포지션 정리 흐름으로 보고 되돌림을 노린다.

## 지원도 통과 기준

- fit(2020-07-15~2022): 400회 이상
- 2023 selection: 100회 이상
- fit의 각 반기: 50회 이상
- 2023 각 반기: 40회 이상
- fit과 2023 각각 long/short 최소 비중 30%
- fit 월 최대 집중도 12% 이하, 2023은 20% 이하
- fit 단일 거래 계약 집중도 20% 이하, 2023은 35% 이하

후보별 non-overlap 스케줄은 각 평가 구간 시작에서 독립적으로 초기화한다.
rolling 피처 이력은 실시간 운용과 동일하게 구간 시작 전에 이미 관측된 최대
7일의 **가격 결과가 아닌 소스 상태**를 사용할 수 있으므로 live-causal이며,
스케줄의 포지션 상태만 split-contained다.
한 항목이라도 실패하면 그 후보의 진입 후 수익률은 계산하지 않는다. 지원도나
수익률을 본 뒤 방향, 임계치, 보유기간을 보정하지 않는다.

## 오염 경계

- checksum 검증된 contract-specific Binance Vision 일별 ZIP으로 만든
  `2020-07-01`~`2023-12-31 23:50` 신호봉만 사용
- 마지막 가능한 진입은 `2023-12-31 23:55`, 2024 데이터는 물리적으로 없음
- 만기 달력으로 front/next를 먼저 정하고 누락된 front를 후순위 계약으로
  승격하지 않음
- 결과 파일은 exclusive-create이며, event clock·구간별 schedule hash와 소스
  SHA-256을 함께 고정
- 소스 SHA `d107b6d...`와 build manifest SHA `cdb1ea8...`를 코드에 핀하고,
  둘 중 하나라도 달라지면 피처 계산 전에 중단
- 엄격한 역선형 COIN-M 손익 평가는 별도 평가기 코드와 테스트를 먼저 커밋한
  뒤에만 허용

## 재현

```bash
PYTHONPATH=. uv run python -m training.preregister_coinm_roll_migration_alpha
PYTHONPATH=. uv run --with pytest==8.4.1 python -m pytest -q \
  tests/test_preregister_coinm_roll_migration_alpha.py
```
