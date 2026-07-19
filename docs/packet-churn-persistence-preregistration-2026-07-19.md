# PCP-6: packet-churn persistence preregistration — 2026-07-19

## 목적과 선행 실패 경계

Minute Packet Topology의 단일 5분 `cross_venue_churn_breakout`은
2020–2022에서 양의 gross 흔적은 보였지만 strict risk-adjusted alpha가 아니어서
폐기됐다. 이 사실은 이미 알려져 있으며, 기존 신호의 방향·hold·임계치를 뒤집어
수리하지 않는다.

PCP-6은 **진입 전에 별도의 30분 상태전이를 완성**시키는 새 후보다. 원 사건 뒤에도
순방향 주문흐름이 상쇄되는 packet churn이 지속되는데 Spot과 USD-M 가격이 모두
원 이동 방향을 유지하면, 단순 일시 충격보다 재고 재가격화가 지속된 것으로 본다.
PCP-6의 2020–2022 수익과 2023 수익은 이 문서/지원 artifact 생성 중 열지 않는다.
2024 이후도 봉인한다.

## 고정 사건과 실행 시계

1. 선행 outcome-blind grid와 동일한 네 개
   `cross_venue_churn_breakout` 사건 셀만 만든다.
   - USD-M minute ticket dispersion prior `q70/q80`
   - USD-M absolute net-flow prior `q20/q35`
2. 사건 봉 뒤 정확히 6개 완료 5분봉을 관찰한다.
3. 6개 봉의 USD-M 누적 가격 이동과 Spot 누적 가격 이동이 모두 원 사건 방향으로
   양수여야 한다.
4. 6개 중 최소 3개에서 USD-M 1분 flow-sign switch rate가 `0.50` 이상이어야
   한다.
5. 여섯 번째 확인 봉 close에서 신호가 확정된다. 계산·전송을 위해 한 개 5분봉을
   완전히 비운 뒤 그 다음 시가에 원 사건 방향으로 진입한다.
6. 고정 96봉(8시간) 뒤 시가에 청산한다. 한 후보가 예약된 뒤에는 exit까지 다른
   사건을 받지 않는다. 확인 실패 사건은 포지션을 예약하지 않는다.

`signed_impact_bp = sign(net flow) × completed-bar return_bp`이므로 flow 부호를
다시 곱해 확인 구간의 이미 완료된 raw return을 복원한다. 이것은 진입 전 상태이며
진입 이후 open/high/low/close나 funding을 읽지 않는다.

## outcome-blind 지원도 선택

네 셀 가운데 **2020–2022 incidence만으로** 아래 조건을 모두 만족하는 셀만
허용한다.

- train 전체 130회 이상
- 2020·2021·2022 각각 40회 이상
- train long/short 각각 25% 이상
- train 단일 월 집중도 15% 이하

유일한 통과 셀은 `q70/q35`, 확인 6봉, hold 96봉이다.

| 셀 | 총 사건 | 2020 | 2021 | 2022 | 2023 | H1/H2 | L/S | 통과 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| q70/q20 | 126 | 35 | 36 | 25 | 30 | 17/13 | 63/63 | 아니오 |
| **q70/q35** | **192** | **48** | **54** | **45** | **45** | **25/20** | **104/88** | **예** |
| q80/q20 | 67 | 15 | 22 | 14 | 16 | 11/5 | 33/34 | 아니오 |
| q80/q35 | 110 | 21 | 34 | 28 | 27 | 16/11 | 52/58 | 아니오 |

2023의 45회와 H1/H2 `25/20`은 실행 가능성 진단으로 공개하지만 셀 선택에는
사용하지 않았다. 따라서 이후 2023 거래 수는 이미 알려진 support이지 독립적인
성과 증거가 아니다. 이 선택에는 진입 이후 수익, CAGR, MDD, 승률, funding이
사용되지 않았다.

## 다음 평가 계약

별도 evaluator와 소스 hash를 먼저 커밋·봉인한 뒤에만 다음 순서로 연다.

1. train: 2020–2022. 양의 절대수익, CAGR/strict-MDD `>=1.5`, MDD `<=15%`,
   거래 `>=100`, 10bp/side stress 양수, weekly-cluster `p<0.10`을 모두 통과해야
   2023을 연다.
2. selection: 2023 전체와 H1/H2. 전체 절대수익 양수, 비율 `>=3`, MDD
   `<=15%`, 각 반기 양수, stress 양수,
   weekly-cluster `p<0.10`을 요구한다.
3. 그 뒤에만 2024 test, 2025 eval을 순서대로 연다. 각 연도는 절대수익 양수,
   비율 `>=3`, MDD `<=15%`, 거래 `>=40`, `p<0.10`이어야 한다. 2026은
   report-only holdout이다.

기본 비용은 notional 편도 6bp, stress는 10bp, 레버리지는 0.5x다. CAGR은
무포지션 기간을 포함한 전체 달력이고, strict MDD는 global/pre-entry HWM,
진입 비용, 보유 중 favorable-before-adverse 5분 high/low, funding, 가상 청산
비용과 실제 청산 비용을 포함한다.

선행 MPT 결과와 repo의 BTC 역사 전체가 이미 연구에 노출됐으므로 시장 전체를
pristine clean room이라고 주장하지 않는다. 다만 PCP-6의 정확한 순차 clock은
수익을 읽기 전에 고정하며, 결과를 본 뒤 방향·확인 길이·hold·threshold를 바꾸지
않는다.

## 재현

```bash
PYTHONPATH=. .venv/bin/python -m training.preregister_packet_churn_persistence
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_preregister_packet_churn_persistence.py \
  tests/test_packet_churn_persistence_support_artifact.py
```
