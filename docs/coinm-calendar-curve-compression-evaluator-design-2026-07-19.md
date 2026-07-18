# COIN-M calendar-curve compression evaluator design — 2026-07-19

## 목적

support commit `68b8c15`의 단일 clock을 결과 확인 전에 전용 2-leg strict
evaluator에 묶는다. 기존 COIN-M roll evaluator는 단일 계약 방향성 원장이므로
재사용하지 않는다.

## pair ledger

- 차월물/전월물은 반대 방향, 동일 USD face로 보유한다.
- 계정 총 gross 0.5, leg당 0.25다.
- inverse contract의 coin PnL `side * face * (1/entry - 1/mark)`를 각 mark
  가격으로 USD 환산한다.
- 같은 비율로 두 계약이 움직이면 1차 USD 방향 PnL은 0이다.
- 각 leg 각 side 6bp, stress 10bp를 진입과 청산에 모두 부과한다.
- delivery futures funding은 없다.

## strict MDD

각 보유 5분봉마다 두 leg를 독립적으로 가장 유리한 극값 조합에 먼저 mark해
HWM을 올린 다음, 두 leg의 독립적인 가장 불리한 극값 조합과 가상 청산 비용을
반영한다. 이는 실제 동시 발생 가능성보다 보수적인 상계 없는 bound다.
global/pre-entry HWM, 진입 비용, 실제 청산 비용도 포함한다. CAGR은 거래하지
않은 날을 포함한 전체 구간 시간으로 계산한다.

## clock 및 controls

- evaluator freeze는 정확한 support commit, support manifest hash, support artifact
  SHA와 봉인 기간 목록에 결합되며 재해시된 변조도 거부한다.
- fit/selection 부모 non-overlap clock을 support hash와 다시 대조한다.
- 하위 구간은 부모 clock을 entry time으로만 자르고 재스케줄링하지 않는다.
- direction flip은 같은 clock에서 두 leg를 동시에 반전한다.
- 1시간/24시간 delay는 동결 clock을 일괄 이동할 뿐 event를 다시 검출하지
  않는다. 계약쌍이 바뀌거나 delivery buffer를 위반하면 fail-closed로 제외한다.
- scheduled path에 OHLC 누락 또는 계약 전환이 있으면 결과를 만들지 않고
  evaluator 전체가 실패한다.

## 통과 조건

사전등록에 명시한 fit/2023 절대수익, CAGR/strict MDD 3, MDD 15%, 평균 gross
curve compression 12bp, weekly cluster p-value 0.10, 2023 양 반기, 10bp stress
조건을 모두 요구한다. direction flip은 음수여야 하고 원 clock은 1시간/24시간
지연 control보다 높은 절대수익을 내야 한다. 2024 이후는 통과 후에도 별도
승격 없이는 열지 않는다.

fractional contract만 사용하며 integer rounding, BTC collateral beta, margin
transfer, order-book impact는 모델링하지 않는다. 따라서 pre-2024 통과는
shadow 후보 자격일 뿐 실거래 승격이 아니다.
