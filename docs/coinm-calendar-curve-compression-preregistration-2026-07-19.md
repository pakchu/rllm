# COIN-M calendar-curve compression preregistration — 2026-07-19

## 가설

동일 COIN-M 담보의 실제 전월물과 차월물 가격비는 BTC 방향 노출을 대부분
상쇄한다. 계약쌍 내부의 차월물/전월물 로그 곡선이 과거 분포 밖으로 처음
이탈한 뒤 별도의 완료 봉에서 수렴을 시작하면, 남은 곡선 왜곡이 12시간 안에
추가 압축될 수 있다. 이는 직전 연구의 단일 계약 방향성 롤 이동과 다른
**2-leg calendar spread** 가설이다.

## 결과를 보기 전에 고정한 규칙

- 원천: checksum으로 봉인된 COIN-M front/next v2 5분봉, 2020-07-01부터
  2023-12-31까지만 사용한다.
  - source SHA-256:
    `d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b`
  - manifest SHA-256:
    `29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af`
- feature: `log(next_close/front_close)`.
- 계약쌍마다 초기화되는 14일 rolling median/IQR를 쓰되 현재 봉은 제외하고
  최소 7일의 선행 이력만 허용한다.
- shock: 직전 완료 봉이 `|z| >= 2`로 처음 진입한다.
- confirmation: 다음 완료 봉에서 shock 방향과 반대로 최소 2bp 움직이고
  `|z|`가 감소해야 한다. median 반대편으로 넘은 경우는 제외한다.
- 진입 시 남은 curve 왜곡은 최소 15bp다. 두 계약 모두 shock/confirmation
  봉 거래량이 각자의 strictly-prior 25% 분위수 이상이어야 한다.
- 전월물 만기는 10~75일이고, 12시간 보유와 추가 12시간 delivery buffer를
  모두 만족해야 한다.
- confirmation 다음 open에서 진입한다. 차월물이 비싸면 차월물 short/전월물
  long, 싸면 반대로 한다.
- 계정 총 gross 0.5를 두 inverse leg에 0.25씩 동일 USD face로 배분한다.
- 고정 보유 12시간, 중복 포지션 금지, delivery futures라 funding은 없다.
- 비용은 각 leg 각 side 6bp, stress는 10bp다.

## 데이터 분리와 통과 기준

- fit: `[2020-07-15, 2023-01-01)`
- selection: `[2023-01-01, 2024-01-01)` 및 2023 H1/H2
- 2024 이후는 열지 않는다.
- 단일 후보만 평가한다.
- fit과 2023 모두 절대수익 양수, full-calendar CAGR/strict MDD `>=3`,
  strict MDD `<=15%`, weekly-cluster sign-flip `p<=0.10`이어야 한다.
- 2023 양 반기 모두 양수여야 하며 10bp stress에서도 양수여야 한다.
- 평균 gross curve compression은 최소 12bp여야 한다.
- 통제실험은 같은 동결 clock의 direction flip, 1시간 지연, 24시간 지연이다.
- support 집중도 상한은 fit/2023 단일 월 15%/25%, 단일 계약쌍
  25%/40%다. 분기별 계약 교체 구조 때문에 반기 내부 계약쌍 집중도는 gate로
  사용하지 않는다.

## strict ledger 경계

평가기에서는 두 inverse leg를 모두 보유하고 각 leg의 독립적인 favorable
extreme을 먼저, adverse extreme을 다음으로 표시해 보수적인 intratrade
strict MDD를 계산한다. global/pre-entry HWM, 진입 비용, adverse 지점의 가상
청산 비용, 실제 청산 비용을 모두 포함한다. 동일 USD face는 1차 BTC 방향을
상쇄하지만 collateral beta, integer contract rounding, 주문장 충격은 아직
모델링하지 않으므로 통과하더라도 shadow 검증 전 실거래 승격은 금지한다.

## 지원도만으로 선택한 이유

15bp residual은 2-leg 왕복 비용의 curve break-even 약 12bp보다 커야 한다는
경제적 하한이다. 12시간 hold는 결과를 열지 않은 support count에서 fit 150회,
2023 50회, 각 반기 25회라는 최소 통계 지원도를 유지한 유일한 단일 고정
후보다. 이 문서와 구현을 커밋하기 전에는 post-entry return을 계산하지 않았다.
