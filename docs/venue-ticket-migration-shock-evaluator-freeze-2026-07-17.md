# VTMS-288 strict evaluator 결과 비개봉 봉인

## 봉인 상태

- evaluator source commit:
  `315418a24e76cfe81dda7e39cbe5829a097592d2`
- evaluator source SHA-256:
  `9456e3f51b21cdce11d42860cd6ec9d1dce3ce4889fd007b11ee541c6b70cd27`
- freeze artifact SHA-256:
  `9af4523bc031a4b4017aa6430ddf7e69469cc47d4ffcbd4f3d2dc9a32f2e6ac9`
- manifest hash:
  `1eeb879d38d4757a3c01b2bccdb25d33798630b65763c6d96dc5af3e0fda1239`
- opened windows: 없음
- mutable parameters: 없음
- freeze 중 읽은 execution OHLC: 0행
- freeze 중 읽은 funding settlement mark: 0행
- freeze 중 실행한 성과 simulation: 없음

2020–2022 train, 2023 selection, 2024 이후 데이터는 이 봉인 시점에 모두
닫혀 있다. 봉인은 support replay와 primary 포함 10개 clock의 행 수 및 hash만
검증했다.

## 고정 회계와 gate

- 0.5x fixed-quantity linear USD-M ledger
- 기본 비용 6bp/notional/side, 스트레스 비용 10bp/notional/side
- funding은 `[entry, exit)` 실제 rate와 고정 settlement-mark proxy로 반영
- global/pre-entry HWM
- held bar마다 favorable mark를 먼저, adverse mark와 가상 청산 비용을 나중에 반영
- 실제 진입·청산 비용 포함
- warm-up과 idle cash를 포함한 전체 달력 CAGR
- train 최소 250건, 2023 최소 75건, 2023 각 반기 최소 35건
- 절대수익 양수, CAGR/strict MDD 3 이상, strict MDD 15% 이하
- weekly cluster sign-flip p-value 0.10 이하
- 평균 gross underlying move 20bp 초과, 10bp/side stress 절대수익 양수
- primary ratio가 ticket level/shock, coherence, price acceptance 제거 대조군을
  각각 상회해야 함
- 1시간/1일 time shift 또는 random-side가 full gate를 통과하면 후보 폐기

## 누수 방어

- Stage 1 parser는 2023 첫 행을 수치 변환하기 전에 물리적으로 중단한다.
- Stage 2 전에 Stage 1 결과를 같은 evaluator로 결정론적으로 전부 재실행해
  byte-level payload까지 대조한다.
- 후보 ID, evaluator source, support commit, execution/funding source hash,
  전체 control clock 집합을 freeze에서 다시 검증한다.
- 신호 시점 source gap과 직후 24봉만 격리한다. 진입 뒤 새로 발생한 source
  gap을 보고 이미 예정된 거래를 사후 삭제하지 않는다. 이는 미래 데이터
  가용성으로 거래를 censor하는 누수를 막기 위한 인과적 계약이다.
- Spot-dominant와 USD-M-dominant branch는 고정 정책 변경에 사용하지 않는
  별도 진단 결과로 함께 출력한다.

## 개봉 순서

1. 2020–2022 train만 한 번 연다.
2. 모든 train gate를 통과할 때만 2023을 연다.
3. train 실패 시 2023과 2024 이후는 계속 봉인하고 VTMS-288을 폐기한다.
4. 2023까지 standalone gate를 통과하기 전에는 기존 portfolio와의 직교성
   결과를 열지 않는다.
