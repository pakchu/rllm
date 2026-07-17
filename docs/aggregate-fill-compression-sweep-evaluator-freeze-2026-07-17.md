# AFCS-144 strict evaluator 결과 비개봉 봉인

## 봉인 상태

- evaluator source commit: `ae675ddd9599eb420fccad3c691295221a9a0353`
- evaluator source SHA-256: `1e73f2e12c0b9588ad882aea34b424dc99cb702265137d1f221dc88094d3c71c`
- freeze artifact SHA-256: `78ef1f8f72fa3cfee81f45a317d8044dcd217753f070bde08f6173ef98ad4012`
- manifest hash: `48b9e5653f2926456e71ce0be1adffa9de435d5caf54ba1f56d9ade3dab3af8a`
- opened windows: 없음
- mutable parameters: 없음
- freeze 중 읽은 execution OHLC: 0행
- freeze 중 읽은 funding settlement mark: 0행
- freeze 중 실행한 성과 simulation: 없음

2020–2022 train, 2023 selection, 2024 이후 데이터는 이 봉인 시점에 모두 닫혀
있다. 봉인은 support replay와 8개 primary/control clock의 행 수 및 hash만 검증했다.

## 고정 회계와 개봉 방어

- 0.5x fixed-quantity linear USD-M ledger
- 기본 비용 6bp/notional/side, 스트레스 비용 10bp/notional/side
- funding은 `[entry, exit)` 실제 rate와 고정 settlement-mark proxy로 반영
- global/pre-entry HWM
- held bar마다 favorable mark를 먼저, adverse mark와 가상 청산 비용을 나중에 반영
- 실제 진입·청산 비용 포함
- warm-up과 idle cash를 포함한 전체 달력 CAGR
- Stage 2 전에 Stage 1 manifest hash, 전체 gate, freeze hash, evaluator commit과
  물리 cutoff를 재검증
- 1시간/1일 time shift와 random-side placebo의 full qualification을 거부

## 개봉 순서

1. 물리 parser가 2023 첫 행을 만나면 수치 변환 전에 중단한다.
2. 2020–2022가 모든 사전등록 gate를 통과할 때만 2023을 연다.
3. 실패하면 2023 및 2024 이후는 계속 봉인하고 AFCS-144를 폐기한다.
4. standalone 통과 전에는 기존 portfolio와의 직교성 결과를 열지 않는다.
