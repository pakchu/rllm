# COIN-M quarterly source v2 repair — 2026-07-19

## 발견

v1 strict evaluator는 성과 artifact를 쓰기 전에 중단됐다. 동결된 next-led 거래
중 한 건(`2023-12-20 23:10` signal, `BTCUSD_240329`)이 60분 보유 중
`2023-12-21 00:00~00:15` OHLC 결측을 통과했기 때문이다. 해당 거래를 사후
삭제하면 데이터 가용성에 의한 선택 편향이므로 허용하지 않는다.

일별 Binance Vision listing에는 그 날짜의 두 필요 계약 ZIP이 없었지만, 공식
월별 ZIP에는 `2023-12-21` 전체 288개 5분봉이 존재했다.

- `BTCUSD_231229-5m-2023-12.zip` checksum:
  `f9b7c4aab310052fe3f7b6a5865e821dd40ade7868ddd39ee4805fbb0b3e2a78`
- `BTCUSD_240329-5m-2023-12.zip` checksum:
  `72a9516fefd147e89516928d67e6caacaef74e29e4d691080664f6ed8fe73827`

v1 실행에서는 candidate return, CAGR, MDD, p-value가 하나도 생성되지 않았고
`results/coinm_roll_migration_pre2024_selection_2026-07-19.json`도 없다.

## v2 고정 보강 규칙

1. 공식 checksum 검증 일별 contract-specific 5분봉을 primary로 사용한다.
2. 고정 front/next 달력에 필요한 `(symbol, timestamp)`가 일별 raw에 없을 때만
   해당 symbol/month의 공식 월별 ZIP을 받는다.
3. 공식 월별 archive에는 일별 archive를 사후 정정한 소수 행이 실제 존재한다.
   겹치는 행은 항상 일별을 유지하고, 불일치 행의 개수·비율·전체 차이 payload
   SHA-256을 manifest에 기록한다. 월별 중첩 값은 출력에 사용하지 않는다.
4. 월별은 오직 absent key만 추가한다.
5. invalid 일별 행을 월별 행으로 교체하지 않는다.
6. 2024 이후 봉은 계속 열지 않는다.

알파 방향, 임계치, hold, support gate와 evaluator gate는 v1에서 고정한 값을
그대로 재사용한다. 소스 보강 뒤 새 SHA로 support부터 다시 exclusive-freeze하고,
새 evaluator를 동결하기 전에는 진입 후 성과를 다시 계산하지 않는다.
