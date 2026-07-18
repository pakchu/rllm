# COIN-M roll-migration v2 evaluator design — 2026-07-19

v2 evaluator는 v1 성과를 보고 바꾼 전략이 아니다. v1 실행은 결측 경로에서
candidate statistic 생성 전에 예외로 중단됐고 selection artifact도 없다.

## 고정 재사용

- v1 strict evaluator logic SHA:
  `b206c5cfd076777557bcb044791a61af157498a0d36360582fbc219786522ca0`
- 0.5x fractional fixed-USD-face inverse ledger
- 6bp/side base, 10bp/side stress
- full-calendar CAGR
- global/pre-entry HWM + held 5분봉 high/low + 양쪽 비용 strict MDD
- 100,000회 weekly cluster sign-flip
- 방향 반전, 동일 frozen schedule의 1시간/24시간 지연 대조군
- `CAGR/strict MDD >=3`, strict MDD `<=15%`를 포함한 모든 gate

v2에서 달라지는 것은 repaired source/manifest SHA, v2 support freeze hash, artifact
경로뿐이다. support에서 자동 탈락한 front-fade는 성과를 계산하지 않고,
next-led 한 후보만 fit/2023을 평가한다. evaluator source와 config를 별도 freeze
artifact로 먼저 커밋한 뒤에만 진입 후 OHLC를 연다. 2024 이후는 계속 봉인한다.
