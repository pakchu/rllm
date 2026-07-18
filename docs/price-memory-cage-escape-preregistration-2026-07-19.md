# Price-memory cage escape preregistration — 2026-07-19

## 목적

이미 각각 약한 방향성만 보였던 피처를 점수로 더하지 않고, 서로 다른
시장 기억이 **같은 순간에 같은 방향으로 해제되는 사건**만 고정한다.
이번 단계에서는 미래 수익률, 고가/저가, 펀딩 결과를 읽지 않고 사건의
지원도와 시간 분산만 검사한다. 2024년 이후 데이터는 열지 않는다.

## 고정 사건

시간 `t`의 5분봉이 끝났을 때 아래 조건을 모두 만족해야 한다.

1. 직전 minute-55 봉에서, 직전 30 UTC일로 매일 동결한 64-bin
   시간×거래대금 점유 지형의 saddle을 돌파했다.
2. 점유 지형에 saddle이 최소 2개 존재한다.
3. 직전 1시간 이동이 그보다 앞서 동결한 2일 또는 7일 local-extrema
   persistence 장벽을 같은 방향으로 최소 3개 제거했다.
4. 직전 24시간 거래대금을 `shift(1)`해 만든 0.25배 volume clock의
   taker signed-flow speed 부호도 같은 방향이다.
5. 세 방향이 같을 때만 제약 해제 후 continuation으로 진입한다.

모든 피처가 완료되는 `t+5분`의 다음 5분봉 시가보다 일찍 진입하지
않는다. 고정 보유기간은 12시간과 24시간이며 후보는 총 4개다.

## 지원도 통과 기준

- fit(2020-10-15~2022): 48회 이상
- 2023 selection: 24회 이상
- 2021H1/H2, 2022H1/H2 각각 6회 이상
- 2023H1/H2 각각 8회 이상
- fit과 2023 각각 long/short 최소 비중 20%
- fit 월 최대 집중도 20% 이하, 2023은 25% 이하

한 항목이라도 실패하면 수익률을 계산하지 않고 후보를 폐기한다.
지원도 통과 후에도 방향 반전, 임계치 완화, 보유기간 추가는 허용하지
않는다.

## 오염 경계

- 물리적으로 `2024-01-01` 전에 잘린 Binance USD-M BTCUSDT 5분봉만 사용
- 결과를 보지 않은 상태에서 방향과 모든 임계치를 이 문서와 코드로 고정
- 지원도 산출물은 exclusive-create로 작성해 같은 경로를 덮어쓰지 않음
- 수익률 평가는 별도 평가기와 별도 동결 커밋 뒤에만 허용

## 재현

```bash
PYTHONPATH=. .venv/bin/python -m training.preregister_price_memory_cage_escape_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_preregister_price_memory_cage_escape_alpha.py
```
