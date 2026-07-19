# PCP-6 strict sequential evaluator contract — 2026-07-19

## 목적

이 문서는 `47cfd8a`에서 수익을 보지 않고 고정한 PCP-6 clock을 평가하는 계약이다.
지원도만 통과한 후보를 수익성 alpha로 오인하지 않으며, **train artifact가 모든
gate를 통과하기 전에는 2023 OHLC와 funding rate를 파싱하지 않는다.** 2024 이후
minute-packet feature는 아직 만들지 않았으므로 계속 봉인한다.

## 고정 데이터와 단계

| 단계 | 값 컬럼을 읽는 기간 | 다음 단계 조건 |
|---|---|---|
| freeze | 없음. timestamp만 읽어 물리적 경계를 확인 | evaluator/source/clock hash 고정 |
| train | 2020-01-01 ≤ t < 2023-01-01 | train gate 전부 통과 |
| selection | 2023-01-01 ≤ t < 2024-01-01 | selection gate 전부 통과 |
| forward | 봉인 | 별도 2024+ feature source를 먼저 생성·hash 고정 |

freeze는 feature와 official USD-M 5분 OHLC의 timestamp grid가 완전히 같은지
확인하고, funding timestamp의 2023 경계를 기록한다. 이 과정에서
`open/high/low/close`와 `funding_rate`는 읽지 않는다. train loader는 freeze에
기록된 `nrows`만 읽으므로 2023 첫 행 이후의 값이 파싱되지 않는다. selection
loader는 write-once train 결과의 hash, 정책명, 모든 gate, `open_selection_2023`
결정을 확인한다. 이어서 동결된 pre-2023 prefix로 train 전체 통계·stress·control을
재연산해 artifact와 정확히 일치할 때만 호출된다. self-consistent한 수기 JSON은
selection을 열 수 없다.

## 실행·비용·위험 계약

- 주 신호: `pcp_cross_venue_churn_breakout_p70_s35_h96_confirm6`
- signal 확정: 여섯 번째 확인 5분봉 close
- 진입: 계산·전송용 5분봉 하나를 완전히 비운 뒤 다음 open
- 청산: 진입 96봉 뒤 open
- 레버리지: `0.5x`
- 기본 편도 비용: fee 5bp + slippage 1bp = 6bp
- stress 편도 비용: 10bp
- funding: 보유 구간의 Binance USD-M realized funding. 진입/청산 timestamp와
  settlement가 정확히 겹치면 debit만 포함하고 credit은 제외한다.
- CAGR: 무포지션 시간을 포함한 window 전체 달력
- strict MDD: global/pre-entry HWM, 진입 비용, 보유 중 5분 high/low의
  favorable-before-adverse 순서, realized funding debit, 가상/실제 청산 비용을
  포함한다.

TP/SL, regime gate, 방향 수정, threshold 수정은 없다. 결과를 본 뒤 주 신호를
control로 교체할 수 없다.

## 결과와 무관하게 고정한 control

1. 같은 clock의 side flip
2. signal 확정 시각에 바로 진입하는 immediate-entry(`+1` bar) control
3. 주 신호보다 한 봉 더 늦게 진입하는 extra-latency(`+3` bars) control

각 control은 진입과 청산을 함께 이동해 96봉 hold를 보존하며, freeze artifact에
실행 clock hash가 기록된다. control은 기전 진단만 담당하고 승격 대상이 아니다.

## Gate

### Train: 2020–2022

- 절대수익률 `> 0%`
- CAGR / strict MDD `>= 1.5`
- strict MDD `<= 15%`
- 거래 수 `>= 100`
- 편도 10bp stress 절대수익률 `> 0%`
- weekly-cluster sign-flip one-sided `p < 0.10`

연도별 2020/2021/2022 통계도 공개하지만 추가 최적화나 gate에는 쓰지 않는다.

### Selection: 2023

- 전체 절대수익률 `> 0%`
- CAGR / strict MDD `>= 3`
- strict MDD `<= 15%`
- H1과 H2 절대수익률이 각각 `> 0%`
- 편도 10bp stress 절대수익률 `> 0%`
- weekly-cluster sign-flip one-sided `p < 0.10`

2023 거래 수와 H1/H2 incidence는 support 단계에서 이미 공개됐으므로 독립 gate로
재사용하지 않는다. selection을 통과해야만 2024+ raw minute packet source를 새로
구축하는 비용을 지불한다.

## 실행 순서

```bash
# 1. outcome 값 미개봉 상태에서 evaluator를 봉인한다.
PYTHONPATH=. .venv/bin/python \
  -m training.evaluate_packet_churn_persistence_pre2024 --stage freeze

# 2. freeze commit 이후 train만 연다.
PYTHONPATH=. .venv/bin/python \
  -m training.evaluate_packet_churn_persistence_pre2024 --stage train

# 3. train artifact가 통과했을 때만 실행 가능하다.
PYTHONPATH=. .venv/bin/python \
  -m training.evaluate_packet_churn_persistence_pre2024 --stage selection
```

검증:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_evaluate_packet_churn_persistence_pre2024.py
uvx ruff check \
  training/evaluate_packet_churn_persistence_pre2024.py \
  tests/test_evaluate_packet_churn_persistence_pre2024.py
```
