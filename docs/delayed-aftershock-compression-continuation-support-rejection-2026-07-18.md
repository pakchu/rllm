# DACC-48 outcome-blind 지원도 기각

## 판정

**REJECTED BEFORE OUTCOMES.** DACC-48은 사전등록한 지원도 게이트를 통과하지
못했다. post-entry return, PnL, funding, CAGR, strict MDD는 계산하거나 열지
않았고 2024·2025·2026 YTD도 봉인 상태다.

## 사건 수

| 구간 | 사전등록 최소 | 비중첩 primary |
|---|---:|---:|
| 2020–2022 | 150 | **1** |
| 2020 | 30 | **1** |
| 2021 | 30 | **0** |
| 2022 | 30 | **0** |
| 2023 | 40 | **0** |
| 2023 H1 | 15 | **0** |
| 2023 H2 | 15 | **0** |

primary는 long 한 건뿐이므로 long/short 30–70%, 월 15%, UTC 주 8% 제한도
통과하지 못했다. 정확한 primary raw event도 한 건이다.

## 어느 단계에서 사라졌는가

| outcome-blind clock | raw events | 비중첩 events |
|---|---:|---:|
| 충격 직후 즉시 진입 대조군 | 1,070 | 815 |
| 압축 게이트 제거 | 7 | 7 |
| flow 게이트 제거 | 1 | 1 |
| range/breakout 게이트 제거 | 1 | 1 |
| **DACC-48 primary** | **1** | **1** |

q99.5 단일봉 충격 자체는 충분했지만, 고정된 30분 압축과 15분 flow/range
재가속을 모두 요구하면 거의 모든 사건이 제거됐다. 특히 압축 게이트를 제거해도
7건뿐이므로 단순히 support 기준만 조금 낮추면 해결되는 문제가 아니다.

## 직교성 진단

남은 한 사건은 재구성한 기존 jump continuation, jump-volume-clock,
efficient-recovery **activation proxy**와 exact entry, position-time 및 ±6시간
overlap이 모두 0이었다. 그러나 기존 전략은 TP/SL에 따라 다음 진입 시점이 달라지고
exact committed trade clock artifact가 없다. 따라서 proxy 수치는 진단용으로만
기록하고 `exact_baseline_clock_binding=false`로 fail-closed 했다. 표본도 한 건이라
직교성 통과나 경제적 증거로 사용하지 않는다.

## 무결성

- preregistration commit: `1ecda026e2fd1e568d53da16c2e153e676d17a5d`
- preregistration manifest hash:
  `4fc7b7e56cf1d691e050f2fd20d7f18afb8c863302b1856dcc199e790160419d`
- support manifest hash:
  `299c099a752429fed39887fa6e174fcbd25852ed347cc35cc3813c29d5b53713`
- clock SHA256:
  `8fc098af70cfef698c09dcad4aebd8ea124b9e858f948000c5b0f7dcf919d0ff`
- source: official Binance USD-M 5분 kline 420,768행
- source의 zero-volume 26봉은 결측 보간하지 않고 signal window에서 fail-closed
- clock에는 위치·시각·방향·고정 hold만 있고 가격·수익률·funding 열이 없다.

## 연구 결정

q99.5, shock scale 4배, 6봉 압축, 3봉 재가속, flow/range 기준 또는 48봉 hold를
완화해 DACC-48을 수리하지 않는다. 이 exact family는 outcome을 열지 않은 채
종료한다. 다음 후보는 jump→compression 순서가 아닌 별도의 경제적 전달 경로와
지원도가 높은 clock을 먼저 요구한다.
