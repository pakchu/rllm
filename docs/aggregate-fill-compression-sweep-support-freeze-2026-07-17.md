# AFCS-144 outcome-blind 지원도·클록 동결

## 판정

**PASS_SUPPORT. 가격 결과는 열지 않았다.**

지원도 프로그램은 market gzip의 SHA256을 검증한 뒤 `date` 열만 읽었다.
`open`, `high`, `low`, `close`, 미래 수익률, funding PnL, CAGR, MDD는 로드하거나
계산하지 않았다. aggTrade 입력도 사전등록된 동시점 피처 6개만 읽었다.

## 고정 사건 수

| 구간 | 비중첩 사건 |
|---|---:|
| 2020 | 123 |
| 2021 | 61 |
| 2022 | 237 |
| train 2020–2022 | **421** |
| 2023 H1 | 87 |
| 2023 H2 | 65 |
| selection 2023 | **152** |
| 전체 | **573** |

- 전체 방향: long 310 / short 263
- 가장 큰 단일 월 비중: 5.41%
- train 반기 최소 사건 수: 22
- 모든 사전등록 support floor 통과

2023 수치는 feature/event-clock 지원도 진단일 뿐이며 2023 진입 이후 가격이나
손익은 보지 않았다.

## 소스와 인과성

- aggTrade feature SHA256:
  `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- market SHA256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- source audit SHA256:
  `5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6`
- missing feature bar 36개, 격리 bar 1,682개
- q97.5/q90/q80/median 기준은 현재 봉을 제외한 직전 clean 관측치만 사용
- signal 이후 한 봉을 비우고 `t+2` open 진입, 144봉 hold
- 반기 경계 밖으로 나가는 사건은 버리고 다음 반기에서 scheduler를 재시작

## 동결 클록

- primary rows: 573
- primary clock SHA256:
  `bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688`

| 대조군 | rows | clock SHA256 앞 12자 |
|---|---:|---|
| direction flip | 573 | `f8375fd91d92` |
| compression 제거 | 2,109 | `cf63849e84f9` |
| coherence 제거 | 927 | `09f77445c175` |
| aligned response 제거 | 739 | `4acb26724061` |
| 1시간 지연 | 572 | `290e57730579` |
| 1일 이동 | 563 | `f4a5ef70a111` |
| 고정-seed random side | 573 | `b17737c85899` |

지원 artifact manifest hash는
`670b56bd57e5466e805c596668330b0544da997ef0e832e29a9ed6f56ec177f9`다.

다음 단계는 evaluator 소스와 모든 대조군 클록 hash를 먼저 커밋·동결한 뒤,
2020–2022 train 손익을 단 한 번 여는 것이다. train gate가 실패하면 2023과
2024+는 계속 봉인한다.
