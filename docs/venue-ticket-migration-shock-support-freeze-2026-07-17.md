# VTMS-288 결과 비공개 지원도·클록 동결

## 판정

**PASS_SUPPORT. 가격 결과는 열지 않았다.**

지원 빌더는 실행시장 gzip의 SHA256을 검증한 뒤 `date` 열만 읽었다. Spot과
USD-M 입력에서도 사전등록된 동시점 ticket/flow/response 열만 사용했다.
`open`, `high`, `low`, `close`, 미래 수익률, funding PnL, 절대수익률, CAGR,
strict MDD는 로드하거나 계산하지 않았다.

## 고정 사건 수

| 구간 | 비중첩 사건 |
|---|---:|
| 2020 | 105 |
| 2021 | 117 |
| 2022 | 114 |
| train 2020–2022 | **336** |
| 2023 H1 | 47 |
| 2023 H2 | 54 |
| selection 2023 | **101** |
| 전체 | **437** |

- 방향: long 229 / short 208
- 지배 venue: Spot 222 / USD-M 215
- 가장 큰 단일 월 비중: 3.43%
- 모든 사전등록 support floor 통과

2023 수치는 event-clock 지원도 진단일 뿐이며 2023 진입 이후 가격이나 손익은
보지 않았다.

## 소스와 인과성

- Spot feature SHA256:
  `d558239fa7085083aa002b7898b632df0774425719467709680ecb99718035a9`
- USD-M feature SHA256:
  `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- execution market SHA256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- Spot missing/incomplete bars: 509
- USD-M missing/source-gap bars: 1,466
- joint quarantined bars: 2,575
- 모든 q95/q5, q97.5/q2.5, q75 기준은 현재 봉을 제외한 직전 clean
  `8,640`개 관측치만 사용한다.
- 신호 후 한 봉을 완전히 비우고 `t+2 open`에 진입해 288봉(24시간) 보유한다.
- 각 반기 경계 밖으로 나가는 사건은 버리고 다음 반기에서 scheduler를
  재시작한다.

## 동결 클록

- primary rows: 437
- primary clock SHA256:
  `7baf6f7de33e66417061dbea6f51efc6ea4993b2b5f2b9e0c09627a68adc57e2`

| 대조군 | rows | clock SHA256 앞 12자 |
|---|---:|---|
| direction flip | 437 | `41c82dba9b5f` |
| ticket level 제거 | 747 | `b4d0b1637133` |
| ticket shock 제거 | 951 | `b554433f0a2f` |
| coherence 제거 | 580 | `9304c0c32344` |
| price acceptance 제거 | 942 | `e03933f9e594` |
| 반대 venue 방향 | 437 | `48f0ef2dbfae` |
| 1시간 지연 | 437 | `b047a2c1f637` |
| 1일 이동 | 434 | `4cde710368ac` |
| 고정-seed random side | 437 | `240bf45d68f2` |

지원 artifact manifest hash는
`bf1bb61319aa434d2f5d1e0fe0da6450b606a0392ded36bf007bf2ba35797890`다.
빌더 소스 SHA256은
`5dde6454df4e4d54048443ffe71fbd29f25a7a9b0e7fae02012b45e958f01eef`다.

다음 단계는 strict evaluator 소스와 모든 대조군 검증 계약을 먼저
커밋·동결한 뒤 2020–2022 train 손익을 한 번만 여는 것이다. train gate가
실패하면 2023과 2024+는 계속 봉인한다.
