# SQFD-6 stablecoin quote-flow diffusion — outcome-blind support freeze

## 판정

**PASS — 사전등록된 표본수·방향 균형·월 집중·기존 clock 독립성 조건을 모두
통과했다. 가격, funding, 수익률, PnL은 아직 열지 않았다.**

이 판정은 SQFD-6가 수익성 있는 알파라는 뜻이 아니다. 다음 단계에서 별도 strict
evaluator를 코드·테스트·해시로 먼저 동결한 뒤, `2023 H2 train` outcome만 열 수
있다는 의미다. Train이 한 항목이라도 실패하면 2024 test 이후는 계속 봉인하고
후보를 수리하지 않는다.

## 고정 신호

완료된 Binance Spot 1시간 봉의 BTCUSDT, BTCUSDC, BTCFDUSD taker flow만 쓴다.
각 book imbalance를 자기 직전 720개 positional hour의 median/IQR로 표준화한다.
현재 행은 `shift(1)`로 기준 분포에서 제외하며 최소 672개 선행 관측을 요구한다.

다음 조건이 false에서 true로 바뀌는 첫 시간만 후보로 잡는다.

1. USDC와 FDUSD z-score 부호가 같고 0이 아니다.
2. 두 절대 z-score의 최솟값이 0.75 이상이다.
3. 두 대체 quote book의 평균 z 부호를 거래 방향으로 쓴다.
4. 같은 방향의 USDT z-score는 0.50 미만이어야 한다.
5. USDC+FDUSD 거래량 비중이 자기 strictly-prior median 이상이다.

source hour 경계에서 판단하고 완전한 5분 latency bucket 하나를 비운 뒤
`boundary+5m`에 진입하는 clock만 기록했다. 6시간 고정 보유, 전역 비중첩,
`entry >= previous_exit`의 exit-exclusive 예약을 전체 타임라인에서 먼저 적용한 후
split-contained 사건만 남겼다.

## 표본 지원도

| 구간 | 사건 | Long | Short | 최대 월 비중 |
|---|---:|---:|---:|---:|
| 2023 H2 train | 55 | 32 | 23 | 27.27% |
| 2023 Q3 | 15 | 9 | 6 | 100.00% |
| 2023 Q4 | 40 | 23 | 17 | 37.50% |
| 2024 test | 185 | 96 | 89 | 12.97% |
| 2024 H1 / H2 | 99 / 86 | 56 / 40 | 43 / 46 | 24.24% / 22.09% |
| 2025 eval | 217 | 106 | 111 | 12.90% |
| 2025 H1 / H2 | 94 / 123 | 57 / 49 | 37 / 74 | 21.28% / 22.76% |
| 2026 H1 final | 93 | 50 | 43 | 25.81% |
| 2026 Q1 / Q2 | 40 / 53 | 22 / 28 | 18 / 25 | 42.50% / 45.28% |

Q3의 사건은 720시간 warm-up 뒤인 9월부터 시작하므로 월 비중 100%다. 사전등록은
Q3/Q4에는 최소 사건 수만 요구하고 월 집중도는 parent train에만 적용했다. 이
규칙은 결과를 보기 전에 고정됐으며 지금 변경하지 않는다.

## 기존 clock과의 독립성

| 비교 대상 | 공통 범위 SQFD 사건 | 비교 사건 | exact Jaccard | ±6h 최대 양방향 포함률 |
|---|---:|---:|---:|---:|
| OPDR-24 | 550 | 145 | 0.00144 | 29.66% |
| PCBR-12 | 550 | 187 | 0.00136 | 26.74% |
| PSR-30/6 | 550 | 750 | 0.00000 | 16.00% |
| FQPR-3 (2023 H2) | 55 | 11 | 0.00000 | 9.09% |

모두 사전등록된 exact Jaccard 0.10 이하, ±6시간 양방향 포함률 0.35 이하를
통과했다. 이는 진입 시각 표현이 기존 후보와 충분히 다르다는 지원도 검사다.
수익 상관이나 포트폴리오 기여도를 의미하지 않으며, 그런 outcome 기반 비교는
단독 sequential gate 통과 뒤에만 허용된다.

## outcome-blind 대조군

동일 source state에서 다음 clock도 결과 전에 고정했다.

- 대체 quote breadth 제거: 1,435건;
- USDT lag 제거: 1,158건;
- 대체 quote 참여율 제거: 1,023건;
- USDT 단독: 3,098건;
- primary 동일 clock 방향 반전: 550건;
- primary 동일 clock deterministic random side: 550건;
- primary 동일 신호 1시간 추가 지연: 550건.

독립 control은 자기 clock에 비중첩을 적용했고, 방향 반전·random side·추가 지연은
primary 예약을 그대로 재사용했다. 이후 train 평가에서 primary는 가장 강한
사전등록 mechanism control보다 CAGR/strict-MDD가 최소 0.25 높아야 한다.

## 무결성 앵커

- preregistration commit: `31b0d0d`;
- preregistration JSON SHA-256:
  `3fed620146b98e920175445a12e2a8684c2a3431e42b1a784ea0e3076577aee3`;
- preregistration canonical manifest hash:
  `74fd535b8b2256d41e39513466bd697d553ee5c80aece8308e3de637745225b3`;
- support builder SHA-256:
  `15f2b6cc34ddd6331be61aeeabed3c878ba8cb8d1091f42ca1ebf006ad242d17`;
- frozen all-control clock: 8,914행, primary 550행;
- clock SHA-256:
  `a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b`;
- support JSON SHA-256:
  `07230e9e579f1b16e07712a022e572026b4fbfa17070e998970b3fd8ee21d4b5`;
- support canonical manifest hash:
  `e37503c54a2a8b9e9f2f5c20cc188d762117d69e04d7e9ba8689480c135023fa`.

두 번의 전체 build가 clock gzip과 support JSON에서 바이트 단위로 동일했다.
clock schema에는 source-derived z-score, 시각, 방향만 있고 가격·수익률·funding·PnL
열은 없다. Support report는 execution OHLC 0행, funding 0행을 기록한다.

## 오염·운영 경계

- 0.75 threshold는 train의 **source-only 사건 수**로 고정했다. 초기 탐색 중 미래
  source support shape를 실수로 확인한 이력은 preregistration에 공개돼 있다.
  가격 outcome은 보지 않았지만 완전히 pristine한 support discovery라고 부르지
  않는다.
- BTCUSDT/BTCUSDC/BTCFDUSD는 2026년 현재 생존 book을 기준으로 고정했으므로
  universe survivor/source-selection bias가 있다.
- checksum historical archive의 close timestamp는 live collector가 항상
  `boundary+5m` 전에 세 book을 최종 확정한다는 증거가 아니다. 실거래 승격 전
  세 book의 forward latency/parity를 fail-closed로 검증해야 한다.
- FQPR 비교는 가용한 2023 H2 공통 범위만 사용한다.

## 다음 허용 작업

1. exact conservative funding boundary, full-calendar CAGR, 5분 OHLC의
   favorable-then-adverse strict MDD, entry/exit/virtual-exit cost를 구현한 전용
   evaluator와 regression tests를 작성한다.
2. evaluator source·policy·support artifact hash를 별도 freeze artifact로 묶고
   outcome을 열기 전에 커밋한다.
3. 그 후에만 2023 H2 train을 한 번 평가한다. 절대수익, CAGR/strict-MDD 3,
   strict MDD 15%, weekly clustered sign-flip p-value 0.10, 평균 gross move 20bp,
   양 절반, 10bp stress 및 mechanism-control margin을 모두 요구한다.
