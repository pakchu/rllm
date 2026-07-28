# RREM-1 preregistration: residual-recovery ExtraTrees

## 목표

Gross9에 없는 두 정보축만 사용한다.

1. 현물–무기한 선물 괴리의 인과적 잔차
2. 가격·carry로 설명되지 않는 지연 OI 재고 변화 잔차

정적 규칙과 9상태 전이표는 이미 실패했다. 이번에는 12개 정규화 잔차
피처의 약한 비선형 상호작용만 얕은 ExtraTrees가 학습한다. 성공 기준은
독립 수익률이 아니라 **동일 gross의 Gross9 레버리지 대조군보다
CAGR/strict-MDD가 2023 포함 train과 2024에서 모두 개선되는지**다.

## 결과를 보기 전에 고정한 구조

- 물리적 데이터 컷오프: `2025-01-01`
- 연단위 expanding fit: 2023, 2024
- 모델: 256 trees, depth 4, min leaf 128, max-features 0.75, 3 seeds
- 행 입장 조건: 완료된 신호봉에서 12개 피처가 모두 finite
- 출력: long/short의 정확한 비용차감 수익과 strict adverse excursion
- adverse target: 보유 중 각 봉 시가 기준 canonical leverage-adjusted adverse
  OHLC 손실의 최댓값
- 점수: `return - 0.5 × adverse - 0.5 × seed uncertainty`
- 기준점: 각 hold별 전년도 모든 유효 hourly OOS 점수의 80% 분위수와
  0 중 큰 값. 점수는 선택된 방향의 seed 평균 utility에서 population
  standard deviation(`ddof=0`)의 0.5배를 뺀 값이다. long/short와 세 운용
  모드는 같은 기준점을 공유
- 진입: 완료된 5분봉 `t`에서 판단, `t+1` 시가
- 보유: 6시간 또는 12시간
- 레버리지: 0.5x
- 기본 비용: 명목당 편도 6bp
- 후보 스트레스: 명목당 편도 10bp
- TP/SL 없음
- 모델 fit에는 진입·모든 보유봉·time exit가 fit cutoff보다 엄격히 앞선
  target만 포함한다. 각 경계 직전 `entry_delay + hold` anchor는 purge한다.
- stress는 모델·점수·방향·진입을 바꾸지 않고 후보 비용만 10bp로 재생
- split 경계를 넘는 거래는 제외한다. 단일 후보 sleeve가 열려 있으면
  `signal_position <= exit_position`인 모든 신호를 버리고 포지션을 쌓지 않는다.
- 인덱스는 `signal=t`, `entry=e=t+1`, 보유봉 `[e,e+H-1]`,
  `time exit=x=e+H`로 고정한다. 다음 허용 signal은 `x+1`이다.

## 정확히 여섯 정책

두 보유기간과 아래 세 운용 모드의 곱만 평가한다.

1. 무제한
2. 신호 시점 Gross9가 완전히 비어 있을 때만
3. 신호 시점 Gross9가 이전 고점 대비 5% 이상 drawdown일 때만

Gross9 상태는 완료된 신호봉까지의 포지션·자산곡선만 사용한다. 미래
Gross9 손익은 피처, 게이트, 임계값에 들어갈 수 없다. 이 상태 자산곡선은
2020-09-01부터 2024-12-31까지 연속이며 2024 경계에서 리셋하지 않는다.

## 선별

2023과 2024는 모두 pre-2025 **선별 구간**이다. 각 정책은 두 해 모두에서
다음을 만족해야 한다.

코드 내부의 기존 배열명 `test2024`는 `selection_2024`의 역사적 alias일
뿐이며 held-out test를 뜻하지 않는다.

- 절대수익 양수
- full-calendar CAGR / strict MDD ≥ 1.5
- strict MDD ≤ 15%
- 거래 30회 이상
- 후보만 10bp/side로 스트레스해도 절대수익 양수

그 뒤 후보 비중 `c ∈ {0.25, 0.50, 0.75, 1.00}`을 Gross9에 더한다. Gross9
고정 비중 `b_i`의 합은 9이고 후보 포트는 `{b_i}+c`, 동일 gross 대조군은
`b_i × (9+c)/9`이다. 이는 실시간 활성 notional이 아니라 기존 엔진과 같은
고정 배분 gross다. Gross9만 비례 확대한 대조군보다 두 선별 구간 모두 비율이 0.05 이상
좋아야 하고, 원래 Gross9 대비 적어도 한 구간 strict MDD를 줄여야 한다.

2025/2026은 pre-2025 top1이 모든 기준을 통과할 때만 연다. 미래는
거부만 가능하며 재선별·수리·차순위 교체에는 사용할 수 없다.

기계 고정 계약:
`results/residual_recovery_extratrees_marginal_preregistration_2026-07-28.json`
