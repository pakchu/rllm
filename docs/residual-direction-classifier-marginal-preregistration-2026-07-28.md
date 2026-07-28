# RDC-1 preregistration: residual direction confidence classifier

## RREM-1과의 경계

RREM-1은 미래를 열지 않은 채 거부됐다. 6h/12h 모두 전년도 OOS의
위험조정 utility 80분위수가 음수였고, 사전등록된 `score >= 0`에서 거래가
0건이었다.

RDC-1은 그 임계값을 낮추거나 기존 예측을 재사용하지 않는다. 새 모델을
처음부터 학습하며 질문 자체를 바꾼다. 이 실험은 동일 12피처 계열의
**마지막 배터리**다. 실패하면 threshold·target·모델·hold·운용 게이트를
바꿔 이 계열을 다시 열지 않는다.

> 절대 수익과 adverse excursion을 동시에 맞힐 수 있는가?

대신:

> 비용차감 후 6h/12h 동안 long과 short 중 어느 방향이 상대적으로 나은가?

## 고정 모델

- 입력: RREM-1과 같은 12개 인과적 residual 피처
- 행 조건: 완료된 신호봉에서 12개 모두 finite
- 모델: `ExtraTreesClassifier`
- scikit-learn: `1.7.2`
- 256 trees, depth 4, min leaf 128, max-features 0.75
- class-weight balanced
- seeds `{7, 71, 715}`
- 각 seed는 `random_state=seed`
- class: short=`0`, long=`1`; `classes_ == 1`인 열만 `p_long`
- label: exact long net return `>=` exact short net return이면 long
- 방향: `mean(p_long) >= 0.5`면 long, 아니면 short
- empirical confidence:
  `max(mean(p_long), 1-mean(p_long)) - 0.5 × std(p_long, ddof=0)`
- 이 값은 보정된 확률이 아니라 rank score다.
- 각 hold의 전년도 모든 finite·split-contained hourly anchor에서 계산한
  empirical confidence 90분위수(`np.quantile(method="linear")`)를 다음 해
  threshold로 사용한다. coordination/non-overlap 적용 전 모집단이다.
- 입장 연산자는 `score >= threshold`
- 절대 confidence floor 없음

## 연단위 누수 방지

- 2022 예측: classifier와 inventory residual beta 모두 2022 이전만 fit
- 2023 예측: classifier와 inventory residual beta 모두 2023 이전만 fit
- 2024 예측: classifier와 inventory residual beta 모두 2024 이전만 fit
- 각 fit/window 경계에서 `entry_delay + hold` anchor purge
- 2025 이후 원천은 classifier graph에서 물리적으로 제외

## 정확히 여섯 정책

보유기간 `{72, 144}` bars와 아래 운용 상태의 곱이다.

1. unrestricted
2. completed signal bar에서 Gross9 실제 포지션이 flat
3. 연속 Gross9 자산곡선이 고점 대비 5% 이상 drawdown

진입·청산·비용·non-overlap·same-gross 대조군·standalone 및 portfolio
합격 기준은 RREM-1과 동일하다. 후보 비중은
`{0.25, 0.50, 0.75, 1.00}`이고 총 24개 셀이다.

정상 선별은 Gross9와 후보 모두 6bp/side다. stress에서는 모델·피처
state·점수·threshold·방향·진입을 그대로 두고 후보 비용만 10bp/side로
바꾼다. Gross9와 same-gross 대조군 비용은 계속 6bp/side다.

2023과 2024는 모두 pre-2025 선별 구간이다. 통과한 top1만 역사적 미래
veto를 열 수 있고, 미래는 재순위·수리·차순위 교체에 사용할 수 없다.
2025/2026은 이미 다른 연구에서 관찰돼 승격 인증에는 쓸 수 없다.

따라서 pre-2025와 역사적 veto를 모두 통과해도 research shadow다. 실제
승격에는 exact top1을 커밋한 뒤 파라미터 변경 없이 90일 연속 prospective
paper/testnet이 필요하다: 후보 30거래 이상, 후보 정상/10bp stress 절대수익
모두 양수, 결합 포트의 절대수익과 strict MDD가 unscaled Gross9보다 나쁘지
않고 same-gross 대조군 대비 CAGR/MDD 개선이 엄격히 양수여야 한다.

기계 고정 계약:
`results/residual_direction_classifier_marginal_preregistration_2026-07-28.json`
