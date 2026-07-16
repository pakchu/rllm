# POWR-12 기각 — Perp-Only Wick Rejection

## 결론

**기각. 2024·2025·2026 YTD는 열지 않았다.**

POWR-12는 outcome-blind support를 통과했지만, 동결된 2020–2022 train에서
비용 전 평균 방향 움직임이 `+2.17bp`에 불과했다. 6bp/notional/side와
0.5x를 적용한 뒤 절대수익 `-27.01%`, CAGR `-9.96%`, strict MDD
`30.06%`가 됐다. 정확히 같은 이벤트의 방향 반전도 train과 2023 모두
손실이므로 실패 정책을 뒤집어 새 후보로 재활용하지 않는다.

## Primary 결과

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 비용 전 평균 움직임 | 거래 수 |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 train | -27.01% | -9.96% | 30.06% | -0.33 | +2.17bp | 594 |
| 2023 | +0.24% | +0.24% | 3.68% | 0.07 | +13.38bp | 43 |
| 2023 H1 | -2.68% | -5.34% | 3.42% | -1.56 | -6.51bp | 29 |
| 2023 H2 | +3.01% | +6.06% | 1.87% | 3.24 | +54.58bp | 14 |

8bp stress의 절대수익은 train `-35.19%`, 2023 `-0.62%`다. Weekly-cluster
sign-flip p-value는 train `0.9425`, 2023 `0.4714`로 사전 기준 `<=0.10`을
통과하지 못했다. 2023 H2의 국소 성과만 양수였으며 H1은 음수이므로 안정적인
시계열 일반화로 볼 수 없다.

## 강건성·대조군

| 정책 | Train 절대수익 | 2023 절대수익 | Train 평균 gross | 2023 평균 gross |
|---|---:|---:|---:|---:|
| direction flip (진단 전용) | -35.47% | -5.36% | -2.17bp | -13.38bp |
| 진입 1봉 추가 지연 | -42.80% | -0.61% | -6.13bp | +9.41bp |
| Spot-only wick | -98.96% | -78.67% | -1.76bp | -2.34bp |
| common wick | -98.74% | -77.43% | -1.47bp | -1.81bp |
| basis-free Perp wick | -98.63% | -77.42% | -1.18bp | -1.80bp |
| stale Spot 1h | -98.99% | -76.92% | -1.17bp | -0.60bp |
| stale Spot 1d | -99.52% | -80.28% | -3.84bp | -2.48bp |

Primary를 한 봉 늦추면 train과 2023 모두 음수가 된다. 이는 진입 시점에
민감한 국소 반응조차 안정적인 비용 후 알파가 아님을 보여준다. 대조군도 모두
실패해 메커니즘 위양성 문제 이전에 경제적 edge 자체가 부족하다.

## 무결성

- 사전등록 commit: `e168d6ff560d429138f586e625ba54e1c9710c97`
- support freeze commit: `4202671697a3a31117d7e2caabbfa536295d5837`
- evaluator source commit: `cea52ae8c08c323fc151462ab677abb96e5abd07`
- evaluator freeze commit: `33e2b3a5d31bb08309cf1a664c628acec9c12211`
- 결과:
  `results/perp_only_wick_rejection_selection_2026-07-17.json`
- 결과 SHA256:
  `b2bf1b96793e088271c1ce43cbb87f03c07f72655e2372286a2a4f979ca95c03`
- 결과 manifest hash:
  `5c7e338ad1d5c874f9723ea273f4c29fcbbff6f2cfad2671d8745778f4a31c70`

평가는 full-wall-clock CAGR, global/pre-entry HWM, held OHLC의
favorable-before-adverse 순서, 실제 funding rate의 `[entry, exit)` 적용,
entry/exit/가상 청산 비용을 사용했다. 결과는 canonical manifest hash로
자체 검증된다.

## 연구 결정

q95, 6bp wick floor, Spot/Perp 0.5 비율, 방향, 3봉 지연 또는 12봉 hold를
수정하지 않는다. 2023 H2만 보고 임계값을 수리하거나 2024를 열면 선택 편향이
되므로 POWR-12 계열은 여기서 종료한다. 다음 후보는 wick/basis/funding이 아닌
별도의 원천 데이터와 경제적 메커니즘을 사용한다.
