# LVRT-R0 기각 — pre-2024 strict 평가

## 판정

**REJECTED. 2024·2025·2026 YTD는 열지 않았다.**

LVRT-R0는 충분한 이벤트 수와 방향 균형을 가졌지만, 한 시간 뒤 움직임의 평균이 비용 전부터 거의 0이거나 잘못된 방향이었다. 이는 표본 부족이나 strict MDD만의 문제가 아니라 경제적 edge 부재다. 사전등록대로 q80, 반전 확인, 방향, 12봉 hold, 비용 또는 게이트를 수정하지 않는다.

## primary 결과

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 비용 전 평균 움직임 | 거래수 |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | **-55.47%** | -23.63% | 55.47% | -0.43 | -2.41bp | 1,112 |
| 2023 | **-8.58%** | -8.58% | 8.75% | -0.98 | -0.17bp | 147 |
| 2023 H1 | **-4.48%** | -8.83% | 5.15% | -1.71 | -0.34bp | 74 |
| 2023 H2 | **-4.29%** | -8.34% | 4.44% | -1.88 | -0.01bp | 73 |

8bp/notional/side 스트레스에서는 train `-64.35%`, 2023 `-11.23%`였다. 주간 cluster sign-flip p-value는 train `1.00000`, 2023 `0.99983`로 유의하지 않았다.

## 방향을 뒤집어도 알파가 아닌 이유

| 방향 반전 | 절대수익 | CAGR | strict MDD | 비용 전 평균 움직임 | 거래수 |
|---|---:|---:|---:|---:|---:|
| train 2020–2022 | -41.54% | -16.38% | 42.74% | +2.41bp | 1,112 |
| 2023 | -8.40% | -8.40% | 9.44% | +0.17bp | 147 |

왕복 손익분기 움직임은 0.5x와 6bp/notional/side에서 기초자산 기준 약 12bp다. 방향 반전의 `+2.41bp / +0.17bp`는 비용의 작은 일부뿐이므로 reverse 정책도 후보로 승격하지 않는다.

## 대조군 결과가 말하는 것

2023 절대수익은 reversal 확인 제거 `-23.77%`, 진입 1봉 지연 `-9.08%`, setup 1일 이동 `-8.62%`, confirmation 부호 순열 `-4.95%`였다. 모든 대조군도 기각됐고, placebo가 우연히 전체 게이트를 통과한 문제는 없었다.

## 원인

1. burst + HHI는 거래 집중과 도착 불규칙성을 측정하지만 사용 가능한 passive depth 자체를 관측하지 않는다.
2. 다음 봉의 반대 공격 흐름과 가격 반전은 refill 지속성을 의미하지 않았다.
3. 60분 고정 horizon의 조건부 평균 움직임이 2020–2022와 2023 모두 왕복 비용보다 한참 작았다.
4. 표본은 총 1,259개였고 2023 H1/H2도 74/73개라, 실패를 단순 표본 부족으로 설명하기 어렵다.

따라서 후속 연구는 LVRT의 threshold/hold 튜닝이 아니라 **다른 데이터 축과 다른 경제적 전달 경로**를 사용해야 한다.

## 재현성

- preregistration commit: `625f8df32d1ea4a227f1e0322d82584c48eba290`
- support commit: `70746f8fc6673913df0b272f3b623f321f5fa220`
- evaluator source commit: `d784101492828ae5e2a1dd7aa39674d47b53acb1`
- evaluator freeze commit: `b9c7ae7`
- selection result SHA256: `9333035ea4df4d715e01348042e0c79f9ec90994355579e098ba142230456790`
- strict accounting: global/pre-entry HWM, favorable-before-adverse held OHLC, hypothetical liquidation cost, funding, full-calendar CAGR
- opened: 2020–2022 train, 2023 selection/H1/H2
- sealed: 2024, 2025, 2026 YTD
