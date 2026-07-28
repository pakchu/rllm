# OFBR-1 pre-2024 selection

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

- tested: 96
- passed frozen shortlist gate: 0
- status: **reject_pre2024**

| # | candidate | calibration | 2023 | 2023 H1 | 2023 H2 |
|---:|---|---:|---:|---:|---:|

## 가장 가까웠던 실패 셀

| # | candidate | calibration | 2023 | 2023 H1 | 2023 H2 |
|---:|---|---:|---:|---:|---:|
| 1 | `ofbr_d90_f20_w6_oi0_r65_h36` | 4.05/1.72/7.42/0.23/84 | 0.19/0.19/0.87/0.21/8 | 0.51/1.04/0.66/1.56/6 | -0.33/-0.64/0.69/-0.93/2 |
| 2 | `ofbr_d90_f35_w6_oi0_r65_h72` | 1.64/0.70/14.23/0.05/133 | 0.68/0.68/2.91/0.23/13 | 1.21/2.47/1.59/1.55/9 | -0.53/-1.05/2.41/-0.43/4 |
| 3 | `ofbr_d90_f20_w12_oi-50_r65_h72` | 0.93/0.40/11.79/0.03/75 | 1.83/1.83/1.82/1.01/9 | 2.35/4.80/1.05/4.58/6 | -0.51/-1.01/1.61/-0.63/3 |
| 4 | `ofbr_d90_f20_w12_oi0_r80_h36` | 0.24/0.10/3.82/0.03/32 | 0.46/0.46/0.66/0.70/3 | 0.46/0.93/0.66/1.40/2 | 0.00/0.00/0.16/0.01/1 |
| 5 | `ofbr_d90_f35_w12_oi-50_r65_h144` | 0.65/0.28/25.78/0.01/128 | 0.76/0.76/3.18/0.24/14 | 1.26/2.56/2.03/1.26/9 | -0.49/-0.97/1.95/-0.50/5 |

수치는 `absolute return / CAGR / strict MDD / CAGR-MDD / trades`다.

## 판정

- 96개 중 calibration과 2023이 모두 양수인 셀은 6개뿐이었다.
- q90 OI divergence가 상대적으로 나았지만, 최고 셀도 calibration ratio가
  `0.23`에 불과했다.
- 2023의 겉보기 이익은 대부분 H1에 집중됐고, 거래가 있었던 상위 셀의
  H2는 모두 음수였다.
- 따라서 이 메커니즘은 Gross9 손실을 상쇄할 독립 알파가 아니라
  **상승 반등 국면에 편향된 약한 조건부 베타**로 판정한다.
- 사전 계약에 따라 2024, 2025, 2026 결과는 열지 않고 이 정확한
  96-cell family를 종료한다.

- The returned source frame contains no row at or after `2024-01-01`.
- OI is delayed one complete 5-minute bar and missing values fail closed.
- No 2024+ outcome is opened in this artifact.
