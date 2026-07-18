# COIN-M calendar-curve compression support freeze — 2026-07-19

## 판정

결과 수익률을 열지 않은 support 단계는 **PASS**다.

| 구간 | 거래 수 | next long | next short |
|---|---:|---:|---:|
| fit 2020-07-15~2022 | 179 | 85 | 94 |
| 2020 partial | 48 | 13 | 35 |
| 2021 | 74 | 33 | 41 |
| 2022 | 57 | 39 | 18 |
| 2023 selection | 60 | 24 | 36 |
| 2023 H1 | 30 | 10 | 20 |
| 2023 H2 | 30 | 14 | 16 |

거래 수, 양방향 비중, 단일 월 집중도, 단일 계약쌍 집중도 gate를 모두
통과했다. 하위 구간 통계는 fit/selection 부모 non-overlap clock을 다시
스케줄링하지 않고 entry time으로 잘라 계산했다.

## 봉인 정보

- preregistration commit: `187cf53`
- support artifact:
  `results/coinm_calendar_curve_compression_support_2026-07-19.json`
- artifact SHA-256:
  `1377015da4f12bb90441d5f3f3bfbf1788ca0416b75bb87ec3b62dc25d6b0dfc`
- manifest hash:
  `6b196ad49af34d6305871a1f07d1b8229fe3a1c2f4953d3364eda7344f7ab881`
- source SHA-256:
  `d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b`
- source manifest SHA-256:
  `29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af`

support builder는 front/next의 완료 봉 close, 거래량, 계약/만기 및
availability metadata만 읽었다. open/high/low, post-entry return, PnL,
funding, 2024 이후 데이터는 읽지 않았다. 다음 단계는 이 artifact와 정확한
clock hash를 묶는 새 2-leg inverse strict evaluator를 결과 확인 전에
동결하는 것이다.
