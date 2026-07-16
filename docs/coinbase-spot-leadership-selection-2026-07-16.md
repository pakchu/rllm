# Coinbase–Binance Venue-Leadership Selection

> 2020–2022만 열었으며 2023과 2024+는 봉인 상태다.

| Rank | Policy | Family | Side | Hold | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래 | 판정 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | P12 | activity_confirmed_relative | 1 | 3 | 2.631% | 0.869% | 11.889% | 0.073 | 162 | REJECT |
| 2 | P10 | activity_confirmed_relative | 1 | 1 | 1.900% | 0.629% | 9.766% | 0.064 | 171 | REJECT |
| 3 | P11 | activity_confirmed_relative | -1 | 3 | -18.464% | -6.576% | 20.208% | -0.325 | 192 | REJECT |
| 4 | P03 | relative_return_lead | -1 | 3 | -66.300% | -30.405% | 68.543% | -0.444 | 1922 | REJECT |
| 5 | P09 | activity_confirmed_relative | -1 | 1 | -10.668% | -3.690% | 11.096% | -0.333 | 197 | REJECT |
| 6 | P13 | activity_premium_confluence | -1 | 3 | -66.889% | -30.812% | 68.261% | -0.451 | 1950 | REJECT |
| 7 | P01 | relative_return_lead | -1 | 1 | -65.453% | -29.827% | 66.587% | -0.448 | 1956 | REJECT |
| 8 | P04 | relative_return_lead | 1 | 3 | -65.437% | -29.816% | 65.448% | -0.456 | 1838 | REJECT |
| 9 | P08 | premium_shock | 1 | 3 | -94.912% | -62.935% | 94.921% | -0.663 | 5225 | REJECT |
| 10 | P02 | relative_return_lead | 1 | 1 | -67.235% | -31.054% | 67.253% | -0.462 | 1863 | REJECT |
| 11 | P06 | premium_shock | 1 | 1 | -94.771% | -62.597% | 94.776% | -0.660 | 5561 | REJECT |
| 12 | P07 | premium_shock | -1 | 3 | -94.649% | -62.308% | 94.729% | -0.658 | 5315 | REJECT |
| 13 | P05 | premium_shock | -1 | 1 | -95.989% | -65.760% | 96.075% | -0.684 | 5675 | REJECT |
| 14 | P14 | activity_premium_confluence | 1 | 3 | -59.567% | -26.049% | 62.174% | -0.419 | 1825 | REJECT |
| 15 | P16 | return_premium_confluence | 1 | 3 | -99.999% | -97.598% | 99.999% | -0.976 | 20668 | REJECT |
| 16 | P15 | return_premium_confluence | -1 | 3 | -99.999% | -98.107% | 99.999% | -0.981 | 20991 | REJECT |

## 판정

- 상태: **rejected_before_2023_holdout**
- 통과 정책 수: 0
- 선택 정책: `None`
- 절대수익과 CAGR은 거래하지 않은 기간까지 포함한 전체 2020–2022 달력으로 계산했다.
- strict MDD는 global/pre-entry HWM, 보유 중 favorable-before-adverse OHLC, 진입·가상청산 비용 및 funding debit/credit의 최악 순서를 포함한다.
- 2023은 선택 정책 manifest가 별도 커밋되기 전에는 열지 않는다.
