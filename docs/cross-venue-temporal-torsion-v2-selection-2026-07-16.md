# CVTT v2 2020–2022 selection

> 2020–2022만 열었고 2023은 봉인 상태다. 모든 CAGR은 무거래 시간을 포함한다.

| Rank | Policy | Route | Hold | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래 | 판정 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | V04 | um_preload_spot_echo | 18 | -57.573% | -24.854% | 59.275% | -0.419 | 1372 | REJECT |
| 2 | V03 | um_preload_spot_echo | 6 | -61.188% | -27.051% | 61.436% | -0.440 | 1403 | REJECT |
| 3 | V02 | spot_preload_um_echo | 18 | -69.733% | -32.852% | 70.360% | -0.467 | 1360 | REJECT |
| 4 | V01 | spot_preload_um_echo | 6 | -61.788% | -27.429% | 61.799% | -0.444 | 1393 | REJECT |

## 판정

- 상태: **rejected_before_2023_holdout**
- 통과 정책 수: 0
- 선택 정책: `None`
- strict MDD는 global/pre-entry HWM, 보유 중 favorable-before-adverse OHLC, 진입·가상청산 비용, funding debit/credit을 포함한다.
- 정책 선택 결과를 별도 커밋하기 전에는 2023을 열지 않는다.

## 실패 진단

- 정방향의 2020–2022 평균 gross edge는 정책별 `-2.6840`~`-0.1300` bps/trade로 6 bps/side 집행비용을 감당하지 못했다.
- 방향 반전 통제도 평균 gross edge가 `0.1302`~`2.6840` bps/trade에 그쳐 모든 정책의 net edge가 음수였다. 따라서 이 결과를 사후 반전 알파로 승격하지 않는다.
- 주별 cluster sign-flip Bonferroni p-value는 네 정책 모두 `1.0`이었다. 거래수는 충분하지만 통계적 알파 증거가 없다.
- 2023 홀드아웃은 열지 않았으며 CVTT v2 계열은 여기서 종료한다.
