# HVIHR-1 source support — 2026-08-11

The preregistered `r3 -> r19` reversal candidate passed every outcome-blind source gate.

| Split | Events | Long | Short | Minority share | Maximum month share |
|---|---:|---:|---:|---:|---:|
| train | 29 | 15 | 14 | 0.483 | 0.345 |
| test | 64 | 25 | 39 | 0.391 | 0.219 |
| eval | 59 | 28 | 31 | 0.475 | 0.186 |
| final | 28 | 19 | 9 | 0.321 | 0.393 |

- Required minima were `8/12/12/8`, minority side share was at least `0.20`, and maximum month share was at most `0.45`.
- The source query opened only completed BTCUSDT OHLC rows. Post-entry prices, returns, PnL, funding values, and Gross9 rows remained sealed.
- Authorized next stage: Gross9 structural-clock novelty only.
