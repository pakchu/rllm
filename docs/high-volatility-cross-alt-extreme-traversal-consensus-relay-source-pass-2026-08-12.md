# HVCATCR-8 source-support pass

The frozen source-only evaluator produced byte-identical artifacts in two
independent runs. Every preregistered split passed its event, side-balance, and
monthly-concentration gates:

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 65 | 29 | 36 | 0.4462 | 0.2462 |
| test | 119 | 58 | 61 | 0.4874 | 0.1681 |
| eval | 114 | 50 | 64 | 0.4386 | 0.1491 |
| final | 37 | 16 | 21 | 0.4324 | 0.2973 |

The source pass authorizes only the frozen Gross9 structural novelty test.
No execution price, post-entry return, funding value, PnL, Gross9 row, or
economic metric was opened while establishing source support.
