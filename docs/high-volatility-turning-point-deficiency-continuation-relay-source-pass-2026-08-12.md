# HVTPDCR-8 source-support pass

The frozen source-only evaluator passed unchanged and produced byte-identical
artifacts in two independent executions from the same committed evaluator.

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 26 | 13 | 13 | 0.5000 | 0.3462 |
| test | 57 | 24 | 33 | 0.4211 | 0.1930 |
| eval | 38 | 19 | 19 | 0.5000 | 0.2368 |
| final | 17 | 9 | 8 | 0.4706 | 0.4118 |

All frozen 8/12/12/8 incidence, 0.20 minority-side, and 0.45 monthly
concentration gates passed. No post-entry price, return, PnL, funding, Gross9
row, or economic metric was opened. The unchanged candidate may advance only
to Gross9 structural novelty.
