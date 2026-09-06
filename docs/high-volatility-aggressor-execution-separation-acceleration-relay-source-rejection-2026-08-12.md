# HVAESAR-8 source-support rejection

The frozen source evaluator reproduced byte-identically. Event counts and
monthly concentration passed, but the execution-price separation side was
structurally one-sided:

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 28 | 27 | 1 | 0.0357 | 0.2857 |
| test | 49 | 48 | 1 | 0.0204 | 0.2245 |
| eval | 38 | 37 | 1 | 0.0263 | 0.3684 |
| final | 20 | 20 | 0 | 0.0000 | 0.4500 |

Aggressive buys normally execute later/higher in a rising intrablock path than
residual aggressive sells, so the raw aggregate buy/sell VWAP ordering is not
a balanced directional statistic. HVAESAR-8 is terminal before Gross9 rows,
execution prices, returns, funding, PnL, or economics. No formula, side,
normalization, threshold, block, clock, hold, subset, or control repair is
permitted.
