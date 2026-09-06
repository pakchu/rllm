# HVLIR-8 source-support result

HVLIR-8 passed its frozen source-only gate.  The read-only BTCUSDT one-minute
source produced complete causal hourly liquidity-impact states without opening
Gross9 rows, funding, execution prices, or post-entry PnL.

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| TRAIN 2023H2 | 133 | 74 | 59 | 0.444 | 0.301 |
| TEST 2024 | 280 | 154 | 126 | 0.450 | 0.186 |
| EVAL 2025 | 273 | 159 | 114 | 0.418 | 0.194 |
| FINAL to 2026-08-01 | 121 | 58 | 63 | 0.479 | 0.380 |

Every split clears the frozen `8/12/12/8` count floor, `0.20` minority-side
floor, and `0.45` maximum monthly concentration ceiling.  This authorizes only
the separately frozen Gross9 novelty gate.  It is not economic evidence, and
the diagnostic controls remain non-promotable.
