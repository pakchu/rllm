# HVOIPSCR-8 source-support rejection

The frozen source-only evaluator produced byte-identical artifacts in two
independent runs. Event minimums and side balance passed in every split, but
the frozen test-period monthly concentration gate failed:

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 42 | 25 | 17 | 0.4048 | 0.2143 |
| test | 15 | 10 | 5 | 0.3333 | **0.6667** |
| eval | 55 | 24 | 31 | 0.4364 | 0.2000 |
| final | 31 | 18 | 13 | 0.4194 | 0.2903 |

The required maximum is 0.45. HVOIPSCR-8 is terminally rejected before any
Gross9 row, execution price, return, funding, PnL, or economic metric was
opened. No aggregation, pair treatment, rank, clock, threshold, side, or hold
repair is permitted.
