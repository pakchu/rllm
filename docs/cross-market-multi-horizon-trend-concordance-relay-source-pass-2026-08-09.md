# CMMTCR-24 source-support pass

The preregistered and evaluator-frozen CMMTCR-24 source clock passed every
minimum-count, side-balance, and month-concentration gate without opening
Gross9 rows, execution prices, funding, or post-entry outcomes.

| stage | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 32 | 20 | 12 | 0.3750 | 0.3125 |
| test | 71 | 37 | 34 | 0.4789 | 0.1690 |
| eval | 57 | 30 | 27 | 0.4737 | 0.1754 |
| final | 27 | 14 | 13 | 0.4815 | 0.2963 |

Two executions against the same candle database snapshot reproduced identical
artifacts:

- source snapshot SHA-256: `bba21dbda0da734bfa96ae2f6979c73465938dbd732f7e797ff687e3bcc47927`
- clock SHA-256: `e0c5e05fabcc76c8dc63f8026fb364f789dff368249b2e441436bb68c21878c0`
- result SHA-256: `a1eb56b4ac72f1aafc805b35994d8b8fe82bd51fdcbf3f627dd45b7844a1268e`

The unchanged candidate may advance only to the frozen Gross9 novelty gate.
