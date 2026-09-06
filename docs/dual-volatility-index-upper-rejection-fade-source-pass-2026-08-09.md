# DVURF-6 source-support pass

The preregistered and evaluator-frozen DVURF-6 paired source clock passed all
minimum-count, side-balance, and month-concentration gates without opening
Gross9 rows, execution prices, funding, or post-entry outcomes.

| stage | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 124 | 65 | 59 | 0.4758 | 0.2500 |
| test | 230 | 123 | 107 | 0.4652 | 0.1130 |
| eval | 224 | 96 | 128 | 0.4286 | 0.1339 |
| final | 161 | 77 | 84 | 0.4783 | 0.2236 |

Two executions reproduced identical artifacts:

- source snapshot SHA-256: `d83b3d90eb2253f953cdfb89573df2d7c2a1311b1e68b5174a6986ca96abebb5`
- clock SHA-256: `bfd90aadc2644360eb4d1acddde97954c7e9061064fe9502ecd7866fbf568fb1`
- result SHA-256: `d00260620b2ed807f2a189794e39f5f89b070c9bd75681787ae59de92235feb0`

The unchanged singleton may advance only to the frozen Gross9 novelty gate.
