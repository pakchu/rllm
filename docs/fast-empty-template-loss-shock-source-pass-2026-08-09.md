# FETLS-6 source-support pass

The preregistered FETLS-6 source evaluator opened only canonical block metadata
and completed pre-entry BTC bars. The 166,682-row block prefix passed contiguous
height and previous-hash validation. Two complete executions produced identical
support and clock hashes.

| stage | events | long / short | minority share | max month share |
|---|---:|---:|---:|---:|
| train | 64 | 34 / 30 | 0.4688 | 0.2969 |
| test | 135 | 65 / 70 | 0.4815 | 0.1704 |
| eval | 85 | 37 / 48 | 0.4353 | 0.1294 |
| final | 30 | 12 / 18 | 0.4000 | 0.2667 |

All frozen `8/12/12/8`, side-balance, and month-concentration gates pass.
Execution prices, funding, post-entry outcomes, RV20, and Gross9 rows remain
unopened. FETLS-6 advances unchanged to Gross9 novelty.

- block source SHA-256: `bdacc354120c526e1672df52f67912b527f6e03bfc3bc2191f3c4ba7ba47e3aa`
- clock SHA-256: `76a810be097e036816a69e15eaf06f9d1abe9ef4bd3ab09f070fb9dc27e0ef89`
- support result SHA-256: `414f51dff9e64a467de060ed176d7c2f62dbfc747f614cb53e021fae8e768d83`
