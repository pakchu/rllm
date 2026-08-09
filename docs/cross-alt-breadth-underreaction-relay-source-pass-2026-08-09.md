# CABUR-8 source-support pass

CABUR-8 passed every source count, direction-balance, and month-concentration
gate without opening Gross9 rows, execution prices, funding, or outcomes.

| stage | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 70 | 38 | 32 | 0.4571 | 0.3000 |
| test | 151 | 76 | 75 | 0.4967 | 0.1788 |
| eval | 157 | 81 | 76 | 0.4841 | 0.1847 |
| final | 66 | 29 | 37 | 0.4394 | 0.3939 |

Two complete seven-symbol PostgreSQL executions reproduced identical files:

- feature SHA-256: `8270b0318d11d16b6b384e64bfaac77ef4bbc4a701dd347c2ababbb093061eae`
- clock SHA-256: `30c5e9fe282a495cc5b9b1638c6c7d7c22939e7bc4af29d1485ad91580b1421d`
- result SHA-256: `9297550b7e7e9adf4c7576f71987fa2e49f4d00063394b4733fa67d40850fb8d`

The unchanged candidate may advance only to Gross9 novelty.
