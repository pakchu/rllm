# HVVRCR-12 source-support pass

HVVRCR-12 passed every frozen source-support gate without opening Gross9,
execution, funding, or outcome rows.

| stage | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 31 | 16 | 15 | 0.4839 | 0.2581 |
| test | 46 | 18 | 28 | 0.3913 | 0.1739 |
| eval | 43 | 20 | 23 | 0.4651 | 0.2093 |
| final | 17 | 8 | 9 | 0.4706 | 0.3529 |

Two executions reproduced identical artifacts:

- source SHA-256: `3fc359bc721892ed627d279797bbdaad7a82b20d065753d9e9e3e3750fe2bd4c`
- clock SHA-256: `8b7e2dea20844e3487297f7bd67167e3bd1d5affb5b0b5ec452d80d19db1beb6`
- result SHA-256: `94deb8a2c0c2102ff42823bd91df23ee1857eedff79de283c1acd2603bf7cba4`

The unchanged singleton may advance only to Gross9 novelty.
