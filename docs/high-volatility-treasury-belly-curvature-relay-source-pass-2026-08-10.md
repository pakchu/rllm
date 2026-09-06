# HVTBCR-24 source-support pass

The frozen HVTBCR-24 source evaluator was run only after its preregistration
and evaluator commits were pushed.  It parsed the hash-bound official Treasury
XML, introduced the five-year belly, and built the unchanged
`2*5y-2y-10y` curvature-change clock without opening Gross9 rows, execution
prices, funding or post-entry outcomes.

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train 2023H2 | 32 | 17 | 15 | 0.46875 | 0.34375 |
| test 2024 | 98 | 49 | 49 | 0.50000 | 0.20408 |
| eval 2025 | 82 | 43 | 39 | 0.47561 | 0.23171 |
| final to 2026-08-01 | 44 | 24 | 20 | 0.45455 | 0.36364 |

Every 8/12/12/8 count, 0.20 minority-side, and 0.45 monthly-concentration
gate passed.  An immediate replay reproduced the state, primary clock and
result hashes byte-for-byte:

- state: `46d64f7c9d42f9e25eab406da2e29454855ff3cd867ff54eecf8cb7a4b4dce6b`
- primary clock: `4d458f6ae879d15e3184c057f6c85bd089c7a8117e7c35420644fa39e3c18886`
- result: `cdbbfff8b9be04e6e11a261f5bdb6cf2eb6d2d7586dbef53858c0c9a3f726b87`

This authorizes only the separately frozen Gross9 novelty evaluator.  No
economic outcome is authorized yet, and no diagnostic control may be
promoted.
