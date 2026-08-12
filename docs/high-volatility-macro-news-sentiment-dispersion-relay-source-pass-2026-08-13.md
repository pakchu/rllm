# HVMNSD-24 source-support pass

The frozen HVMNSD-24 source evaluator was run only after the outcome-blind
preregistration and evaluator commits were pushed. It used the hash-bound SF
Fed Daily News Sentiment snapshot under the conservative D+8 embargo and built
the unchanged two-window population-dispersion clock without opening Gross9
rows, execution prices, funding, or post-entry outcomes.

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train 2023H2 | 45 | 25 | 20 | 0.44444 | 0.33333 |
| test 2024 | 159 | 67 | 92 | 0.42138 | 0.16352 |
| eval 2025 | 112 | 48 | 64 | 0.42857 | 0.21429 |
| final to 2026-08-01 | 80 | 40 | 40 | 0.50000 | 0.31250 |

Every 8/12/12/8 count, 0.20 minority-side, and 0.45 monthly-concentration
gate passed. An immediate replay reproduced every state, primary-clock,
control, manifest, and result artifact byte-for-byte. Key hashes are:

- state: `b929abb10376c9e8da8ee1383887afa0a2ed428120539ca92a1878cee9be8cef`
- source-artifact manifest: `44da9402aa74234867fcf4d1d2c1a3539147ab0a2ff7a77d60eb46f7bc680645`
- primary clock: `d5665424eff2bf1abb1474a3e1868ae46c49b364f7f2181095a27bcd49431048`
- result: `4a117c0de3b75d9225e9f0f122bdaf253917cc9e1fd4fa4f68921b439b3ecf39`

This authorizes only a separately frozen Gross9 novelty evaluator. No economic
outcome is authorized yet, and no diagnostic control may be promoted.
