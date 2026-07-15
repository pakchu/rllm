# Frozen ExtraTrees rank-7 stability — 2026-07-15

Artifact JSON: `results/expanding_extratrees_rank7_stability_2026-07-15.json`

Manifest: `c6e7d78a328118456eacf70bc42cb12a48f33e26d13edbe21f2edb3aedea4f8e`, frozen rank: `7`

Spec: `max_depth=2,min_samples_leaf=32,max_features=.8,lambda=.25,funding_q=.40,premium_q=.55,risk_q=.75`; 1h delayed features; source-owned exact labels; purged exits; next-open; exact costs/funding/strict MDD; prediction `n_jobs=1` forced.

## Summary

Individual seed passes: `1/5`
- 300-tree 5-seed mean ensemble: **PASS**, hash `0cc647284179f772`
- 1000-tree 5-seed mean ensemble: **PASS**, hash `790e88d2ebb36955`
- 2000-tree 5-seed mean ensemble: **PASS**, hash `3234b4eaffa98dc0`

## Metrics

| Case | Period | Abs ret | CAGR | MDD | Ratio | Trades | Pass | Hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| seed7_300 | 2023 | 12.3418% | 12.3508% | 3.1173% | 3.9620 | 18 | FAIL | `4b801b979556` |
| seed7_300 | 2024 | 11.6661% | 11.6409% | 3.4631% | 3.3614 | 19 | FAIL | `4b801b979556` |
| seed7_300 | 2025 | 17.1079% | 17.1206% | 4.9844% | 3.4348 | 20 | FAIL | `4b801b979556` |
| seed7_300 | 2026h1 | 7.3132% | 18.4835% | 4.3007% | 4.2978 | 12 | FAIL | `4b801b979556` |
| seed7_300 | all | 57.6530% | 14.2513% | 4.9844% | 2.8592 | 69 | FAIL | `4b801b979556` |
| seed71_300 | 2023 | 12.2003% | 12.2092% | 3.1173% | 3.9166 | 20 | FAIL | `cef0e3bd56ab` |
| seed71_300 | 2024 | 14.8499% | 14.8173% | 3.4631% | 4.2786 | 23 | FAIL | `cef0e3bd56ab` |
| seed71_300 | 2025 | 12.6979% | 12.7071% | 4.9844% | 2.5494 | 18 | FAIL | `cef0e3bd56ab` |
| seed71_300 | 2026h1 | 9.3938% | 24.0788% | 4.3007% | 5.5988 | 13 | FAIL | `cef0e3bd56ab` |
| seed71_300 | all | 58.8667% | 14.5080% | 4.9844% | 2.9107 | 74 | FAIL | `cef0e3bd56ab` |
| seed715_300 | 2023 | 13.5439% | 13.5538% | 3.1173% | 4.3479 | 19 | FAIL | `7e51d98a847f` |
| seed715_300 | 2024 | 16.7268% | 16.6899% | 3.4631% | 4.8194 | 19 | FAIL | `7e51d98a847f` |
| seed715_300 | 2025 | 12.7733% | 12.7826% | 5.2179% | 2.4498 | 20 | FAIL | `7e51d98a847f` |
| seed715_300 | 2026h1 | 9.3938% | 24.0788% | 4.3007% | 5.5988 | 13 | FAIL | `7e51d98a847f` |
| seed715_300 | all | 63.5059% | 15.4767% | 5.2179% | 2.9661 | 71 | FAIL | `7e51d98a847f` |
| seed2026_300 | 2023 | 12.2034% | 12.2122% | 3.1173% | 3.9175 | 20 | PASS | `55804400c1a3` |
| seed2026_300 | 2024 | 16.9371% | 16.8997% | 3.4631% | 4.8799 | 20 | PASS | `55804400c1a3` |
| seed2026_300 | 2025 | 17.1079% | 17.1206% | 4.9844% | 3.4348 | 20 | PASS | `55804400c1a3` |
| seed2026_300 | 2026h1 | 7.3132% | 18.4835% | 4.3007% | 4.2978 | 12 | PASS | `55804400c1a3` |
| seed2026_300 | all | 64.8912% | 15.7622% | 4.9844% | 3.1623 | 72 | PASS | `55804400c1a3` |
| seed71515_300 | 2023 | 12.2287% | 12.2375% | 3.1173% | 3.9257 | 18 | FAIL | `26d33cee92ed` |
| seed71515_300 | 2024 | 15.3441% | 15.3104% | 3.9754% | 3.8513 | 21 | FAIL | `26d33cee92ed` |
| seed71515_300 | 2025 | 11.7695% | 11.7780% | 5.2179% | 2.2572 | 21 | FAIL | `26d33cee92ed` |
| seed71515_300 | 2026h1 | 7.6026% | 19.2528% | 3.5225% | 5.4657 | 10 | FAIL | `26d33cee92ed` |
| seed71515_300 | all | 55.6844% | 13.8319% | 5.2179% | 2.6509 | 70 | FAIL | `26d33cee92ed` |
| ensemble5_300 | 2023 | 12.8641% | 12.8735% | 3.1173% | 4.1297 | 19 | PASS | `0cc647284179` |
| ensemble5_300 | 2024 | 16.3961% | 16.3599% | 3.4631% | 4.7241 | 22 | PASS | `0cc647284179` |
| ensemble5_300 | 2025 | 16.3620% | 16.3740% | 4.9844% | 3.2850 | 21 | PASS | `0cc647284179` |
| ensemble5_300 | 2026h1 | 7.3132% | 18.4835% | 4.3007% | 4.2978 | 12 | PASS | `0cc647284179` |
| ensemble5_300 | all | 64.0433% | 15.5877% | 4.9844% | 3.1273 | 74 | PASS | `0cc647284179` |
| ensemble5_1000 | 2023 | 12.8641% | 12.8735% | 3.1173% | 4.1297 | 19 | PASS | `790e88d2ebb3` |
| ensemble5_1000 | 2024 | 15.0636% | 15.0305% | 3.4631% | 4.3402 | 22 | PASS | `790e88d2ebb3` |
| ensemble5_1000 | 2025 | 16.3620% | 16.3740% | 4.9844% | 3.2850 | 21 | PASS | `790e88d2ebb3` |
| ensemble5_1000 | 2026h1 | 7.3132% | 18.4835% | 4.3007% | 4.2978 | 12 | PASS | `790e88d2ebb3` |
| ensemble5_1000 | all | 62.1653% | 15.1988% | 4.9844% | 3.0493 | 74 | PASS | `790e88d2ebb3` |
| ensemble5_2000 | 2023 | 12.8641% | 12.8735% | 3.1173% | 4.1297 | 19 | PASS | `3234b4eaffa9` |
| ensemble5_2000 | 2024 | 16.0576% | 16.0222% | 3.4631% | 4.6266 | 21 | PASS | `3234b4eaffa9` |
| ensemble5_2000 | 2025 | 16.3620% | 16.3740% | 4.9844% | 3.2850 | 21 | PASS | `3234b4eaffa9` |
| ensemble5_2000 | 2026h1 | 7.3132% | 18.4835% | 4.3007% | 4.2978 | 12 | PASS | `3234b4eaffa9` |
| ensemble5_2000 | all | 63.5662% | 15.4892% | 4.9844% | 3.1075 | 73 | PASS | `3234b4eaffa9` |

## Determinism

- ensemble5_300: MATCH first `0cc647284179f7728e83a7ed6c160f9600c3509f25e468224e6a5d2f2e029eef`, repeat `0cc647284179f7728e83a7ed6c160f9600c3509f25e468224e6a5d2f2e029eef`
- ensemble5_1000: MATCH first `790e88d2ebb36955bcf2ea08160427ad8bdb890c090d2fe3c8a8a82204b00cc0`, repeat `790e88d2ebb36955bcf2ea08160427ad8bdb890c090d2fe3c8a8a82204b00cc0`
- ensemble5_2000: MATCH first `3234b4eaffa98dc08613550627bab99ec35cb19c0611977c4fc59fb9dd2bdb26`, repeat `3234b4eaffa98dc08613550627bab99ec35cb19c0611977c4fc59fb9dd2bdb26`
