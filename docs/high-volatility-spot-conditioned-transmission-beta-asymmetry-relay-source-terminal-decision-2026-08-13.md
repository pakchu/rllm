# HVSCTBA-8 terminal source-support rejection

The hash-bound HVSCTBA-8 evaluator was run only after its singleton
preregistration and source evaluator were pushed. Event counts and monthly
concentration passed in every split, but the frozen 0.20 minority-side gate
failed in train and test:

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train 2023H2 | 17 | 14 | 3 | 0.17647 | 0.35294 |
| test 2024 | 35 | 29 | 6 | 0.17143 | 0.25714 |
| eval 2025 | 33 | 22 | 11 | 0.33333 | 0.15152 |
| final to 2026-08-01 | 18 | 12 | 6 | 0.33333 | 0.38889 |

An immediate replay reproduced every panel, clock, control, manifest, and
result artifact byte-for-byte. Key hashes are:

- source panel: `78cdc9580af6bc905dd869647d7555a709ffe1331a448d21900f2ed962a93511`
- source manifest: `da5fb397aaf0e7d4bf0a660945c42ff8110a18132e170f6d5805aa17d11ad070`
- primary clock: `f08c40cd0246268f64d1f37c28e3ff6af1cd5f88c5be2f125b73ab5e377152ca`
- result: `ac079c607d935e6b6bb3163bd0dcacb6bfa9794f953764cc9da71f7d961b28f8`

HVSCTBA-8 is rejected unchanged. Gross9 rows, execution prices, funding, PnL,
strict MDD, CAGR, and RV20 remain unopened. No venue, return, beta, sign
support, rank, variation, onset, side, clock, hold, subset, threshold,
comparator, or diagnostic control may be repaired or promoted.
