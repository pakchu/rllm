# Weak signal feature ensemble pre-2024 test — 2026-07-15

This tests combinations of previously meaningful weak signal policies. Weights are fit on 2020–2022 only; 2023 is holdout. 2024+ remains sealed.

Candidate sleeves: 24

## Top combinations

| Rank | Train abs | Train CAGR/MDD | Train MDD | 2023 abs | 2023 CAGR/MDD | 2023 MDD | 2023 trades | Weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | -3.98% | -0.07 | 18.47% | -0.25% | -0.04 | 6.23% | 249 | rift:direction_flip=0.30, rift:stale_setup_1h=0.60 |
| 2 | -0.80% | -0.10 | 2.58% | 0.20% | 0.32 | 0.63% | 85 | rift:direction_flip=0.10 |
| 3 | -0.09% | -0.01 | 3.10% | -0.12% | -0.11 | 1.17% | 164 | rift:stale_setup_1h=0.10 |
| 4 | -1.62% | -0.11 | 5.10% | 0.40% | 0.32 | 1.25% | 85 | rift:direction_flip=0.20 |
| 5 | -2.05% | -0.11 | 6.34% | 0.50% | 0.32 | 1.56% | 85 | rift:direction_flip=0.25 |

## Interpretation

This is a weak-signal ensemble test, not a deployment decision. Passing requires the 2023 holdout to remain positive with acceptable strict MDD before any 2024+ evaluation is justified.

## Result

**Fail.** Independent sleeve allocation does not convert the weak signals into a
tradable ensemble. The best train-ranked allocation is still negative in train
and 2023. The few 2023-positive rows are single low-weight `rift:direction_flip`
variants, but they are train-negative and therefore cannot be selected without
using 2023 outcome information.

Carry-forward: independent capital allocation across weak event clocks is the
wrong composition operator for these features.
