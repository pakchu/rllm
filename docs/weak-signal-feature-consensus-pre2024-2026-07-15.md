# Weak signal feature ensemble pre-2024 test — 2026-07-15

This tests combinations of previously meaningful weak signal policies. Weights are fit on 2020–2022 only; 2023 is holdout. 2024+ remains sealed.

Candidate sleeves: 24

## Top combinations

| Rank | Train abs | Train CAGR/MDD | Train MDD | 2023 abs | 2023 CAGR/MDD | 2023 MDD | 2023 trades | Weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 17.29% | 0.51 | 10.62% | -1.09% | -0.23 | 4.84% | 43 | equal thr=0.2 hold=72 counts={'train': 122, 'select2023': 43, 'select2023_h1': 12, 'select2023_h2': 31} |
| 2 | 9.91% | 0.35 | 9.14% | -3.82% | -0.84 | 4.55% | 45 | equal thr=0.2 hold=48 counts={'train': 123, 'select2023': 45, 'select2023_h1': 12, 'select2023_h2': 33} |
| 3 | 12.07% | 0.35 | 11.09% | -1.43% | -0.30 | 4.81% | 41 | equal thr=0.2 hold=96 counts={'train': 120, 'select2023': 41, 'select2023_h1': 11, 'select2023_h2': 30} |
| 4 | 6.87% | 0.24 | 9.52% | -4.11% | -0.82 | 5.02% | 45 | equal thr=0.2 hold=36 counts={'train': 124, 'select2023': 45, 'select2023_h1': 12, 'select2023_h2': 33} |
| 5 | 13.72% | 0.12 | 35.99% | -8.71% | -0.47 | 18.43% | 469 | gross_weighted thr=0.05 hold=96 counts={'train': 1317, 'select2023': 469, 'select2023_h1': 223, 'select2023_h2': 246} |

## Interpretation

This is a weak-signal ensemble test, not a deployment decision. Passing requires the 2023 holdout to remain positive with acceptable strict MDD before any 2024+ evaluation is justified.

## Result

**Fail, but informative.** Equal-weight consensus at threshold `0.20` can create
a much stronger train signal: +17.29% absolute return, 5.46% CAGR, 10.62% strict
MDD, and +39.35 bp gross on 122 trades. However, it fails the 2023 holdout:
-1.09% absolute return and only 43 trades.

The only saved consensus row with positive 2023 was `gross_weighted`, threshold
`0.20`, hold `72`, with +2.30% in 2023, +1.14% in both H1 and H2, and 112 trades;
but it was train-negative (-0.81%) and therefore is not selectable under the
train-only protocol.

Carry-forward: weak-signal agreement is a better structure than independent
sleeve allocation, but the present weak feature set is regime-unstable. The next
step should make the consensus operator itself causal/adaptive, e.g. online
experts or rolling feature reliability, rather than freezing one global vote
threshold.
