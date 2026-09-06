# HVLSRA-24 train economic rejection

Date: 2026-08-12

## Decision

HVLSRA-24 is terminally rejected unchanged at the first strict economic stage.
The source-support and Gross9 novelty passes remain valid source/structural
evidence, but they are not alpha evidence. No test, eval, final, or RV20-q90
outcome was opened.

## Frozen train result

Window: 2023-07-01 through 2024-01-01, 17 trades, fixed 0.5 gross exposure,
exact funding, 6 bp base and 10 bp stress costs per notional side, full-calendar
CAGR, and favorable-then-adverse held-5m strict MDD.

| metric | base | stress |
|---|---:|---:|
| absolute return | -0.7562% | -1.4296% |
| full-calendar CAGR | -1.4955% | -2.8178% |
| strict MDD | 5.5266% | 5.6966% |
| CAGR / strict MDD | -0.2706 | -0.4947 |
| mean gross underlying move | 5.64 bp | 5.64 bp |

The one-sided UTC-week cluster sign-flip p-value is 0.5637. Both calendar
halves are negative. Only the strict-MDD ceiling passed; return, ratio, gross
move, significance, stress, and half-year gates failed.

## Reproduction and boundary

Two consecutive frozen runs produced artifact SHA-256
`0ecb9bf6f565c9a04ca1064d467efd211cabe35815be8866b6c252e223a6bbd4`.

The registered stopping rule prohibits changing the lottery, report field,
availability delay, variation threshold, side, hold, clock, subset, weekday
adjustment, jackpot control, or comparator. Diagnostic controls cannot be
promoted. The candidate is therefore closed without repair.
