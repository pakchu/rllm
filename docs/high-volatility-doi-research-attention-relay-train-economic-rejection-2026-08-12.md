# HVDRA-24 train economic rejection

Date: 2026-08-12

## Decision

HVDRA-24 is terminally rejected unchanged at the first strict economic stage.
Its source-support and Gross9 novelty results remain valid structural evidence,
but the train risk-adjusted and statistical gates do not establish deployable
alpha. Test, eval, final, and RV20-q90 outcomes remain unopened.

## Frozen train result

Window: 2023-07-01 through 2024-01-01, 43 trades, fixed 0.5 gross exposure,
exact funding, 6 bp base and 10 bp stress costs per notional side, full-calendar
CAGR, and favorable-then-adverse held-5m strict MDD.

| metric | base | stress |
|---|---:|---:|
| absolute return | +3.6272% | +1.8565% |
| full-calendar CAGR | +7.3289% | +3.7190% |
| strict MDD | 12.3543% | 12.5195% |
| CAGR / strict MDD | 0.5932 | 0.2971 |
| mean gross underlying move | 31.17 bp | 31.17 bp |

Both calendar halves are positive and mean gross move exceeds 20 bp. However,
base CAGR/MDD is below 3.0, stress CAGR/MDD is below 2.5, and the one-sided
UTC-week cluster sign-flip p-value is 0.3534 rather than at most 0.1.

Two frozen runs produced artifact SHA-256
`86181944c03372b42f490e6a98c7d9fb72893dc10d7c4eb495f7a273a2d379a8`.
The stopping rule prohibits query, title grammar, work type, redeposit rule,
delay, lag, variation threshold, side, hold, clock, subset, source, or control
repair. Diagnostic controls cannot be promoted.
