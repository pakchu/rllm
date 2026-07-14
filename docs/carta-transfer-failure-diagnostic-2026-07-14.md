# CARTA v1 temporal-transfer failure diagnostic — 2026-07-14

## Boundary

This is post-hoc analysis of the already rejected 2020–2023 CARTA v1 result.
It cannot promote, repair, rebalance, or launch Gemma for CARTA. The 2024,
2025, and 2026 windows remain sealed.

- frozen rejection: `bfa27de`
- diagnostic: `results/carta_transfer_failure_diagnostic_2026-07-14.json`
- diagnostic SHA256: `804271a210d6f9e21c400746405b49413463e2a24f62b8c55a9a53ca4bb1e464`

## Annual delayed labels

| year | candidates | abstain | follow | fade | mean follow utility | mean fade utility |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 205 | 118 | 45 | 42 | -0.00357 | -0.00307 |
| 2021 | 60 | 27 | 13 | 20 | -0.00449 | -0.00724 |
| 2022 | 58 | 30 | 14 | 14 | -0.00328 | -0.00199 |
| 2023 | 236 | 180 | 25 | 31 | -0.00228 | -0.00203 |

The oracle best-action class remains mostly `ABSTAIN`, but the conditional
meaning of individual relation tokens is not stable.

## Token-effect transfer

For every supported `action × token-field=value` cell, the diagnostic subtracts
that year's action-wide mean utility, then compares the same cells across eras.

| transfer | shared cells | effect correlation | sign agreement |
|---|---:|---:|---:|
| 2020 → 2021 | 232 | -0.277 | 50.4% |
| 2021 → 2022 | 216 | +0.113 | 53.2% |
| 2022 → 2023 | 218 | +0.191 | 60.6% |
| pooled 2020–2022 → 2023 | 244 | +0.085 | 50.0% |

The pooled token effects are essentially uncorrelated with 2023 and have
coin-flip sign agreement. This directly rejects the idea that only old sample
weight caused CARTA's failure.

## Recent-history models transferred to 2023

All rows below use the exact frozen CARTA state, reward, alpha, action floor,
next-open execution, 72-bar exit, costs, and strict-MDD simulator. Only the
historical rows available to the already frozen model are restricted.

### Relational ridge

| training history | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long / short |
|---|---:|---:|---:|---:|---:|---:|
| 2020 only | -0.67% | -0.67% | 0.98% | -0.69 | 7 | 6 / 1 |
| 2021 only | -0.20% | -0.20% | 0.70% | -0.28 | 5 | 5 / 0 |
| 2022 only | -0.15% | -0.15% | 0.78% | -0.19 | 8 | 8 / 0 |
| 2020–2021 | -0.69% | -0.69% | 2.00% | -0.34 | 27 | 24 / 3 |
| 2021–2022 | -1.07% | -1.07% | 2.49% | -0.43 | 37 | 31 / 6 |
| 2020–2022 | -0.74% | -0.75% | 2.04% | -0.36 | 31 | 28 / 3 |

Every ridge history is negative and every variant is severely long-biased.
The 2022-only model is not a rescue; it executes only eight positions, all
long, and still loses.

### Categorical Naive Bayes

| training history | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long / short |
|---|---:|---:|---:|---:|---:|---:|
| 2020 only | -8.68% | -8.69% | 9.42% | -0.92 | 142 | 78 / 64 |
| 2021 only | -8.19% | -8.20% | 8.69% | -0.94 | 117 | 73 / 44 |
| 2022 only | -2.09% | -2.10% | 4.13% | -0.51 | 94 | 49 / 45 |
| 2020–2021 | -9.21% | -9.22% | 9.39% | -0.98 | 116 | 62 / 54 |
| 2021–2022 | -4.20% | -4.20% | 6.03% | -0.70 | 86 | 44 / 42 |
| 2020–2022 | -5.65% | -5.65% | 6.48% | -0.87 | 89 | 56 / 33 |

NB preserves both directions but every history loses. Recency reduces the loss
without changing its sign.

## Root cause

CARTA is not failing only because 2020 contributes too many rows. Its symbolic
cells themselves do not preserve utility sign, and recent-only learners also
fail. The event trigger identifies large movement opportunities in hindsight,
but the public aggregate-trade relations do not determine which side earns the
move six hours later with stable conditional expectation.

Therefore these are invalid repairs:

- discarding 2020 or using 2022 only;
- lowering ridge shrinkage to force more trades;
- balancing FOLLOW/FADE labels or executed sides;
- allowing Gemma to memorize nonlinear combinations of the same 36 tokens;
- opening 2024+ to select a recency window.

## Successor constraint

The next independent alpha must add a mechanism that is absent from CARTA,
not a more expressive classifier over the same event. It should predict a
**completed state transition with an observable causal consequence**—for
example a change-point in impact persistence or event-time intensity followed
by a measured relaxation/propagation phase. Recent adaptation can then be a
secondary policy feature, not the source of direction itself.
