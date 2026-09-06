# HVMCPAC-8 pairwise condition battery preregistration

The prior independent-source exhaustion blocker is no longer operative because the
research direction now authorizes combinations of already available, frozen
conditions. This battery is exploratory discovery rather than fresh confirmatory
evidence: every component's standalone outcome was previously observed, while no
combination incidence or combination PnL was opened before this lock.

## Frozen family

The four mechanically and semantically selected components are:

- CARSC-8: cross-alt return synchrony,
- HVTCCR-8: quote-turnover concentration,
- HVTFR-8: time-price trend fit,
- HVLZC-8: Lempel-Ziv path compressibility.

All use exact 00:00/08:00/16:00 UTC entries at decision plus five minutes, an
eight-hour hold, and frozen component formulas. The family contains exactly the
six unordered pairwise intersections. A pair fires only when both component clock
artifacts contain exactly the same entry timestamp and exactly the same strict
nonzero side. No timestamp tolerance, side reconciliation, threshold change, or
higher-order combination is allowed.

## Selection and stopping

All six pairs count in the Bonferroni family (`alpha=0.10`, raw winner weekly
sign-flip `p<=0.10/6`). Source support and unchanged Gross9 novelty are evaluated
before economics. Among eligible pairs, the raw train rank one is selected by
base-cost CAGR/strict-MDD, then absolute return, then fixed candidate order. The
winner must pass every train gate before test is opened. A rank-one failure, or
any later test/eval/final failure, terminates the battery without substituting
another pair.

The canonical preregistration artifact is
`results/high_volatility_multi_condition_pairwise_concordance_preregistration_2026-08-14.json`
with SHA-256
`3cdd3edbedfda4e581bb95b9fac2db7309a7b54fe72c37ff5e5004ce8bba8d14`.
It was reproduced byte-for-byte twice; the targeted five-test contract suite
passed twice before any combination incidence was materialized.
