# DCLB-864 source-support result — rejected before comparators and outcomes

## Decision

**Retire DCLB-864 unchanged.**

The frozen dollar-collateral/bank-relay clock failed the preregistered
source-support gate. No comparator row, BTC market row, funding row, future
return, PnL, CAGR, or MDD value was opened.

## Bound artifacts

- evaluator commit: `c83a55b`
- report:
  `results/dollar_collateral_liquidity_bank_relay_support_2026-07-24.json`
- report SHA-256:
  `05db079878d8ad218ab4350e79ec7899bb66d36de4ae4ecc8b0c0b884cd988c5`
- source-only clock:
  `data/dollar_collateral_liquidity_bank_relay_clocks_2020_2023.csv.gz`
- clock SHA-256:
  `1973a3a79c574b6cc53f93e36f2b6c1550f7d8050e3ac34fe903f28a0253cb37`

## Outcome-blind funnel

| Item | Count |
|---|---:|
| H.4.1 rows decoded | 313 |
| ON RRP rows decoded | 1,498 |
| H.8 rows decoded | 365 |
| common causal rows | 133 |
| raw primary-eligible rows | 107 |
| globally accepted primary clocks | 105 |
| train primary clocks | 77 |
| 2023 selection primary clocks | 28 |
| comparator rows decoded | 0 |
| BTC/funding/future-return/PnL rows decoded | 0 |

## First failure

The first ordered failure was:

```text
source_support: train_maximum_entry_gap
```

The frozen train maximum New York calendar gap was **196 days**, against the
preregistered maximum of **60 days**. This was not a marginal miss: the
calendar was sparse in both 2020 and 2021, whose within-year maximum gaps were
196 and 118 days.

## Additional failures

| Check | Observed | Gate |
|---|---:|---:|
| 2023 Q1 events | 0 | at least 2 |
| train RRP-only same-side reproduction | 85.71% | at most 85% |
| selection stale-RRP same-side reproduction | 85.71% | at most 85% |

The 2023 selection aggregate remained superficially adequate—28 events across
9 active months, with 17 long and 11 short—but had no Q1 event. The train
calendar had 77 events and balanced sides, yet its long source-free gaps
violated the operational continuity requirement.

## Interpretation

DCLB-864 did not fail because of a market outcome or an optimized threshold.
It failed before comparator and market evidence because the exact three-source
intersection was not a stable recurring clock throughout the frozen period.
The two RRP-reproduction misses additionally show that the composite boundary
was only marginally distinct from an RRP timing component.

Relaxing the 60-day gap, allowing an empty selection quarter, or raising the
85% reproduction ceilings would be post-result gate repair and is forbidden.
The next candidate must use a different observable geometry and may not be a
retuned DCLB intersection.
