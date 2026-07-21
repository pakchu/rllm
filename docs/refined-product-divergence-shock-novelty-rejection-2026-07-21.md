# RPDS-576 source-support and novelty result — rejected before outcomes

## Decision

RPDS-576 is retired. The frozen refined-product divergence predicate passed
its source-support gates, but failed the preregistered novelty limits against
existing EPSB source-control clocks. No BTC price, funding, return, PnL,
absolute return, CAGR, MDD, or strict simulation was opened.

- result artifact:
  `results/refined_product_divergence_shock_source_support_2026-07-21.json`;
- artifact SHA-256:
  `e9ab44864ddb0e5c92c69c4eb50bc32a941f50f9fe7ab064df388e0f618993b6`;
- result manifest hash:
  `b1e03654bc2ebf2ef2dd90df935e4264c8f58b77f725633081653a944532cadf`;
- clock artifact:
  `data/refined_product_divergence_shock_clocks_2019_2023.csv.gz`;
- clock SHA-256:
  `729f9a236923909ff906f499fb2fe1bada8c38b1db5382b38e1cf4a189f9f52e`;
- evaluator SHA-256:
  `aa91aea9d9213c696615c561ca848b9c62dd2c5c46678da9696f80a5a1ca465a`;
- source rows: 259 physical, 258 quality-ready, one quarantined; and
- emitted clock rows: 923 across the primary and seven frozen controls.

## Source support

The primary clock passed every frozen density, direction-balance, calendar-
concentration, quality, causal-clock, split-containment, and non-overlap check.

| Split | Events | LONG | SHORT | Distribution check |
|---|---:|---:|---:|---|
| train 2020–2022 | 54 | 25 | 29 | 21/17/16 by year; maximum month 7.41% |
| selection 2023 | 15 | 7 | 8 | 8/7 by half-year; maximum month 20.00% |

This establishes only that the rule is executable often enough to evaluate.
It is not evidence of return or predictive value.

## Novelty failure

Novelty compared the 69 combined train-plus-selection RPDS primary clocks,
not RPDS control clocks, against each individual candidate in the frozen
comparison cohort on the full `[2020-01-01, 2024-01-01)` five-minute grid.

| Comparator | Exact-entry Jaccard | Max +/-6h containment | Signed exposure correlation | Failed limits |
|---|---:|---:|---:|---|
| EPSB `crude_only` | 33.50% | 100.00% | -0.4741 | all three |
| EPSB `refined_products_only` | 57.98% | 100.00% | +0.6219 | all three |
| EPSB `one_release_delay` | 15.69% | 32.65% | +0.0545 | Jaccard, containment |

Frozen limits were Jaccard at most 10%, maximum bidirectional containment at
most 25%, and absolute signed exposure correlation at most 0.35. RPDS is the
intersection of an opposite crude sign with agreeing refined-product signs,
so every RPDS release is necessarily contained in both the prior `crude_only`
and `refined_products_only` control clocks. The failure therefore exposes a
structural reuse of previously evaluated execution timing, not a threshold
that can be repaired.

The mutually exclusive EPSB `primary` comparison behaved as expected: zero
exact release overlap, zero exact entry overlap, zero tolerant containment,
and signed exposure correlation -0.0017.

## Outcome boundary

The evaluator read 259 frozen EIA source rows and 3,461 outcome-free comparator
clock rows. It made no network or subprocess calls and read:

```text
BTC market rows          = 0
funding rows             = 0
future-return rows       = 0
return or PnL fields     = 0
post-2023 source rows    = 0
economic outcomes opened = false
```

The prefreeze disclosure remains one EIA source row and ten comparator rows
printed for schema inspection. Those exposures did not compute RPDS incidence,
overlap, or any market outcome.

## No-repair rule

The following are prohibited:

- dropping EPSB controls from the already frozen novelty cohort;
- comparing only with EPSB `primary` because it happens to be disjoint;
- relaxing the 10% Jaccard, 25% containment, or 0.35 correlation limits;
- adding a magnitude, seasonality, regime, latency, or hold filter to escape
  the inherited clocks; and
- opening RPDS BTC outcomes.

The next search must use an independent source clock. The preserved next axis
is Bitcoin coinbase first-payout topology, whose bounded source-only probe was
approved in
`docs/bitcoin-coinbase-payout-topology-source-feasibility-2026-07-21.md`.
