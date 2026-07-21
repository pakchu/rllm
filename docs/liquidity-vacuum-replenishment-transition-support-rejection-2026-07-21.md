# LVRT-72 source-support result — rejected before economic evaluation

## Decision

LVRT-72 is retired. The preregistered sequential mechanism produced too few
causal trade clocks to justify novelty or economic evaluation. No threshold,
deadline, side, hold, or feature was changed after observing the result.

- immutable result:
  `results/liquidity_vacuum_replenishment_transition_source_gate_2026-07-21.json`;
- artifact SHA-256:
  `82ce4a9bfbd6e5346c8d4cdc33c897e47a238ef6982b4786fcf3dd183f3b1cb3`;
- result hash:
  `8fbc4ebf93fd8213f769fbcac1bdcff61552881c62277ffdf8082556ef676536`;
- pure-clock artifact: not created; and
- failure action: `retire_before_economic_evaluation`.

## Frozen mechanism incidence

The source covered 420,732 five-minute aggTrade microstructure rows from 2020
through 2023. After the strict-prior 2,016-bar rank window, 399,705 rows were
rank-ready. The frozen vacuum condition appeared 49 times, but only one setup
received the required opposite-sign replenishment confirmation within the
inclusive twelve-bar deadline.

| Stage | Count |
|---|---:|
| vacuum setups | 49 |
| confirmed transitions | **1** |
| expired without confirmation | 48 |
| gap cancellations | 0 |
| accepted train clocks | **1** |
| accepted 2023 selection clocks | **0** |

The one accepted clock was SHORT. Therefore the total, annual, half-year,
directional, active-week, and concentration support floors failed. Novelty was
correctly skipped and no pure-clock file was published.

## Diagnostic controls

Controls were frozen with the mechanism and are incidence diagnostics, not
alternative strategies.

| Clock construction | Train | 2023 selection |
|---|---:|---:|
| vacuum only | 40 | 4 |
| replenishment only | 1,676 | 613 |
| relay without flow-flip requirement | 2 | 0 |
| replenishment-before-vacuum reverse order | 1 | 0 |
| exact direction flip | 1 | 0 |
| deterministic random side | 1 | 0 |
| one-bar later execution | 1 | 0 |

This isolates the failure: neither raw data availability nor replenishment
incidence was scarce. The fully specified transition from a severe low-count,
bursty, concentrated one-sided vacuum to normalized participation plus an
opposite flow sign almost never occurred. Removing individual requirements
after seeing this would create a new, outcome-informed hypothesis rather than
validate LVRT-72.

## Outcome boundary

```text
source columns read              = 7 allowlisted microstructure fields
forbidden source columns read    = 0
market rows loaded               = 0
funding rows loaded              = 0
performance artifacts parsed     = 0
return/PnL fields read           = 0
strict simulation calls          = 0
post-2023 rows loaded             = 0
network calls                    = 0
economic outcomes computed       = false
```

The result is only a source-support rejection. It contains no claim about
return, CAGR, MDD, or profitability.

## No-repair rule

The following are prohibited for LVRT-72:

- lowering the 0.90 vacuum or flow thresholds;
- extending the twelve-bar confirmation deadline;
- dropping the opposite-sign flow transition;
- changing the 72-bar hold or reversing the side;
- treating `vacuum_only` or `replenishment_only` as the same candidate; and
- opening economic outcomes to choose among such repairs.

The next search must be registered as an independent mechanism with its own
source-support gate before any BTC outcome is evaluated.
