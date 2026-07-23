# CVICR-72 source-support result — rejected before outcomes

## Decision

**Retire CVICR-72 unchanged.**

The preregistered cross-venue intrinsic-clock conflict-to-resolution sequence
is too sparse and fails temporal support. No threshold, transition, direction,
latency, hold, or support gate may be repaired under this candidate identity.

## Frozen execution

- preregistration commit: `7a57710`
- source-support evaluator commit: `d542f42`
- report:
  `results/cross_venue_intrinsic_clock_resolution_support_2026-07-24.json`
- report SHA-256:
  `d330cebf95a1af16162ac67847f81c2828a898d9a2e7cfe2a3efb12835523886`
- clock:
  `data/cross_venue_intrinsic_clock_resolution_clocks_2020_2023.csv.gz`
- clock SHA-256:
  `9f05b372686805539dbf56fb9b7ea7a8f90f8887d6731e1a8e1b1c1db14d8c0e`

The deterministic rerun returned `verified_existing` for both artifacts.

## Outcome-blind result

The source evaluator decoded the frozen 420,768-row eight-column allowlist and
no comparator or economic source.

```text
UTC days                                1,461
complete historical reference days     1,336
paired non-tied valid prefixes            954
strictly-prior gap reference ready        864
q60 gap pass                              356
initial conflict                          274
late two-venue alignment                  636
raw primary resolution sequence             9
globally reserved primary                   9
train 2020-2022                              9
selection 2023                               0
```

Train primary events were distributed as:

- 2020: 2;
- 2021: 3;
- 2022: 4;
- LONG/SHORT: 4/5;
- Spot-led/USD-M-led: 6/3;
- active months: 9;
- maximum entry gap: 133.93 days.

Selection contained no accepted event. The first deterministic failure was
`source_support: train_events_min`. Additional failures included every
per-year count floor, train active months and maximum gap, all selection
count/calendar/side/leader gates, and fixed-time-clock selectivity.

## Mechanism diagnosis

Clock dislocation and either component predicate are not rare:

- `gap_only`: 356 globally reserved rows;
- `initial_conflict_only`: 100;
- `late_alignment_only`: 232;
- `no_gap_tail`: 21.

The exact conjunction is the failure. Only nine days satisfy material clock
separation, opposite early cumulative flows, leader persistence, and laggard
resolution together. None survive in 2023. This is structural sparsity rather
than a scheduling or side-balance artifact.

The result does not authorize weakening the conjunction. Such a change would
be a new mechanism requiring a new boundary and preregistration.

## Closed evidence boundary

- comparator rows decoded: `0`;
- post-entry price rows decoded: `0`;
- funding rows decoded: `0`;
- future-return rows decoded: `0`;
- return/PnL fields decoded: `0`;
- CAGR/MDD values decoded: `0`;
- network calls: `0`.

No claim about profitability is made. CVICR failed before novelty and economic
evaluation.

## Research implication

The next candidate should not require a four-way exact same-day sign
conjunction. The reusable observation is that cross-venue clock dislocation,
early conflict, and late alignment each have ample independent incidence.
The next axis must change the state representation—not relax CVICR gates—and
should aggregate a causal multi-observation state or use a different source
family so weak signals can combine without collapsing support to nine events.
