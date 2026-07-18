# Minute Packet Topology battery — pre-2024 rejection

## Decision

**Rejected before 2023 selection and before all 2024+ OOS windows.** None of the
21 outcome-blind supported cells passed the frozen 2020–2022 train admission.
No direction, quantile, hold, or cost rule is repaired after this result.

- preregistration commit: `2d9d0a5`
- evaluator logic commit: `9cc1dd6`
- evaluator/source seal commit: `a9c9988`
- result: `results/minute_packet_topology_pre2024_selection_2026-07-19.json`
- result SHA-256:
  `ca030ff6a37704862a5da4505dbec0ab6d40b87a2ac861305ab6c21675d34d77`
- 2023 selection opened for candidates: **0**
- 2024/2025/2026 outcomes opened: **false**

## Best train result by mechanism

| frozen policy | absolute return | CAGR | strict MDD | CAGR/MDD | trades | mean gross |
|---|---:|---:|---:|---:|---:|---:|
| `cross_venue_churn_breakout_p70_s20_h96` | +6.02% | +1.97% | 16.65% | 0.12 | 220 | +19.11 bp |
| `um_swarm_absorption_p10_s20_h24` | -8.93% | -3.07% | 13.29% | -0.23 | 129 | -2.25 bp |

The churn-breakout branch had positive gross edge, but the complete three-year
return was too small and path risk too high: ratio `0.12` versus the train floor
of `1.5`. The swarm-fade direction was wrong even before costs. This is not a
transaction-cost-only failure.

## What was learned

Minute-level packet topology is reconstructable and causal, but a single
five-minute event followed by a fixed hold does not identify enough durable
inventory pressure. The only positive branch needed a 96-bar hold and still
had 16.65% strict drawdown. A successor must add a genuinely persistent state
or a separately observed constraint-release mechanism; it may not gate or
invert this failed battery using the unopened 2023/2024 outcomes.
