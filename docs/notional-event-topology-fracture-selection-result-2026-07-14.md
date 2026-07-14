# NETF v1 frozen selection result — 2026-07-14

## Verdict

**NETF v1 is rejected.** No candidate passed the frozen train/2023 selection gate, so 2024, 2025, and 2026 NETF outcomes remain sealed.

- preregistration commit: `26e6b3d`
- evaluator commit: `47ae208`
- result: `results/notional_event_topology_fracture_selection_2026-07-14.json`
- result SHA256: `478e6562ce59d6bfc93257d096381ccc19d5989a008771dfa9b797fb1334522a`
- execution: next 5-minute open, scheduled-open exit, `0.5x`, `6 bp` notional cost per side
- account flat round-trip cost: `5.9991 bp`
- CAGR: full split clock including idle cash
- MDD: complete held path, favorable extreme first then adverse extreme

## `netf_fast`

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly-cluster p |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | -4.26% | -1.44% | 16.62% | -0.09 | 246 | 0.6368 |
| select 2023 | -5.37% | -5.37% | 5.87% | -0.91 | 73 | 0.9891 |
| 2023 H1 | -1.76% | -3.52% | 2.62% | -1.34 | 29 | 0.8899 |
| 2023 H2 | -3.67% | -7.15% | 4.55% | -1.57 | 44 | 0.9700 |

Mean net trade return deteriorated from `-0.0156%` in train to `-0.0751%` in 2023.

## `netf_slow`

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly-cluster p |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | +24.75% | +7.65% | 16.63% | 0.46 | 205 | 0.0412 |
| select 2023 | -6.07% | -6.07% | 6.42% | -0.95 | 61 | 0.9844 |
| 2023 H1 | -0.67% | -1.34% | 1.51% | -0.89 | 20 | 0.7522 |
| 2023 H2 | -5.31% | -10.26% | 5.73% | -1.79 | 40 | 0.9768 |

The slow candidate exhibits a genuine-looking train effect—positive absolute return and weekly-cluster `p=0.0412`—but it fails the economic target even there (`CAGR/MDD=0.46`, MDD above `15%`) and reverses sign in both 2023 halves. Mean net trade return changes from `+0.1112%` in train to `-0.1020%` in 2023.

## Interpretation boundary

The train significance cannot promote the strategy because 2023 was the frozen selection window and is uniformly negative. The sign reversal is evidence of nonstationarity or a changing meaning of aggregate-trade topology, not permission to:

- select only pre-2023 years;
- invert the 2023 direction;
- modify the 87.5th-percentile threshold;
- change confirmation/hold lengths;
- drop one side or structure mark;
- open sealed OOS windows to rescue the policy.

NETF v1 is closed. Post-hoc analysis may examine which causal structure marks or calendar eras changed, but any successor must be independently named and preregistered.
