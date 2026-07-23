# CXRT-288 source-support result — rejected before comparators and outcomes

## Decision

**Retire CXRT-288 unchanged.**

The frozen cross-surface majority clock was dense enough, but it failed the
predeclared source-support and relational-composition gates. No comparator row,
BTC market row, funding row, future return, PnL, CAGR, or MDD value was opened.

## Bound artifacts

- evaluator commit: `a2e8972`
- report:
  `results/cboe_cross_surface_risk_transfer_support_2026-07-24.json`
- report SHA-256:
  `58b33695b861a213af1f177adb986b2025c4276116f2ebbc6a461278456e678d`
- source-only clock:
  `data/cboe_cross_surface_risk_transfer_clocks_2020_2023.csv.gz`
- clock SHA-256:
  `b3cc6f3d6a19cb39ef63ec0ba9908c983ce03c56a0c7dd8786e51c2ef1c0885f`

## Outcome-blind funnel

| Item | Count |
|---|---:|
| term rows decoded | 1,509 |
| tail rows decoded | 1,507 |
| option rows decoded | 1,006 |
| exact common dates | 1,006 |
| rank-complete common dates | 879 |
| schedulable common dates | 878 |
| globally accepted primary clocks | 870 |
| comparator rows decoded | 0 |
| BTC/funding/future-return/PnL rows decoded | 0 |

The primary had 498 contained train events and 246 contained 2023 selection
events. Both sides, all months, both selection halves, every selection quarter,
and maximum entry-gap requirements had support.

## First failure

The first ordered failure was:

```text
source_support: selection_max_same_side_run
```

The frozen selection maximum was 30 consecutive same-side events versus the
predeclared maximum of 20. The run occurred in the second half of 2023; Q4 alone
contained a 21-event same-side run.

## Additional composition failures

| Check | Train | Selection | Gate |
|---|---:|---:|---:|
| unanimous share | 8.84% | 13.82% | 10–80% |
| option-only same-side reproduction | 88.15% | 86.99% | at most 80% |
| tail RELIEF share | 53.21% | 6.91% | at least 15% |

The random-side reproduction remained near chance (50.20% train, 52.44%
selection), and the one-common-date stale control remained distinct (45.18%
train, 49.19% selection). The failure is therefore not a generic clock
duplication problem. It is specifically:

1. a late-selection tail surface collapse into STRESS;
2. an option surface that reproduces the composite side too often; and
3. insufficient unanimous participation in train.

## Interpretation

CXRT demonstrated that dense CBOE cross-surface states are operationally
available, but the equal-vote relay does not create a sufficiently balanced,
multi-surface decision boundary. Relaxing the 20-run cap, lowering the
unanimity floor, or raising the option reproduction cap would be post-result
gate repair and is forbidden.

The next candidate must change the observable or state geometry rather than
retune CXRT. In particular, it should not use the same equal-vote majority or
promote option flow from a weak relation token into a dominant direction
generator.
