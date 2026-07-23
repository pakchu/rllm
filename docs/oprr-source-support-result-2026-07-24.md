# OPRR-288 source-support result — rejected before comparators and outcomes

## Decision

**Retire OPRR-288 unchanged.**

The frozen option-rank-rotation clock was far too sparse. It failed the
predeclared source-support gates before any comparator row, BTC row, funding
row, future return, PnL, CAGR, or strict MDD value was opened.

## Bound artifacts

- evaluator commit: `3b9efaa`
- report:
  `results/cboe_option_pressure_rank_rotation_support_2026-07-24.json`
- report SHA-256:
  `bf0e15622a52ae6f748fea87abde3728ac550b3d69e52187d8a500c29c8d8968`
- source-only clock:
  `data/cboe_option_pressure_rank_rotation_clocks_2020_2023.csv.gz`
- clock SHA-256:
  `a5c15e0d6444f79239276fb9c3da0555dea27a52eda254e7425d9b223d30d46c`
- report manifest hash:
  `190c2447bfe79d190ff14f4de988e1ef5f934367b50b6b54d3bafa99164cf7d4`

## Outcome-blind funnel

| Item | Count |
|---|---:|
| term rows decoded | 1,509 |
| tail rows decoded | 1,507 |
| option rows decoded | 1,006 |
| exact common dates | 1,006 |
| rank-complete common dates | 879 |
| adjacent rank-complete transitions | 878 |
| globally accepted primary clocks, full source | 28 |
| contained train events | 11 |
| contained 2023 selection events | 13 |
| comparator rows decoded | 0 |
| BTC/funding/future-return/PnL rows decoded | 0 |

The retained train clock covered only seven active months, with a 237-day
maximum local-calendar entry gap. Selection covered eight active months, with
a 104-day maximum gap.

## First failure

The first ordered failure was:

```text
source_support: train_events_min
```

The frozen train minimum was 100 events; OPRR produced 11. It also missed both
annual floors, both selection-half floors, every selection-quarter floor, and
multiple concentration/gap gates.

## Source-geometry diagnosis

The sparse primary was not caused by a sparse underlying CBOE calendar:

| Clock | Full raw events | Train | Selection |
|---|---:|---:|---:|
| rank rotation only | 238 | 126 | 65 |
| option-own confirmed | 223 | 116 | 60 |
| non-option pair only | 304 | 164 | 97 |
| frozen OPRR primary | 28 | 11 | 13 |
| term sponsor permutation | 5 | 2 | 3 |
| tail sponsor permutation | 0 | 0 | 0 |

Requiring option rank rotation while option, term, and tail pressure all move in
that same direction collapses several individually dense weak relations into a
rare event. The failure is structural, not a threshold miss.

## No-repair rule

OPRR may not be repaired by:

- dropping either non-option confirmation;
- replacing both confirmations with their mean;
- promoting the 223-event option-own control;
- lowering the 100/45 event floors or relaxing gap/concentration gates;
- removing the two-step or sponsor-control gates; or
- opening comparator or market outcomes for any OPRR control.

Those changes would condition a successor on observed OPRR incidence. A later
candidate must change observable/state geometry and receive a new boundary,
mechanism, preregistration, and evaluator before its incidence is opened.
