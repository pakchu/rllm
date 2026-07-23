# SCAF-48 source-support result — rejected before comparators and outcomes

## Decision

**Retire SCAF-48 unchanged.**

The frozen SOMA collateral-allocation fracture clock was dense, balanced, and
operationally continuous, but failed the preregistered source-selectivity and
weak-component composition gates. No SLCS comparator row, BTC market row,
funding row, future return, PnL, CAGR, or MDD value was opened.

## Bound artifacts

- evaluator commit: `f500a98`
- report:
  `results/soma_collateral_allocation_fracture_support_2026-07-24.json`
- report SHA-256:
  `714263911817a8d4d7a820ff80a4315e6e7710223944e5338f11835cb7b976bd`
- source-only clock:
  `data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz`
- clock SHA-256:
  `64e07005d70442bfa7a110b1e6bea9802ee94be16d95f6e7db9228f4790a28e6`

## Outcome-blind funnel

| Item | Count |
|---|---:|
| operation rows decoded | 1,259 |
| detail rows decoded | 182,616 |
| causal batches | 1,249 |
| valid batches | 1,249 |
| invalid batches / continuity resets | 0 / 0 |
| valid transitions | 1,248 |
| raw primary opportunities | 660 |
| globally accepted primary clocks | 368 |
| train accepted clocks | 259 |
| 2023 selection accepted clocks | 109 |
| SLCS comparator rows decoded | 0 |
| BTC/funding/future-return/PnL rows decoded | 0 |

Train contained 101 long and 158 short events across all 36 months. Selection
contained 48 long and 61 short events across all 12 months. Maximum UTC
calendar gaps were 20 and 8 days. Count, side, month, quarter, gap, run, and
required-control gates all passed.

## First failure

The first ordered failure was:

```text
source_support: selection_raw_consensus_share_max
```

The three-of-four relation fired on **169 / 248 = 68.15%** of
split-contained 2023 transitions, above the preregistered 65% maximum. Train
selectivity was 371 / 749 = 49.53% and passed.

## Composition failures

| Train component | Agreement with raw primary side | Gate |
|---|---:|---:|
| inventory mismatch | 85.44% | 55–95% |
| award distortion | **97.30%** | 55–95% |
| unmet-demand mass | **52.56%** | 55–95% |
| fee distortion | 84.64% | 55–95% |

Selection component agreement was 76.92% / 92.31% / 83.43% / 81.66%, so all
selection composition gates passed. Train failed because award distortion was
almost always aligned while unmet-demand mass was below its minimum
participation. This is not a random-side or stale-clock artifact:

- train deterministic-random reproduction: 51.35%;
- train one-batch stale reproduction: 7.34%;
- train five-batch stale reproduction: 19.69%;
- train permutation exact-entry Jaccard: 28.21%.

## Interpretation

The daily source and 48-hour execution cadence are viable, but SCAF did not
behave as a balanced combination of four weak signals. In train it collapsed
toward award-distortion direction, while in selection the consensus became too
dense to remain a selective fracture event.

Raising the 65% density ceiling, raising the 95% dominance ceiling, lowering
the 55% component floor, or removing unmet-demand mass would be post-incidence
repair and is forbidden. A successor may retain only the general lesson that
CUSIP allocation distributions are operationally rich; it must use a genuinely
different state geometry or source family rather than retune SCAF.
