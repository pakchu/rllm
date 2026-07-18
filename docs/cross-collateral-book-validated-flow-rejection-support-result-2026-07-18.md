# CBFR-72 outcome-blind support result — 2026-07-18

## Verdict

**PASS support and clock-independence gates.** CBFR-72 may advance to a
separately committed and hash-frozen calendar-2023 evaluator. This is not a
profitability result.

No entry-or-later OHLC, future return, funding, PnL, equity, CAGR, strict MDD,
hit rate, or payoff was loaded. The only price fields read were the completed
signal bar's `open` and `close`.

## Outcome-blind selected clock

The frozen incidence-only selection rule chose:

- absolute taker-flow rolling quantile: `0.75`;
- mean cross-collateral defense threshold: `0.25`;
- hold: 72 five-minute bars (six hours);
- non-overlapping events: **140**;
- long / short: **75 / 65**;
- H1 / H2: **57 / 83**;
- Q1 / Q2 / Q3 / Q4: **20 / 37 / 46 / 37**.

The stricter `defense=0.50` cell failed the preregistered Q1 floor (19 versus
20), so it was not eligible. No return statistic participated in this choice.

## Prior-clock overlap (±12 five-minute bars)

| Prior clock | Matches | Jaccard | CBFR containment |
|---|---:|---:|---:|
| PDF-10 | 6 | 0.00828 | 0.04286 |
| CRRC-72 | 4 | 0.01370 | 0.02857 |
| CSPR | 8 | 0.03653 | 0.05714 |
| UMFR-36 | 12 | 0.02948 | 0.08571 |

Every comparison is below the frozen maximums (`0.20` Jaccard and `0.30` new
clock containment). The low overlap supports—but cannot prove—that this clock
is behaviorally distinct from the closest prior flow and book candidates.

## Frozen artifacts

- support:
  `results/cross_collateral_book_validated_flow_rejection_support_2026-07-18.json`
  (`5c2793a504b63c0b928b5a75407d0099e03a6c30f41cc0bce768837fbed3aa93`)
- event clock:
  `results/cross_collateral_book_validated_flow_rejection_event_clock_2026-07-18.json`
  (`b95e49600611c21a090efb43d9949607384c0a39188da4a5a069bd99bd152631`)
- canonical event-list hash:
  `8aca1a9d0071ec2cf3ca37a02a496bdaa3a3f58e257628e160a071467bc51710`

## Integrity correction made before freeze

The first dry support run exposed a scheduling defect: future source
availability could cancel a trade selected at entry. No post-entry market
outcome had been opened, but that still constituted non-causal clock logic.
The scheduler was corrected and committed before this artifact was generated:
only signal-time validity may suppress a signal; later source gaps cannot
retroactively cancel it. PDF-10 overlap was also replayed on its own frozen
book-only validity contract (591 events), not on CBFR's market-validity mask.

## Next irreversible evidence step

Implement the strict calendar-2023 evaluator, commit it, then freeze its source
hash while the CBFR outcome remains sealed. Only then may the 140 trades' held
paths be evaluated. Calendar 2024 and later remain sealed.
