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
- mean cross-collateral defense threshold: `0.50`;
- hold: 72 five-minute bars (six hours);
- non-overlapping events: **144**;
- long / short: **74 / 70**;
- H1 / H2: **66 / 78**;
- Q1 / Q2 / Q3 / Q4: **22 / 44 / 43 / 35**.

The stricter `defense=0.75` cell failed the preregistered total and Q1 floors,
so `defense=0.50` was the highest eligible threshold. No return statistic
participated in this choice.

## Prior-clock overlap (±12 five-minute bars)

| Prior clock | Matches | Jaccard | CBFR containment |
|---|---:|---:|---:|
| PDF-10 | 9 | 0.01240 | 0.06250 |
| CRRC-72 | 5 | 0.01695 | 0.03472 |
| CSPR | 7 | 0.03125 | 0.04861 |
| UMFR-36 | 10 | 0.02421 | 0.06944 |

Every comparison is below the frozen maximums (`0.20` Jaccard and `0.30` new
clock containment). The low overlap supports—but cannot prove—that this clock
is behaviorally distinct from the closest prior flow and book candidates.

## Frozen artifacts

- support:
  `results/cross_collateral_book_validated_flow_rejection_support_2026-07-18.json`
  (`048a8723494a91b082bdd07d466e1741a13a974c3c3c25c8ec81e081f27cc444`)
- event clock:
  `results/cross_collateral_book_validated_flow_rejection_event_clock_2026-07-18.json`
  (`79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f`)
- canonical event-list hash:
  `d2cdcad8f57867722c220e32029d0ccbf1f1aa511e5ae590cf43411a588af4bd`

## Integrity correction made before freeze

The dry support path exposed a scheduling defect: future source
availability could cancel a trade selected at entry. No post-entry market
outcome had been opened, but that still constituted non-causal clock logic.
The first correction cleared the signal table, but evaluator integration found
that the shared scheduler actually read the source frame. The scheduler frame
itself was then corrected and regression-tested before this artifact was
regenerated. Only signal-time validity may suppress a signal; later source gaps
cannot retroactively cancel it. PDF-10 overlap was also replayed on its own
frozen book-only validity contract (591 events), not on CBFR's market-validity
mask.

## Next irreversible evidence step

Implement the strict calendar-2023 evaluator, commit it, then freeze its source
hash while the CBFR outcome remains sealed. Only then may the 140 trades' held
paths be evaluated. Calendar 2024 and later remain sealed.
