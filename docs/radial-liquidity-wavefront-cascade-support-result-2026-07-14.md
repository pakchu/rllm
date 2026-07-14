# RLWC-144 support result — 2026-07-14

## Frozen decision

RLWC-144 was preregistered in commit `c2ff825` before support incidence was
opened. The support run loaded only the frozen calendar-2023 radial shell
panel. It loaded no BTC price, return, PnL, CAGR, MDD, label, or 2024+ row.

- support artifact:
  `results/radial_liquidity_wavefront_cascade_support_2026-07-14.json`
- artifact SHA256:
  `59e162056a443ee78e6f51a99da446dd8871e55463fd78018a52cea4196bc1fa`
- preregistration source SHA256:
  `9f94706ef05750bc08ce7ef56672512ff7d245a31f830ae1064d1d1c2b02a7a9`
- preregistration document SHA256:
  `11542acad4f2a3be6b901dcc3da91d582963d57aee4a899c6cef054da76f9b9f`
- wall time / maximum RSS: `20.75 s / 543,928 KiB`

**Decision: reject RLWC-144 v1 at the outcome-blind support gate.** The 2023
return evaluator will not be built or run, and 2024–2026 shell outcomes remain
sealed.

## Support result

| support statistic | result | required |
|---|---:|---:|
| non-overlapping trades | 0 | at least 120 |
| H1 / H2 | 0 / 0 | at least 45 / 45 |
| Q1 / Q2 / Q3 / Q4 | 0 / 0 / 0 / 0 | at least 20 each |
| long / short | 0 / 0 | each at least 35% |
| raw action candidates | 0 | nonzero |

The prior-clock Jaccards were both zero only because RLWC produced no event.
That is not evidence of useful independence.

## Outcome-blind failure localization

The exact six-bar venue-side wave detector did fire:

| wave | UM count | CM count |
|---|---:|---:|
| bid addition | 25 | 8 |
| bid withdrawal | 22 | 12 |
| ask addition | 27 | 21 |
| ask withdrawal | 16 | 12 |

The failure occurs at cross-venue synchronization, before direction, price, or
execution is considered:

- ask-withdrawal wave on both venues within the fixed current/prior-bar
  tolerance: `0`;
- bid-withdrawal wave on both venues within that tolerance: `0`;
- bid-addition wave on both venues: `0`;
- ask-addition wave on both venues: `0`;
- long primary conjunction before vetoes: `0`;
- short primary conjunction before vetoes: `0`.

The nearest same-kind UM/CM terminal distances were 73 bars for ask withdrawal,
675 for bid withdrawal, 35 for bid addition, and 28 for ask addition. Thus the
strict venue-local cascade is real enough to occur occasionally, but it is not
a synchronized cross-collateral event at a two-bar horizon in this dataset.

## Locked consequence

No threshold relaxation, longer synchronization window, single-venue fallback,
hold change, or side flip is permitted for RLWC-144 v1. Those changes would be
support-driven repairs to the same registered object. A future candidate must
use a materially different predictive object, such as a continuous radial
mass-transport or polarization statistic, and must receive its own source,
tests, preregistration, support clock, and stopping rule before returns open.
