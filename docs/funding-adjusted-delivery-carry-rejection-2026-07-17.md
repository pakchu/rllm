# FADC-21 rejection before 2023 — 2026-07-17

## Decision

**Reject FADC-21 without opening 2023 or any 2024+ outcome.** The policy and
evaluator were hash-frozen before execution OHLC opened. The stage-1 parser
stopped before the first 2023 value.

| Window/cost | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021-02-03–2022, 6 bp | -4.9902% | -2.6482% | 27.5473% | -0.0961 | 30 |
| same, 10 bp stress | -7.2538% | -3.8712% | 27.7157% | -0.1397 | 30 |
| 2021 partial, 6 bp | -3.7295% | -4.0994% | 27.5473% | -0.1488 | 16 |
| 2022, 6 bp | -1.3096% | -1.3105% | 6.6475% | -0.1971 | 14 |

All four entry-cohort halves were negative. Both carry directions lost money:

- long perp / short quarter: -0.9338% compounded across 11 trades;
- short perp / long quarter: -4.0947% compounded across 19 trades.

## Why it failed

The funding forecast did identify positive cash flow, but not enough spread
edge. At base cost the fixed schedule produced approximately:

- price-spread PnL: **-4.1687%** of initial equity;
- realized funding cash: **+2.6725%**;
- transaction costs: **-3.4940%**.

The exact direction flip also lost (-2.1969%), the one-funding-event delay lost
(-5.2806%), and removing funding cash lost more (-7.5352%). This is not a
single-sign typo or a five-minute timing artifact.

The basis-only and constant long-perp/short-quarter controls were positive
(+4.4952% and +3.7364%), but their strict MDD was above 23% and CAGR/MDD below
0.10. They represent weak conventional roll exposure, not a promotable
orthogonal alpha. Using their observed sign to repair FADC would violate the
frozen no-repair rule and would be outcome-driven.

## Boundary retained

- 2023 execution paths and PnL: **sealed**;
- 2024, 2025, 2026: **sealed**;
- no threshold, lookback, sign, exit, cost or side repair is permitted on this
  sample.

The next search should abandon funding-adjusted delivery carry rather than
optimize its gate. A genuinely different event mechanism is required.
