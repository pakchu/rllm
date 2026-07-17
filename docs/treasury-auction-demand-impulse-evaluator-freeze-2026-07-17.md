# TADI-1 strict evaluator freeze — 2026-07-17

## State

The evaluator is frozen before opening any BTC execution outcome.

- candidate: one, unchanged `TADI-1`;
- primary and five falsification/control schedules are hash-bound;
- 5-minute post-source entry and 24-hour hold are exact;
- every clock is globally non-overlapping;
- strict accounting imports the previously audited execution engine;
- the evaluator replaces its one-sided diagnostic with a true two-sided
  weekly-cluster sign-flip test;
- Stage1 physically parses only `[2021-01-01, 2023-01-01)`;
- 2023 is loaded only after exact replay of a passing Stage1 report;
- no threshold, direction, source, hold, gate, or regime repair is mutable.

## Frozen clock family

- primary dual-concordance clock;
- bid-to-cover-only mechanism control;
- indirect-share-only mechanism control;
- exact direction flip;
- one complete same-tenor auction delay;
- deterministic random side.

Every control receives the same base/stress cost, funding, strict-MDD,
significance, trade-count, gross-edge, and contained-subperiod battery. A
standalone primary pass additionally requires a CAGR/strict-MDD margin of at
least 0.25 over both mechanism controls.

## Artifacts

- Evaluator: `training/evaluate_treasury_auction_demand_impulse.py`
- Tests: `tests/test_evaluate_treasury_auction_demand_impulse.py`
- Freeze manifest:
  `results/treasury_auction_demand_impulse_evaluator_freeze_2026-07-17.json`

At freeze time, parsed OHLC rows, parsed funding rows, and simulations are all
zero.
