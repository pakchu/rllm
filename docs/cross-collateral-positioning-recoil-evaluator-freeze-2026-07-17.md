# CCPR-1 strict evaluator freeze — 2026-07-17

## Freeze result

The CCPR-1 strict evaluator is frozen before opening any execution outcome.

| Integrity item | Result |
|---|---:|
| Evaluator source SHA-256 | `ab918bae12237056b413c506a0bac8508efcb75dbd07a84a46cd6755f11b4132` |
| Freeze manifest | `940052687d6b3a7795b0f0d1683315a1c5e169a4af9870ea586bd9cb6a44d262` |
| Schedule invariants | 80 / 80 passed |
| OHLC rows parsed | 0 |
| Funding rows parsed | 0 |
| Simulations run | 0 |
| Mutable parameters | 0 |

The evaluator binds the exact preregistration, source-only support result,
Q=0.85 clock ledger, and strict MDD engine by SHA-256. It uses a true
two-sided weekly-cluster sign-flip test; the reused strict engine's original
one-sided diagnostic is replaced from the frozen trade ledger before any gate
is evaluated.

## Frozen schedules

| Candidate | Hold | Primary | OI-only | Taker-only | USD-M-only | COIN-M-only |
|---|---:|---:|---:|---:|---:|---:|
| CCPR-H4 | 4h | 137 | 302 | 1,046 | 181 | 138 |
| CCPR-H8 | 8h | 120 | 251 | 651 | 149 | 122 |

Each candidate also has a direction flip, deterministic random side, and
one-hour entry-shift schedule. Every schedule has exact source delay and hold,
valid sides, no overlapping positions, and exits before 2024. Control schedules
are independently de-duplicated rather than borrowing the primary's event
count.

## Sequential opening contract

1. Stage1 may physically parse only `[2021-07-08, 2023-01-01)` OHLC and
   funding rows.
2. Both holds receive base cost, 10bp stress, all three Stage1 subperiods, and
   the complete seven-control battery.
3. 2023 stays sealed if neither hold passes every Stage1 gate.
4. Stage2 first verifies the stored Stage1 structure and hash, then physically
   replays Stage1 and requires an exact full-report match before loading one
   2023 row.
5. A Stage2 pass advances to the separately frozen orthogonality audit; it does
   not by itself authorize portfolio inclusion or live trading.

## Verification

- Targeted preregistration/support/evaluator suite: **23 passed**.
- Ruff lint: passed.
- Ruff format check: passed.
- Pyright evaluator module: **0 errors**.

Artifacts:

- `training/evaluate_cross_collateral_positioning_recoil.py`
- `tests/test_evaluate_cross_collateral_positioning_recoil.py`
- `results/cross_collateral_positioning_recoil_evaluator_freeze_2026-07-17.json`
