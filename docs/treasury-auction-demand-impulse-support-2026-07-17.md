# TADI-1 source support freeze — 2026-07-17

## Decision

**PASS SUPPORT; FREEZE THE STRICT STAGE1 EVALUATOR.**

No market OHLC, funding cash flow, future return, PnL, CAGR, MDD, win rate, or
portfolio overlap was opened in this step.

## Primary source clock

| Window | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| 2021 | 13 | 8 | 5 | 15.38% |
| 2022 | 15 | 10 | 5 | 20.00% |
| Stage1 2021-2022 | 28 | 18 | 10 | 10.71% |
| 2023 H1 | 12 | 8 | 4 | 41.67% |
| 2023 H2 | 11 | 1 | 10 | 36.36% |
| Sealed 2023 | 23 | 9 | 14 | 21.74% |

The month-concentration gate applies to full Stage1 and full 2023, not to each
half diagnostic. All preregistered density, side-balance, half-coverage, and
month-share floors pass.

## Quarantine enforcement

- source rows: 445;
- same-day complete: 440;
- later-updated and blanked: 5;
- a quarantined row resets the previous same-tenor observation;
- no demand change is bridged across a quarantined auction.

## Frozen control clocks

- bid-to-cover change only;
- indirect-share change only;
- one complete same-tenor auction delay;
- direction flip and deterministic random side will reuse the primary entries
  inside the evaluator.

Every clock is globally non-overlapping under the same 24-hour reservation.
Source-clock Jaccards are diagnostics only; no return overlap was computed.

## Artifacts

- Builder: `training/build_treasury_auction_demand_impulse_support.py`
- Tests: `tests/test_build_treasury_auction_demand_impulse_support.py`
- Report: `results/treasury_auction_demand_impulse_support_2026-07-17.json`
- Ledger: `results/treasury_auction_demand_impulse_clocks_2026-07-17.csv.gz`
