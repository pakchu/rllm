# REX taker-low + range-high fixed gate rolling validation (2026-07-11)

## Candidate

Fixed gate selected from robust train/test TTE follow-up:

- `taker_imbalance <= -0.06692877021545077`
- `range_vol >= 0.021877719902693965`

This validation does **not** fit or rank thresholds. It only reports the fixed gate across full periods and calendar blocks.

## Artifacts

- Script: `training/validate_fixed_event_gate_by_period.py`
- Quarterly JSON: `results/rex_taker_low_range_high_fixed_gate_period_validation_2026-07-11.json`
- Monthly JSON: `results/rex_taker_low_range_high_fixed_gate_monthly_validation_2026-07-11.json`

## Full-period stats

| Period | Abs return | CAGR | Strict MDD | CAGR/MDD | Trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 2021-2024 | 95.35% | 18.22% | 12.64% | 1.44 | 306 | 0.0046 |
| Test 2025 | 8.68% | 8.68% | 2.63% | 3.31 | 26 | 0.022 |
| Eval 2026H1 | 6.81% | 17.29% | 4.38% | 3.95 | 18 | 0.035 |

## Quarterly block check

| Block | Abs return | CAGR | Strict MDD | CAGR/MDD | Trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025Q1 | 1.04% | 4.27% | 1.42% | 3.01 | 6 | 0.409 |
| 2025Q2 | 2.17% | 8.99% | 2.63% | 3.42 | 9 | 0.337 |
| 2025Q3 | 0.01% | 0.04% | 0.79% | 0.05 | 2 | 0.991 |
| 2025Q4 | 5.27% | 22.62% | 2.01% | 11.23 | 9 | 0.035 |
| 2026Q1 | 6.21% | 27.70% | 4.38% | 6.32 | 14 | 0.050 |
| 2026Q2-to-Jun01 | 0.57% | 3.45% | 0.69% | 5.03 | 4 | 0.300 |

## Monthly block notes

Monthly blocks confirm this is a sparse/opportunistic sleeve, not a steady monthly income engine:

- No trades: 2025-01, 2025-08, 2025-09, 2026-01, 2026-04, 2026-06-01 partial day.
- Negative months with tiny N: 2025-02 (`-0.25%`, 1 trade), 2025-04 (`-0.61%`, 3 trades).
- Strong concentrated months: 2025-11 (`3.16%`, 4 trades, p≈0.009), 2026-02 (`3.56%`, 10 trades), 2026-03 (`2.56%`, 4 trades, p≈0.00009).

## Decision

This fixed gate is currently the best REX-TTE candidate found in this branch:

- It passes full-period test/eval `CAGR/MDD >= 3` at 0.5x validation leverage.
- Strict MDD remains below 5% in test/eval.
- Full test and eval p-values are below 0.05.

Remaining issue: trade count is sparse, especially eval `N=18`; monthly distribution has several no-trade months and a weak 2025Q3. Treat it as a small portfolio sleeve / feature-conditioned REX gate candidate, not a standalone bot yet. Next work should test integration with the portfolio allocator and live feature availability/staleness for `taker_imbalance` and `range_vol`.
