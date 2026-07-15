# Forced-deleveraging pullback OOS replay

The rule frozen in commit `14a7d02` was replayed once on 2024-2026 data. The
OOS seal was written before any source file was opened, and the frozen source
prefix, execution configuration, thresholds, activation, and selection
schedules were hash-checked. No OOS retuning was performed.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| test_2024 | -6.17% | -6.15% | 14.38% | -0.43 | 18 | 13/5 |
| eval_2025 | 5.60% | 5.61% | 5.85% | 0.96 | 23 | 17/6 |
| holdout_2026 | -5.91% | -13.62% | 8.39% | -1.62 | 7 | 4/3 |
| oos_2024_2026 | -6.77% | -2.86% | 14.38% | -0.20 | 48 | 34/14 |

## Decision

Reject this exact static usage. The pre-2024 fit/2023 improvement did not
generalize: both 2024 and 2026 were negative, and the complete OOS period lost
money after 6bp/notional/side, realized funding, next-open execution, and
strict intratrade MDD. Do not tune the range, kimchi, stop, or hold thresholds
on these opened OOS years. A retry requires a materially different mechanism
and a newly frozen forward sample.
