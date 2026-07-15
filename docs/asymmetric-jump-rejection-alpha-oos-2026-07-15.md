# Asymmetric jump/rejection OOS replay

The rule frozen in commit `d01f264` was replayed once on 2024-2026 data. The
OOS seal was written before any source file was opened. Frozen source hashes,
the implementation hash, execution configuration, thresholds, activation,
selection schedules, and the full-run pre-2024 feature prefix were checked.
No OOS retuning was performed.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| test_2024 | 6.91% | 6.90% | 6.88% | 1.00 | 84 | 44/40 |
| eval_2025 | 0.72% | 0.72% | 9.16% | 0.08 | 87 | 43/44 |
| holdout_2026 | 3.60% | 8.88% | 7.07% | 1.26 | 33 | 14/19 |
| oos_2024_2026 | 11.56% | 4.63% | 9.16% | 0.51 | 204 | 101/103 |

## Decision

Reject this exact static strategy as a target alpha. It remained positive and
well balanced across 204 OOS trades, but the complete OOS CAGR/strict-MDD
ratio was only 0.51 and 2025 was effectively flat. The large pre-2024 barrier
improvement therefore did not generalize. Do not retune its 3% stop, 4% take,
24-hour hold, or witness thresholds on these opened years. The next candidate
must use a materially different interaction and a newly sealed forward test.
