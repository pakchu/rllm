# Frozen rank-7 annual vs monthly refit cadence before 2025

The feature graph was physically truncated at `2025-01-01`. Both cadences use the same frozen rank-7 learner, policy, five seeds, exact labels, and execution; only the expanding-refit cutoff changes.

- Parent rank-family manifest: `c6e7d78a328118456eacf70bc42cb12a48f33e26d13edbe21f2edb3aedea4f8e`
- Cadence manifest: `627441e5a7a3bd070e136e771f7dcc93cea6162565c0dd2226c2140c5c836f21`
- Pre-2025 selected cadence: **annual**
- Annual folds: `2`; monthly folds: `24`

## Results

| Cadence | Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean net | Win rate | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| annual | test_2023 | 12.8641% | 12.8735% | 3.1173% | 4.1297 | 19 | 64.19 bps | 78.95% | PASS |
| annual | validation_2024 | 16.3961% | 16.3599% | 3.4631% | 4.7241 | 22 | 69.75 bps | 81.82% | PASS |
| annual | selection_2023_2024 | 31.3695% | 14.6058% | 3.4631% | 4.2176 | 41 | 67.17 bps | 80.49% | PASS |
| monthly | test_2023 | 9.9066% | 9.9137% | 3.1173% | 3.1802 | 20 | 47.77 bps | 70.00% | FAIL |
| monthly | validation_2024 | 12.6059% | 12.5785% | 4.4743% | 2.8113 | 26 | 46.31 bps | 69.23% | FAIL |
| monthly | selection_2023_2024 | 23.7613% | 11.2399% | 4.4743% | 2.5121 | 46 | 46.95 bps | 69.57% | FAIL |

## Selection rule

Cadence is selected lexicographically using only 2023/2024: pass flag, worst yearly CAGR/MDD, combined CAGR/MDD, combined trades, combined absolute return. Exact ties prefer annual refit as the simpler schedule.

## Integrity

- Every monthly fit purges targets whose source-owned exits reach that month start.
- The annual result must exactly reproduce frozen rank-7 2023/2024 schedules.
- No 2025+ source row is opened by this program.
- This artifact freezes cadence choice before the separate future evaluator is run.
