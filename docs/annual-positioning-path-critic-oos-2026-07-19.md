# Annual positioning path critic — frozen OOS replay

## Decision

**REJECT**

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| test_2024 | 31.66% | 31.59% | 16.89% | 1.87 | 146 | 98/48 |
| eval_2025 | -16.59% | -16.60% | 17.19% | -0.97 | 172 | 144/28 |
| holdout_2026 | -7.93% | -18.15% | 16.67% | -1.09 | 79 | 71/8 |
| oos_2024_2026 | 1.12% | 0.46% | 29.26% | 0.02 | 397 | 313/84 |

## Gates

- `each_oos_window_absolute_return_positive`: FAIL
- `each_oos_window_ratio_at_least_3`: FAIL
- `each_oos_window_strict_mdd_at_most_15`: FAIL
- `minimum_trade_support`: PASS
- `stress_return_positive_each_oos_window`: FAIL
- `combined_weekly_signflip_p_at_most_0_10`: FAIL
- `combined_primary_beats_direction_flip`: PASS

## Caveat

2024-2026 BTC history was globally seen by unrelated repository research; this is an exact-policy mechanically frozen replay, not a pristine market clean room.

## Failure diagnosis

- The annual critic generalized for one year (`2024 +31.66%`) but then lost
  `-16.59%` in 2025 and `-7.93%` in 2026. The combined weekly-cluster
  sign-flip p-value was `0.4770`, so the full OOS clock has no statistical edge.
- This is not primarily a fee problem. Mean **gross** BTC move per trade was
  already negative in 2025 (`-6.94 bp`) and 2026 (`-7.01 bp`). Raising cost to
  10 bp/notional-side only made the failure larger.
- The annual model became strongly long-biased: long shares were `67.1%`,
  `83.7%`, and `89.9%` in 2024, 2025, and 2026. An exact direction flip also
  lost (`-42.33%` combined), so this family cannot be repaired by reversing the
  prediction after seeing OOS.
- Strict MDD exceeded the 15% ceiling in every OOS subwindow and reached
  `29.26%` on the combined path. The pre-2024 path-utility ranking therefore
  did not remain calibrated under later regime and feature-distribution shift.

This exact annual HGB family is retired. No threshold, side balance, cadence,
hold, or direction is retuned on the consumed 2024-2026 windows.
