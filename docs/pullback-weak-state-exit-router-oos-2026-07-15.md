# Pullback weak-state exit router — frozen OOS replay

## Verdict: **REJECTED_OOS**

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| test_2024 | +17.91% | +17.87% | 5.29% | 3.38 | 20 |
| eval_2025 | +4.03% | +4.03% | 5.11% | 0.79 | 12 |
| holdout_2026 | +8.48% | +21.61% | 5.11% | 4.23 | 21 |
| oos_2024_2026 | +33.07% | +12.55% | 7.04% | 1.78 | 53 |

Manifest `9045f3fc1f8a92ea5e933e222817114633f58c54710e25e0fe396c8d47f6689c` was reconstructed on the pre-2024 prefix before the OOS-open sidecar was written.
No OOS threshold, action, hold, or sizing was retuned.
