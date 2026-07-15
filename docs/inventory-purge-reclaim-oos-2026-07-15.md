# Inventory purge/reclaim alpha — frozen OOS replay (2026-07-15)

## Verdict

**Rejected.** The pre-2024 champion lost money in 2024, 2025, and 2026. The positioning gate therefore captured development-period selection noise rather than a transferable interaction.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| test_2024 | -10.42% | -10.40% | 12.70% | -0.82 | 54 | 38/16 |
| eval_2025 | -1.59% | -1.59% | 4.46% | -0.36 | 39 | 32/7 |
| holdout_2026 | -3.95% | -9.24% | 5.94% | -1.56 | 20 | 14/6 |
| oos_2024_2026 | -15.18% | -6.59% | 16.68% | -0.39 | 114 | 85/29 |

Frozen 2024/2025 live-grade pass: **False**.

The negative result is selection evidence: 3,760 base variants followed by 1,248 context variants produced one pre-2024 qualifier, while its short gate had only seven 2023 shorts. This family must not be retuned against the opened OOS windows.
