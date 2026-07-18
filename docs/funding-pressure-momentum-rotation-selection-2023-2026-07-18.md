# FPMR-1 strict 2023 one-shot selection — 2026-07-18

- Decision: **rejected_before_2024**
- 2024, 2025, and 2026 outcomes remain sealed.

| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | -15.951% | -15.961% | 69.629% | -0.229 | 45 |
| 2023 H1 | +9.550% | +20.210% | 17.349% | 1.165 | 19 |
| 2023 H2 | -25.448% | -44.175% | 68.076% | -0.649 | 25 |

| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 10 bp/side | -18.979% | -18.991% | 69.953% | -0.271 | 45 |
| Entry/exit +5m | -16.008% | -16.018% | 69.584% | -0.230 | 45 |
| Direction flip | -10.339% | -10.346% | 32.360% | -0.320 | 45 |
| Price level + rotation only | -12.575% | -12.583% | 72.414% | -0.174 | 45 |
| Funding-change only | -28.043% | -28.059% | 36.491% | -0.769 | 45 |
| Static residual level only | -25.428% | -25.443% | 73.835% | -0.345 | 45 |

- Weekly-cluster sign-flip p: `0.587971`
- Failed gates: `['2023_absolute_return_positive', '2023_cagr_to_strict_mdd_at_least_3', '2023_strict_mdd_at_most_15', '2023_h2_absolute_return_positive', 'ten_bp_stress_absolute_return_positive', 'entry_and_exit_delay_plus_5m_absolute_return_positive', 'direction_flip_cagr_lower', 'weekly_cluster_signflip_p_at_most_0_10']`
- CAGR spans the complete declared calendar, including warm-up and idle cash.
- Strict MDD uses the global/pre-entry HWM, favorable-before-adverse two-leg OHLC, exact held funding, and entry/hypothetical-liquidation/exit costs.
- Controls are diagnostics only. No sign, score, coefficient, lookback, hold, pair, or weight repair is permitted.
