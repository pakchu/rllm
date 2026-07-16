# PFCR-2 strict 2023–2024 one-shot selection — 2026-07-17

- Decision: **rejected_before_2025_no_outcome_repair**
- 2025 and 2026 remain sealed.

| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | -11.540% | -11.547% | 13.553% | -0.852 | 38 |
| 2024 | +0.591% | +0.590% | 6.585% | 0.090 | 44 |
| 2023–2024 | -11.017% | -5.665% | 13.831% | -0.410 | 82 |

| Control, 2023–2024 | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 10 bp/side | -16.681% | -8.715% | 18.733% | -0.465 | 82 |
| Entry/exit +5m | -9.442% | -4.835% | 11.870% | -0.407 | 82 |
| Direction flip | -8.214% | -4.192% | 16.385% | -0.256 | 82 |
| Fake settlement +4h | -12.770% | -6.599% | 15.400% | -0.428 | 82 |

- Weekly-cluster sign-flip p: `0.946703`
- Failed gates: `['2023_absolute_return_positive', '2023_cagr_to_strict_mdd_at_least_1_5', '2024_cagr_to_strict_mdd_at_least_1_5', 'combined_cagr_to_strict_mdd_at_least_3', 'ten_bp_stress_absolute_return_positive', 'entry_delay_plus_5m_absolute_return_positive', 'direction_flip_cagr_lower', 'weekly_cluster_signflip_p_at_most_0_10']`
- CAGR includes the complete declared calendar, including idle time.
- Strict MDD uses global/pre-entry HWM, favorable-before-adverse two-leg OHLC, entry/hypothetical-liquidation/exit costs, and exact held-interval funding.
- No threshold, sign, cooldown, hold, pair, or beta repair is permitted after this opening.
