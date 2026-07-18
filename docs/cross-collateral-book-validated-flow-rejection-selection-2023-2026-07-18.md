# CBFR-72 frozen 2023 selection outcome — 2026-07-18

Decision: **rejected_before_2024**

| Window / control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | -15.684% | -15.684% | 16.776% | -0.935 | 144 |
| H1 | -9.003% | -17.325% | 12.047% | -1.438 | 66 |
| H2 | -7.343% | -14.039% | 7.841% | -1.791 | 78 |
| Q1 | -1.234% | -4.910% | 2.631% | -1.866 | 22 |
| Q2 | -7.866% | -28.007% | 10.683% | -2.622 | 44 |
| Q3 | -1.972% | -7.599% | 3.349% | -2.269 | 43 |
| Q4 | -5.478% | -20.031% | 6.458% | -3.102 | 35 |
| Long only | -6.572% | -6.572% | 7.629% | -0.861 | 74 |
| Short only | -9.753% | -9.753% | 11.954% | -0.816 | 70 |
| 10bp stress | -20.412% | -20.412% | 21.285% | -0.959 | 144 |
| +5m delay | -14.993% | -14.993% | 16.192% | -0.926 | 144 |
| Direction flip | -0.536% | -0.536% | 9.040% | -0.059 | 144 |
| Control: without_book_confirmation | -29.601% | -29.601% | 30.075% | -0.984 | 448 |
| Control: um_only_confirmation | -14.958% | -14.958% | 16.158% | -0.926 | 154 |
| Control: cm_only_confirmation | -29.363% | -29.363% | 30.202% | -0.972 | 258 |

- Weekly-cluster sign-flip p: `0.993500`
- Failed gates: `['annual_absolute_return_positive', 'annual_cagr_to_strict_mdd_at_least_3', 'annual_strict_mdd_at_most_15_pct', 'both_halves_absolute_return_positive', 'every_quarter_absolute_return_positive', 'long_only_absolute_return_positive', 'short_only_absolute_return_positive', 'ten_bp_stress_absolute_return_positive', 'delay_plus_5m_absolute_return_positive', 'direction_flip_cagr_lower', 'book_confirmation_improves_mean_net_bps', 'weekly_cluster_signflip_p_at_most_0_10']`
- CAGR spans the full declared calendar, including every idle interval.
- Strict MDD uses global/pre-entry HWM, entry cost, realized funding, favorable-before-adverse held OHLC, hypothetical liquidation cost, and exit cost.
- Mechanism controls are diagnostics and cannot replace or rerank the frozen primary clock.
- Failure keeps 2024 onward sealed and forbids threshold, sign, hold, or feature repair.
