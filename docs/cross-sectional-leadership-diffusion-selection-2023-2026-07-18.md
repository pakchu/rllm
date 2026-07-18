# CLD-72 frozen 2023 selection outcome — 2026-07-18

Decision: **rejected_before_2024**

| Window / control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | -11.523% | -11.523% | 13.420% | -0.859 | 106 |
| H1 | -6.012% | -11.753% | 8.703% | -1.351 | 37 |
| H2 | -5.864% | -11.297% | 7.534% | -1.499 | 69 |
| Q1 | -0.733% | -2.941% | 3.265% | -0.901 | 15 |
| Q2 | -5.318% | -19.681% | 6.616% | -2.975 | 22 |
| Q3 | -3.462% | -13.045% | 4.814% | -2.710 | 29 |
| Q4 | -2.488% | -9.513% | 3.943% | -2.413 | 40 |
| Long only | -1.938% | -1.938% | 4.647% | -0.417 | 41 |
| Short only | -9.775% | -9.775% | 10.663% | -0.917 | 65 |
| 10bp stress | -15.204% | -15.204% | 16.756% | -0.907 | 106 |
| +5m delay | -10.078% | -10.078% | 12.599% | -0.800 | 106 |
| Direction flip | -0.710% | -0.710% | 5.847% | -0.121 | 106 |
| Control: static_alt_breadth | -40.715% | -40.715% | 43.460% | -0.937 | 599 |
| Control: transition_without_flow | -17.917% | -17.917% | 19.382% | -0.924 | 159 |
| Control: transition_without_btc_lag | -12.433% | -12.433% | 14.249% | -0.873 | 136 |
| Control: btc_momentum_at_primary_opportunities | -5.982% | -5.982% | 7.801% | -0.767 | 106 |

- Weekly-cluster sign-flip p: `0.991970`
- Failed gates: `['annual_absolute_return_positive', 'annual_cagr_to_strict_mdd_at_least_3', 'both_halves_absolute_return_positive', 'at_least_three_quarters_absolute_return_positive', 'long_only_absolute_return_positive', 'short_only_absolute_return_positive', 'ten_bp_stress_absolute_return_positive', 'delay_plus_5m_absolute_return_positive', 'direction_flip_cagr_lower', 'alt_direction_beats_btc_momentum_ratio', 'weekly_cluster_signflip_p_at_most_0_10']`
- CAGR uses the full declared calendar, including warm-up and idle cash.
- Strict MDD uses global/pre-entry HWM, held OHLC, realized funding, and entry/liquidation/exit costs.
- Controls are diagnostics only and cannot replace or repair the frozen primary clock.
