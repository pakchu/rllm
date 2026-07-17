# CRRC-72 frozen 2023 selection outcome — 2026-07-17

Decision: **rejected_before_2024**

| Window / control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | -1.018% | -1.018% | 9.498% | -0.107 | 156 |
| Q1 | +3.850% | +16.555% | 4.627% | 3.578 | 32 |
| Q2 | -1.092% | -4.309% | 4.101% | -1.051 | 25 |
| Q3 | -1.747% | -6.753% | 5.905% | -1.144 | 47 |
| Q4 | -1.921% | -7.407% | 6.147% | -1.205 | 52 |
| Long only | +4.046% | +4.046% | 4.926% | 0.821 | 91 |
| Short only | -4.867% | -4.867% | 8.238% | -0.591 | 65 |
| 10bp stress | -7.011% | -7.011% | 14.021% | -0.500 | 156 |
| +5m delay | -0.264% | -0.264% | 8.355% | -0.032 | 156 |
| Direction flip | -16.770% | -16.770% | 17.702% | -0.947 | 156 |
| Control: um_only | -27.549% | -27.549% | 33.023% | -0.834 | 714 |
| Control: cm_only | -24.154% | -24.154% | 31.672% | -0.763 | 681 |
| Control: without_credibility | -41.274% | -41.274% | 47.625% | -0.867 | 637 |
| Control: inner_add_only | -41.136% | -41.136% | 46.456% | -0.885 | 822 |
| Control: outer_withdraw_only | -18.064% | -18.064% | 26.273% | -0.688 | 1067 |

- Monthly-cluster sign-flip p: `0.571421`
- Failed gates: `['annual_absolute_return_positive', 'annual_cagr_to_strict_mdd_at_least_3', 'every_quarter_absolute_return_positive', 'short_only_absolute_return_positive', 'ten_bp_stress_absolute_return_positive', 'delay_plus_5m_absolute_return_positive', 'monthly_cluster_signflip_p_at_most_0_10']`
- CAGR spans the full declared calendar, including warm-up and idle cash.
- Strict MDD uses the global/pre-entry HWM, favorable-before-adverse held OHLC, exact held funding, and entry/hypothetical-liquidation/exit costs.
- Mechanism controls are diagnostics and cannot rerank or replace the frozen primary singleton.
