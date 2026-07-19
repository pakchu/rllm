# TBASR-24 train result — REJECT

This report was produced by the committed, write-once evaluator contract.
No threshold, hold, leverage, cost, or semantic prompt was selected from this outcome.

| run | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long | short |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | -6.6063% | -4.4453% | 40.6370% | -0.1094 | 358 | 209 | 149 |
| delayed stress | -18.1498% | -12.4750% | 41.6369% | -0.2996 | 358 | 209 | 149 |

- weekly clustered sign-flip p: `0.9088545572721364`;
- weekly clusters: `77`;
- mean gross underlying move: `11.4442 bp`;
- minimum mechanism-control ratio margin: `0.27770748300453696`;
- failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15pct', 'weekly_cluster_signflip_p_at_most_10pct', 'mean_gross_underlying_at_least_20bp', 'each_half_year_absolute_return_positive', 'stress_absolute_return_positive', 'stress_cagr_to_strict_mdd_at_least_2_5']`.

The candidate is rejected at this frozen stage. Later windows remain sealed,
and this candidate may not be repaired on the observed outcome.
