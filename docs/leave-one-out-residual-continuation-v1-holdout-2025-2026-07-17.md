# LORC v1 calendar-2025 holdout — 2026-07-17

> This execution opened calendar 2025 only. Calendar 2026 remains sealed.

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2025 | -34.533% | -34.552% | 40.965% | -0.843 | 99 |
| 2025 H1 | -12.438% | -23.511% | 21.372% | -1.100 | 34 |
| 2025 H2 | -25.234% | -43.856% | 27.679% | -1.584 | 65 |
| 2025, 10 bp | -39.547% | -39.568% | 44.693% | -0.885 | 99 |
| 2025, +5m entry | -34.488% | -34.507% | 40.595% | -0.850 | 99 |

## Decision

- status: **rejected_before_2026**
- weekly cluster sign-flip p: `0.992900`
- failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'strict_mdd_at_most_15', 'h1_absolute_return_positive', 'h2_absolute_return_positive', 'ten_bp_cost_stress_absolute_return_positive', 'weekly_cluster_signflip_p_at_most_0_10', 'entry_delay_plus_5m_absolute_return_positive']`
- CAGR uses the full declared calendar, including idle periods.
- strict MDD includes global/pre-entry HWM, two-leg favorable-before-adverse OHLC, entry/hypothetical liquidation/exit costs, and exact funding ordering.
- Diagnostics cannot repair a failed single-policy holdout.

## Failure diagnosis

- The research edge reversed completely: mean trade fell from `+50.38 bp` in 2023–2024 to `-41.67 bp` in untouched 2025.
- Failure is not isolated to one half: H1 absolute return was `-12.438%` and H2 was `-25.234%`.
- It is not an execution-timing artifact: +5m entry delay remained `-34.488%`, while 10 bp/notional/side stress fell to `-39.547%`.
- The weekly-cluster sign-flip p-value is `0.992900`, rejecting the interpretation that 2025 merely sampled an unlucky positive process.
- Flipping back to mean reversion produced only `+17.854%` absolute return, `17.867%` CAGR, `14.915%` strict MDD, ratio `1.198`; a seven-day clock shift and monthly pair permutation were also mildly positive. This pattern indicates unstable sign/regime dependence rather than a robust cross-sectional residual alpha.

LORC and its parent LORE are retired. Their 2025 diagnostics cannot be turned into another threshold/sign repair, and calendar 2026 remains unopened for both families.
