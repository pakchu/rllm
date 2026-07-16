# AFCH v1 2023–2024 selection — 2026-07-17

> Only 2023–2024 execution outcomes were opened. Calendar 2025 and 2026 remain sealed.

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Sleeves | Funding cash | Costs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 fit | +37.631% | +37.661% | 20.023% | 1.881 | 42 | +7.306% | 1.357% |
| 2024 test | -62.435% | -62.359% | 77.082% | -0.809 | 45 | +9.393% | 1.949% |
| 2023–2024 | -43.893% | -25.081% | 77.082% | -0.325 | 90 | +22.594% | 4.401% |
| 2023 H1 | +1.360% | +2.763% | 20.023% | 0.138 | 18 | +5.213% | 0.551% |
| 2023 H2 | +42.966% | +103.303% | 18.708% | 5.522 | 20 | +1.518% | 0.711% |
| 2024 H1 | +47.118% | +117.010% | 16.987% | 6.888 | 22 | +6.979% | 0.878% |
| 2024 H2 | -72.516% | -92.299% | 76.725% | -1.203 | 19 | +1.542% | 0.655% |

## Decision

- status: **rejected_before_2025**
- failed gates: `['each_year_absolute_return_positive', 'each_year_cagr_to_strict_mdd_at_least_1_5', 'combined_cagr_to_strict_mdd_at_least_3', 'combined_strict_mdd_at_most_15', 'ten_bp_cost_stress_absolute_return_positive', 'weekly_cluster_signflip_p_at_most_0_10']`
- weekly cluster sign-flip p: `0.601720`
- Funding cash and transaction cost are reported as cash percentages of initial equity; price PnL is separated in the JSON artifact.
- Recorded settlement `mark_price` is used when present; missing 2023 marks use the separately frozen last-completed 5m mark-price close, with exact/proxy application counts and cash split in JSON.
- CAGR includes every idle day in the declared calendar.
- strict MDD uses aggregate net-symbol favorable-before-adverse OHLC, global HWM, active funding-credit exclusion, and entry/exit/hypothetical liquidation costs.
- Diagnostics cannot repair a failed AFCH01 policy.
