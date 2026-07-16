# DCRM-1 strict 2023 one-shot selection — 2026-07-17

- Decision: **rejected_before_2024**
- 2024, 2025, and 2026 outcomes remain sealed.

| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | +2.225% | +2.227% | 24.619% | 0.090 | 38 |
| 2023 H1 | -9.638% | -18.495% | 15.945% | -1.160 | 12 |
| 2023 H2 | +15.689% | +33.549% | 19.848% | 1.690 | 25 |

| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 10 bp/side | -0.009% | -0.009% | 25.747% | -0.000 | 38 |
| Entry/exit +5m | +2.287% | +2.288% | 24.562% | 0.093 | 38 |
| Direction flip | -12.662% | -12.670% | 31.932% | -0.397 | 38 |
| Full gross | -33.381% | -33.399% | 71.997% | -0.464 | 38 |
| Inverted dispersion scale | -40.044% | -40.065% | 71.279% | -0.562 | 38 |

- Weekly-cluster sign-flip p: `0.461527`
- Failed gates: `['2023_cagr_to_strict_mdd_at_least_2', '2023_strict_mdd_at_most_15', '2023_h1_absolute_return_positive', 'ten_bp_stress_absolute_return_positive', 'weekly_cluster_signflip_p_at_most_0_10']`
- CAGR spans the complete declared calendar, including warm-up and reduced-gross weeks.
- Strict MDD uses the global/pre-entry HWM, favorable-before-adverse two-leg OHLC, exact held funding, and entry/hypothetical-liquidation/exit costs.
- Funding credits cannot create an intratrade peak. No sign, lookback, hold, scale, pair, or beta repair is permitted.
