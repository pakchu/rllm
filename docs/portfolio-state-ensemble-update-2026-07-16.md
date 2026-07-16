# State-ensemble portfolio allocation update (2026-07-16)

Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

## Selection contract

- Portfolio ranking uses train and 2024 only.
- 2025 and 2026 performance metrics are evaluated after rank order is frozen; they may veto rank 1 but never select rank 2+.
- Each new state sleeve is a predeclared strict majority: >= 6 of the pre-evaluation Top-10 family.
- No exact Kalman/BOCPD/Semi-Markov representative was chosen from later-window passers.
- Gross <= 10.0; family gross <= 2.0; non-zero >= 0.25; step 0.05.
- All state sleeves share the funding/premium family cap because they gate the same base setup.

## Decision

- Frozen rank-1 weights: `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` (gross 6.15).
- Frozen rank-1 future veto: **FAIL**.
- Replaces previous added-alpha candidate: **NO**.
- Disposition: **retain_previous_added_alpha_shadow_candidate**.

| Portfolio | Train | 2024 selection | 2025 report | 2026H1 report |
|---|---:|---:|---:|---:|
| Previous added-alpha best | 2274.53/158.73/36.58/4.34/861 | 180.81/180.22/16.05/11.23/203 | 148.35/148.51/12.35/12.03/133 | 69.24/251.14/15.00/16.74/108 |
| State-ensemble frozen rank 1 | 2271.88/158.64/24.84/6.39/971 | 88.94/88.69/13.46/6.59/221 | 48.12/48.16/22.00/2.19/144 | 47.47/152.76/12.60/12.12/125 |

## Top frozen pre-2025 ranks

| # | Gross | Weights | Train | 2024 | 2025 report | 2026H1 report | Veto |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 1 | 6.15 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2271.88/158.64/24.84/6.39/971 | 88.94/88.69/13.46/6.59/221 | 48.12/48.16/22.00/2.19/144 | 47.47/152.76/12.60/12.12/125 | FAIL |
| 2 | 6.20 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2327.99/160.46/25.13/6.38/971 | 88.11/87.87/13.73/6.40/221 | 48.34/48.38/22.13/2.19/144 | 47.60/153.32/12.60/12.17/125 | FAIL |
| 3 | 6.30 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.6, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.4}` | 2403.00/162.85/25.52/6.38/971 | 90.18/89.93/13.72/6.56/221 | 49.51/49.55/22.10/2.24/144 | 48.77/158.11/12.68/12.47/125 | FAIL |
| 4 | 6.25 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.4}` | 2350.10/161.17/25.29/6.37/971 | 89.08/88.83/13.76/6.46/221 | 49.17/49.22/22.10/2.23/144 | 48.26/156.02/12.66/12.33/125 | FAIL |
| 5 | 6.05 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.5, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.3}` | 2200.14/156.27/24.52/6.37/971 | 86.87/86.63/13.48/6.43/221 | 46.97/47.01/22.02/2.13/144 | 46.31/148.06/12.52/11.83/125 | FAIL |
| 6 | 6.10 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 1.95, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2258.35/158.20/24.84/6.37/971 | 87.62/87.38/13.34/6.55/221 | 47.07/47.11/21.88/2.15/144 | 47.00/150.85/12.45/12.12/125 | FAIL |
| 7 | 6.05 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.2, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.3}` | 2195.62/156.12/24.52/6.37/971 | 88.80/88.55/13.16/6.73/221 | 47.08/47.12/21.89/2.15/144 | 46.68/149.54/12.54/11.92/125 | FAIL |
| 8 | 6.25 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.6, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.4}` | 2345.30/161.02/25.29/6.37/971 | 91.03/90.77/13.44/6.75/221 | 49.29/49.33/21.97/2.24/144 | 48.63/157.55/12.68/12.42/125 | FAIL |
| 9 | 6.25 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 1.95, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.6, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.4}` | 2388.72/162.40/25.52/6.36/971 | 88.86/88.61/13.60/6.51/221 | 48.44/48.48/21.98/2.21/144 | 48.29/156.16/12.53/12.46/125 | FAIL |
| 10 | 6.10 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.3}` | 2250.33/157.93/24.83/6.36/971 | 87.97/87.73/13.44/6.53/221 | 47.30/47.34/22.02/2.15/144 | 46.81/150.09/12.54/11.97/125 | FAIL |
| 11 | 6.15 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.2, 'rex_taker_low_range_position': 0.6, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2266.96/158.48/24.91/6.36/971 | 90.88/90.63/13.14/6.90/221 | 48.23/48.27/21.87/2.21/144 | 47.83/154.27/12.62/12.22/125 | FAIL |
| 12 | 6.20 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.6, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2323.22/160.31/25.22/6.36/971 | 90.05/89.80/13.42/6.69/221 | 48.45/48.49/22.00/2.20/144 | 47.97/154.83/12.62/12.27/125 | FAIL |
| 13 | 5.90 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.2, 'rex_taker_low_range_position': 0.45, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.25}` | 2077.89/152.10/23.93/6.36/971 | 85.63/85.40/13.22/6.46/221 | 45.60/45.64/21.91/2.08/144 | 45.03/142.90/12.44/11.49/125 | FAIL |
| 14 | 6.20 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 1.95, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.4}` | 2336.12/160.72/25.29/6.36/971 | 87.76/87.51/13.64/6.41/221 | 48.11/48.15/21.98/2.19/144 | 47.79/154.08/12.51/12.32/125 | FAIL |
| 15 | 6.15 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 1.95, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2314.13/160.02/25.13/6.37/971 | 86.80/86.56/13.62/6.36/221 | 47.29/47.33/22.01/2.15/144 | 47.13/151.40/12.45/12.16/125 | FAIL |
| 16 | 6.00 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 1.95, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.5, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.3}` | 2187.01/155.83/24.52/6.35/971 | 85.57/85.33/13.36/6.39/221 | 45.92/45.96/21.90/2.10/144 | 45.85/146.18/12.37/11.82/125 | FAIL |
| 17 | 6.25 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.3, 'rex_taker_low_range_position': 0.6, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2380.39/162.14/25.52/6.35/971 | 89.21/88.96/13.69/6.50/221 | 48.67/48.72/22.13/2.20/144 | 48.11/155.39/12.62/12.31/125 | FAIL |
| 18 | 6.05 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 1.9, 'cand_rex_veto_7': 1.25, 'rex_taker_low_range_position': 0.55, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.35}` | 2244.87/157.75/24.84/6.35/971 | 86.31/86.07/13.23/6.51/221 | 46.02/46.06/21.75/2.12/144 | 46.53/148.95/12.30/12.11/125 | FAIL |
| 19 | 6.25 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.2, 'rex_taker_low_range_position': 0.65, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.4}` | 2339.96/160.85/25.33/6.35/971 | 92.99/92.73/13.13/7.06/221 | 49.39/49.44/21.85/2.26/144 | 49.00/159.08/12.70/12.52/125 | FAIL |
| 20 | 5.90 | `{'bocpd_top10_strict_majority_long': 1.75, 'frozen_annual_rank7': 2.0, 'cand_rex_veto_7': 1.15, 'rex_taker_low_range_position': 0.5, 'markov_transition_long': 0.25, 'fresh_kimchi_fx': 0.25}` | 2073.60/151.95/23.93/6.35/971 | 87.54/87.30/12.91/6.76/221 | 45.71/45.75/21.79/2.10/144 | 45.39/144.35/12.46/11.58/125 | FAIL |

## Interpretation

- A future-veto failure retains the previous shadow candidate; lower-ranked future passers are not promoted.
- The search is deterministic seeded sampling plus exact 0.05-grid refinement, not a proof of the global combinatorial optimum.
- Reported later windows have prior research exposure, so even a pass remains forward-shadow only.
- The current live config is not modified by this experiment.
