# Rank7-capacity portfolio allocation update (2026-07-28)

Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

## Frozen protocol

- Gross <= 10.0; family caps: `funding_premium=2, kimchi_fx=2, oi=2, rank7=3, rex=2`.
- Non-zero weight >= 0.25; step = 0.05.
- Accounting `same_btc_low_high_v1`; protocol `e5dfde97cc32ba4f686c66e23c843b6abaf9d63e2a03fcd449c1d5b8ccf1f41a`.
- Allocation ranking uses train and 2024 only.
- Two deterministic seed pools plus exact 0.05-grid beam refinement (3 stalled rounds patience) are ranked on the shared 5-minute clock; there is no daily shortlist.
- Exact score ties prefer lower gross, then lexicographically lower sleeve weights.
- 2025 and 2026 may veto frozen rank 1, but never rerank or select rank 2+.
- All future windows have prior research exposure; result is shadow-only.

## Decision

- Frozen rank-1 weights: `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 2.0}` (gross 9.00).
- Frozen rank-1 future veto: **PASS**.
- Deployment disposition: **forward_shadow_candidate_not_live**.

| Portfolio | Train | 2024 selection | 2025 report | 2026H1 report |
|---|---:|---:|---:|---:|
| Previous live | 523.60/73.21/32.20/2.27/818 | 66.94/66.76/13.94/4.79/172 | 61.20/61.25/10.11/6.06/109 | 24.89/70.00/7.38/9.48/65 |
| Previous Gross8 live | 2274.53/158.73/36.58/4.34/861 | 180.81/180.22/16.05/11.23/203 | 148.35/148.51/12.35/12.03/133 | 69.24/251.14/15.00/16.74/108 |
| Frozen rank 1 | 2553.19/167.49/36.58/4.58/861 | 222.48/221.71/16.95/13.08/203 | 185.25/185.46/14.73/12.59/133 | 79.63/304.80/17.93/17.00/108 |

## Top pre-2025 allocation ranks

| # | Gross | Weights | Train | 2024 | 2025 report | 2026H1 report | Future veto |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 1 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 2.0}` | 2553.19/167.49/36.58/4.58/861 | 222.48/221.71/16.95/13.08/203 | 185.25/185.46/14.73/12.59/133 | 79.63/304.80/17.93/17.00/108 | PASS |
| 2 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.45, 'cand_rex_veto_7': 1.55, 'markov_transition_long': 2.0}` | 2551.75/167.45/36.57/4.58/861 | 225.86/225.07/16.74/13.45/203 | 185.47/185.68/14.73/12.61/133 | 80.08/307.26/17.95/17.12/108 | PASS |
| 3 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.35, 'cand_rex_veto_7': 1.65, 'markov_transition_long': 2.0}` | 2554.02/167.51/36.59/4.58/861 | 219.13/218.37/17.18/12.71/203 | 185.03/185.23/14.73/12.58/133 | 79.17/302.35/17.90/16.89/108 | PASS |
| 4 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.5, 'cand_rex_veto_7': 1.5, 'markov_transition_long': 2.0}` | 2549.72/167.38/36.56/4.58/861 | 229.26/228.46/16.61/13.76/203 | 185.69/185.89/14.73/12.62/133 | 80.54/309.73/17.97/17.24/108 | PASS |
| 5 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.3, 'cand_rex_veto_7': 1.7, 'markov_transition_long': 2.0}` | 2554.26/167.52/36.60/4.58/861 | 215.80/215.06/17.41/12.35/203 | 184.80/185.01/14.73/12.56/133 | 78.71/299.91/17.88/16.77/108 | PASS |
| 6 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.55, 'cand_rex_veto_7': 1.45, 'markov_transition_long': 2.0}` | 2547.08/167.30/36.55/4.58/861 | 232.69/231.87/16.47/14.07/203 | 185.90/186.11/14.73/12.64/133 | 80.99/312.21/17.99/17.35/108 | PASS |
| 7 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.25, 'cand_rex_veto_7': 1.75, 'markov_transition_long': 2.0}` | 2553.89/167.51/36.61/4.58/861 | 212.51/211.78/17.64/12.01/203 | 184.57/184.78/14.73/12.55/133 | 78.26/297.48/17.86/16.65/108 | PASS |
| 8 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.6, 'cand_rex_veto_7': 1.4, 'markov_transition_long': 2.0}` | 2543.85/167.21/36.54/4.58/861 | 236.14/235.31/16.34/14.40/203 | 186.11/186.32/14.73/12.65/133 | 81.45/314.69/18.01/17.47/108 | PASS |
| 9 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.65, 'cand_rex_veto_7': 1.35, 'markov_transition_long': 2.0}` | 2540.03/167.09/36.54/4.57/861 | 239.63/238.78/16.21/14.73/203 | 186.32/186.53/14.73/12.66/133 | 81.91/317.18/18.03/17.59/108 | PASS |
| 10 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.7, 'cand_rex_veto_7': 1.3, 'markov_transition_long': 2.0}` | 2535.61/166.96/36.53/4.57/861 | 243.14/242.27/16.08/15.07/203 | 186.52/186.73/14.73/12.68/133 | 82.36/319.68/18.05/17.71/108 | PASS |
| 11 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 1.95}` | 2465.59/164.81/36.08/4.57/861 | 217.71/216.95/16.91/12.83/203 | 183.08/183.28/14.62/12.54/133 | 79.01/301.51/17.79/16.95/108 | PASS |
| 12 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.45, 'cand_rex_veto_7': 1.55, 'markov_transition_long': 1.95}` | 2464.21/164.76/36.08/4.57/861 | 221.03/220.27/16.70/13.19/203 | 183.30/183.50/14.62/12.55/133 | 79.47/303.95/17.81/17.07/108 | PASS |
| 13 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.95, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 2.0}` | 2538.67/167.05/36.58/4.57/861 | 220.29/219.52/16.91/12.98/203 | 183.31/183.51/14.61/12.56/133 | 79.10/302.00/17.78/16.98/108 | PASS |
| 14 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.95, 'rex_taker_low_range_position': 0.45, 'cand_rex_veto_7': 1.55, 'markov_transition_long': 2.0}` | 2537.24/167.01/36.57/4.57/861 | 223.64/222.86/16.69/13.35/203 | 183.53/183.73/14.61/12.58/133 | 79.56/304.44/17.80/17.10/108 | PASS |
| 15 | 9.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.75, 'cand_rex_veto_7': 1.25, 'markov_transition_long': 2.0}` | 2530.60/166.80/36.52/4.57/861 | 246.67/245.79/15.95/15.41/203 | 186.72/186.93/14.73/12.69/133 | 82.82/322.19/18.07/17.83/108 | PASS |
| 16 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.35, 'cand_rex_veto_7': 1.65, 'markov_transition_long': 1.95}` | 2466.40/164.83/36.09/4.57/861 | 214.41/213.67/17.14/12.47/203 | 182.86/183.06/14.62/12.52/133 | 78.56/299.07/17.76/16.84/108 | PASS |
| 17 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.5, 'cand_rex_veto_7': 1.5, 'markov_transition_long': 1.95}` | 2462.25/164.70/36.07/4.57/861 | 224.39/223.61/16.57/13.50/203 | 183.51/183.71/14.62/12.57/133 | 79.92/306.39/17.83/17.19/108 | PASS |
| 18 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.95, 'rex_taker_low_range_position': 0.35, 'cand_rex_veto_7': 1.65, 'markov_transition_long': 2.0}` | 2539.50/167.07/36.59/4.57/861 | 216.96/216.21/17.14/12.62/203 | 183.09/183.29/14.61/12.54/133 | 78.65/299.56/17.76/16.87/108 | PASS |
| 19 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.95, 'rex_taker_low_range_position': 0.5, 'cand_rex_veto_7': 1.5, 'markov_transition_long': 2.0}` | 2535.21/166.94/36.56/4.57/861 | 227.02/226.23/16.56/13.66/203 | 183.75/183.95/14.61/12.59/133 | 80.01/306.89/17.82/17.22/108 | PASS |
| 20 | 8.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 3.0, 'rex_taker_low_range_position': 0.3, 'cand_rex_veto_7': 1.7, 'markov_transition_long': 1.95}` | 2466.62/164.84/36.11/4.57/861 | 211.13/210.41/17.37/12.11/203 | 182.63/182.83/14.62/12.51/133 | 78.10/296.65/17.74/16.72/108 | PASS |

## Candidate and accounting notes

- The old live row is reproduced exactly under its legacy MDD engine before comparison.
- Selection uses the corrected same-bar upper-before-lower strict MDD clock.
- Every sleeve is marked at the same underlying BTC low/high price points; upper is applied before lower on each bar.
- The reported row is the best found in a deterministic seeded candidate search, not a proof of the global discrete-grid optimum.
- Rank7 and Fresh Kimchi retain their canonical execution/funding schedules.
- A Rank7 cap above the common family cap is allowed only when a pre-2025-selected leverage battery proves the exact multiplier; no duplicate Rank7 sleeve is created.
- Advanced-state representatives selected by inspecting future passers were excluded.
- This experiment does not overwrite the current live config.
- Rank7 finished exactly at its authorized 3.0 multiplier (1.5x effective leverage). This is a capacity-bound result, not evidence that leverage above the preregistered cap is optimal or safe.
