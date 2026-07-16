# Added-alpha portfolio allocation update (2026-07-16)

Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

## Frozen protocol

- Gross <= 10.0; family gross <= 2.0.
- Non-zero weight >= 0.25; step = 0.05.
- Accounting `same_btc_low_high_v1`; protocol `f07c1f2990fe8b59fa306eb72224ac2d2fc8f9d44d8d771d6e66e81775395022`.
- Allocation ranking uses train and 2024 only.
- Two deterministic seed pools plus exact 0.05-grid beam refinement (3 stalled rounds patience) are ranked on the shared 5-minute clock; there is no daily shortlist.
- Exact score ties prefer lower gross, then lexicographically lower sleeve weights.
- 2025 and 2026 may veto frozen rank 1, but never rerank or select rank 2+.
- All future windows have prior research exposure; result is shadow-only.

## Decision

- Frozen rank-1 weights: `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 2.0}` (gross 8.00).
- Frozen rank-1 future veto: **PASS**.
- Deployment disposition: **forward_shadow_candidate_not_live**.

| Portfolio | Train | 2024 selection | 2025 report | 2026H1 report |
|---|---:|---:|---:|---:|
| Previous live | 523.60/73.21/32.20/2.27/818 | 66.94/66.76/13.94/4.79/172 | 61.20/61.25/10.11/6.06/109 | 24.89/70.00/7.38/9.48/65 |
| Frozen rank 1 | 2274.53/158.73/36.58/4.34/861 | 180.81/180.22/16.05/11.23/203 | 148.35/148.51/12.35/12.03/133 | 69.24/251.14/15.00/16.74/108 |

## Top pre-2025 allocation ranks

| # | Gross | Weights | Train | 2024 | 2025 report | 2026H1 report | Future veto |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 1 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 2.0}` | 2274.53/158.73/36.58/4.34/861 | 180.81/180.22/16.05/11.23/203 | 148.35/148.51/12.35/12.03/133 | 69.24/251.14/15.00/16.74/108 | PASS |
| 2 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.45, 'cand_rex_veto_7': 1.55, 'markov_transition_long': 2.0}` | 2273.21/158.68/36.57/4.34/861 | 183.76/183.15/15.84/11.56/203 | 148.54/148.70/12.35/12.04/133 | 69.67/253.27/15.02/16.86/108 | PASS |
| 3 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.35, 'cand_rex_veto_7': 1.65, 'markov_transition_long': 2.0}` | 2275.32/158.75/36.59/4.34/861 | 177.90/177.32/16.29/10.89/203 | 148.16/148.31/12.35/12.01/133 | 68.81/249.03/14.98/16.63/108 | PASS |
| 4 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.5, 'cand_rex_veto_7': 1.5, 'markov_transition_long': 2.0}` | 2271.35/158.62/36.56/4.34/861 | 186.72/186.10/15.71/11.85/203 | 148.73/148.89/12.35/12.06/133 | 70.10/255.40/15.04/16.98/108 | PASS |
| 5 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.3, 'cand_rex_veto_7': 1.7, 'markov_transition_long': 2.0}` | 2275.57/158.76/36.60/4.34/861 | 175.00/174.43/16.52/10.56/203 | 147.96/148.11/12.35/12.00/133 | 68.38/246.92/14.96/16.51/108 | PASS |
| 6 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.55, 'cand_rex_veto_7': 1.45, 'markov_transition_long': 2.0}` | 2268.95/158.55/36.55/4.34/861 | 189.71/189.08/15.57/12.14/203 | 148.92/149.07/12.35/12.07/133 | 70.52/257.54/15.07/17.09/108 | PASS |
| 7 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.25, 'cand_rex_veto_7': 1.75, 'markov_transition_long': 2.0}` | 2275.28/158.75/36.61/4.34/861 | 172.13/171.57/16.75/10.24/203 | 147.76/147.91/12.47/11.87/133 | 67.95/244.81/14.93/16.39/108 | PASS |
| 8 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.6, 'cand_rex_veto_7': 1.4, 'markov_transition_long': 2.0}` | 2266.02/158.45/36.54/4.34/861 | 192.72/192.07/15.44/12.44/203 | 149.10/149.26/12.35/12.09/133 | 70.95/259.69/15.09/17.21/108 | PASS |
| 9 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.65, 'cand_rex_veto_7': 1.35, 'markov_transition_long': 2.0}` | 2262.56/158.34/36.54/4.33/861 | 195.75/195.09/15.31/12.75/203 | 149.28/149.44/12.35/12.10/133 | 71.38/261.85/15.11/17.33/108 | PASS |
| 10 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.7, 'cand_rex_veto_7': 1.3, 'markov_transition_long': 2.0}` | 2258.57/158.20/36.53/4.33/861 | 198.81/198.14/15.17/13.06/203 | 149.46/149.61/12.35/12.12/133 | 71.81/264.01/15.13/17.45/108 | PASS |
| 11 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 1.95, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 2.0}` | 2261.24/158.29/36.58/4.33/861 | 178.85/178.26/16.01/11.14/203 | 146.61/146.76/12.23/12.00/133 | 68.72/248.60/14.85/16.74/108 | PASS |
| 12 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 1.95, 'rex_taker_low_range_position': 0.45, 'cand_rex_veto_7': 1.55, 'markov_transition_long': 2.0}` | 2259.92/158.25/36.57/4.33/861 | 181.77/181.17/15.79/11.47/203 | 146.80/146.95/12.23/12.02/133 | 69.15/250.71/14.87/16.85/108 | PASS |
| 13 | 8.00 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.75, 'cand_rex_veto_7': 1.25, 'markov_transition_long': 2.0}` | 2254.05/158.06/36.52/4.33/861 | 201.89/201.20/15.04/13.38/203 | 149.63/149.79/12.35/12.13/133 | 72.24/266.18/15.16/17.56/108 | PASS |
| 14 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 1.95, 'rex_taker_low_range_position': 0.35, 'cand_rex_veto_7': 1.65, 'markov_transition_long': 2.0}` | 2262.02/158.32/36.59/4.33/861 | 175.95/175.38/16.24/10.80/203 | 146.42/146.57/12.23/11.99/133 | 68.30/246.50/14.83/16.62/108 | PASS |
| 15 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 1.95, 'rex_taker_low_range_position': 0.5, 'cand_rex_veto_7': 1.5, 'markov_transition_long': 2.0}` | 2258.07/158.19/36.56/4.33/861 | 184.71/184.10/15.66/11.76/203 | 146.99/147.14/12.23/12.03/133 | 69.58/252.83/14.90/16.97/108 | PASS |
| 16 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.4, 'cand_rex_veto_7': 1.6, 'markov_transition_long': 1.95}` | 2195.89/156.13/36.08/4.33/861 | 176.62/176.04/16.01/10.99/203 | 146.43/146.58/12.23/11.98/133 | 68.64/248.19/14.86/16.71/108 | PASS |
| 17 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.45, 'cand_rex_veto_7': 1.55, 'markov_transition_long': 1.95}` | 2194.62/156.08/36.08/4.33/861 | 179.52/178.93/15.80/11.32/203 | 146.62/146.77/12.23/12.00/133 | 69.07/250.30/14.88/16.82/108 | PASS |
| 18 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 1.95, 'rex_taker_low_range_position': 0.3, 'cand_rex_veto_7': 1.7, 'markov_transition_long': 2.0}` | 2262.27/158.33/36.60/4.33/861 | 173.08/172.52/16.47/10.47/203 | 146.22/146.37/12.23/11.97/133 | 67.87/244.40/14.81/16.50/108 | PASS |
| 19 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.35, 'cand_rex_veto_7': 1.65, 'markov_transition_long': 1.95}` | 2196.65/156.15/36.09/4.33/861 | 173.75/173.18/16.25/10.66/203 | 146.24/146.39/12.23/11.97/133 | 68.21/246.09/14.83/16.59/108 | PASS |
| 20 | 7.95 | `{'fresh_kimchi_fx': 2.0, 'frozen_annual_rank7': 2.0, 'rex_taker_low_range_position': 0.5, 'cand_rex_veto_7': 1.5, 'markov_transition_long': 1.95}` | 2192.82/156.02/36.07/4.33/861 | 182.44/181.84/15.67/11.61/203 | 146.81/146.96/12.23/12.01/133 | 69.50/252.42/14.90/16.94/108 | PASS |

## Candidate and accounting notes

- The old live row is reproduced exactly under its legacy MDD engine before comparison.
- Selection uses the corrected same-bar upper-before-lower strict MDD clock.
- Every sleeve is marked at the same underlying BTC low/high price points; upper is applied before lower on each bar.
- The reported row is the best found in a deterministic seeded candidate search, not a proof of the global discrete-grid optimum.
- Rank7 and Fresh Kimchi retain their canonical execution/funding schedules.
- Advanced-state representatives selected by inspecting future passers were excluded.
- This experiment does not overwrite the current live config.
