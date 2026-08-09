# CAEVC-8 Gross9 novelty rejection

CAEVC-8 is terminally rejected before economics. Source support passed with
`79 / 167 / 154 / 99` scheduled trades across train, test, eval, and final.
Exact-entry Jaccard, occupied-bar Jaccard, and absolute signed-exposure
correlation passed against every authenticated Gross9 sleeve. The frozen
one-to-one ±6-hour matched-share limit failed against one sleeve.

| Gross9 sleeve | ±6h matched share | limit | status |
|---|---:|---:|---|
| cand_rex_veto_7 | 0.240000 | 0.350000 | pass |
| fresh_kimchi_fx | 0.216216 | 0.350000 | pass |
| frozen_annual_rank7 | **0.413793** | 0.350000 | **fail** |
| markov_transition_long | 0.264706 | 0.350000 | pass |
| rex_taker_low_range_position | 0.250000 | 0.350000 | pass |

No BTC execution row, funding row, post-entry return, PnL, CAGR, MDD, or RV20
value was opened. Symbols, elasticity definition, histories, ranks, crossing,
decision phase, hold, side, subset, and controls are not repaired or rerun.
