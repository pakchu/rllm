# VSPCR-8 Gross9 novelty rejection

VSPCR-8 is terminally rejected before economics. Its source-support gate passed
with `107 / 223 / 213 / 141` scheduled trades across train, test, eval, and
final, and its exact-entry Jaccard, occupied-bar Jaccard, and absolute signed-
exposure correlation passed against every Gross9 sleeve. The frozen one-to-one
±6-hour matched-share limit failed against one sleeve.

| Gross9 sleeve | ±6h matched share | limit | status |
|---|---:|---:|---|
| cand_rex_veto_7 | 0.300000 | 0.350000 | pass |
| fresh_kimchi_fx | 0.297297 | 0.350000 | pass |
| frozen_annual_rank7 | 0.241379 | 0.350000 | pass |
| markov_transition_long | 0.264706 | 0.350000 | pass |
| rex_taker_low_range_position | **0.354167** | 0.350000 | **fail** |

No BTC execution row, funding row, post-entry return, PnL, CAGR, MDD, or RV20
value was opened. The volume-stratified clock is not sufficiently orthogonal to
the authenticated Gross9 roster under the preregistered limit. Its cohort size,
rank threshold, decision phase, onset, hold, direction, subset, and diagnostic
controls are not repaired or rerun.
