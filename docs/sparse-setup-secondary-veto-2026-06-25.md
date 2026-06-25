# Sparse setup secondary candidate veto audit — 2026-06-25

## Question

The top4 sparse setup portfolio is dragged down by 2025H2. Determine whether this can be fixed by turning off weaker secondary CVD candidates using only past fold evidence.

## Diagnostic

Using `data/sparse_setup_action_policy_top4_risk_profiles_2026-06-25.jsonl`, each candidate was replayed separately with source/base actions.

2025H2 candidate split:

| candidate | trades | CAGR | strict MDD | CAGR/MDD | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `wave__mom_12 low & wave__mom_288 low` | 24 | 35.24% | 11.98% | 2.94 | robust top1 remains good |
| `wave__mom_12 low & wave__cvd_mom_55 low` | 7 | -13.98% | 12.86% | -1.09 | hurts fold |
| `wave__mom_24 low & wave__cvd_mom_55 low` h36 | 10 | -10.58% | 12.83% | -0.82 | hurts fold |
| `wave__mom_24 low & wave__cvd_mom_55 low` h72 | 8 | -9.53% | 12.83% | -0.74 | hurts fold |

The secondary CVD candidates have very low fold-level trade counts, often 3-14 trades per half-year, so prior ratios are noisy and easily overtrusted.

## Veto sweep

A simple past-only rule was tested: keep top1 always, enable secondary candidates only if the previous fold had at least N trades and ratio above a threshold.

Output:

`results/sparse_setup_action_policy_2026-06-25/secondary_candidate_veto_sweep.json`

Best observed variants were around:

- CAGR: `~22.1%`
- strict MDD: `~14.75%`
- CAGR/MDD: `~1.50`

This is below the contextual ranker best:

- CAGR: `23.21%`
- strict MDD: `14.51%`
- CAGR/MDD: `1.60`

## Conclusion

The 2025H2 loss is not solved by a simple candidate-level past-performance veto. The top1 price-action/momentum setup is robust, but secondary CVD setups need stronger regime/context features before they can be safely used. The next useful work is new feature/event discovery around regime and price-action structure, not more gate tuning on this candidate pool.
