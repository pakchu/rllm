# Sparse setup train-only descriptor veto TTE (2026-06-26)

## Purpose

The failure-cluster miner found strong-looking 2025 bad-event descriptors, especially weak multi-day HTF return plus rolling-extreme compression. That artifact was diagnostic only because it used 2025 realized losses to mine thresholds.

This pass tests the same idea with stricter leakage control:

1. Build sparse setup candidate events over train/test/eval folds.
2. Label **train-only** rows as good or bad from realized utility/MAE.
3. Mine descriptor thresholds from train good-vs-bad rows only.
4. Fit ridge event scorer on train only.
5. Select score quantile + fixed descriptor veto on 2025 test only.
6. Refit scorer on train+test and evaluate untouched 2026H1 with the selected quantile/veto.

Implementation:

- `training/sparse_setup_descriptor_veto_tte.py`
- `tests/test_sparse_setup_descriptor_veto_tte.py`

## Main strict run

Artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/descriptor_veto_tte_trainonly_v1.json`

Config summary:

- train folds: `2020H1` through `2024H2`
- test folds: `2025H1`, `2025H2`
- eval folds: `2026H1`
- candidate limit: `80`
- ridge alpha: `300`
- quantiles: `0.90,0.925,0.95,0.975`
- max veto size: `1`
- train-good: utility >= `0.25%`, MAE <= `2.5%`
- train-bad: utility <= `-0.25%`
- min descriptor coverage edge: `0.08`
- max train-good block rate: `0.65`

Rows:

| Bucket | Count |
| --- | ---: |
| All events | 212,431 |
| Train events | 180,767 |
| Test events | 22,632 |
| Eval events | 9,032 |
| Train-good | 68,967 |
| Train-bad | 92,058 |

Selected by test:

```json
{"q": 0.9, "veto": [], "score": -996.5578256888998}
```

| Period | CAGR | Strict MDD | CAGR/MDD | Trades |
| --- | ---: | ---: | ---: | ---: |
| Train 2020-2024 | 43.42% | 19.24% | 2.26 | 471 |
| Test 2025 | -0.70% | 10.08% | -0.07 | 71 |
| Eval 2026H1 | -34.66% | 14.03% | -2.47 | 36 |

The strict train-only descriptor filter produced only short-side descriptor candidates, so the selected long-heavy event stream was effectively unchanged.

## Lenient descriptor run

Artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/descriptor_veto_tte_trainonly_v2_lenient.json`

Relaxed descriptor filters:

- min coverage edge: `0.03`
- max train-good block rate: `0.90`
- top descriptors per scope: `12`

Selected by test again chose no veto:

```json
{"q": 0.9, "veto": [], "score": -996.5578256888998}
```

The top train-only descriptors were much weaker than the prior 2025-bad diagnostic descriptors. Examples:

| Scope | Feature/rule | Edge | Train-good block | Bad coverage |
| --- | --- | ---: | ---: | ---: |
| overall | `pa__pa_ext_144_to_max_high_pct >= -0.057527` | 0.053 | 0.697 | 0.750 |
| overall | `mkt__htf_3d_return_4 <= 0.092644` | 0.037 | 0.723 | 0.760 |
| long | `pa__pa_ext_144_to_max_high_pct >= -0.058003` | 0.050 | 0.700 | 0.750 |
| short | `pa__pa_ext_576_max_high_bar_spread_pct <= 0.009064` | 0.153 | 0.597 | 0.750 |

## Conclusion

The strong-looking 2025 failure descriptors did **not** survive when the descriptor thresholds were discovered from train only. This is important negative evidence:

- The 2025-bad mining artifact was likely an after-the-fact explanation, not a stable failure-regime alpha.
- Price-action extrema and HTF return features are still useful context, but not as single hard vetoes over this sparse setup pool.
- More threshold/gate optimization on this sparse pool is unlikely to solve the objective.

Next direction should move away from hard veto selection and toward either:

1. a different event pool with stronger base expectancy, or
2. a sequence/LLM-friendly objective that predicts path structure and abstention jointly rather than applying one-feature vetoes after a weak sparse trigger.
