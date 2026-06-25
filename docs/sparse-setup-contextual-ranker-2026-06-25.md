# Sparse setup contextual ranker — 2026-06-25

## Purpose

Test whether an LLM/RL-shaped state/action token policy can improve over fixed sparse setup source actions without using current-fold rewards.

## Implementation

Added `training/sparse_setup_contextual_ranker.py`.

The ranker reads sparse event-action JSONL records and creates tokenized examples from:

- event `state_tokens`
- candidate key/index
- action side
- hold length
- risk profile
- stop/take-profit metadata

For each fold, it fits a ridge model only on previous folds, then ranks all executable actions in the current fold. If the predicted score is below threshold, it falls back to the sparse source action.

## Validation

```bash
.venv/bin/python -m unittest tests/test_sparse_setup_contextual_ranker.py
.venv/bin/python -m unittest tests/test_sparse_setup_action_policy.py
```

Observed unit validation:

- contextual ranker: `2 tests OK`
- sparse action policy: `9 tests OK`

## Main sweep

Input JSONL:

`data/sparse_setup_action_policy_top4_risk_profiles_2026-06-25.jsonl`

Summary output:

`results/sparse_setup_contextual_ranker_2026-06-25/summary.json`

Best config:

- target: `utility`
- l2: `1.0` or `10.0`
- min predicted utility: `-0.002`

Final metrics:

- trades: `217`
- CAGR: `23.21%`
- strict MDD: `14.51%`
- CAGR/MDD: `1.60`
- win rate: `57.14%`
- risk profiles: `base=215`, `sl5tp8=2`

Baseline top4 source fallback was:

- trades: `218`
- CAGR: `21.55%`
- strict MDD: `14.51%`
- CAGR/MDD: `1.49`

Fold detail for best config:

| fold | trades | CAGR | strict MDD | CAGR/MDD | note |
| --- | ---: | ---: | ---: | ---: | --- |
| 2023H1 | 11 | 33.60% | 3.39% | 9.90 | cold start |
| 2023H2 | 11 | 46.70% | 2.32% | 20.15 | fallback source |
| 2024H1 | 52 | 40.36% | 11.40% | 3.54 | ranker used on 384 records, 2 sl5tp8 trades |
| 2024H2 | 43 | 32.69% | 14.51% | 2.25 | fallback source |
| 2025H1 | 36 | 15.71% | 8.59% | 1.83 | fallback source |
| 2025H2 | 30 | 0.63% | 12.83% | 0.05 | main bottleneck |
| 2026H1 | 34 | 34.36% | 14.37% | 2.39 | fallback source |

## Interpretation

The contextual ranker is directionally useful but not transformative. It improves CAGR and win rate without increasing strict MDD, but mostly rejects its own predictions and falls back to source actions. This means current state tokens do not yet explain enough of when risk profiles should change.

The remaining bottleneck is not the action selector alone. The weak 2025H2 fold needs better event features or a higher-level regime veto; otherwise the strategy remains below the target CAGR/MDD >= 3.
