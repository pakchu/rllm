# Score-geometry side-map transform selection (2026-06-23)

## Purpose

Prior validation health alone found a weak 2026 inversion candidate, but not enough alpha. This test adds current prediction-score geometry to the transform selector:

- prior month validation score;
- current `score_long`, `score_short`, `score_wait`;
- side gap `abs(score_long - score_short)`.

The transform can pass, invert, or block existing generated trades. Configs are ranked only on 2024-01 through 2025-12 and replayed on untouched 2026-01 through 2026-05.

## Implementation

Added:

- `training/nested_score_geometry_transform_selection.py`
- `tests/test_nested_score_geometry_transform_selection.py`

Transform families:

- `pass`
- `invert`
- `block`
- `invert_low_gap`
- `invert_high_gap`
- `block_low_gap`
- `block_high_gap`

The script deletes per-candidate prediction/backtest files by default to avoid WSL disk growth.

## Bounded grid result

Command output:

- `results/nested_score_geometry_transform_h288_fast_2026-06-23.json`

Best selection config:

| Config | Selection trades | Selection CAGR | Selection strict MDD | Selection ratio | Eval trades | Eval CAGR | Eval strict MDD | Eval ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold `0.5`, below=`block_high_gap`, above=`pass`, score_gap `0.20`, TP `3%` | 346 | 31.52% | 11.89% | 2.65 | 0 | 0.00% | 0.00% | 0.00 |

Interpretation: selection still prefers blocking 2026-like severe-decay months, so the clean top candidate does not trade eval.

## Inversion-focused result

Command output:

- `results/nested_score_geometry_transform_h288_invert_focus_2026-06-23.json`

Best inversion-focused clean-picked config:

| Config | Selection trades | Selection CAGR | Selection strict MDD | Selection ratio | Eval trades | Eval CAGR | Eval strict MDD | Eval ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold `-500`, below=`invert`, above=`pass`, TP `3%` | 418 | 26.51% | 11.92% | 2.22 | 80 | 14.29% | 10.22% | 1.40 |

Score-gap transforms did not improve this result.

## Decision

No-go for score-geometry side-map transforms.

What this rules out:

- Simple side inversion based on prior validation score.
- Simple row-level inversion/blocking based on `abs(score_long-score_short)`.
- Passive reliability token plus same action space.

Current best causal eval with trades remains far below target:

- 80 trades over 2026-01 through 2026-05;
- CAGR 14.29%;
- strict MDD 10.22%;
- ratio 1.40.

Next direction:

The base ranker seems to have weak entry timing but unstable side mapping. A better approach is not another gate sweep, but a new label/action design:

1. Train an explicit side-map reliability target from prior rolling windows: `normal|inverse|unreliable`.
2. Use it to construct training examples where the LLM reasons about when side evidence is non-stationary.
3. Evaluate with strict nested splits before any deployability claim.
