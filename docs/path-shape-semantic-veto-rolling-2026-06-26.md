# Path-shape semantic veto rolling check (2026-06-26)

## Purpose

Exact token vetoes improved one OOS split but failed another rolling split, indicating token-list overfit. This pass folds exact tokens into semantic units and excludes `recent=` exact bar-sequence tokens by default.

Examples:

- `aug.micro.w12.return=-2..-0.75pct` -> `aug.micro.return=-2..-0.75pct`
- `aug.pa.w576.to_high=NEAR` -> `aug.pa.to_high=NEAR`

Implementation:

- `training/path_shape_val_token_veto_tte.py`
  - new `--veto-unit-mode token|semantic`
  - new `--exclude-veto-regex`
- `tests/test_path_shape_val_token_veto_tte.py`

## Rolling result

Config:

- PA+micro path-shape rows
- `min_count=3`, `top_k_tokens=24`
- `veto_unit_mode=semantic`
- `exclude_veto_regex='^recent='`
- `min_token_trades=16`
- `max_veto_mean_ret_pct=-0.05`
- stop/take `0.6% / 1.0%`

| Split | Selected veto | Selected p/m | Val CAGR | Val MDD | Val ratio | Val trades | Eval CAGR | Eval MDD | Eval ratio | Eval trades | Eval p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 12 | 0.55 / 0.15 | 18.12% | 4.20% | 4.32 | 104 | 1.25% | 6.54% | 0.19 | 96 | 0.904 |
| r2 | 12 | 0.34 / 0.30 | 30.32% | 5.03% | 6.03 | 76 | 9.12% | 7.52% | 1.21 | 64 | 0.458 |

Comparison to exact-token rolling:

| Split | Exact-token eval | Semantic eval |
| --- | --- | --- |
| r1 | -17.10% CAGR / 9.71% MDD / -1.76 ratio | 1.25% CAGR / 6.54% MDD / 0.19 ratio |
| r2 | 19.50% CAGR / 7.82% MDD / 2.49 ratio | 9.12% CAGR / 7.52% MDD / 1.21 ratio |

## Interpretation

Semantic vetoes reduce overfit and stabilize drawdown, but they also dilute the edge:

- r1 no longer loses materially, but has no meaningful positive expectancy.
- r2 remains positive but weaker than exact-token veto.
- Both eval p-values are weak.

This supports the current architectural direction: the useful component is an abstention model, but the veto selector must be more causal and less one-window opportunistic.

Next work:

1. Use semantic veto as a safety prior, not as a full strategy.
2. Add rolling/decayed bad-token statistics so the veto set adapts over time without peeking at eval.
3. Only then consider LLM fine-tuning as an abstention reasoner.
