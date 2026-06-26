# Path-shape adaptive semantic veto backtest (2026-06-26)

## Purpose

Fixed val-selected vetoes still overfit rolling splits. This pass tests a live-like monthly adaptive semantic veto:

1. At each month start, fit the token model using rows before that month only.
2. Use only prior rows in a trailing history window to mine bad semantic veto units.
3. Trade the current month with fixed probability/margin and veto settings.
4. Combine monthly predictions into one strict OHLC backtest.

Implementation:

- `training/path_shape_adaptive_semantic_veto_backtest.py`
- `tests/test_path_shape_adaptive_semantic_veto_backtest.py`

Config:

- input: all PA+micro path-shape rows, 2023-01 through 2026-02
- eval start: `2025-01-01`
- history window: `6` months
- `min_count=3`, `top_k_tokens=24`
- semantic veto mode, `exclude_veto_regex='^recent='`
- fixed thresholds: `prob=0.34`, `margin=0.30`
- `veto_size=12`
- stop/take: `0.6% / 1.0%`

Artifact:

- `results/path_shape_adaptive_semantic_veto_h144_t1p0_s0p6_pa_micro/report.json`

## Result

Combined 2025-01 to 2026-02:

| Metric | Value |
| --- | ---: |
| CAGR | -10.47% |
| Strict MDD | 17.48% |
| CAGR/MDD | -0.60 |
| Trades | 169 |
| Mean trade | -0.073% |
| p approx | 0.219 |

Monthly behavior was unstable:

- Large losses in early 2025 and 2026-01 dominated.
- Some months were profitable, but trade counts per month were tiny and not reliable.
- Adaptive monthly updating did not solve the regime instability observed in rolling splits.

## Conclusion

The adaptive semantic veto is safer from fixed-val overfit in principle, but this implementation is not profitable. Current evidence:

- Exact token veto: strongest r2 result, but overfits r1.
- Semantic veto: more stable, weak/no edge.
- Monthly adaptive semantic veto: live-like but negative.

Next direction should change the **base event/action generation** or label target, not continue threshold/veto tweaking. If LLM is used, it should be trained on stable semantic abstention rationales only after a non-LLM rolling selector shows durable positive expectancy.
