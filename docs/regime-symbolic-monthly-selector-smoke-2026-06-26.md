# Regime symbolic monthly selector smoke — 2026-06-26

## Objective

After symbolic ridge showed 2024-only strength and failed on later holdouts, test whether a monthly prior-validation selector can decide when to trade or abstain.

This is the intended reliability layer for an eventual RLLM/Gemma compressor: the LLM should not imitate raw side labels; it should help describe context, while a causal selector decides whether the action-value surface is valid.

## Full grid attempt

Attempted:

- history: `data/event_action_verifier_text_v3k8_history_2020_2025.jsonl`
- eval: `data/event_action_verifier_text_v3k8_2026_jan_may.jsonl`
- targets: `utility,net_return,risk_adjusted,tail_risk,distributional_safety`
- thresholds: `-0.003,-0.001,0,0.001,0.003`
- eval: 2026-01 through 2026-05

This was stopped after ~12 minutes because `regime_symbolic_monthly_selector.py` refits rolling experts for every target/threshold/month candidate. RSS grew to ~18.9GB and the work dir reached ~588MB. Disk stayed below the 300GB WSL limit after cleanup.

Conclusion: the implementation is too expensive for broad grids without caching fitted monthly feature matrices/expert fits.

## Reduced utility smoke

Command artifact:

- `results/regime_symbolic_monthly_selector_utility_smoke_2026-06-26/report.json`

Inputs:

- history: `data/event_action_verifier_text_v3k8_2024_2025.jsonl`
- eval: `data/event_action_verifier_text_v3k8_2026_jan_may.jsonl`
- eval months: 2026-01 through 2026-02
- target: `utility`
- thresholds: `0,0.001,0.003`
- validation months: 3
- gates: `min_val_trades=20`, `min_val_cagr_pct=10`, `max_val_mdd_pct=15.5`, `max_val_p_value=0.25`, `min_val_positive_months=2`, `max_val_worst_month_ret_pct=-3`

Aggregate result:

| period | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|
| 2026 Jan-Feb strict output | 0.00% | 0.00% | 0.00 | 0 |

Month decisions:

| eval month | status | selected target | threshold | validation CAGR | validation MDD | validation trades | reject reasons |
|---|---|---|---:|---:|---:|---:|---|
| 2026-01 | ABSTAIN | utility | 0.000 | -58.45% | 22.02% | 71 | cagr below min; mdd above max; positive months below min; worst month below min |
| 2026-02 | ABSTAIN | utility | 0.003 | -5.53% | 9.43% | 60 | cagr below min; p-value above max; positive months below min |

## Decision

The reliability selector is directionally correct in protocol: it refuses to trade when prior validation is bad. However, this does not create a monetizable strategy because current prior windows do not produce enough valid months.

Next engineering step before more alpha search:

1. Cache monthly feature matrices / expert fits in `regime_symbolic_monthly_selector.py`; current broad grids are computationally wasteful.
2. After caching, run a broader target/threshold/expert grid over 2025H2 and 2026 Jan-May.
3. Only if monthly validation authorizes enough months with positive strict holdout should Gemma/Gemma4 be introduced as a symbolic context compressor.
