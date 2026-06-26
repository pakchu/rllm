# Regime symbolic monthly selector grid — 2026-06-26

## Objective

After adding threshold-variant caching to `regime_symbolic_monthly_selector.py`, retry a broader monthly reliability grid for 2026 Jan-May.

The selector is evaluated as a deployment gate: if prior validation fails, the month emits `NO_TRADE` and untouched eval is not traded.

## Full-history attempt

Attempted with `data/event_action_verifier_text_v3k8_history_2020_2025.jsonl` as history.

Even after threshold caching, this remained too slow because each `(month, target)` still rebuilds row caches, feature matrices, and expert fits over a large 2020-2025 history. The run was stopped and cleaned at ~20 minutes while processing 2026-02. Disk returned to ~292GB used.

Conclusion: threshold caching is useful but not enough for full-history broad grids. Further optimization needs target-independent row/feature matrix reuse or a history lookback mode.

## Recent-history strict grid

Artifact:

- `results/regime_symbolic_monthly_selector_recent_grid_cached_2026-06-26/report.json`

Inputs:

- history: `data/event_action_verifier_text_v3k8_2024_2025.jsonl`
- eval: `data/event_action_verifier_text_v3k8_2026_jan_may.jsonl`
- targets: `utility,net_return,risk_adjusted,tail_risk,distributional_safety`
- thresholds: `-0.003,-0.001,0,0.001,0.003`
- validation months: 3
- gates: `min_val_trades=20`, `min_val_cagr_pct=10`, `max_val_mdd_pct=15.5`, `max_val_p_value=0.25`, `min_val_positive_months=2`, `max_val_worst_month_ret_pct=-3`

Aggregate strict result:

| period | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|
| 2026 Jan-May | 0.00% | 0.00% | 0.00 | 0 |

Month decisions:

| eval month | status | selected | validation CAGR | validation MDD | validation trades | reject reasons |
|---|---|---|---:|---:|---:|---|
| 2026-01 | ABSTAIN | `tail_risk @ -0.001` | -50.03% | 25.17% | 81 | CAGR below min; MDD above max; positive months below min; worst month below min |
| 2026-02 | ABSTAIN | `distributional_safety @ -0.001` | 41.51% | 6.79% | 102 | p-value above max |
| 2026-03 | ABSTAIN | `distributional_safety @ -0.003` | 44.35% | 16.97% | 105 | MDD above max; p-value above max; positive months below min |
| 2026-04 | ABSTAIN | `risk_adjusted @ 0.003` | 7.84% | 20.52% | 102 | CAGR below min; MDD above max; p-value above max |
| 2026-05 | ABSTAIN | `distributional_safety @ -0.001` | -79.50% | 34.73% | 145 | CAGR below min; MDD above max; positive months below min; worst month below min |

Interpretation: strict prior-validation gate correctly refuses every 2026 month.

## Relaxed p-value diagnostic

Artifact:

- `results/regime_symbolic_monthly_selector_recent_grid_cached_relaxedp_2026-06-26/report.json`

Same as strict grid, except `max_val_p_value=1.0`.

Aggregate relaxed result:

| period | CAGR | strict MDD | CAGR/MDD | trades | p approx |
|---|---:|---:|---:|---:|---:|
| 2026 Jan-May | -20.80% | 13.60% | -1.53 | 40 | 0.4225 |

Only one month traded:

| eval month | status | selected | validation result | eval result |
|---|---|---|---|---|
| 2026-02 | TRADED | `distributional_safety @ -0.001` | val CAGR 41.51%, MDD 6.79%, 102 trades | eval CAGR -72.38%, MDD 13.60%, 40 trades |

Interpretation: relaxing the p-value gate allowed a high-CAGR validation candidate that immediately failed in the next month. The statistical gate is not merely conservative; it prevented a real loss.

## Decision

The current regime-symbolic selector is useful as a safety layer but does not unlock profit. It says "do not trade" under strict evidence. That is a valid live behavior but not a monetizable strategy.

Next direction:

1. Do not relax validation evidence gates to force trades; the relaxed diagnostic lost money.
2. Optimize full-history target-independent caches only if continuing this line.
3. More importantly, look for new alpha features/action families. The selector can protect against bad regimes, but it cannot create edge from unstable action surfaces.
