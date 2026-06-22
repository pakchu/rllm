# Single-policy real Gemma backtest check — 2026-06-23

## Purpose

After resetting away from the analyzer/trader cascade, the first safety issue was
that `training/backtest_single_policy_predictions.py` could silently fall back to
future-derived `target` labels when `policy_prediction` was absent. That behavior
made oracle target-echo backtests too easy to misread as model performance.

The backtester now requires `policy_prediction` by default. Target fallback is
available only through `--allow-target-echo`, and reports set
`target_echo_oracle_mode=true` in that case.

## Real model backtest refresh

Both runs below use existing Gemma prediction JSONL files with actual
`policy_prediction` objects. No target echo was allowed.

### `balanced_m1536_step160_val512`

Input:
`results/pred_gemma4_single_policy_nt0p004_balanced_m1536_step160_val512_model.jsonl`

Strict result:

- `target_echo_oracle_mode`: `false`
- CAGR: `-11.16%`
- strict MDD: `12.48%`
- CAGR / strict MDD: `-0.89`
- trades: `108`
- mean trade return: `-0.0453%`
- approximate p-value: `0.661`

### `random2048_r8_step96_val512`

Input:
`results/pred_gemma4_single_policy_nt0p004_random2048_r8_step96_val512_model.jsonl`

Strict result:

- `target_echo_oracle_mode`: `false`
- CAGR: `-26.62%`
- strict MDD: `21.19%`
- CAGR / strict MDD: `-1.26`
- trades: `106`
- mean trade return: `-0.1305%`
- approximate p-value: `0.194`

## Interpretation

These are real model predictions, not oracle labels, and both fail. This confirms
that the previous target-echo/oracle style results are not deployable evidence.
The next useful work is not more two-stage prompt engineering; it is either:

1. stronger causal feature/alpha discovery, or
2. a simulator-trained single policy that is evaluated only through
   `policy_prediction` rows with `target_echo_oracle_mode=false`.

Any future profitable claim must report the oracle flag and should be rejected if
`target_echo_oracle_mode=true`.
