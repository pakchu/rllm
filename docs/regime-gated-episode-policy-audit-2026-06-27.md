# Regime-gated episode policy audit (2026-06-27)

## Purpose

Static episode templates and per-template online paper gates failed 2026. This pass tested an explicit market-regime gate before episode execution.

Fixed input policy:

- `results/price_action_episode_policy_wavefull_seqmacro_train2024_test2025_eval2026jm_2026-06-27/report.json`

New script:

- `training/regime_gated_episode_policy.py`

Protocol:

1. Keep selected episode templates fixed from the prior policy report.
2. Build past-only regime buckets at each signal bar.
3. Select `(side, regime_key)` buckets using train/test only.
4. Trade eval only when the current signal's regime key was selected.

Default regime fields:

- `side`
- `trend_phase`
- `vol_phase`
- `macro_phase`
- `drawdown_phase`

Regime inputs are derived from historical market features: higher-timeframe trend, volatility/volume state, DXY/USDKRW/Kimchi/funding/OI pressure, and drawdown state.

## Results

### Loose default

Report: `results/regime_gated_episode_seqmacro_train2024_test2025_eval2026jm_default_2026-06-27/report.json`

- selected keys: `16`
- eval CAGR: `-21.85%`
- strict MDD: `14.04%`
- trades: `48`
- side: LONG `47`, SHORT `1`

Loose regime gating still admits too many weak keys and fails.

### Strict key selection

Report: `results/regime_gated_episode_grid_seqmacro_train2024_test2025_eval2026jm_2026-06-27/c016.json`

Config:

- fields: `side,trend_phase,vol_phase,macro_phase,drawdown_phase`
- min train trades: `8`
- min test trades: `5`
- min train mean return: `0.10%`
- min test mean return: `0.05%`
- max test loss rate: `0.50`

Eval result:

- selected keys: `8`
- eval CAGR: `4.64%`
- strict MDD: `8.80%`
- CAGR/MDD: `0.53`
- trades: `34`
- p-value: `0.815`
- side: LONG `33`, SHORT `1`

Selected keys are mostly long entries in `down` trend / `dd_high` regimes, plus two sparse short keys:

- `SHORT, down, low, neutral, dd_high`
- `SHORT, flat, mid, neutral, dd_mid`

## Diagnosis

This is the first gate that turns 2026 eval positive, but it is not tradable:

- CAGR is far below target.
- trade count is too low for statistical confidence.
- p-value is effectively random (`0.815`).
- side exposure remains almost entirely LONG.

Useful signal:

- explicit regime filtering is directionally better than static template selection or per-template paper gating.
- however, these hand-built regime buckets are too crude and mostly act as exposure reduction.

## Next direction

The next step should not be more static gate thresholds. It should build a regime classifier target explicitly:

- label each bar/month as `long_enabled`, `short_enabled`, `abstain` based on rolling forward-safe template performance;
- train/evaluate the classifier with purged chronological splits;
- feed regime class + recent episode sequence to the LLM later.

In other words, the LLM/RL stage should learn **when a family of episode policies is enabled**, not directly memorize raw templates or numeric returns.
