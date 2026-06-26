# Price-action episode policy audit (2026-06-27)

## Purpose

After symbolic verifier/ridge labels failed 2026 holdout, this pass replaced gate optimization with a causal price-action episode representation:

- breakout continuation
- breakdown continuation
- liquidity sweep reversal
- failed breakout / failed breakdown reversal
- range-mid reclaim / reject

Episode side is fixed semantically before training. Template inclusion uses train/test only; eval is untouched.

Script: `training/price_action_episode_policy.py`

Leakage guard:

- prior range levels are shifted rolling highs/lows;
- features use rows at or before signal bar;
- entry occurs after `entry_delay_bars`;
- strict MDD includes intrabar adverse excursion;
- eval is not used for template selection.

## Run 1: train 2020-2023 / test 2024-2025 / eval 2026 Jan-May

Report: `results/price_action_episode_policy_wavefull_train2020_2023_test2024_2025_eval2026jm_portfolio_dedup_2026-06-27/report.json`

Selection constraints:

- individual template train/test filters;
- incremental portfolio train/test filters;
- trigger-overlap dedupe at Jaccard `0.80`;
- `min_train_ratio=0.5`, `min_test_ratio=0.5`.

Selected templates:

- `pae_w2016_low_sweep_reclaim`, LONG, horizon 72
- `pae_w8640_low_sweep_reclaim`, LONG, horizon 72

Portfolio:

| split | CAGR | strict MDD | ratio | trades | p-value | side |
|---|---:|---:|---:|---:|---:|---|
| train 2020-2023 | 9.28 | 16.04 | 0.58 | 232 | 0.0566 | LONG only |
| test 2024-2025 | 7.86 | 8.07 | 0.97 | 140 | 0.1334 | LONG only |
| eval 2026 Jan-May | -14.13 | 9.44 | -1.50 | 32 | 0.2876 | LONG only |

## Run 2: train 2023-2024 / test 2025 / eval 2026 Jan-May

Report: `results/price_action_episode_policy_wavefull_train2023_2024_test2025_eval2026jm_portfolio_dedup_2026-06-27/report.json`

Selected templates are again low-sweep / failed-breakdown LONG variants.

Portfolio:

| split | CAGR | strict MDD | ratio | trades | p-value | side |
|---|---:|---:|---:|---:|---:|---|
| train 2023-2024 | 13.53 | 13.75 | 0.98 | 188 | 0.1350 | LONG only |
| test 2025 | 14.58 | 13.78 | 1.06 | 117 | 0.2243 | LONG only |
| eval 2026 Jan-May | -18.25 | 14.22 | -1.28 | 44 | 0.5196 | LONG only |

## Diagnosis

The new episode representation is cleaner and more causal than the previous verifier/ridge surface, but it still does not meet the target. It repeatedly finds a weak `low_sweep_reclaim -> LONG` dip-reversal pattern that worked in 2024-2025 and failed in 2026. Short-side semantic templates did not survive the train/test filters.

This is useful negative evidence:

1. The problem is not just LLM numeric weakness or gate optimization.
2. The current pool lacks a durable short/regime-transition alpha.
3. Any LLM/RL stage trained on this surface would mostly learn long dip-buying and would likely fail the 2026 eval regime.

## Next direction

The next alpha surface should explicitly model regime transition and short-side episodes, not just prior-range event triggers. Candidate additions:

- downtrend continuation after failed reclaim;
- lower-high / lower-low sequence tokens across multiple windows;
- volatility expansion after long compression with direction confirmed by higher timeframe trend;
- external macro/premium-conditioned short pressure tokens;
- purged rolling selection that requires both long and short templates or abstains.

## Run 3: structure-transition episode expansion

Added causal structure-transition events inside `training/price_action_episode_policy.py`:

- `lower_high_mid_reject` SHORT
- `lower_low_mid_fail` SHORT
- `downtrend_pullback_reject` SHORT
- `failed_mid_reclaim_short` SHORT
- bullish analogues for higher-low / higher-high reclaim

Report: `results/price_action_episode_policy_wavefull_struct_train2023_2024_test2025_eval2026jm_2026-06-27/report.json`

Portfolio result:

| split | CAGR | strict MDD | ratio | trades | p-value | side |
|---|---:|---:|---:|---:|---:|---|
| train 2023-2024 | 34.34 | 13.57 | 2.53 | 289 | 0.0124 | LONG only |
| test 2025 | 11.98 | 12.71 | 0.94 | 154 | 0.3558 | LONG only |
| eval 2026 Jan-May | -24.50 | 15.39 | -1.59 | 67 | 0.3492 | LONG only |

Short-side audit from the same run:

- best SHORT candidates were mostly 8640-window `reject_mid_from_above`, `lower_high_mid_reject`, and `failed_mid_reclaim_short`.
- they looked strong on 2025 test, e.g. `pae_w8640_reject_mid_from_above` SHORT h432 had test CAGR `26.06`, MDD `5.21`, ratio `5.01`, p `0.0013`.
- but the same candidate failed train stability and lost in 2026 eval: eval CAGR `-5.25`, MDD `6.99`, trades `11`.

Conclusion: the added short/regime-transition features are not enough. Current short candidates are validation-regime artifacts, not durable alpha.

Updated next direction:

- require side-diversity only after a genuinely train-stable short alpha exists; forcing shorts now would just add unstable losses.
- short alpha likely needs richer context than prior-range shape alone: higher-timeframe trend phase, macro pressure, funding/OI stress, and failed bounce sequence over multiple events.
- next implementation should build sequence-level episode tokens, not single-bar event triggers.

## Run 4: sequence-level macro/funding/OI episode expansion

Added sequence-context events:

- `seq_bear_reject_macro` SHORT
- `seq_bear_breakdown_macro` SHORT
- `seq_bear_failed_bounce` SHORT
- `seq_bull_reclaim_macro` LONG
- `seq_bull_breakout_macro` LONG
- `seq_bull_failed_dump` LONG

These combine recent prior event sequences with DXY/USDKRW/Kimchi/funding/OI pressure.  Event sequence windows use shifted rolling prior event counts; current event uses only the completed signal bar and entry is still delayed.

### train 2023-2024 / test 2025 / eval 2026 Jan-May

Report: `results/price_action_episode_policy_wavefull_seqmacro_train2023_2024_test2025_eval2026jm_2026-06-27/report.json`

Portfolio:

| split | CAGR | strict MDD | ratio | trades | p-value | side |
|---|---:|---:|---:|---:|---:|---|
| train 2023-2024 | 49.33 | 21.99 | 2.24 | 372 | 0.0041 | LONG only |
| test 2025 | 11.00 | 16.78 | 0.66 | 184 | 0.4348 | LONG only |
| eval 2026 Jan-May | -17.00 | 15.92 | -1.07 | 82 | 0.5983 | LONG only |

Observation: sequence/macro features improved train but still selected long-only portfolios and failed eval.

### train 2024 / test 2025 / eval 2026 Jan-May

Report: `results/price_action_episode_policy_wavefull_seqmacro_train2024_test2025_eval2026jm_2026-06-27/report.json`

Portfolio:

| split | CAGR | strict MDD | ratio | trades | p-value | side |
|---|---:|---:|---:|---:|---:|---|
| train 2024 | 21.18 | 16.30 | 1.30 | 190 | 0.3059 | LONG 169 / SHORT 21 |
| test 2025 | 17.62 | 16.03 | 1.10 | 198 | 0.2216 | LONG 168 / SHORT 30 |
| eval 2026 Jan-May | -8.36 | 13.73 | -0.61 | 76 | 0.8370 | LONG 71 / SHORT 5 |

This is the least bad recent split so far, but still not viable. Train/test significance is weak and eval remains negative.

### Top4 prefix diagnostic

Report: `results/price_action_episode_policy_wavefull_seqmacro_train2024_test2025_eval2026jm_top4_2026-06-27/report.json`

Selected first four templates included two SHORT templates and two LONG templates. Test looked very strong, but eval broke harder:

| split | CAGR | strict MDD | ratio | trades | p-value | side |
|---|---:|---:|---:|---:|---:|---|
| train 2024 | 15.46 | 15.49 | 1.00 | 106 | 0.3347 | LONG 79 / SHORT 27 |
| test 2025 | 37.40 | 5.86 | 6.38 | 113 | 0.0014 | LONG 72 / SHORT 41 |
| eval 2026 Jan-May | -24.49 | 16.75 | -1.46 | 39 | 0.1360 | LONG 29 / SHORT 10 |

Conclusion: the apparently strong 2025 short/mixed setup is a regime artifact. It does not transfer to 2026.

## Updated diagnosis

Sequence-level context and macro/funding/OI conditioning helped expose short candidates, but not durable alpha. The main recurring failure mode is now clearer:

- 2025 rewards both long dip-buying and 8640-window short mid-reject structures.
- 2026 invalidates both, especially the 2025-strong 8640 short reject templates.
- Selection on 2025, even with train filtering, is still too weak because the regime changed in 2026.

Next work should move from static template selection to online regime adaptation / abstention:

1. score templates only if their recent realized paper-trade performance remains positive;
2. use purged rolling monthly re-selection with strict walk-forward state;
3. feed the LLM a compact sequence of recent episode outcomes, not just current episode tokens.
