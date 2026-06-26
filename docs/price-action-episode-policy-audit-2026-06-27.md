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
