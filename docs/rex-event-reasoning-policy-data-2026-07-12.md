# REX event reasoning policy data (2026-07-12)

## Purpose
Move RLLM back toward its intended strength: symbolic/deductive reasoning over event context, not raw numeric regression.

This work builds a single-policy SFT surface for `rex_htf_pullback_reclaim` events:

- Prompt: past-only symbolic/bucketed facts.
- Target: offline executable path-utility label for `LONG`, `SHORT`, or `NO_TRADE`.
- Target labels are for training only and are never included in prompts.

## Artifacts
- Builder: `training/build_rex_event_reasoning_policy_data.py`
- Token baseline: `training/evaluate_rex_event_token_policy.py`
- SFT rows: `data/rex_event_reasoning_policy_sft_20260712.jsonl`
- Dataset summary: `results/rex_event_reasoning_policy_sft_summary_2026-07-12.json`
- Token NB eval: `results/rex_event_token_nb_policy_eval_2026-07-12.json`

## Dataset
Input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz`

Split:
- train: `< 2025-01-01`
- test: `2025-01-01..2026-01-01`
- eval: `2026-01-01..2026-06-01`

Rows:

| split | rows | target LONG | target SHORT | target NO_TRADE | base LONG | base SHORT |
|---|---:|---:|---:|---:|---:|---:|
| train | 1259 | 367 | 303 | 589 | 713 | 546 |
| test | 104 | 31 | 22 | 51 | 38 | 66 |
| eval | 81 | 21 | 21 | 39 | 16 | 65 |

Prompt shape:
- Avg prompt length: about 1.2k chars.
- No raw price path or future return appears in prompt.
- Facts are symbolic buckets such as `rsi_state=oversold`, `short_context_hint=range_stress_ok`, `long_range_location=near_long_low`, `taker_flow=sell_flow`.

## Token baseline check
A simple train-only token Naive Bayes model was used as a non-LLM learnability baseline.

Backtest using predicted actions:

| split | accuracy | abs return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| train | 44.24% | 42.84% | 7.65% | 27.40% | 0.28 | 330 |
| test 2025 | 47.12% | 7.50% | 8.80% | 6.96% | 1.26 | 29 |
| eval 2026H1 | 38.27% | -0.12% | -0.46% | 5.21% | -0.09 | 22 |

## Interpretation
- Symbolic prompt labels are balanced enough for SFT: train has 367 LONG / 303 SHORT / 589 NO_TRADE.
- A simple token-frequency model fails on eval, so this is not a trivial token shortcut.
- This supports trying a small Gemma/RLLM model: it must reason over combinations of event, side context, and regime rather than memorize single-token gates.
- The data is not a trading result. It is a no-leak supervised training surface for the next LLM policy experiment.

## Leakage guard
- Event threshold is fit on train only.
- Prompt tokens use signal-time/current-or-past features only.
- Future OHLC path is used only to create offline SFT labels.
- NB baseline trains only on train rows; test/eval targets are metrics only.
