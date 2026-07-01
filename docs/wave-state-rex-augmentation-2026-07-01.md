# Wave text-state REX augmentation (2026-07-01)

## Why

The previous direct pairwise LLM policy failed when evaluated on the full chronological set. The next better direction is a smaller single-LLM state card: keep a robust pre-generated candidate source, then let the LLM/policy reason over compact bucketed state rather than dense numeric OHLC text. Rolling-extrema price-action context is an important part of that state card.

## Change

- `training/build_wave_llm_state_dataset.py` now includes rolling-extrema state tokens in newly built wave LLM state rows.
- `training/augment_wave_llm_state_with_rex.py` can add the same tokens to existing wave state datasets without rebuilding the wave-trading teacher module.

Added token groups for 36/144/576/2016/8640-bar windows:

- `{prefix}_loc`: near_low / lower_half / middle / upper_half / near_high
- `{prefix}_width`: low / medium / high
- `{prefix}_upper_gap`: touching / near / mid / far
- `{prefix}_lower_gap`: touching / near / mid / far

The token source is `build_market_feature_frame`; all REX values use rows at or before `signal_pos`.

## Runs

Existing wave state rows were augmented instead of rebuilding the wave teacher because the local `wave_trading` module currently requires `polars`, which is not installed in this environment and we are avoiding new dependencies.

- Train input: `data/wave_llm_state_ranker_train_2021_2024h1.jsonl` (481 rows)
- Eval input: `data/wave_llm_state_ranker_eval_2024h2_2026.jsonl` (42 rows)
- Train output: `data/wave_llm_state_rex_ranker_train_2021_2024h1.jsonl`
- Eval output: `data/wave_llm_state_rex_ranker_eval_2024h2_2026.jsonl`
- Rule eval: `results/wave_llm_state_rex_rule_eval_2026-07-01.json`
- Baseline rule eval: `results/wave_llm_state_baseline_rule_eval_2026-07-01.json`

## Sanity result

Token-rule evaluation uses train-only token weights and train-score quantile thresholds. Eval rewards are used only for reporting.

| dataset | best frozen threshold | eval rows selected | mean reward | compound return | positive rate |
|---|---:|---:|---:|---:|---:|
| baseline state | q0.50 | 28/42 | 0.571% | 16.90% | 53.57% |
| REX-augmented state | q0.00 | 42/42 | 0.355% | 15.27% | 47.62% |
| REX-augmented state | q0.50 | 26/42 | 0.538% | 14.35% | 50.00% |

REX tokens did not improve this tiny wave-state rule benchmark. They did surface plausible reward-associated tokens (for example short-side 36-bar upper-gap/location), but 42 eval rows are far too few to claim edge.

## Decision

Keep REX tokens in the LLM state-card exporter because they are semantically useful and leak-safe, but do not claim they improve the current wave-state policy yet. The next step needs more candidate rows / broader wave-state coverage before any Gemma fine-tune; otherwise the LLM will overfit sparse labels again.
