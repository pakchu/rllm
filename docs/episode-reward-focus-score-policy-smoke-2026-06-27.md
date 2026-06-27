# Gemma focus-score policy smoke — 2026-06-27

## Purpose

After the train-only NB baseline failed to preserve the oracle edge, the next deployable path is to use Gemma's
causal focus-label logprob scores as downstream policy features. This smoke test verifies the plumbing:
Gemma focus eval rows now preserve candidate metadata, and `training/focus_score_policy.py` can convert those
scores into strict backtest-compatible `policy_prediction` rows.

## Smoke command

- Eval source: `data/episode_reward_focus_v1_clauses_2026-06-27/plain/eval.jsonl`
- Adapter: `checkpoints/episode_reward_focus_v1_clauses_gemma4_sft64_2026-06-27`
- Sample: random 32 rows, seed 7
- Output dir: `results/episode_reward_focus_score_policy_smoke_2026-06-27/`

## Result

Gemma focus eval over 32 random eval rows:

- `path_shape` accuracy: 25.00% (8/32)
- `utility_bucket` accuracy: 43.75% (14/32)
- exact match: 12.50%
- runtime: ~44.9s wall

Score-policy conversion:

- rows with scores: 32/32
- actions: `NO_TRADE=32`
- trade rate: 0.00%

## Interpretation

This is a schema/plumbing validation, not a profitability test. It confirms the downstream policy can be built
from causal Gemma scores plus candidate metadata without reading future target labels. The default strict rule
requires both predicted `CLEAN_WIN_PATH` and predicted `UTILITY_HIGH`; no random smoke row satisfied both.

The next meaningful experiment is a larger score extraction or a calibrated threshold sweep over Gemma score
probabilities/margins, followed by strict backtest. The action rule must continue to avoid `focus_target` and
`target_audit` for action decisions.
