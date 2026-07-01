# Wave state top-5 token-rule check (2026-07-01)

## Purpose

The 42-row single-policy wave state eval was too sparse. Existing `top5` wave-state datasets provide broader candidate coverage, so they are a better sanity check before any Gemma fine-tune.

## Inputs

- Baseline train: `data/wave_llm_state_ranker_top5_train_2021_2024h1.jsonl` (2,173 rows)
- Baseline eval: `data/wave_llm_state_ranker_top5_eval_2024h2_2026.jsonl` (181 rows)
- REX train: `data/wave_llm_state_rex_ranker_top5_train_2021_2024h1.jsonl`
- REX eval: `data/wave_llm_state_rex_ranker_top5_eval_2024h2_2026.jsonl`
- Baseline rule eval: `results/wave_llm_state_baseline_top5_rule_eval_2026-07-01.json`
- REX rule eval: `results/wave_llm_state_rex_top5_rule_eval_2026-07-01.json`

## Result

Token-rule scoring fits token reward means on train only and freezes thresholds from train-score quantiles. Eval rewards are reporting only.

| state | best frozen threshold | selected eval rows | mean reward | compound return | positive rate |
|---|---:|---:|---:|---:|---:|
| baseline top5 | q0.50 | 116/181 | 0.598% | 96.29% | 54.31% |
| REX top5 | q0.70 | 82/181 | 0.702% | 74.75% | 53.66% |
| all eval candidates | none | 181/181 | 0.278% | 60.61% | 45.30% |

REX tokens are informative in the learned token weights (`side:SHORT|rex_36_upper_gap=mid`, `rex_36_upper_gap=far`, `side:SHORT|rex_576_loc=near_low`), but the naive equal-weight token model performs better without them on compound return. This likely means REX is useful but needs a model that can learn interactions/attention instead of simple additive averaging.

## Decision

Use the top5 wave-state dataset as the next LLM/RL candidate surface, not the raw pairwise option-choice dataset. Do not fine-tune on the 42-row single-policy surface. For REX, keep the augmented data available but compare baseline-vs-REX with a real model rather than assuming more tokens always help.
