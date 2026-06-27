# Gemma focus-score test/eval validation — 2026-06-27

## Purpose

The eval random1000 threshold sweep was exploratory and not valid for final selection. This run restores the
proper order: score a random test sample, select the best threshold on test, then apply that exact threshold once
to the already-scored random eval1000 sample.

## Test score extraction

- Input split: `data/episode_reward_focus_v1_clauses_2026-06-27/plain/test.jsonl`
- Sample: random 1000, seed 42
- Adapter: `checkpoints/episode_reward_focus_v1_clauses_gemma4_sft64_2026-06-27`
- Runtime: 44m 21.59s wall
- Output: `results/episode_reward_focus_score_policy_test_random1000_2026-06-27/gemma_focus_test1000_predictions.jsonl`

Label quality on test random1000:

- `path_shape` accuracy: 26.30%
- `utility_bucket` accuracy: 35.50%
- exact match: 8.10%

## Test threshold selection

Small threshold grid over causal Gemma score probabilities:

- `clean_prob`: `0.02,0.05,0.1,0.2,0.3,0.4`
- `high_prob`: `0.0,0.2,0.333`
- margins disabled (`-999`)

Best test threshold:

| clean_prob | high_prob | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.333 | 97 | 2.45% | 9.14% | 0.27 | 0.055% | 0.600 |

Even the best test threshold is weak and statistically insignificant.

## Held-out eval application

Applied the selected test threshold (`clean_prob>=0.10`, `high_prob>=0.333`) once to the random eval1000 score set:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eval random1000 | 54 | -0.32% | 9.84% | -0.03 | 0.004% | 0.978 |

## Decision

Reject the current Gemma focus-score policy path as-is. The failure is not primarily threshold tuning: the selected
test threshold is already weak, and it does not transfer to eval. The current focused Gemma SFT adapter does not
produce a strong enough causal reward signal for deployable trading decisions.

## Next direction

Stop spending cycles on gate optimization over this adapter. The next intervention should target the learning
setup itself:

1. Improve target formulation so the model learns a ranking/utility decision rather than brittle categorical labels.
2. Increase hard-negative contrast around similar price-action clauses.
3. Consider pairwise or listwise preference training over candidates at the same timestamp instead of independent
   absolute label prediction.
4. Preserve train/test/eval ordering and keep final eval as a single-shot validation.
