# Episode Reward Focus Gemma4-E4B SFT64 PoC (2026-06-27)

## Question
If the reward target is narrowed to the two useful fields (`path_shape`, `utility_bucket`), does Gemma4-E4B produce an out-of-sample auxiliary signal above majority baseline?

## SFT setup
- Dataset: `data/episode_reward_focus_v1_clauses_2026-06-27/plain/train.jsonl`
- Model: `gemma4-e4b` (`google/gemma-4-E4B-it`)
- Adapter: `checkpoints/episode_reward_focus_v1_clauses_gemma4_sft64_2026-06-27/`
- Train sample: 2,048 rows
- Sample mode: balanced by `utility_bucket + path_shape`
- Steps: 64
- Max sequence length: 1,152
- LoRA: r=16, alpha=32, dropout=0.05
- No 4-bit quantization

Training result:
- runtime: 417.8 seconds (~7.0 minutes)
- train loss: 1.024
- final observed token accuracy: ~0.939
- prompt mean chars: 1,013.6
- target mean chars: 63.3

## Evaluation
Evaluator: `training/eval_reward_focus_logprob.py`

It compares candidate labels via teacher-forced option logprob for only:
- `path_shape`
- `utility_bucket`

### Sequential first 500 eval rows
This slice is distribution-skewed, so it is useful only as a smoke probe.

| metric | majority | Gemma SFT64 |
| --- | ---: | ---: |
| exact both fields | 2.6% | 7.0% |
| path_shape | 5.4% | 43.0% |
| utility_bucket | 13.4% | 31.4% |

### Random 1,000 eval rows (`seed=42`)
This is the main representative probe.

| metric | majority | Gemma SFT64 | delta |
| --- | ---: | ---: | ---: |
| exact both fields | 19.3% | 9.8% | -9.5pp |
| path_shape | 22.9% | 30.5% | +7.6pp |
| utility_bucket | 29.2% | 39.4% | +10.2pp |

## Interpretation
This is the first LLM result in this branch with a meaningful out-of-sample advantage over a simple baseline on the intended fields.

What works:
- `path_shape` and `utility_bucket` both beat majority on random eval1000.
- The focused target avoids the direct A/B positional collapse.
- The target is short enough for stable SFT.

What does not work yet:
- exact two-field match is worse than majority on random eval1000.
- evaluation is too slow: random eval1000 took roughly 32 minutes.
- accuracy is not high enough to use directly as a trading decision.
- no strict portfolio backtest has been built on these predictions.

## Decision
Do not promote to trading/RL yet, but keep this direction.

Next practical step:
1. Export Gemma focus predictions/scores as auxiliary features.
2. Train a calibrated downstream model using only train predictions/features.
3. Validate on test/eval strict backtest.
4. Optimize evaluator speed or reduce prompts before larger sweeps.

The correct role for Gemma here is not “final trader”; it is an auxiliary causal path/utility annotator whose outputs may improve a smaller calibrated policy.
