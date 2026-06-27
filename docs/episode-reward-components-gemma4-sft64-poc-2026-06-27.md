# Episode Reward Components Gemma4-E4B SFT64 PoC (2026-06-27)

## Question
After direct `TRADE/NO_TRADE` and pairwise `A/B` rankers failed, can Gemma4-E4B learn decomposed reward/path components from clause prompts?

## Code changes
- `training/train_text_sft.py`
  - balanced sampling now recognizes reward-component targets via `utility_bucket + path_shape`.
  - dry-run summaries now count component labels.
- `training/eval_reward_component_generation.py`
  - JSON generation evaluator with per-key accuracy and majority baseline mode.
  - Generation was too slow for iteration even at small sample sizes.
- `training/eval_reward_component_logprob.py`
  - teacher-forced component-option logprob evaluator.
  - Scores candidate labels per component key instead of generating full JSON.

## SFT setup
- Dataset: `data/episode_reward_components_v1_clauses_2026-06-27/plain/train.jsonl`
- Model: `gemma4-e4b` (`google/gemma-4-E4B-it`)
- Adapter: `checkpoints/episode_reward_components_v1_clauses_gemma4_sft64_2026-06-27/`
- Train sample: 2,048 rows, balanced by `utility_bucket + path_shape`
- Steps: 64
- Max sequence length: 1,280
- LoRA: r=16, alpha=32, dropout=0.05
- No 4-bit quantization

Training result:
- runtime: 434.5 seconds (~7.2 minutes)
- train loss: 0.5759
- final observed token accuracy: ~0.946
- checkpoint size: ~745MB

## Evaluation
Full JSON generation was not usable for fast iteration:
- eval200 generation was stopped after ~8.5 minutes.
- eval50 generation was stopped after ~7 minutes.

Teacher-forced component logprob eval on first 50 eval rows:

| component | Gemma SFT64 | majority baseline | delta |
| --- | ---: | ---: | ---: |
| net_bucket | 22% | 26% | -4pp |
| mae_bucket | 54% | 62% | -8pp |
| mfe_bucket | 44% | 48% | -4pp |
| mfe_to_mae_bucket | 14% | 34% | -20pp |
| utility_bucket | 36% | 26% | +10pp |
| path_shape | 30% | 16% | +14pp |
| exact all keys | 0% | 0% | 0pp |

## Interpretation
This is not promotable yet, but it is more informative than the failed pairwise ranker.

What improved:
- `path_shape` and `utility_bucket` beat the majority baseline on the small eval probe.
- Clause prompts plus component targets avoid the extreme A/B positional collapse seen in pairwise SFT.
- Training speed remains practical.

What failed:
- `net_bucket`, `mae_bucket`, `mfe_bucket`, and especially `mfe_to_mae_bucket` underperformed majority on this sample.
- Exact component JSON match is 0%.
- Evaluation is still too slow for large sweeps because every key/option scoring uses long causal prompts.

## Decision
Do not promote this adapter to trading/backtest/RL.

Keep this direction, but change the next experiment:
1. Remove or simplify weak/noisy targets like `mfe_to_mae_bucket`.
2. Focus on two useful targets first: `path_shape` and `utility_bucket`.
3. Build a faster evaluator that scores only those two fields, ideally with shorter prompts or cached prefix states.
4. Use component predictions as auxiliary features for a calibrated downstream model, not as direct trade decisions.
