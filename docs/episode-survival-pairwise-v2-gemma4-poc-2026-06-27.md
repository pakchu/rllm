# Episode Survival Pairwise v2 Gemma4-E4B PoC (2026-06-27)

## Question
Can `google/gemma-4-E4B-it` fine-tuned with LoRA on the pairwise v2 survival-ranking prompt beat the cheap numeric logistic baseline before we build a strict backtest/RL layer around it?

## Setup
- Dataset: `data/episode_survival_pairwise_v2_2026-06-27/`
  - train: 31,010 pairs
  - test: 15,447 pairs
  - eval: 3,257 pairs
- Plain JSONL copies were created under `data/episode_survival_pairwise_v2_2026-06-27/plain/` because the existing SFT/eval scripts read uncompressed JSONL.
- Model alias: `gemma4-e4b` -> `google/gemma-4-E4B-it`
- SFT command shape:
  - balanced train sample: 1,024 rows, 512 A / 512 B
  - max steps: 32
  - max seq length: 2,048
  - LoRA: r=16, alpha=32, dropout=0.05
  - batch size 1, grad accumulation 8
  - no 4-bit quantization
- Output adapter: `checkpoints/episode_survival_pairwise_v2_gemma4_sft32_2026-06-27/`

## Training result
From `sft_summary.json` and trainer output:
- train runtime: 1,104 seconds (~18.4 minutes)
- train loss: 1.731
- final observed token accuracy around 0.83 on the tiny training stream
- checkpoint size: ~745MB
- RTX 5090 32GB was nearly saturated during training (~32GB used), but no OOM occurred.

## Evaluation changes
`training/eval_pairwise_choice.py` was updated to make PoC evaluation practical:
- added `--max-samples`
- aligned `model_logprob` completions with the SFT target JSON, including `reason`
- added batched logprob scoring
- added `model_choice_token`, which compares the A/B token probability after the JSON prefix `{"choice":"`.

The full-completion logprob mode remained too slow for broad iteration: even 500 rows took about 9 minutes. The choice-token probe is faster and more directly measures the pairwise decision token, but still expensive with Gemma4-E4B on full datasets.

## Accuracy results
Reference baseline from `docs/episode-survival-pairwise-v2-audit-2026-06-27.md`:
- logistic pairwise v2 test accuracy: 54.93%
- logistic pairwise v2 eval accuracy: 52.35%

Gemma4-E4B LoRA PoC:

| split | rows | mode | accuracy | prediction counts |
| --- | ---: | --- | ---: | --- |
| test | 500 | `model_logprob` | 47.2% | A=240, B=260 |
| test | 500 | `model_choice_token` | 48.0% | A=352, B=148 |
| eval | 500 | `model_choice_token` | 47.2% | A=367, B=133 |

## Interpretation
This PoC failed the promotion gate.

The model learned the output format and training stream, but out-of-sample pairwise choice accuracy is below random and far below the cheap logistic baseline. This means the current prompt/SFT setup is not yet extracting useful ranking signal from pairwise v2. Building an execution/backtest layer on this adapter would likely amplify noise, not produce alpha.

Likely causes:
1. **Tiny SFT is format learning, not ranking learning**: 1,024 rows / 32 steps is enough to learn JSON shape, not stable causal preference.
2. **Prompt is still numeric-heavy**: Gemma is asked to infer subtle utility rank from many scalar descriptors; the advantage over logistic regression is not expressed.
3. **Pairwise labels are noisy**: baseline only reaches ~52% eval, so the label surface may be weak even before LLM fine-tuning.
4. **Evaluation is too slow**: current autoregressive scoring over long prompts blocks wide sweeps.

## Decision
Do not promote this Gemma4-E4B SFT adapter to strict backtest/RL.

Next useful direction is not more steps on the same prompt. The next unit should reshape the task so the LLM has a real advantage:
- convert numeric features into compact regime/price-action clauses instead of raw scalar dumps,
- reduce prompt length and candidate ambiguity,
- train/evaluate with a cheaper classifier-style head or token-prob probe from the start,
- require beating logistic eval accuracy before any portfolio backtest.
