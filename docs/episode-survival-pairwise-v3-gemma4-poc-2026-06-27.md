# Episode Survival Pairwise v3 Clause Gemma4-E4B PoC (2026-06-27)

## Question
Does the v3 clause prompt fix the v2 failure mode enough for Gemma4-E4B LoRA to beat the pairwise logistic baseline before promotion to backtest/RL?

## Setup
- Dataset: `data/episode_survival_pairwise_v3_clauses_2026-06-27/`
- Prompt style: causal price-action/regime clauses, not raw JSON scalar dump
- Model alias: `gemma4-e4b` -> `google/gemma-4-E4B-it`
- Adapter: `checkpoints/episode_survival_pairwise_v3_clauses_gemma4_sft32_2026-06-27/`
- Training:
  - 1,024 balanced rows
  - 32 steps
  - max sequence length 1,536
  - LoRA r=16, alpha=32, dropout=0.05
  - no 4-bit quantization

## Training result
- runtime: 209.3 seconds (~3.5 minutes)
- train loss: 1.399
- final observed token accuracy: ~0.9345
- train samples/sec: 1.223
- checkpoint size: ~745MB

Compared with v2 numeric prompt:
- v2 runtime: 1,104 seconds (~18.4 minutes)
- v3 runtime: 209 seconds
- speedup: ~5.3x

The shorter clause prompt materially improved iteration speed.

## Evaluation
Reference baseline from pairwise v2 logistic audit:
- test accuracy: 54.93%
- eval accuracy: 52.35%

Gemma4-E4B v3 clause LoRA, `model_choice_token`:

| split | rows | accuracy | prediction counts |
| --- | ---: | ---: | --- |
| test | 500 | 50.8% | A=430, B=70 |
| eval | 500 | 51.0% | A=426, B=74 |
| eval | 3,257 | 49.28% | A=2,813, B=444 |

## Interpretation
The representation change improved speed and small-sample behavior, but did not solve ranking quality.

The full eval result is below random and below the 52.35% logistic baseline. The adapter also developed a strong A bias on full eval (~86.4% A predictions) despite balanced SFT sampling. This makes it unsuitable for strict backtest/RL promotion.

Important distinction:
- **Success**: prompt compression made Gemma training/eval practical.
- **Failure**: the current pairwise SFT objective still does not produce robust out-of-sample preference ranking.

Likely causes:
1. The first-N eval/test samples are not representative; full eval exposed A-side bias.
2. Pairwise target noise remains high; label edge is weak even for the logistic baseline.
3. Balanced row sampling does not guarantee calibrated A/B token probabilities out of distribution.
4. The prompt now describes regimes better, but the answer target still asks for a brittle binary preference rather than decomposed reasoning or reward components.

## Decision
Do not promote v3 clause Gemma4-E4B SFT to backtest/RL.

Next direction should keep the clause representation because it is much faster, but change the learning target. Better candidates:
- predict decomposed reward components first (expected favorable excursion bucket, adverse excursion bucket, timeout/failure bucket), then derive trade preference;
- train a calibrated lightweight classifier/head on frozen clause embeddings instead of relying on autoregressive A/B token probability;
- add side-balanced and time-balanced evaluation slices so A/B prompt-order bias is detected before full eval;
- consider explicit pair augmentation with swapped A/B copies for every pair, not random one-sided assignment only.
