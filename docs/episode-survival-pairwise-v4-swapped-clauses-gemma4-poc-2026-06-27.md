# Episode Survival Pairwise v4 Swapped Clause Gemma4-E4B PoC (2026-06-27)

## Question
The v3 clause model failed full eval with strong A-side bias. Does explicit A/B swap augmentation remove prompt-order bias and improve Gemma4 pairwise ranking?

## Change
`training/export_episode_survival_pairwise_data.py` now supports:

```bash
--augment-swaps
```

When enabled, every accepted best/loser pair is emitted twice:
- A=best, B=loser, target A
- A=loser, B=best, target B

This preserves the same no-leak pair construction while forcing exact A/B label balance and symmetric prompt positions.

## Dataset
Output directory:

`data/episode_survival_pairwise_v4_clauses_swaps_2026-06-27/`

| split | rows | A labels | B labels | mean utility gap |
| --- | ---: | ---: | ---: | ---: |
| train | 62,020 | 31,010 | 31,010 | 1.3983% |
| test | 30,894 | 15,447 | 15,447 | 1.1224% |
| eval | 6,514 | 3,257 | 3,257 | 1.1410% |

## SFT setup
- model: `gemma4-e4b` (`google/gemma-4-E4B-it`)
- rows sampled for SFT: 1,024 balanced
- steps: 32
- max sequence length: 1,536
- LoRA r=16, alpha=32, dropout=0.05
- no 4-bit quantization
- adapter: `checkpoints/episode_survival_pairwise_v4_clauses_swaps_gemma4_sft32_2026-06-27/`

Training result:
- runtime: 215.9 seconds
- train loss: 1.346
- final observed token accuracy: ~0.9167

## Evaluation
`model_choice_token` on full eval:

| split | rows | accuracy | prediction counts |
| --- | ---: | ---: | --- |
| eval | 6,514 | 47.88% | A=5,412, B=1,102 |

## Interpretation
Swap augmentation did not fix the failure. The model still strongly prefers A on out-of-sample eval (~83.1% A predictions), despite exact A/B-balanced train and eval data.

This suggests the issue is not only random pair ordering. More likely:
1. Autoregressive A/B token scoring is poorly calibrated for this task.
2. The tiny SFT learns answer format and superficial prompt priors but not robust candidate comparison.
3. The pairwise target remains noisy/weak; the model collapses to a positional prior under distribution shift.
4. The model needs either a different target decomposition or a non-autoregressive calibration layer.

## Decision
Do not scale this pairwise SFT path further as-is.

Keep the useful pieces:
- clause prompt representation: faster and easier to inspect;
- swap augmentation: correct guard against order bias for future datasets;
- full eval requirement: 500-row probes were misleading.

Next direction should stop treating Gemma as an A/B direct ranker. A better RLLM structure is to let the LLM produce structured causal descriptors / reward-component buckets, then train a calibrated downstream policy or reward model on those outputs.
