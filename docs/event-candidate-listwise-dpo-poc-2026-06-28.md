# Event candidate listwise Gemma DPO PoC — 2026-06-28

## Purpose

Test whether the same-signal LONG/SHORT/NO_TRADE preference data can produce a measurable held-out preference
margin with a small Gemma 4 E4B DPO run.

## Training

- Dataset: `data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_train_2026-06-27.jsonl`
- Adapter: `checkpoints/event_candidate_listwise_pref_gemma4_dpo_s64_2026-06-27`
- Model: `google/gemma-4-E4B-it`
- Samples: 1,024 gate-balanced
- Steps: 64
- LoRA: r16 alpha32 dropout 0.05
- Max length: 2048
- Runtime: 1h 30m 21s
- Train loss: 0.692

Training sample mix:

- Chosen: NO_TRADE 512, LONG 274, SHORT 238
- Rejected: NO_TRADE 262, LONG 368, SHORT 394

## Held-out preference scoring

Evaluation script: `training/eval_preference_logprob.py`

Sample:

- Eval dataset: `data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_eval_2026-06-27.jsonl`
- 128 gate-balanced rows, seed 42
- Metric: chosen response logprob > rejected response logprob

| Model | Accuracy | Correct | Mean margin | Margin std |
| --- | ---: | ---: | ---: | ---: |
| Base Gemma 4 E4B | 23.44% | 30/128 | -0.414 | 3.667 |
| DPO s64 adapter | 22.66% | 29/128 | -0.410 | 3.758 |

Pair-level adapter accuracy:

- `NO_TRADE > LONG`: 38.10%
- `NO_TRADE > SHORT`: 33.33%
- `LONG > NO_TRADE`: 22.73%
- `LONG > SHORT`: 0.00%
- `SHORT > NO_TRADE`: 29.17%
- `SHORT > LONG`: 10.53%

## Decision

Reject the 64-step DPO PoC. It does not improve held-out preference accuracy over the base model, and both base and
adapter strongly underperform random preference choice. This means the current prompt/response scoring format is
not a useful RLLM learning surface yet.

## Likely causes

1. Chosen/rejected JSON strings are too similar and too short relative to long prompts; response logprob is dominated
   by formatting priors rather than trading preference.
2. DPO batch has only 1,024 samples and 64 steps, but loss staying near 0.693 suggests the setup is not separating
   preferences, not merely undertrained.
3. Same-signal listwise rows are structurally better than categorical labels, but the model needs a stronger response
   surface: explicit option letters/rationales or compressed prompts with option descriptions.

## Next direction

Do not scale this exact DPO setup. Next experiment should change the response format to option scoring, e.g.
`A/B/C` choices with concise candidate summaries and evaluate logprob over single-token options. That directly tests
whether the LLM can rank actions without long JSON response likelihood bias.
