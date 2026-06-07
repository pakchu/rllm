# JSON-key candidate log-prob diagnostic (2026-06-07)

## Purpose
Generation-based evaluation is slow and can conflate parsing/output-format issues with model ranking. This diagnostic evaluates single-key JSON tasks (`gate`, `side`) by scoring fixed valid candidates.

## Correction
Candidate-logprob evaluation now forces right padding and records `batch_size` plus `score_normalization` in the output. Older reports that used tokenizer defaults should be regenerated before comparison.

Fix commit: `5aa59e0`.

## Corrected side SFT eval candidate-logprob
Dataset: `data/stable_trader_policy_h144_t1p8_s1p5_split_side_eval.jsonl`
Report: `results/stable_trader_side_gemma4_e4b_h144_t1p8_s1p5_step16_eval_logprob_batched.json`

- Adapter: `checkpoints/stable_trader_side_gemma4_e4b_h144_t1p8_s1p5_step16`
- Samples: 96.
- Accuracy: 57.29%.
- Confusion:
  - LONG→LONG: 44
  - LONG→SHORT: 21
  - SHORT→LONG: 20
  - SHORT→SHORT: 11

This remains below the eval LONG-majority baseline (67.71%), so the side specialist is not useful.

## Corrected gate eval128 candidate-logprob
Report: `results/stable_trader_gate_gemma4_e4b_h144_t1p8_s1p5_step16_eval128_logprob_batched.json`

- Adapter: `checkpoints/stable_trader_gate_gemma4_e4b_h144_t1p8_s1p5_step16`
- Samples: 128.
- Accuracy: 88.28%.
- Confusion:
  - NO_TRADE→NO_TRADE: 113
  - TRADE→NO_TRADE: 15

The corrected gate scorer collapses to abstention, not trade. It is safe-looking but economically useless without trade recall.

## Interpretation
The corrected single-key diagnostics reject the current split heads:
- Gate learned abstention but not opportunity recall.
- Side ranking is weaker than a simple majority baseline.

## Decision
Keep candidate-logprob evaluator as a fast diagnostic. Do not deploy current gate/side adapters.

## Next step
The next viable path is not more gate-threshold tuning. Improve the analyzer state/teacher target so the LLM receives a learnable economic distinction, then validate on leak-free train/test/eval with explicit minimum trades and statistical power constraints.
