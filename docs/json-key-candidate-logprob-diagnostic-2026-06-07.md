# JSON-key candidate log-prob diagnostic (2026-06-07)

## Purpose
Generation-based evaluation is slow and can conflate parsing/output-format issues with model ranking. This adds candidate log-prob evaluation for single-key JSON tasks (`gate`, `side`) and uses it to diagnose current gate/side adapters.

## Side SFT eval candidate-logprob
Dataset: `data/stable_trader_policy_h144_t1p8_s1p5_split_side_eval.jsonl`

- Adapter: `checkpoints/stable_trader_side_gemma4_e4b_h144_t1p8_s1p5_step16`
- Accuracy: 57.29%.
- Confusion:
  - LONG→LONG: 45
  - LONG→SHORT: 20
  - SHORT→LONG: 21
  - SHORT→SHORT: 10

This is slightly better than generation for the same adapter (56.25%) but still below eval LONG-majority baseline (67.71%).

## Side DPO eval candidate-logprob
- Adapter: `checkpoints/side_pref_dpo_gemma4_e4b_h144_t1p8_s1p5_step16`
- Accuracy: 55.21%.
- Worse than side SFT.

## Gate eval128 candidate-logprob
- Adapter: `checkpoints/stable_trader_gate_gemma4_e4b_h144_t1p8_s1p5_step16`
- Accuracy: 11.72%.
- Predicted TRADE for all 128 rows.

This confirms candidate scoring is unsafe for current gate adapter because it collapses to TRADE even when generation was conservative.

## Interpretation
Candidate scoring removes parse noise but reveals the same structural issue: side ranking is weak and gate ranking is miscalibrated. The current models have not learned reliable economic ranking even when constrained to valid JSON candidates.

## Decision
Keep candidate-logprob evaluator for diagnostics. Do not use it for deployment with the current adapters.

## Next step
Further progress likely needs better prompt/state representation or supervised non-LLM teacher improvements before more Gemma training. For the immediate code path, use candidate-logprob as a fast rejection/selection tool, not as a trading policy.
