# Side preference DPO check (2026-06-07)

## Purpose
Use realized LONG-vs-SHORT return preference pairs to improve the failed side specialist.

## Training
- Base model: `gemma4-e4b`.
- Initial adapter: `checkpoints/stable_trader_side_gemma4_e4b_h144_t1p8_s1p5_step16`.
- Preference data: `data/side_preference_h144_t1p8_s1p5_diff0p05_train.jsonl`.
- Samples: 512 balanced chosen-side pairs.
- DPO steps: 16.
- Learning rate: 5e-7.
- Beta: 0.1.
- Runtime: 153.4s.
- Final train loss: ~0.693.

## Metrics after DPO
Val side:
- Rows: 92.
- Accuracy: 58.70%.
- Confusion: LONG→LONG 28, LONG→SHORT 21, SHORT→LONG 17, SHORT→SHORT 26.
- Same as pre-DPO side SFT.

Eval side:
- Rows: 96.
- Accuracy: 54.17%.
- Confusion: LONG→LONG 38, LONG→SHORT 27, SHORT→LONG 17, SHORT→SHORT 14.
- Worse than pre-DPO side SFT (56.25%).

## Interpretation
The DPO run did not move the model meaningfully. Training loss stayed near the random-preference baseline and eval accuracy fell. Likely causes:
- responses are extremely short and symmetric, so DPO signal is weak at low LR/step count,
- the initial side adapter already collapses toward a similar output prior,
- preference prompts are too compressed or identical-length to produce strong gradients.

## Decision
Reject this DPO checkpoint. Keep the preference dataset, but use it for either stronger candidate scoring/evaluation or a higher-signal training setup rather than the current tiny DPO continuation.

## Next step
Before more GPU training, implement a fast candidate log-prob evaluator for side/gate so full val/eval can be scored without sample-by-sample generation. Then use log-prob margins to diagnose whether models rank the correct side even when generated output is wrong.
