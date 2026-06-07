# Stable trader side Gemma4 SFT (2026-06-07)

## Purpose
After splitting gate and side tasks, train a trade-only side specialist to see whether Gemma4 can learn LONG/SHORT direction better when NO_TRADE is removed from the task.

## Training
- Model: `gemma4-e4b` (`google/gemma-4-E4B-it`).
- Dataset: `data/stable_trader_policy_h144_t1p8_s1p5_split_side_train.jsonl`.
- Rows: 372 trade-only rows.
- Train distribution: LONG 227 / SHORT 145.
- LoRA: r=16, alpha=32, dropout=0.05.
- Steps: 16.
- Runtime: 110.8s.
- Final train loss: 0.786.
- Local adapter: `checkpoints/stable_trader_side_gemma4_e4b_h144_t1p8_s1p5_step16` (not committed).

## Val side metrics
- Rows: 92.
- Accuracy: 58.70%.
- Confusion:
  - LONG→LONG: 28
  - LONG→SHORT: 21
  - SHORT→LONG: 17
  - SHORT→SHORT: 26

Val majority baseline is LONG 49/92 = 53.26%, so this is only a small lift.

## Eval side metrics
- Rows: 96.
- Accuracy: 56.25%.
- Confusion:
  - LONG→LONG: 40
  - LONG→SHORT: 25
  - SHORT→LONG: 17
  - SHORT→SHORT: 14

Eval majority baseline is LONG 65/96 = 67.71%, so this model underperforms the trivial majority side baseline.

## Interpretation
Removing NO_TRADE helps make the task cleaner, but the current prompt/features and 16-step SFT do not learn robust side direction. The side bottleneck remains the main blocker for a profitable LLM trader.

## Decision
Reject this side step16 checkpoint. Continue using the split-task structure, but the side specialist needs stronger labels/features or preference pairs before more SFT scaling.

## Next step
Build side preference/DPO data from each trade row: chosen stable side vs opposite side using realized net return difference. This should teach the model comparative side quality instead of only imitating noisy hard labels.
