# Stable trader Gemma4 random-prior SFT (2026-06-07)

## Purpose
The first balanced SFT over-traded because training artificially raised trade frequency. This run preserved the original action prior by using random sampling from the stable trader dataset.

## Training
- Model: `gemma4-e4b` (`google/gemma-4-E4B-it`).
- Dataset: `data/stable_trader_policy_h144_t1p8_s1p5_train.jsonl`.
- Sampling: random 512 rows.
- Action prior in sample:
  - `NO_TRADE`: 357/512 (69.7%)
  - `LONG`: 85/512
  - `SHORT`: 70/512
- LoRA: r=16, alpha=32, dropout=0.05.
- Steps: 16.
- Runtime: 112.4s.
- Final train loss: 0.589.
- Local adapter: `checkpoints/stable_trader_gemma4_e4b_h144_t1p8_s1p5_random512_step16` (~745MB, not committed).

## Val128 metrics
- Action accuracy: 53.13%.
- Exact accuracy: 4.69%.
- Target trades: 27/128.
- Predicted trades: 51/128.
- Side accuracy when target trade: 22.22%.

Val128 strict backtest:
- Trades: 44.
- CAGR: +61.23%.
- Strict MDD: 2.72%.
- CAGR/MDD: 22.51.
- Mean trade: +0.1278%.
- p-value: 0.272.

## Eval128 metrics
- Action accuracy: 75.00%.
- Exact accuracy: 2.34%.
- Target trades: 15/128.
- Predicted trades: 27/128.
- Side accuracy when target trade: 20.00%.

Eval128 strict backtest:
- Trades: 24.
- CAGR: -31.81%.
- Strict MDD: 6.27%.
- CAGR/MDD: -5.07.
- Mean trade: -0.1817%.
- p-value: 0.117.

## Comparison to balanced step16
Random-prior sampling improved abstention:
- Balanced Eval128 predicted trades: 50/128.
- Random-prior Eval128 predicted trades: 27/128.

But it did not solve directional/economic generalization:
- Random-prior Val128 is strongly positive but Eval128 is strongly negative.
- Trade side accuracy on target trades is low (20% Eval128).

## Decision
Do not promote the random-prior checkpoint. It is a better training direction than balanced sampling because it respects trade rate, but it still fails held-out economic validation.

## Next step
The next iteration needs side-quality improvement, not just abstention. Candidate options:
1. train separate gate and side heads/tasks,
2. keep `NO_TRADE` prior while oversampling only LONG/SHORT examples for a side-specialist model,
3. add reward-aware preference/DPO examples comparing chosen action vs opposite side/no-trade,
4. select checkpoints using full validation backtest after faster batched evaluation is implemented.
