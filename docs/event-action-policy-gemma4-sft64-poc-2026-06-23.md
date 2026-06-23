# Event-action policy Gemma4 SFT64 POC (2026-06-23)

## Why this path

The prior side-map / rationale-gate work mostly optimized weak gates. This pass switches to a more RLLM-compatible representation: one Gemma policy sees a past-only state plus a candidate action book and learns a full action template (`gate`, `family`, `side`, `hold_bars`, `confidence`).

## Data

Generated with `training.event_action_policy_data` using BTCUSDT 5m market data and wave_trading external features from `../workspace/wave_trading`.

- Train/calibration: `data/event_action_policy_train_pre2026_2026-06-23.jsonl`
  - Period: 2022-01-01 02:55:00 to 2025-12-31 20:55:00
  - Rows: 5,844
  - Target gates: TRADE 4,726 / NO_TRADE 1,118
  - Top target families: NONE 1,118, macro_pressure 1,015, micro_exhaustion_reversal 902, kimchi_extreme_fade 860
- Eval: `data/event_action_policy_eval2026_2026-06-23.jsonl`
  - Period: 2026-01-01 02:55:00 to 2026-05-30 02:55:00
  - Rows: 597
  - Target gates: TRADE 487 / NO_TRADE 110

The target is oracle-labeled from future path utility and is for training only; prompts are past-only.

## Oracle upper bound diagnostic

Target echo on 2026 eval is intentionally cheating, but verifies that the action book/label surface has economic upside if selected correctly:

- Samples: 597
- Trades: 111
- CAGR: 2903.99%
- Strict MDD: 2.56%
- CAGR/strict MDD: 1134.48

This is not a model result.

## SFT run

Checkpoint: `checkpoints/event_action_policy_gemma4_e4b_sft64_2026-06-23`

Config:

- Model: `google/gemma-4-E4B-it`
- Rows: 4,096 gate-balanced from pre-2026 train
- Steps: 64
- LoRA: r=16, alpha=32, dropout=0.05
- Max seq length: 1600
- Runtime: 442s
- Final train loss: 0.4352

## Evaluation bottleneck and fix

The existing candidate-logprob evaluator was unusably slow because it re-forwarded the full prompt for every candidate action. `training.fast_eval_text_trader` adds prompt KV-cache scoring and progress output.

Even the fast scorer is still not cheap on Gemma4 E4B:

- 100 eval rows took 263.5s after model load.
- Full 597-row eval would be roughly 25-30 minutes unless further optimized/batched.

## 100-row smoke result

Fast candidate-logprob, candidates: `NO_TRADE` + LONG/SHORT × {72,144,288,432}, score normalization mean.

Classification:

- Gate accuracy: 76.0%
- Side accuracy when target trade: 50.0%
- Hold accuracy when target trade: 51.3%
- Exact action accuracy: 17.0%
- Critical failure: predicted no `NO_TRADE`; it trades every sample and is biased toward 432-bar holds.

Strict smoke backtest on those 100 rows:

- Trades: 15
- CAGR: -50.95%
- Strict MDD: 7.05%
- CAGR/strict MDD: -7.22
- Mean trade return: -0.306%
- p approx: 0.252

## Conclusion

This representation is more aligned with the RLLM goal than side-only labels, and the oracle upper bound confirms the action book can express profitable decisions. However, SFT64 did not learn a deployable selector. The model copied the action JSON surface but failed the key live skill: abstaining and choosing side/hold under uncertainty.

## Next direction

Do not scale steps blindly. The next fix should change supervision:

1. Use candidate-level TAKE/SKIP value rows or DPO pairs so NO_TRADE/abstention is directly contrasted against risky trades.
2. Add explicit negative pairs for high-MAE / negative utility long-hold trades.
3. Keep fast KV-cache evaluator, but add partial writes and score-prior subtraction before any full 597-row eval.
