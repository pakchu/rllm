# LLM context-regime SFT64 checkpoint (2026-06-23)

## What changed

A new single-policy RLLM data path was tested after DXY/Kimchi gates and broad alpha scans failed to generalize.

Data path:

- `training.llm_context_regime_miner`
- Train-only bucket cut points.
- Prompt contains causal bucket tokens only.
- Target uses future path labels for supervised learning only.
- No analyzer/trader cascade.

Generated dataset:

- Full: `data/llm_context_regime_policy_h288_2026-06-23.jsonl`
- Train-only SFT input: `data/llm_context_regime_policy_h288_train_2026-06-23.jsonl`
- Test split: `data/llm_context_regime_policy_h288_test_2026-06-23.jsonl`
- Eval split: `data/llm_context_regime_policy_h288_eval_2026-06-23.jsonl`

Split/action distribution:

| Split | Rows | LONG | SHORT | NO_TRADE |
| --- | ---: | ---: | ---: | ---: |
| train | 6572 | 2583 | 2332 | 1657 |
| test | 2196 | 835 | 739 | 622 |
| eval | 598 | 204 | 230 | 164 |

## Training

Command shape:

```bash
.venv/bin/python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/llm_context_regime_policy_h288_train_2026-06-23.jsonl \
  --output-dir checkpoints/llm_context_regime_gemma4_e4b_sft64_2026-06-23 \
  --max-samples 900 \
  --sample-mode balanced \
  --max-steps 64 \
  --max-seq-length 1536 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32
```

Training summary:

- Model alias resolved to `google/gemma-4-E4B-it`.
- Balanced SFT rows: 900 = 300 LONG / 300 SHORT / 300 NO_TRADE.
- Runtime: 433.9s.
- Final train loss: 0.971 overall; step losses fell from about 4.3 to 0.05-0.08 late in training.
- Mean token accuracy late in training: ~0.96-0.98.

## Evaluation

Test split, balanced 300, candidate-logprob over single `action` key:

- Accuracy: 33.3%.
- Confusion: all predictions were `NO_TRADE`.

Test split, balanced 60, generation mode:

- Accuracy: 26.7%.
- Prediction distribution: mixed LONG/SHORT/NO_TRADE, but no useful alignment with targets.

## Interpretation

This is **not ready for backtesting**. The model learned output formatting, but it did not learn a reliable mapping from causal buckets to oracle best-side labels.

Likely root cause:

1. The label is still too close to a future-path oracle: choose the best side over the next 288 bars if net/gap/MAE passes.
2. The causal token set is descriptive but may not contain enough stable information to infer those path labels at 5m resolution.
3. Candidate-logprob under single-key JSON is format-mismatched with full JSON targets, so it is diagnostic only; generation confirms broader target learnability is still weak.

## Next direction

Do not continue by simply increasing steps. The right next step is to mine **stable context labels** first:

- Group causal token contexts on train.
- Keep only contexts with enough train support and stable test behavior.
- Label actions from context-level expected utility, not one-row oracle best side.
- Then SFT Gemma on these smoother labels and evaluate action accuracy before any PnL backtest.
