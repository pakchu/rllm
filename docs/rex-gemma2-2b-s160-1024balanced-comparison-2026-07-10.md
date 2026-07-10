# REX Gemma2-2B 4bit LoRA comparison (2026-07-10)

## Purpose

Check whether a small Gemma-family LLM can replace or complement the current live-size REX selector on a 3060Ti-class deployment path.

## Protocol

- Base model: `google/gemma-2-2b-it`.
- Fine-tune: 4bit QLoRA-style SFT via `training/train_text_sft.py`.
- Train data: `data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl` only.
- Sample: 1,024 balanced rows (`TRADE` 512 / `ABSTAIN` 512).
- Steps: 160, LoRA r=16, alpha=32, dropout=0.05, LR=2e-5, max sequence length 1280.
- Adapter: `checkpoints/rex_regime_thesis_gemma2_2b_lora_s160_1024balanced_4bit_2026-07-10`.
- Evaluation: held-out 2025 test and 2026H1 eval predictions, followed by generated-action OHLC bar-by-bar strict-MDD backtest with entry delay 1 bar.
- CAGR denominator: full calendar window, including idle/no-trade periods.

## 1.0x backtest comparison

| Model / selector | Split | Abs return | Full-window CAGR | Strict MDD | CAGR / strict MDD | Trades | Mean-ret p-value |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma2-2B 4bit s160/1024 | 2025 test | 35.34% | 40.97% | 5.05% | 8.12 | 27 | 0.000151 |
| Gemma2-2B 4bit s160/1024 | 2026H1 eval | 11.10% | 51.95% | 7.15% | 7.26 | 22 | 0.170 |
| Qwen2.5-1.5B s160/1024 | 2025 test | 36.01% | 41.76% | 5.05% | 8.27 | 28 | 0.000120 |
| Qwen2.5-1.5B s160/1024 | 2026H1 eval | 8.05% | 36.00% | 7.10% | 5.07 | 21 | 0.287 |
| Qwen2.5-1.5B s80/512 recheck | 2025 test | 24.36% | 29.07% | 5.47% | 5.31 | 22 | 0.00463 |
| Qwen2.5-1.5B s80/512 recheck | 2026H1 eval | 7.71% | 32.72% | 5.01% | 6.54 | 19 | 0.156 |
| Gemma4 E4B s32/512 historical | 2025 test | 43.87% | 51.27% | 5.06% | 10.12 | 33 | 0.0000285 |
| Gemma4 E4B s32/512 historical | 2026H1 eval | 9.51% | 41.31% | 7.15% | 5.78 | 25 | 0.253 |
| no-LLM rule gate reference | 2025 test | 38.77% | 38.80% | 5.12% | 7.57 | 33 | n/a |
| no-LLM rule gate reference | 2026YTD | 11.02% | 28.33% | 7.37% | 3.84 | 24 | n/a |

## Label accuracy

| Model | Split | Rows | Accuracy | Confusion summary |
|---|---:|---:|---:|---|
| Gemma2-2B 4bit s160/1024 | 2025 test | 132 | 99.24% | ABSTAIN→ABSTAIN 94, TRADE→ABSTAIN 1, TRADE→TRADE 37 |
| Gemma2-2B 4bit s160/1024 | 2026H1 eval | 100 | 99.00% | ABSTAIN→ABSTAIN 61, TRADE→ABSTAIN 1, TRADE→TRADE 38 |

## Interpretation

- Gemma2-2B 4bit is a viable small Gemma candidate. On 2026H1 it beats the fresh Qwen2.5-1.5B s160 selector on absolute return, full-window CAGR, and CAGR/strict-MDD ratio.
- It does **not** yet prove a statistically strong live edge: 2026H1 has only 22 trades and mean trade return p≈0.17.
- The 2025 result is strong and consistent with the REX gate family, but the 2026H1 statistical weakness means this should be treated as a shadow selector candidate, not an automatic live replacement.
- The no-LLM rule gate remains important as a transparent baseline. The LLM variants mostly learn to preserve/trim that gate, not invent a fresh independent alpha.

## Deployment implication

- For 3060Ti-class hardware, Gemma2-2B with 4bit loading is a more realistic Gemma-family target than the earlier Gemma4 E4B path.
- Qwen2.5-1.5B remains lighter and operationally safer, but Gemma2-2B is worth keeping as the best small-Gemma shadow candidate from this run.

## Artifacts

- Adapter: `checkpoints/rex_regime_thesis_gemma2_2b_lora_s160_1024balanced_4bit_2026-07-10`
- 2025 predictions: `results/rex_regime_thesis_gemma2_2b_s160_1024balanced_test_2025_decision_sum_predictions_2026-07-10.jsonl`
- 2025 backtest: `results/rex_regime_thesis_gemma2_2b_s160_1024balanced_test_2025_decision_sum_backtest_2026-07-10.json`
- 2026H1 predictions: `results/rex_regime_thesis_gemma2_2b_s160_1024balanced_eval_2026h1_decision_sum_predictions_2026-07-10.jsonl`
- 2026H1 backtest: `results/rex_regime_thesis_gemma2_2b_s160_1024balanced_eval_2026h1_decision_sum_backtest_2026-07-10.json`

## Verification commands

- `HfApi().model_info("google/gemma-2-2b-it")` succeeded after terms/auth refresh.
- `training/train_text_sft.py` completed for Gemma2-2B 4bit, 160 steps, 1,024 balanced rows.
- `training/eval_text_label.py` completed for 2025 test and 2026H1 eval.
- `training/backtest_decision_label_predictions.py` completed for 2025 test and 2026H1 eval.

## Cleanup

- Removed intermediate Trainer `checkpoint-*` directories under `checkpoints/`, freeing about 8.2GB while keeping final adapter directories.
- Removed the aborted Gemma4 E4B 4bit retry directory because it only contained partial run metadata and no useful adapter.
