# REX Gemma4 12B Coder LoRA trading SFT comparison (2026-07-10)

## Purpose

Train the Hugging Face `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF` family as a REX trading selector and compare strict trading stats against the smaller REX LLM baselines.

## Training protocol

- Base model: `models/hf/gemma4-12b-coder-patched`
  - Local patch maps upstream `gemma4_unified` / `gemma4_unified_text` config names to the local Transformers Gemma4 classes.
- Adapter: `checkpoints/rex_regime_thesis_gemma4_12b_coder_patched_lora_s160_1024balanced_4bit_textloss_2026-07-10`
- Train data: `data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl`
- Holdout/test: `data/rex_regime_thesis_range_kimchi_label_test_2025.jsonl`
- Eval: `data/rex_regime_thesis_range_kimchi_label_eval_2026h1.jsonl`
- Samples: 1,024 balanced (`TRADE=512`, `ABSTAIN=512`)
- Steps: 160, batch 1, grad accumulation 8, LR `2e-5`
- LoRA: r=8, alpha=16, dropout=0.05
- Target modules: language-model attention and MLP projections only
- Quantization: 4-bit NF4 for training and eval
- Loss mode: whole-text loss (`--no-completion-only-loss`)
  - Completion-only masking currently mismatches this patched Gemma4 tokenizer/template path, so the run is usable as a smoke/trading comparison but not the final desired training objective.
- Training runtime: ~1,270s (~21m10s), train loss 1.455, final token accuracy around 0.90.

## 1.0x strict trading stats

All CAGR values use the evaluated calendar/window span rather than only active position time. Strict MDD is bar-by-bar adverse excursion from generated actions on actual OHLC bars.

| Model / scoring | Period | Predictions | Abs return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean-ret p approx | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gemma4 12B s160 sum | 2025 test | 132 TRADE / 0 ABSTAIN | 38.46% | 44.09% | 9.46% | 4.66 | 76 | 0.0123 | invalid selector: all-trade collapse |
| Gemma4 12B s160 sum | 2026H1 eval | 100 TRADE / 0 ABSTAIN | 12.11% | 51.38% | 8.46% | 6.07 | 43 | 0.2775 | invalid selector: all-trade collapse, weak p |
| Gemma4 12B s160 mean | 2025 test | 130 TRADE / 2 ABSTAIN | 39.27% | 45.03% | 9.46% | 4.76 | 76 | 0.0108 | near all-trade collapse |
| Gemma4 12B s160 mean | 2026H1 eval | 97 TRADE / 3 ABSTAIN | 13.06% | 56.10% | 8.12% | 6.91 | 43 | 0.2524 | near all-trade collapse, weak p |
| Gemma2 2B s160 | 2025 test | 37 TRADE / 95 ABSTAIN | 35.34% | 40.97% | 5.05% | 8.12 | 27 | 0.000151 | stronger selector |
| Gemma2 2B s160 | 2026H1 eval | 38 TRADE / 62 ABSTAIN | 11.10% | 51.95% | 7.15% | 7.26 | 22 | 0.170 | better risk/selector balance |
| Qwen2.5 1.5B s160 | 2025 test | 39 TRADE / 93 ABSTAIN | 36.01% | 41.76% | 5.05% | 8.27 | 28 | 0.000120 | strongest compact baseline |
| Qwen2.5 1.5B s160 | 2026H1 eval | 39 TRADE / 61 ABSTAIN | 8.05% | 36.00% | 7.10% | 5.07 | 21 | 0.287 | weaker eval return, still selective |
| Historical Gemma4 E4B s32 | 2025 test | n/a | 43.87% | 51.27% | 5.06% | 10.12 | 33 | 0.0000285 | best historical 2025 stat |
| Historical Gemma4 E4B s32 | 2026H1 eval | n/a | 9.51% | 41.31% | 7.15% | 5.78 | 25 | 0.253 | selective but weak p |
| no-LLM REX rule | 2025 test | n/a | 38.77% | 38.80% | 5.12% | 7.57 | 33 | n/a | strong non-LLM reference |
| no-LLM REX rule | 2026 YTD | n/a | 11.02% | 28.33% | 7.37% | 3.84 | 24 | n/a | strong non-LLM reference |

## Interpretation

The 12B model trained successfully, but the trading selector behavior is worse than the smaller baselines. Both sum and mean candidate scoring collapse toward almost always `TRADE`. The absolute returns and CAGR/MDD therefore look acceptable on paper, but they are mostly measuring the underlying REX action stream with little useful LLM gating.

The smaller Gemma2/Qwen/Gemma4-E4B variants are better for this REX selector role because they keep ABSTAIN selectivity and roughly halve strict MDD while preserving similar returns. The 12B coder model is not recommended as the current live REX selector unless we fix label calibration / completion-only loss / abstention objective.

## Artifacts

- Adapter: `checkpoints/rex_regime_thesis_gemma4_12b_coder_patched_lora_s160_1024balanced_4bit_textloss_2026-07-10`
- Training log: `logs/train_rex_gemma4_12b_coder_patched_lora_s160_1024balanced_4bit_textloss_2026-07-10.log`
- Sum eval reports:
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_test_2025_decision_sum_2026-07-10.json`
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_eval_2026h1_decision_sum_2026-07-10.json`
- Mean eval reports:
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_test_2025_decision_mean_2026-07-10.json`
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_eval_2026h1_decision_mean_2026-07-10.json`
- Backtests:
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_test_2025_decision_sum_backtest_2026-07-10.json`
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_eval_2026h1_decision_sum_backtest_2026-07-10.json`
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_test_2025_decision_mean_backtest_2026-07-10.json`
  - `results/rex_regime_thesis_gemma4_12b_coder_s160_1024balanced_eval_2026h1_decision_mean_backtest_2026-07-10.json`

## Next fixes before using this model family

1. Restore true completion-only label loss for Gemma4 patched chat templates.
2. Add an explicit abstention/calibration objective, not just balanced label sampling.
3. Evaluate a compact Gemma4 text model first; the 12B coder model is too heavy and did not improve selector quality.
4. Keep the 3060Ti live path on compact models unless the 12B path proves a real selectivity edge.
