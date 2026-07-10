# REX Qwen2.5-1.5B live-size SFT comparison (2026-07-10)

## Purpose

Train a 3060Ti-friendly REX selector model and compare it against the previous small Qwen checkpoint, prior Gemma4 checkpoint, and the no-LLM REX rule gate.

## Model / training

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter: `checkpoints/rex_regime_thesis_qwen25_1p5b_lora_s160_1024balanced_2026-07-10`
- Rows: 1,024 balanced rows (`TRADE` 512 / `ABSTAIN` 512)
- Dataset: `data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl`
- LoRA: r=16, alpha=32, dropout=0.05
- Steps: 160
- Runtime on RTX 5090 host: 499.8s
- Final reported train loss: 0.1149
- Intermediate `checkpoint-160` was deleted after final adapter save; final adapter directory is 82MB.

## Protocol

- Evaluation mode: `TRADE` vs `ABSTAIN` candidate logprob, `score_normalization=sum`.
- Backtest: `training.backtest_decision_label_predictions`, 1.0x leverage row shown below for apples-to-apples selector comparison.
- Entry: signal+1 open.
- Cost: fee 4bp + slippage 1bp per leg via existing strict backtest config.
- CAGR uses full split window including idle time.
- Strict MDD includes intraposition adverse excursion.

## Accuracy

| model | split | rows | accuracy | confusion |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-1.5B s160/1024 | 2025 test | 132 | 99.24% | ABSTAIN→TRADE 1, TRADE→TRADE 38, ABSTAIN→ABSTAIN 93 |
| Qwen2.5-1.5B s160/1024 | 2026H1 eval | 100 | 98.00% | ABSTAIN→TRADE 1, TRADE→ABSTAIN 1, TRADE→TRADE 38, ABSTAIN→ABSTAIN 60 |

## 1.0x backtest comparison

| model | split | pred TRADE/ABSTAIN | abs return | CAGR | strict MDD | CAGR/MDD | trades | p approx |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B s160/1024 new | 2025 test | 39 / 93 | 36.01% | 41.76% | 5.05% | 8.27 | 28 | 0.00012 |
| Qwen2.5-1.5B s80/512 old | 2025 test | 27 / 105 | 24.36% | 29.07% | 5.47% | 5.31 | 22 | 0.00463 |
| Gemma4 E4B s32 old | 2025 test | 49 / 83 | 43.87% | 51.27% | 5.06% | 10.12 | 33 | 0.00003 |
| Qwen2.5-1.5B s160/1024 new | 2026H1 eval | 39 / 61 | 8.05% | 36.00% | 7.10% | 5.07 | 21 | 0.28732 |
| Qwen2.5-1.5B s80/512 old | 2026H1 eval | 27 / 73 | 7.71% | 32.72% | 5.01% | 6.54 | 19 | 0.15551 |
| Gemma4 E4B s32 old | 2026H1 eval | 47 / 53 | 9.51% | 41.31% | 7.15% | 5.78 | 25 | 0.25263 |

## No-LLM REX rule reference

From `docs/rex-no-llm-standalone-check-2026-07-10.md`:

| baseline | split | abs return | CAGR | strict MDD | CAGR/MDD | trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no-LLM rule_gate | 2025 | 38.77% | 38.80% | 5.12% | 7.57 | 33 |
| no-LLM rule_gate | 2026YTD | 11.02% | 28.33% | 7.37% | 3.84 | 24 |

## Readout

- The new 1.5B model is runnable-size and clearly improved over the previous 1.5B checkpoint on 2025 test: higher return, higher CAGR/MDD, similar strict MDD, and stronger p-value.
- On 2026H1 eval, the new 1.5B checkpoint improves absolute return and CAGR over the old 1.5B checkpoint but worsens strict MDD, CAGR/MDD, and p-value. It is not a statistically stronger eval model.
- Gemma4 remains slightly stronger on absolute 2026H1 return, but not by enough to justify live friction on an 8GB 3060Ti host without a separate loading/latency proof.
- The no-LLM rule gate is still a hard baseline: the small LLM is mostly learning/reproducing the REX rule prior, not yet adding a statistically clean new edge.

## Decision

Use `Qwen2.5-1.5B s160/1024` only as a shadow/live-size candidate. Do not replace the no-LLM rule gate yet. For live promotion, require a selector threshold or calibration layer that improves 2026H1 p-value/trade quality without increasing stale/live risk.

## Artifacts

- Adapter: `checkpoints/rex_regime_thesis_qwen25_1p5b_lora_s160_1024balanced_2026-07-10`
- Test eval: `results/rex_regime_thesis_qwen25_1p5b_s160_1024balanced_test_2025_decision_sum_2026-07-10.json`
- Eval eval: `results/rex_regime_thesis_qwen25_1p5b_s160_1024balanced_eval_2026h1_decision_sum_2026-07-10.json`
- Test backtest: `results/rex_regime_thesis_qwen25_1p5b_s160_1024balanced_test_2025_decision_sum_backtest_2026-07-10.json`
- Eval backtest: `results/rex_regime_thesis_qwen25_1p5b_s160_1024balanced_eval_2026h1_decision_sum_backtest_2026-07-10.json`
