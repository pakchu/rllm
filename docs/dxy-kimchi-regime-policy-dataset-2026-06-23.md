# DXY-low / Kimchi RLLM policy dataset — 2026-06-23

## Purpose

Convert the strongest reset-path family into an RLLM-friendly supervised policy dataset:

- Prior family: `dxy_zscore:low -> kimchi_premium_zscore`.
- Horizon: 144 bars.
- Train-fitted thresholds only:
  - DXY low threshold: `-0.3410015673`.
  - Kimchi low threshold: `-0.7480158502`.
  - Kimchi high threshold: `1.2292722084`.
  - Direction mapping: high Kimchi => `LONG`, low Kimchi => `SHORT`.

The prompt contains causal text state plus the frozen prior signal.  The target JSON asks whether to activate or abstain.  Future path reward is used only for SFT labels and audit metadata, not in prompts.

## Generation command

```bash
.venv/bin/python -m training.dxy_kimchi_regime_policy_dataset \
  --market-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output data/dxy_kimchi_regime_policy_sft_2026-06-23.jsonl \
  --summary-output results/dxy_kimchi_regime_policy_sft_summary_2026-06-23.json \
  --sample-output results/dxy_kimchi_regime_policy_sft_sample_2026-06-23.jsonl \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --external-tolerance 30min \
  --binance-funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --binance-premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --train-start 2023-01-01 \
  --train-end '2024-06-30 23:59:59' \
  --test-start 2024-07-01 \
  --test-end '2025-08-31 23:59:59' \
  --eval-start 2025-09-01 \
  --eval-end '2026-05-31 15:00:00' \
  --horizon 144 \
  --stride-bars 144 \
  --leverage 0.5
```

Generated JSONL/results are ignored and not committed.

## Summary

- Rows: 2,492.
- Split counts:
  - train: 1,094
  - test: 854
  - eval: 544
- Prior signal counts:
  - LONG: 176
  - SHORT: 167
  - NONE: 2,149
- Target action counts:
  - LONG: 74
  - SHORT: 55
  - NO_TRADE: 2,363
- Activate labels:
  - True: 129
  - False: 2,363
- Prompt length: ~1.3k chars.

## Immediate issue

The activation label is highly imbalanced.  Fine-tuning directly on the raw JSONL will likely learn `NO_TRADE` too strongly.  Next step should export balanced chat-SFT splits that oversample activation rows or downsample no-signal rows while preserving chronological test/eval files for evaluation.
