# DXY/Kimchi policy SFT export — 2026-06-23

## Purpose

The raw DXY-low/Kimchi RLLM dataset is too imbalanced for direct SFT:

- Raw rows: 2,492.
- Activate labels: 129.
- NO_TRADE labels: 2,363.

This export keeps test/eval untouched and only balances the train split.

## Command

```bash
.venv/bin/python -m training.export_dxy_kimchi_policy_sft \
  --input-jsonl data/dxy_kimchi_regime_policy_sft_2026-06-23.jsonl \
  --train-output data/dxy_kimchi_regime_policy_train_balanced_2026-06-23.jsonl \
  --test-output data/dxy_kimchi_regime_policy_test_2026-06-23.jsonl \
  --eval-output data/dxy_kimchi_regime_policy_eval_2026-06-23.jsonl \
  --summary-output results/dxy_kimchi_regime_policy_export_summary_2026-06-23.json \
  --no-trade-per-activate 3.0 \
  --seed 42
```

Generated JSONL/results are ignored and not committed.

## Output summary

- Raw chronological counts:
  - train: 1,094
  - test: 854
  - eval: 544
- Balanced train:
  - rows: 200
  - activate true: 50
  - activate false: 150
  - action LONG: 31
  - action SHORT: 19
  - action NO_TRADE: 150
- Untouched test:
  - rows: 854
  - activate true: 52
  - activate false: 802
- Untouched eval:
  - rows: 544
  - activate true: 27
  - activate false: 517

## Next use

Use `data/dxy_kimchi_regime_policy_train_balanced_2026-06-23.jsonl` for a small Gemma 4 / text SFT dry run.  Evaluate on untouched test/eval, not on the balanced train distribution.

## Dry-run verification

`train_text_sft.py` compatibility was verified after exporting both chat `messages` and top-level `prompt`/`target` fields:

```bash
.venv/bin/python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/dxy_kimchi_regime_policy_train_balanced_2026-06-23.jsonl \
  --output-dir checkpoints/dxy_kimchi_policy_gemma4_e4b_dryrun_2026-06-23 \
  --max-steps 1 \
  --max-seq-length 2048 \
  --dry-run
```

Result:

- resolved model: `google/gemma-4-E4B-it`
- rows loaded: 200
- target counts: `NO_TRADE=150`, `LONG=31`, `SHORT=19`
- dry_run: true
