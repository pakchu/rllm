# Side-map reliability SFT data (2026-06-23)

## Purpose

The monthly side-map audit showed that 2026 is not uniformly inverse. A separate reliability head needs to classify each target month as:

- `normal`: trust base side mapping;
- `inverse`: flip base side mapping;
- `unreliable`: avoid trading the base side map.

This document records the first SFT-style dataset for that head.

## Implementation

Added:

- `training/build_side_map_reliability_sft.py`
- `tests/test_build_side_map_reliability_sft.py`

Prompt inputs are prior-only:

- target month;
- bucketed prior validation score for the target month;
- previous N months of realized side-map labels and coarse pass/invert return buckets.

Target is the current month audit label from `monthly_side_map_reliability_audit.py`.

Leakage distinction:

- prompt does not include target-month outcomes;
- target label is derived from target-month audit and is for training/evaluation only;
- the dataset is not a live selector until a rolling model predicts eval months from prior-only prompts.

## Generated dataset

Command output:

- `data/side_map_reliability_sft_h288_2026-06-23.jsonl`
- `results/side_map_reliability_sft_h288_summary_2026-06-23.json`

Rows: 29 months.

Split/class distribution:

| Split | normal | inverse | unreliable |
| --- | ---: | ---: | ---: |
| train 2024 | 9 | 1 | 2 |
| val 2025 | 5 | 5 | 2 |
| eval 2026-01..2026-05 | 1 | 1 | 3 |

## Decision

This is the right target shape for returning to LLM/RLLM, but it is too small for a high-capacity fine-tune by itself.

Next step should be one of:

1. Extend monthly labels back before 2024 by generating earlier rolling predictions, increasing side-map head data.
2. Convert the head into a rule/nearest-neighbor memory first, then distill into Gemma after more labels exist.
3. Add richer prior-only monthly state from wave_trading/binance aux data before any fine-tune.

Do not fine-tune Gemma on these 29 rows and expect robust eval performance.
