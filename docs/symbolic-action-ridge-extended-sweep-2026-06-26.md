# Symbolic action ridge extended sweep — 2026-06-26

## Objective

Retest the strongest prior symbolic/action-value line under stricter train/val/holdout discipline, after the h36 path-shape token probe showed regime-flipping side labels.

The intent is to use LLM-style symbolic prompts as a compressed feature surface, but select actions with a cheap auditable ridge ranker before spending Gemma/Gemma4 fine-tuning budget.

## Code change

`training/symbolic_action_ridge.py` sweep now supports:

- configurable target list: `--targets utility,net_return,risk_adjusted,tail_risk,distributional_safety`
- configurable alpha/threshold/gap grids
- validation gates: `--min-val-cagr-pct`, `--min-val-ratio`, `--max-val-mdd-pct`, `--max-val-p-value`
- strict deployment behavior: `--abstain-on-validation-fail`

The strict option matters: if no validation candidate passes, untouched holdout must be all `NO_TRADE`, not the best failed validation candidate.

## Split A: train 2020-2023, select on 2024, holdout 2025

Command artifact:

- `results/symbolic_action_ridge_extended_train2020_2023_val2024_hold2025/report.json`

Selected config:

```json
{"target":"net_return","alpha":10000.0,"threshold":0.003,"min_gap":0.0}
```

| split | CAGR | strict MDD | CAGR/MDD | trades | p approx |
|---|---:|---:|---:|---:|---:|
| val 2024 | 51.46% | 11.01% | 4.67 | 215 | 0.0241 |
| holdout 2025 | -15.40% | 21.23% | -0.73 | 193 | 0.3936 |

Interpretation: this reproduces why the older result looked promising. It genuinely passed on 2024, but failed immediately on 2025. Do not treat the 2024 target-hit as durable alpha.

## Split B: train 2020-2024, select on 2025, holdout 2026 Jan-May

Non-strict artifact for research comparison:

- `results/symbolic_action_ridge_extended_train2020_2024_val2025_hold2026janmay/report.json`

Strict artifact:

- `results/symbolic_action_ridge_extended_train2020_2024_val2025_hold2026janmay_strict/report.json`

Selected candidate before strict abstention:

```json
{"target":"utility","alpha":10000.0,"threshold":0.001,"min_gap":0.0015}
```

Validation failed because CAGR was below the configured minimum:

```text
validation_reject_reasons = ["cagr_below_min"]
```

| mode | split | CAGR | strict MDD | CAGR/MDD | trades | p approx |
|---|---|---:|---:|---:|---:|---:|
| non-strict research | val 2025 | 8.34% | 5.49% | 1.52 | 132 | 0.2327 |
| non-strict research | holdout 2026 Jan-May | 4.28% | 4.25% | 1.01 | 92 | 0.7251 |
| strict deployable | holdout 2026 Jan-May | 0.00% | 0.00% | 0.00 | 0 | n/a |

Interpretation: 2025 validation was not good enough to justify trading 2026. The non-strict holdout happened to be slightly positive but statistically weak and not deployable under the stated standard.

## Decision

This line has a weak structural signal but still does not meet the target:

- It can find profitable 2024 setups.
- The selected 2024 policy does not survive 2025.
- The 2025-selected policy is too weak to authorize 2026 trading.

Next work should not be another threshold sweep. The useful part is the LLM-style symbolic action representation; the missing part is a stable regime/context selector that predicts when the action-value surface is valid.

Concrete next direction:

1. Build a regime-conditioned reliability target from prior windows only: whether the symbolic action ranker should be `trade`, `invert`, or `abstain` for the next period.
2. Train/evaluate that selector on rolling periods before touching eval.
3. Only after the selector produces a non-zero strict holdout with acceptable MDD should Gemma/Gemma4 be used as a compressor over the symbolic prompt.
