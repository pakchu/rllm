# Price-action event tokens in symbolic ranker — 2026-06-27

## Objective

Use the causal price-action event scan output as LLM/RLLM-style context, not as a standalone rule.

The change adds price-action event tokens to event-action verifier rows and makes the symbolic ridge ranker consume `state_tokens`.

## Code changes

- `training/augment_event_candidate_price_action_events.py`
  - Adds `pae:*` tokens from shifted prior rolling range events.
  - Uses backward-asof join to market features.
  - Does not change reward/action audit labels.
- `training/symbolic_action_ridge.py`
  - `row_tokens()` now consumes `row["state_tokens"]`.
  - Adds side/family/horizon interactions for each state token.

## Data generated

Artifacts are generated/ignored data, not committed:

| file | rows | match rate | event token cols |
|---|---:|---:|---:|
| `data/event_action_verifier_text_v3k8_2024_pae_2026-06-27.jsonl` | 46,848 | 100% | 90 |
| `data/event_action_verifier_text_v3k8_2025_pae_2026-06-27.jsonl` | 42,688 | 100% | 90 |
| `data/event_action_verifier_text_v3k8_2026_jan_may_pae_2026-06-27.jsonl` | 19,104 | 100% | 90 |

Disk stayed below the WSL 300GB limit at about 293GB used.

## Strict recent split comparison

Protocol:

- train: 2024
- validation: 2025
- holdout: 2026 Jan-May
- targets: `utility,net_return,risk_adjusted,tail_risk,distributional_safety`
- alpha: `1000,10000`
- thresholds: `-0.003,-0.001,0,0.001,0.003`
- validation gates: `min_val_trades=80`, `min_val_cagr_pct=10`, `min_val_ratio=1`, `max_val_mdd_pct=15.5`, `max_val_p_value=0.25`
- strict behavior: `--abstain-on-validation-fail`

### Base symbolic ranker

Artifact:

- `results/symbolic_ridge_recent_base_2024_2025_2026jm_strict_2026-06-27/report.json`

Selected:

```json
{"target":"net_return","alpha":10000.0,"threshold":0.0,"min_gap":0.0}
```

| split | CAGR | strict MDD | CAGR/MDD | trades | p approx |
|---|---:|---:|---:|---:|---:|
| val 2025 | 30.10% | 12.86% | 2.34 | 306 | 0.1221 |
| holdout 2026 Jan-May | -15.89% | 18.14% | -0.88 | 159 | 0.5840 |

### PAE-token symbolic ranker, strict

Artifact:

- `results/symbolic_ridge_recent_pae_2024_2025_2026jm_strict_2026-06-27/report.json`

Selected candidate failed validation because p-value was too weak:

```json
{"target":"net_return","alpha":10000.0,"threshold":-0.003,"min_gap":0.0}
```

| split | CAGR | strict MDD | CAGR/MDD | trades | p approx |
|---|---:|---:|---:|---:|---:|
| val 2025 | 17.32% | 13.77% | 1.26 | 311 | 0.3675 |
| holdout 2026 Jan-May | 0.00% | 0.00% | 0.00 | 0 | 1.0000 |

### PAE-token symbolic ranker, relaxed p-value diagnostic

Artifact:

- `results/symbolic_ridge_recent_pae_2024_2025_2026jm_relaxedp_2026-06-27/report.json`

With `max_val_p_value=1.0`, the candidate traded holdout and lost:

| split | CAGR | strict MDD | CAGR/MDD | trades | p approx |
|---|---:|---:|---:|---:|---:|
| val 2025 | 21.86% | 13.77% | 1.59 | 308 | 0.2756 |
| holdout 2026 Jan-May | -21.85% | 21.23% | -1.03 | 155 | 0.4473 |

## Decision

PAE tokens do not solve the alpha problem in this ridge setup.

Observed effect:

- feature count increased from 1,667 to 3,295;
- strict validation rejected PAE candidates due weak statistical evidence;
- relaxing the p-value gate caused a larger 2026 loss than the base model.

This supports the current rule: do not weaken validation gates to force trades.

Next direction:

1. Keep PAE tokens available for LLM context and future feature-compressor experiments.
2. Do not expect linear symbolic ridge to monetize them directly.
3. Search for stronger setup labels or event-family stability targets before Gemma/Gemma4 SFT.
