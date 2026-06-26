# PAE symbolic ridge min-feature-count sweep — 2026-06-27

## Objective

Check whether PAE-token failure was caused by sparse-token overfit. The first PAE run doubled features from 1,667 to 3,295 and failed/abstained under strict validation.

## Code change

`training/symbolic_action_ridge.py sweep` now exposes:

```bash
--min-feature-count
```

This controls `FeatureSpace.fit(..., min_count=...)` for sweep mode instead of the previous hard-coded `5`.

## Protocol

- train: `data/event_action_verifier_text_v3k8_2024_pae_2026-06-27.jsonl`
- validation: `data/event_action_verifier_text_v3k8_2025_pae_2026-06-27.jsonl`
- holdout: `data/event_action_verifier_text_v3k8_2026_jan_may_pae_2026-06-27.jsonl`
- targets: `utility,net_return,risk_adjusted,tail_risk,distributional_safety`
- alpha: `1000,10000`
- thresholds: `-0.003,-0.001,0,0.001,0.003`
- min gaps: `0,0.0015`
- strict validation: min trades 80, CAGR >=10, ratio >=1, MDD <=15.5, p <=0.25

## Results

| min feature count | features | selected | val CAGR | val MDD | val ratio | val p | holdout CAGR | holdout MDD | holdout ratio | holdout trades |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3295 | `net_return / alpha=10000 / threshold=-0.003` | 17.32% | 13.77% | 1.26 | 0.3675 | 0.00% | 0.00% | 0.00 | 0 strict abstain |
| 50 | 2361 | `net_return / alpha=1000 / threshold=-0.003` | 38.64% | 9.00% | 4.29 | 0.0612 | -30.72% | 20.64% | -1.49 | 158 |
| 100 | 2007 | `net_return / alpha=10000 / threshold=0` | 64.59% | 9.76% | 6.62 | 0.0039 | -6.20% | 20.35% | -0.30 | 157 |
| 200 | 1755 | `distributional_safety / alpha=1000 / threshold=0.003` | 20.35% | 9.52% | 2.14 | 0.1201 | -19.19% | 14.93% | -1.29 | 197 |

## Decision

Sparse-token overfit is not the only problem. Raising min-count can make validation look much stronger, including statistically significant validation at `mfc=100`, but holdout remains negative.

This reinforces the main diagnosis:

- 2025 validation patterns do not transfer to 2026 Jan-May;
- PAE tokens are useful context but not sufficient alpha under linear symbolic ridge;
- validation gates are necessary but not sufficient when the selected family itself is regime-unstable.

Next work should shift from token/ridge tuning to regime-stability labels or new action families, not larger sweeps of the same setup.
