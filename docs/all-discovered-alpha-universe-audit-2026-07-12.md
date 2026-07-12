# All-discovered alpha universe audit

Date: 2026-07-12

## Scope

Replayed all 381 sleeves used by the broad portfolio search and hashed their complete Train/2024/2025/2026 return and adverse-excursion paths. Multiple event records belonging to the same sleeve and split are aggregated before hashing.

## Findings

- Sleeves: 381
- Replayed event records: 13,596
- Exact duplicate PnL groups: 19
- Redundant sleeve aliases inside those groups: 71

Examples include:

- `oi_raw` and `oi_high_sel`
- `cand_path_gate_0` and `cand_path_gate_1`
- duplicate jump-volume gates
- duplicate calendar rules whose different stride labels produce the same executed path
- duplicate REX-veto candidates

These aliases must be collapsed before subsequent portfolio optimization. Otherwise the optimizer can allocate separately to the same realised alpha path and overstate diversification or search breadth.

## Current Train-MDD-40 candidate

The three selected sleeves are not exact duplicates and have low realised active-bar overlap.

| pair | active Jaccard | overlap bars | return correlation on overlap |
|---|---:|---:|---:|
| OI/Upbit vs funding/premium | 0.0288 | 4,444 | -0.0262 |
| OI/Upbit vs REX veto | 0.0297 | 2,258 | -0.0575 |
| funding/premium vs REX veto | 0.1048 | 19,702 | 0.1833 |

Therefore the current three-sleeve result is not explained by exact duplicate PnL stacking. It has genuine path diversity. This does **not** make its reported 2025/2026 numbers pristine.

## Provenance status

| sleeve | trades | status | reason |
|---|---:|---|---|
| `oi_upbit_ratio288_low` | 475 | legacy unknown | requires its original candidate-selection audit |
| `new_long_minimal_funding_premium` | 250 | research seen | promoted during iterative 2025/2026 research |
| `cand_rex_veto_7` | 439 | contaminated | source top/TTE family used post-Train selection evidence |

The portfolio may remain a research/live-shadow candidate, but it cannot be described as clean final OOS. A pristine forward window or newly frozen rolling protocol is required.

## Mandatory optimizer changes

1. Collapse exact PnL hashes to one canonical sleeve.
2. Group near-duplicates by active-event Jaccard and return correlation.
3. Apply family-level gross caps, not only per-name weights.
4. Record candidate split provenance and forbid contaminated metrics from being labelled final OOS.
5. Report net unique entries and concurrent gross exposure rather than sleeve trade-count sums.

## Artifacts

- Script: `training/audit_all_discovered_alpha_universe.py`
- Result: `results/all_discovered_alpha_universe_audit_2026-07-12.json`
