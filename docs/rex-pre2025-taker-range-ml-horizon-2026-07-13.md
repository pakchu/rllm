# Pre-2025 REX ML execution-horizon search (2026-07-13)

## Verdict

Changing only the time exit did not solve 2025.  The 2024 selection favored a
four-hour ridge policy, but that policy failed to generalize.  The original
12-hour ExtraTrees q30 critic remained the strongest later-window candidate.
No alpha was promoted.

## Protocol

- Same pre-2025 taker/range gate and 2021-2023 fitted critics.
- Execution holds tested: 4h, 6h, 8h, 12h, 18h, and 24h.
- Horizon and policy selected on 2024 full/H1/H2.
- 2025 and 2026 opened after the new manifest.
- Manifest hash:
  `939e534367258089693e8900a78f947c1f1618ad85747870b5af3531c7df28e9`
- Corrected strict MDD, 0.5x, 6 bp per side.

## 2024-selected winner

Ridge utility alpha100, q70, both sides, 4-hour hold:

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p approx |
|---|---:|---:|---:|---:|---:|---:|
| 2024 Selection | +6.54% | 6.52% | 1.23% | 5.32 | 29 | <0.001 |
| 2025 Eval | +2.74% | 2.74% | 1.97% | 1.39 | 22 | 0.294 |
| 2026 Holdout | +0.52% | 1.25% | 1.51% | 0.83 | 15 | 0.602 |
| 2025-2026 Combined | +3.27% | 2.30% | 1.97% | 1.17 | 37 | 0.240 |

## Decision

The short hold is a 2024-specific fit.  Longer holds did not improve 2025 path
efficiency either.  Keep the 12-hour ExtraTrees q30 policy as the research base
and test only executable stop/take-profit exits selected on 2024.

## Artifacts

- Search: `training/search_rex_pre2025_gate_ml_alpha.py`
- Tests: `tests/test_search_rex_pre2025_gate_ml_alpha.py`
- Manifest: `results/rex_pre2025_taker_range_ml_horizon_top10_manifest_2026-07-13.json`
- Result: `results/rex_pre2025_taker_range_ml_horizon_alpha_scan_2026-07-13.json`
