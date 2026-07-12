# Pre-2025 taker/range REX ML alpha (2026-07-13)

## Verdict

The frozen sparse critic improved the later-window edge, but did not pass the
per-year alpha target because 2025 remained below `CAGR / strict MDD = 3`.
No candidate was promoted yet.

## Protocol

- Base gate: Top-10 rank 8 from the corrected 2021-2023 + 2024 conjunction
  scan (`taker_imbalance` low and seven-day REX range position not high).
- Model fit: completed 12-hour paths in 2021-2023.
- Selection: 2024 full/H1/H2.
- 2025 and 2026 files opened only after the Top-10 manifest.
- Manifest hash:
  `5846f30d40c469953dd170774392df67555e96b5879165e1828360dd7a6d127b`
- Costs: 0.5x and 6 bp per side.
- Full-window CAGR and corrected intraposition high-water strict MDD.

## Best later-window critic

ExtraTrees TAKE-probability critic, q30, both sides:

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p approx |
|---|---:|---:|---:|---:|---:|---:|
| 2024 Selection | +11.42% | 11.40% | 7.26% | 1.57 | 44 | 0.135 |
| 2025 Eval | +4.84% | 4.84% | 2.55% | 1.90 | 13 | 0.094 |
| 2026 Holdout | +7.07% | 17.83% | 1.74% | 10.27 | 13 | 0.005 |
| 2025-2026 Combined | +12.24% | 8.50% | 2.55% | 3.33 | 26 | 0.002 |

## Interpretation

The model sharply improves statistical quality over the raw gate and preserves
the 2026 edge.  The remaining miss is path efficiency in 2025, not sign or
combined-window significance.  The supervised target and execution both use a
fixed 12-hour hold, so the next bounded experiment freezes the model and lets
2024 select among shorter and longer execution horizons before reopening the
same later windows.

## Artifacts

- Search: `training/search_rex_pre2025_gate_ml_alpha.py`
- Tests: `tests/test_search_rex_pre2025_gate_ml_alpha.py`
- Manifest: `results/rex_pre2025_taker_range_ml_top10_manifest_2026-07-13.json`
- Result: `results/rex_pre2025_taker_range_ml_alpha_scan_2026-07-13.json`
