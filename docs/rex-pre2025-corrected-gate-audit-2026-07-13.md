# Pre-2025 REX gate audit with corrected MDD (2026-07-13)

## Verdict

The previously highlighted REX8640/USDKRW gate did not survive 2026.  A
different candidate inside the train+2024-selected Top-10 did generalize across
the combined 2025-2026 window and is promoted to the next research stage, not
to live trading.

## Protocol

- Candidate source: REX reclaim q75, 12-hour hold.
- Primitive thresholds: train feature quantiles.
- Conjunction ranking: 2021-2023 train plus 2024 selection only.
- Replay: 2025 and 2026 are split and reported separately.
- Costs: 0.5x and 6 bp per side.
- CAGR: full configured calendar windows including idle time.
- strict MDD: corrected favorable-to-adverse intraposition high-water path.
- 2024 is selection evidence, not OOS proof.

## REX8640 width + USDKRW gate

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p approx |
|---|---:|---:|---:|---:|---:|---:|
| 2024 Selection | +19.73% | 19.68% | 5.12% | 3.85 | 40 | 0.008 |
| 2025 Eval | +5.70% | 5.70% | 1.50% | 3.81 | 13 | 0.063 |
| 2026 Holdout | +0.81% | 1.96% | 3.31% | 0.59 | 17 | 0.816 |
| 2025-2026 Combined | +6.56% | 4.59% | 3.31% | 1.39 | 30 | 0.182 |

This candidate is rejected because its 2026 edge disappears.

## Top-10 rank 8: taker-low + range-position gate

- `taker_imbalance <= -0.07073595391836504`
- `rex_2016_range_pos <= 0.6865011402825759`

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p approx |
|---|---:|---:|---:|---:|---:|---:|
| 2024 Selection | +12.91% | 12.88% | 6.22% | 2.07 | 65 | 0.135 |
| 2025 Eval | +4.51% | 4.51% | 2.63% | 1.71 | 31 | 0.247 |
| 2026 Holdout | +6.98% | 17.60% | 1.74% | 10.14 | 23 | 0.024 |
| 2025-2026 Combined | +11.80% | 8.20% | 2.63% | 3.12 | 54 | 0.021 |

Short-only contributes most of 2026, but both-side execution is stronger over
the complete later window.  The combined ratio and approximate p-value justify
an ML refinement stage.  The remaining defect is concentrated 2025 path noise,
so the next model is fitted through 2023, selected in 2024, and frozen before
opening 2025-2026.

## Artifacts

- Corrected conjunction scan:
  `results/rex_pre2025_conjunctive_gate_corrected_scan_2026-07-13.json`
- Fixed replay tool: `training/audit_rex8640_usdkrw_gate.py`
- Tests: `tests/test_audit_rex8640_usdkrw_gate.py`
- REX8640 replay: `results/rex8640_usdkrw_gate_strict_audit_2026-07-13.json`
- Taker/range replay: `results/rex_taker_rangepos_gate_strict_audit_2026-07-13.json`
