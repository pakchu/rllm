# Side-balanced pre-2024 REX ML alpha (2026-07-13)

## Verdict

Reserving five Top-10 slots for long specialists and five for short specialists
did not create a standalone all-regime alpha.  It did expose complementary
regime behavior that the global ranking had hidden: a long specialist worked in
2024, while a conservative short utility critic worked in 2025-2026.

## Frozen protocol

- Same 2021-2022 candidate-threshold fit and completed-path model fit as the
  global sparse REX experiment.
- 2023 ranks long and short specialists independently.
- Manifest slots: five `side=long`, five `side=short`.
- Pre-future manifest hash:
  `b286354490bfb2f8e815b4c55d3cc2de5c7398224fd4d373c1128cfb51b8c456`
- Pre-2024 candidates and OHLC are physically bounded before the manifest;
  full OHLC and future candidate files are opened afterward.
- 0.5x leverage and 6 bp per side total cost.
- Full-window CAGR and corrected intraposition high-water strict MDD.

## Complementary specialists

| Specialist | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| ExtraTrees TAKE q80, long, all families | 2024 Test | +16.34% | 16.31% | 5.11% | 3.19 | 41 |
| same | 2025 Eval | +3.27% | 3.27% | 1.49% | 2.20 | 13 |
| same | 2026 YTD | -0.64% | -1.54% | 1.20% | -1.28 | 3 |
| ExtraTrees utility q70, short, all families | 2024 Test | -2.48% | -2.47% | 5.64% | -0.44 | 12 |
| same | 2025 Eval | +7.43% | 7.43% | 3.87% | 1.92 | 26 |
| same | 2026 YTD | +4.80% | 11.93% | 2.01% | 5.95 | 11 |

## Interpretation

The REX critic is not a universal directional alpha.  It behaves as two weak,
complementary specialists whose useful side changes with the slow market state.
Selecting one side globally guarantees regime failure.

The next experiment must combine only the frozen 2023 specialists and route
between them using a signal-time long-horizon price state such as the four-week
return or 30-day range location.  The router must be selected on 2023 and must
not use future realized rewards at execution time.

## Artifacts

- Search: `training/search_rex_pre2024_ml_alpha.py`
- Tests: `tests/test_search_rex_pre2024_ml_alpha.py`
- Manifest: `results/rex_pre2024_ml_side_balanced_top10_manifest_2026-07-13.json`
- Result: `results/rex_pre2024_ml_side_balanced_alpha_scan_2026-07-13.json`
