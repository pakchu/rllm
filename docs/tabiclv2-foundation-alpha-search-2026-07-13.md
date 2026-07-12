# TabICLv2 foundation-model alpha search

Date: 2026-07-13

## Summary

TabICLv2 was tested as a recent tabular foundation model for standalone 48-hour
BTC long/short alpha discovery. A pre-evaluation Top-10 manifest was frozen from
the 2023 holdout before 2024-2026 metrics were computed.

The fixed 2020-2022 models did not produce a live-grade candidate. An
ExtraTrees baseline inside the same frozen Top-10 generalized through 2025 but
lost its edge in 2026, indicating model-age/drift rather than a durable static
predictor.

Official implementation: <https://github.com/soda-inria/tabicl>.

## Protocol

- Causal 5-minute feature frame; no date feature.
- Signal anchors every 72 five-minute bars (6 hours).
- Label: next 576-bar (48-hour) open-to-open log return.
- Entry: next 5-minute open.
- Fit: 2020-2022 only, with label-crossing samples purged at the boundary.
- Score threshold and Top-10 rank: 2023 holdout only.
- The manifest was written before Test 2024, Eval 2025 or 2026 were simulated.
- Strict backtest: 6 bp/side, 0.5x, full-window CAGR, intraposition strict MDD.
- 4,376 fit anchors, 1,452 holdout anchors, 1,456 Test-2024 anchors,
  1,452 Eval-2025 anchors and 594 2026 anchors.

Models:

- TabICLv2 regression, four estimators;
- HistGradientBoosting regression;
- ExtraTrees regression.

Feature groups:

- compact: 52 features;
- price-only: 80 features;
- full: 100 features after removing availability flags.

The search tested seven model/feature combinations and 15 fixed score policies
per model. Twenty-six candidates passed basic 2023 requirements; ten distinct
signals were frozen.

## Foundation-model diagnostics

| model | features | 2023 Spearman | direction accuracy |
|---|---:|---:|---:|
| TabICLv2 compact | 52 | -0.072 | 48.76% |
| TabICLv2 price | 80 | -0.048 | 49.52% |
| TabICLv2 full | 100 | -0.036 | 51.52% |
| HistGB full | 100 | +0.028 | 51.58% |
| ExtraTrees full | 100 | +0.029 | 52.41% |

The pretrained TabICLv2 prior did not align with 48-hour BTC return ranking in
the 2023 holdout. Its high-ranked policies were path-profitable in 2024 but not
stable in 2025.

## Best Top-10 outcomes

### ExtraTrees full, rank 3, long top 20%

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +31.08% | 31.01% | 7.04% | 4.40 | 32 |
| Eval 2025 | +22.28% | 22.30% | 5.67% | 3.93 | 40 |
| 2026 YTD | +1.82% | 4.43% | 5.02% | 0.88 | 24 |

This is the only static Top-10 member that passed both 2024 and 2025 with enough
trades. It fails the 2026 live-grade criterion.

### TabICLv2 price, rank 6, long top 30%

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +82.84% | 82.62% | 7.43% | 11.12 | 70 |
| Eval 2025 | +7.22% | 7.23% | 9.86% | 0.73 | 65 |
| 2026 YTD | -3.10% | -7.28% | 11.61% | -0.63 | 27 |

The large 2024 return is not persistent and is rejected.

## Verdict

- No fixed-model live-grade alpha.
- Static TabICLv2 is rejected for this target/feature formulation.
- ExtraTrees demonstrates a real 2024-2025 effect but decays in 2026.
- The next justified experiment is an algorithmically frozen annual
  retraining/calibration schedule, using only data that would have been known
  before each evaluation year.

## Artifacts

- Search: `training/search_tabicl_foundation_alpha.py`
- Frozen manifest: `results/tabicl_top10_manifest_2026-07-13.json`
- Result: `results/tabicl_foundation_alpha_scan_2026-07-13.json`
- Tests: `tests/test_tabicl_foundation_alpha.py`
