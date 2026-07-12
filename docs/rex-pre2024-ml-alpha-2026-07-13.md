# Pre-2024 sparse REX ML alpha (2026-07-13)

## Verdict

The first leak-safe sparse-event critic did not produce a standalone alpha.
All 192 policies were selected without 2024-2026 data, but the 2023 ranking
collapsed toward long policies.  Those policies weakened in 2025 and failed in
2026 as the candidate-side distribution shifted bearish.  No alpha or live
candidate was promoted.

## Protocol

- REX candidate thresholds: fitted on 2021-2022 feature history only.
- ML fit: labels whose complete 12-hour path exits strictly before 2023.
- Models: regularized ridge, shallow histogram gradient boosting, and shallow
  extra-trees critics over 46 signal-time REX/price/flow/macro features.
- Selection: 2023 full/H1/H2, then physical Top-10 manifest write.
- Future files: opened only after the manifest write.
- Manifest hash:
  `b395d02f86882e8aefd5a497ea0a922f7905b234c35726a7281aa7745955bd01`
- Both the candidate JSONL and OHLC reader are physically cut at 2024 before
  the manifest; the full market file is opened only after the write.
- Costs: 0.5x, 5 bp fee plus 1 bp slippage per side.
- CAGR: complete configured calendar window including idle time.
- strict MDD: worst-order favorable-to-adverse OHLC high-water drawdown,
  including intraposition high-water marks.

## Representative selected policies

| Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| ExtraTrees TAKE q70, reclaim, both sides | 2023 Select | +11.01% | 11.02% | 4.09% | 2.70 | 58 |
| same | 2024 Test | +13.03% | 13.00% | 7.39% | 1.76 | 62 |
| same | 2025 Eval | +4.81% | 4.82% | 1.85% | 2.60 | 14 |
| same | 2026 YTD | +0.54% | 1.31% | 3.74% | 0.35 | 9 |
| ExtraTrees TAKE q80, all families, long | 2023 Select | +6.02% | 6.02% | 3.73% | 1.61 | 46 |
| same | 2024 Test | +16.34% | 16.31% | 5.11% | 3.19 | 41 |
| same | 2025 Eval | +3.27% | 3.27% | 1.49% | 2.20 | 13 |
| same | 2026 YTD | -0.64% | -1.54% | 1.20% | -1.28 | 3 |

## Interpretation

The critic found a real but weak 2024 long filter.  The failure is structural:
ranking a single Top-10 on bullish 2023 naturally removes short specialists,
even though the frozen REX generator emits mostly short candidates in 2025-2026.
More threshold tuning on the same one-list selection would only overfit.

The next experiment therefore pre-allocates Top-10 slots by side before future
evaluation.  Long and short specialists must be selected independently, and a
causal regime state must decide which specialist is active.

## Artifacts

- Dataset summary: `results/rex_clean_q065_h144_dataset_summary_2026-07-13.json`
- Search: `training/search_rex_pre2024_ml_alpha.py`
- Tests: `tests/test_search_rex_pre2024_ml_alpha.py`
- Manifest: `results/rex_pre2024_ml_top10_manifest_2026-07-13.json`
- Result: `results/rex_pre2024_ml_alpha_scan_2026-07-13.json`
