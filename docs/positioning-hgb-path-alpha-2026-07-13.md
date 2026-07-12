# Positioning HGB path-utility alpha (2026-07-13)

## Verdict

Adding nonlinear price-regime/positioning interactions and a path-risk target
materially improved the prior rule search, but no frozen Top-10 policy passed the
full target.  Different long policies reached the target in 2024, but none
preserved it in 2025 and 2026.  This remains research evidence, not a live alpha.

## Model and causal protocol

- Inputs: 138 past-only features
  - delayed Binance top/global trader positioning and taker ratios
  - price returns, rolling range position, realized volatility
  - OI change and OI-price divergence
  - futures taker imbalance and quote-volume state
  - DXY, kimchi premium, and USDKRW context
  - causal time-of-day/day-of-week encodings
- Model: separate scikit-learn `HistGradientBoostingRegressor` long/short
  utility critics
- Target:
  - executable next-open return
  - minus the same 0.5x account-level two-side cost used by the backtest
  - minus `lambda *` side-specific intratrade MAE
- Fit: 2020-10-15 through 2022-12-31; every label exits before 2023
- Selection: 2023 full year plus H1/H2 robustness
- Manifest: physically written before future feature/prediction/metric creation
- Manifest SHA-256:
`69af70c287de1b8046baa961432a4a2f0179f3f83a30c3c9c6157625e1ad294b`
- OOS: 2024 Test, 2025 Eval, 2026 YTD
- Execution: 0.5x, 5 bp fee + 1 bp slippage per side, next-bar-open entry,
  fixed hold, full-window CAGR, strict intraposition MDD

The phase-2 implementation hashes both the complete pre-2024 feature prefix and
model prediction prefix and rejects the run if admitting future rows changes
either prefix.

## Best selected policies

| Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| 24h mean path utility, 365d q80, both, stride 12 | 2023 Select | +57.48% | 57.53% | 8.50% | 6.77 | 183 |
| same | 2024 Test | +8.57% | 8.55% | 19.05% | 0.45 | 157 |
| same | 2025 Eval | -6.08% | -6.08% | 12.12% | -0.50 | 144 |
| same | 2026 YTD | -6.26% | -14.39% | 10.72% | -1.34 | 55 |
| 24h mean path utility, 365d q80, long, stride 12 | 2023 Select | +45.71% | 45.75% | 8.50% | 5.38 | 87 |
| same | 2024 Test | +20.09% | 20.05% | 6.32% | 3.17 | 55 |
| same | 2025 Eval | -2.33% | -2.34% | 7.67% | -0.30 | 75 |
| same | 2026 YTD | -4.37% | -10.19% | 10.14% | -1.01 | 35 |
| 48h mean path utility, 365d q80, long, stride 6 | 2023 Select | +37.29% | 37.32% | 9.68% | 3.86 | 47 |
| same | 2024 Test | +28.88% | 28.81% | 6.36% | 4.53 | 48 |
| same | 2025 Eval | +8.05% | 8.06% | 10.65% | 0.76 | 65 |
| same | 2026 YTD | -6.99% | -15.98% | 12.82% | -1.25 | 39 |

## Interpretation

The audit-corrected strict MDD uses the conservative worst ordering of each
trade's favorable and adverse OHLC extremes, and the utility target now uses the
same 0.5x account-level cost convention as the backtest.  Under those stricter
rules, the new positioning data is still useful: a fixed nonlinear long critic
achieved a 2024 ratio above 3 without rule-gate optimization.  It did not
generalize through 2025/2026.  The remaining failure is temporal adaptation.

The justified follow-up is therefore a monthly prequential refit: at each month
boundary, admit only labels whose complete execution path has already ended,
retrain on the expanding/rolling past, and predict the next month.  The model,
feature set, update cadence, and 2023 selection policy must be frozen before
2024+ prequential evaluation.

## Artifacts

- Search: `training/search_positioning_hgb_path_alpha.py`
- Tests: `tests/test_search_positioning_hgb_path_alpha.py`
- Frozen manifest: `results/positioning_hgb_path_top10_manifest_2026-07-13.json`
- Result: `results/positioning_hgb_path_alpha_scan_2026-07-13.json`
