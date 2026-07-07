# Wave alpha search checkpoint (2026-07-08)

## Scope

- Objective: preserve the newly found non-PB30/non-REX price-structure alpha evidence and scan wave-trading-style alpha candidates.
- Main splits: `train < 2024`, `test = 2024`, `eval = 2025`, `ytd2026` where available.
- Cost assumptions for local wave-structure scan: fee `4bp` + slippage `1bp` per trade.
- Contamination rule: train-fit/threshold evidence is separated from OOS diagnostic ranking. OOS-ranked threshold variants are not treated as selection-safe live choices by themselves.

## Saved evidence files

- `results/context_session_alpha_scan_fast_2026-07-08.json`
- `results/price_structure_breakout_alpha_scan_limited_2026-07-08.json`
- `results/wave_structure_alpha_scan_2026-07-08.json`
- `results/wave_trading_best_tte_validation_2026-07-08.json`
- `results/wave_trading_lr_threshold_sweep_tte_2026-07-08.json`

## Previous non-wave alpha saved

### `fade_low_72_rangevol_high`

Different mechanism from PB30/REX/OI/funding: high-volatility price-structure mean reversion after prior 72-bar low pressure.

| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|
| train `<2024` | 93.73% | 21.95% | 19.32% | 1.14 | 125 | 59.2% | 2.81 |
| test 2024 | 29.43% | 29.36% | 6.18% | 4.75 | 20 | 90.0% | 3.29 |
| eval 2025 | 10.46% | 10.47% | 1.98% | 5.30 | 14 | 78.6% | 1.64 |
| ytd 2026 | 5.78% | 14.36% | 5.54% | 2.59 | 4 | 50.0% | 1.01 |

## Wave scan results

### True wave_trading documented LR/wavelet alpha

Source: sibling `wave_trading` documented 15-feature wavelet + flow + VWAP + non-wavelet LogisticRegression, rolling train-before-test folds with purge gap.

Best train-selected threshold from sweep: `C=0.1`, `long_th=0.66`, `short_th=0.35`.

| split | return | CAGR | MDD | Calmar | trades | win | long/short |
|---|---:|---:|---:|---:|---:|---:|---:|
| train walk 2020-2023 | 254.14% | 37.18% | 13.39% | 2.78 | 490 | 61.4% | 370/120 |
| test 2024 | 27.48% | 27.42% | 13.84% | 1.98 | 221 | 54.8% | 178/43 |
| eval 2025 | 5.37% | 5.38% | 4.78% | 1.12 | 40 | 55.0% | 20/20 |
| ytd 2026 to 2026-06-01 | 6.64% | 16.82% | 2.73% | 6.17 | 11 | 54.5% | 4/7 |

Interpretation: this is a real, independent wavelet/flow alpha, but 2025 frequency and return are much lower than 2024. Keep as diversifying sleeve, not as a main return engine.

### OOS diagnostic threshold variants

These are useful for research direction but not selection-safe unless revalidated by a clean rolling selector.

Example diagnostic top: `C=0.03`, `long_th=0.66`, `short_th=0.30`.

| split | return | CAGR | MDD | Calmar | trades | win | long/short |
|---|---:|---:|---:|---:|---:|---:|---:|
| train walk 2020-2023 | 182.59% | 29.65% | 13.39% | 2.21 | 412 | 61.2% | 366/46 |
| test 2024 | 39.29% | 39.20% | 13.61% | 2.88 | 196 | 57.1% | 180/16 |
| eval 2025 | 14.90% | 14.91% | 2.84% | 5.24 | 26 | 69.2% | 19/7 |
| ytd 2026 to 2026-06-01 | 8.55% | 21.94% | 0.45% | 49.27 | 6 | 66.7% | 3/3 |

Interpretation: threshold relaxing toward fewer shorts looks better OOS, but this was found by OOS diagnostic ranking. Treat as a hypothesis for a future train-only/anchored selector, not a confirmed live parameter.

### Local symbolic wave-structure scan

The hand-built wave/fib/retracement/extreme scan did not produce a strong standalone candidate. Best patterns were sparse and mostly weak in 2025. The only moderately interesting family was fib pullback/short-heavy variants, but it is not yet portfolio-worthy by itself.

Example: `wave_fib_pullback_W144_ret_144_0.5_0.9`, hold 72, TP 2.5%, no SL.

| split | return | CAGR | strict MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| train `<2024` | 45.25% | 9.78% | 13.05% | 0.75 | 118 | 60.2% |
| test 2024 | 10.32% | 10.30% | 4.85% | 2.12 | 24 | 58.3% |
| eval 2025 | 2.66% | 2.66% | 3.42% | 0.78 | 12 | 66.7% |
| ytd 2026 | 8.64% | 17.68% | 0.00% | n/a | 4 | 100.0% |

## Decision

- Save and keep `fade_low_72_rangevol_high` as the better non-wave independent alpha candidate.
- Keep true wave_trading LR/wavelet strategy as a diversifying wave sleeve.
- Do not promote the OOS diagnostic wave threshold variants directly without a clean selector/revalidation pass.
