# Liquidity-recovery bidirectional alpha — 2026-07-12

## Result

`efficient_recovery_continuation_72` is a new BTCUSDT standalone alpha-pool candidate. It combines 72-bar signed path efficiency, 72-bar taker-flow imbalance, and short-vs-long flow recovery. All thresholds are fitted on train only; execution is at the next bar open.

| split | return | CAGR | strict MDD | CAGR/MDD | trades | L/S | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train `<2024` | 10.83% | 2.60% | 9.77% | 0.27 | 294 | 150/144 | 0.85 |
| test `2024` | 14.32% | 14.29% | 5.16% | **2.77** | 107 | 58/49 | 1.74 |
| eval `2025` | 9.22% | 9.23% | 3.26% | **2.83** | 114 | 58/56 | 1.33 |
| YTD `2026` diagnostic | 4.50% | 11.15% | 6.63% | 1.68 | 56 | 26/30 | 0.90 |

The candidate clears the project alpha-pool threshold on test and eval, with more than 100 trades per OOS year and balanced directions. It does not clear standalone live-grade ratio 3, and 2026 remains below target; keep it research-only pending fresh OOS.

## Fixed policy

- LONG: `lr_signed_eff_72 >= 0.1824548212170588`, `lr_flow_72 >= 0.04869572464172154`, `lr_flow_recovery >= 0.06794383030855547`.
- SHORT: `lr_signed_eff_72 <= -0.16331195830141687`, `lr_flow_72 <= -0.055475434124467156`, `lr_flow_recovery <= -0.06783534014527763`.
- TP 2.5%, SL 1.5%, cap 144 bars, stride 12 bars, leverage 0.5x.
- Cost 6bp/side, full-window CAGR, next-open entry, strict intrabar MDD, split-contained exit.

## Selection and contamination notes

- Feature formulas and quantile thresholds use completed bars and train data only.
- Candidate ranking uses test 2024 only. Eval 2025 and 2026 were attached after selecting the test top set.
- The raw test winner had ratio 4.48 but collapsed to 0.01 in eval; it is rejected. This is evidence that test rank alone is noisy, not a reason to tune against eval.
- Independent validator found no negative shift or centered rolling use and confirmed cost, trade counts, both directions, and OOS thresholds.

## Artifacts

- `training/search_liquidity_recovery_bidirectional_alpha.py`
- `results/liquidity_recovery_bidirectional_alpha_scan_2026-07-12.json`
- `results/liquidity_recovery_bidirectional_alpha_validator_2026-07-12.json`

Reproduce:

```bash
PYTHONPATH=. python training/search_liquidity_recovery_bidirectional_alpha.py \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --output results/liquidity_recovery_bidirectional_alpha_scan_2026-07-12.json
```
