# Volume alpha search checkpoint (2026-07-08)

## Scope

User requested a broad volume-based alpha search using available means: Binance BTC futures volume/taker flow, Upbit KRW-BTC volume, candle/volume interaction, and available Binance altcoin volume breadth.

Cost assumptions: fee `4bp` + slippage `1bp` per trade. Splits were forced-exit by split end. All thresholds were derived from train quantiles only, but OOS diagnostic ranking is explicitly not live-safe by itself.

## Saved evidence

- `results/volume_upbit_alpha_scan_2026-07-08.json`
- `results/volume_limited_alpha_scan_2026-07-08.json`
- `results/alt_volume_breadth_alpha_scan_2026-07-08.json`
- `results/alt_volume_top_refine_2026-07-08.json`

## Findings

### 1. BTC + Upbit direct volume: mostly failed

BTC/Upbit single-market volume patterns showed strong regime instability. Several train-positive long-volume regimes worked in 2024 but broke in 2025 and 2026. Treat direct BTC volume spikes/dry-ups as poor standalone alpha.

Example: `vol_z_24_q0.9 LONG hold144`:

| split | CAGR | strict MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|
| train `<2024` | 42.27% | 68.64% | 0.62 | 1931 | 51.9% |
| test 2024 | 33.41% | 35.52% | 0.94 | 440 | n/a |
| eval 2025 | 10.06% | 27.13% | 0.37 | 472 | n/a |
| ytd 2026 | -33.41% | 27.99% | -1.19 | 194 | n/a |

### 2. Upbit/Binance relative BTC volume: weak but all-period positive in one diagnostic

`alt_btc_qv_ratio_z_72_down288 LONG hold72` means alt/BTC quote-volume ratio is high while BTC 288-bar return is weak. It is not Upbit-specific; it came from alt breadth scan. It was one of the few candidates positive across test/eval/2026, but train edge is very weak.

| split | CAGR | strict MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|
| train 2023 | 2.64% | 14.80% | 0.18 | 341 | 49.3% |
| test 2024 | 41.11% | 14.92% | 2.76 | 315 | 57.8% |
| eval 2025 | 4.16% | 23.19% | 0.18 | 306 | 51.6% |
| ytd 2026 | 30.42% | 7.43% | 4.09 | 131 | 48.1% |

Decision: keep as a weak diversifier candidate only. Train selection is too weak to treat as robust alpha by itself.

### 3. ETH volume regime: best high-frequency volume candidate, but MDD too high

Refined candidate: `ETH_qv_288_hi90 LONG hold72 TP2.5%`.

Definition: ETHUSDT quote-volume 288-bar z-score above train 90th percentile, BTC long, hold 72 bars, TP 2.5%, no SL.

| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 2023 | 7.52% | 7.52% | 28.19% | 0.27 | 603 | 43.9% | 0.39 |
| test 2024 | 24.80% | 24.75% | 37.06% | 0.67 | 610 | 51.5% | 0.78 |
| eval 2025 | 20.46% | 20.47% | 23.42% | 0.87 | 589 | 54.7% | 0.77 |
| ytd 2026 | 17.08% | 46.06% | 22.03% | 2.09 | 233 | 53.6% | 0.89 |

Decision: this is the most interesting pure volume candidate found. It has high trade count and all-period positive OOS, but it is not yet capital-efficient: train/test/eval MDD and weak t-stat mean it should only be considered as a small sleeve or as a feature for an LLM/selector, not as standalone live strategy.

### 4. SOL/ADA low-volume regimes

Some SOL/ADA volume dry-up regimes had good test/eval but failed 2026. They are not robust enough for live promotion.

Example `SOL_qv_288_lo20 LONG hold144`:

| split | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|
| train 2023 | 16.11% | 26.46% | 0.61 | 329 |
| test 2024 | 60.00% | 42.79% | 1.40 | 401 |
| eval 2025 | 32.10% | 12.62% | 2.54 | 336 |
| ytd 2026 | -23.50% | 20.63% | -1.14 | 134 |

## Conclusion

- Direct BTC/Upbit volume alpha: failed as standalone.
- Altcoin relative volume/breadth contains more signal than BTC-only volume.
- Best currently usable candidate: `ETH_qv_288_hi90 LONG hold72 TP2.5%`, but it is a weak/high-MDD sleeve, not a main alpha.
- Best all-period positive diagnostic: `alt_btc_qv_ratio_z_72_down288 LONG hold72`, but train edge is too weak.

Recommended next step: use these as selector features or tiny sleeves in the existing portfolio optimizer, not as standalone strategies.
