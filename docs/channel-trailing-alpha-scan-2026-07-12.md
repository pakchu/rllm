# Channel breakout + heterogeneous trailing-exit scan (2026-07-12)

## Protocol
- BTCUSDT 5m, 6bp/side, 1.0x.
- Shifted Donchian entry events (cross only), next-bar open execution.
- Exit is path-dependent ATR trailing stop or shifted opposite Donchian channel; no mandatory holding period.
- Strict intrabar MDD and forced period-end liquidation.
- Parameters ranked on test2024 only; eval2025 and ytd2026 attached afterward.

## Result
- 144 corrected deterministic variants tested.
- test2024/eval2025 CAGR/strict-MDD >=2.5: **0**.
- test2024/eval2025 >=3.0: **0**.
- No selected variant was profitable in test2024; the least-bad row was a 14-day short breakout with ATR(1d) x6 trailing stop: test2024 -3.2%, eval2025 -23.1%, ytd2026 -4.7%.

## Interpretation
Classic raw price-channel trend following at 5m granularity is decisively uneconomic under 6bp/side in this dataset. This is gamma-grade failure evidence for using raw Donchian+ATR as a standalone alpha. It does not invalidate channel position as a contextual feature or a much slower execution horizon.

## Implementation corrections
Two invalid intermediate runs were rejected:
1. breakout state caused repeated re-entry; corrected to first-cross events only;
2. same-bar high/low was incorrectly allowed to update and hit a trail in unknown order; corrected to test the existing trail first and update it only after survival.

## Artifacts
- `training/search_channel_trailing_alpha.py`
- `results/channel_trailing_alpha_corrected_scan_2026-07-12.json`
