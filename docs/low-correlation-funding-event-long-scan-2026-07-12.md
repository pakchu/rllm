# Low-correlation funding-event long alpha scan (2026-07-12)

## Protocol
- BTCUSDT long only.
- Entry clock restricted to Binance funding settlements and 30-minute neighborhoods at 00/08/16 UTC, plus Asia open.
- Train `<2024` thresholds; test2024 selection; eval2025/YTD2026 attached afterward.
- 6bp/side, 0.5x, strict intrabar MDD, next-open entry, TP/SL or time exit.
- Existing long-component maximum activation phi <=0.20.

## Result
- 2,708 eligible low-correlation variants.
- test/eval CAGR/strict-MDD >=2.5: **0**.
- live-grade >=3: **0**.

Closest family: `premium_discount_recovery` during the 30 minutes after funding settlement.

Entry:
- `premium_index_zscore <= -1.8876801640`
- `12-bar taker-flow recovery minus 48-bar flow >= 0.0251275091`
- max activation phi `0.0192`, nearest `premium20_mom90`

Best test/eval-balanced execution: 96-bar cap, TP4%, SL2.5%.

| split | return | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|
| train | -16.04% | -0.19 | 200 | 44.5% |
| test2024 | +6.62% | 1.88 | 45 | 57.8% |
| eval2025 | +7.25% | 1.69 | 42 | 50.0% |
| ytd2026 | -0.31% | -0.25 | 19 | 36.8% |

## Verdict
The feature is genuinely orthogonal and positive in both 2024/2025, but negative train and 2026 plus ratios below 2.5 prevent alpha promotion. Keep as beta event-selector evidence only. Do not retune settlement windows or thresholds on 2026.

## Artifacts
- `training/search_lowcorr_funding_event_long_alpha.py`
- `results/lowcorr_funding_event_long_alpha_scan_2026-07-12.json`
