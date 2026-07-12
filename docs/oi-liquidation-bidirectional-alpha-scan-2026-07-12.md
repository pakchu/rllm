# OI liquidation/build bidirectional alpha scan (2026-07-12)

## Protocol
- BTCUSDT target, OI/price/taker only.
- Train `<2024` thresholds; test2024 selection; eval2025/YTD2026 reporting.
- Long theses: liquidation exhaustion, positive OI-price divergence, downside crowding reversal.
- Short theses: crowded OI build, negative divergence, upside crowding reversal.
- 6bp/side, 0.5x, strict intrabar MDD, both directions required.

## Result
- 7,188 eligible variants.
- test/eval ratio>=2.5: **0**.
- live-grade >=3: **0**.

Best balanced candidate was `crowding_reversal_288`:
- test ratio 1.45, 30 trades
- eval ratio 1.33, 26 trades
- YTD2026 ratio 3.36

The OI/price state contains some 2026 information but did not form a standalone alpha across test/eval. No alpha_pool entry was added.

## Artifacts
- `training/search_oi_liquidation_bidirectional_alpha.py`
- `results/oi_liquidation_bidirectional_alpha_scan_2026-07-12.json`
