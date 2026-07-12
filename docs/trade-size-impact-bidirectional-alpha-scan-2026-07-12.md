# Trade-size / price-impact bidirectional alpha scan (2026-07-12)

## Protocol
- BTCUSDT target, 5m quote volume, number of trades and taker-buy quote volume.
- Features: average notional per trade, trade intensity, signed notional, signed-flow price impact and absorption.
- Train `<2024` thresholds; test2024 selection; eval2025/YTD2026 reporting.
- 6bp/side, 0.5x, strict intrabar MDD, both directions required.

## Result
- 5,696 eligible variants.
- test/eval ratio>=2.5: **0**.
- live-grade >=3: **0**.

Closest candidate: `large_flow_continuation_48`:
- test ratio 3.61, 202 trades
- eval ratio 2.41, 134 trades
- YTD2026 ratio -1.55

Large-trade continuation was real in 2024 and nearly reached alpha-pool threshold in 2025, but reversed in 2026. It is beta evidence, not an alpha.

## Artifacts
- `training/search_trade_size_impact_bidirectional_alpha.py`
- `results/trade_size_impact_bidirectional_alpha_scan_2026-07-12.json`
