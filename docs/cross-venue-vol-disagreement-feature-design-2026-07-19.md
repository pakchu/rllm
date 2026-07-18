# Cross-venue volatility disagreement feature design — 2026-07-19

## Research object

This source aligns Binance `BTCBVOLUSDT`, Deribit BTC DVOL, and completed BTCUSDT
price history on exact UTC hour-close timestamps. It is intended to test whether
venue-level implied-volatility disagreement contains a directionally useful
state transition that is distinct from funding, premium, OI, Kimchi, FX, REX,
and minute-packet families.

## Causal contract

- Binance BVOL is accepted only from checksum-verified, complete 3,600-second
  hours. Missing archives and incomplete hours remain invalid and are not filled.
- Deribit DVOL joins on its published hourly `close_time`; no nearest/as-of join
  is allowed.
- The BTC 4h return uses only completed five-minute closes available at the same
  hour boundary.
- The earliest executable timestamp is five minutes after that boundary.
- The generated research frame is physically truncated before 2024-01-01.
- Invalid rows quarantine all numeric features rather than carrying stale values.
- This stage computes backward-looking features only and does not parse future
  trade returns, fees, funding PnL, or selection statistics.

The relative level is a venue-disagreement proxy only. It must not be described
as investor identity, order-flow ownership, or an options-skew measurement.
