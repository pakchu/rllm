# Binance aggTrades microstructure dataset — build design

## Purpose

Recent alpha failures repeatedly produced novel transformations of the same
five-minute aggregates without robust direction. This build adds genuinely new
observables from individual aggregate trades: event-size distribution, actual
underlying fill count, trade-sign runs and inter-arrival burstiness. No trade
return or future label is computed in this stage.

## Official sources and live parity

- Binance's official public-data repository states that USD-M Futures
  `aggTrades` archives contain aggregate trade ID, price, quantity, first/last
  underlying trade IDs, transaction time and buyer-maker flag, matching
  `/fapi/v1/aggTrades`:
  https://github.com/binance/binance-public-data
- Official USD-M Futures aggregate-trade WebSocket stream, used for eventual
  live parity:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams
- Every archive's published `.CHECKSUM` is verified before parsing. Raw ZIPs
  remain in memory only and are never accumulated on WSL disk.
- A resumed month re-fetches every current checksum sidecar and compares it
  with the recorded archive hash. Changed upstream data invalidates and
  rebuilds the month instead of silently preserving a stale artifact.

Historical inspection confirms BTCUSDT daily archives exist from
`2020-01-01`. Files before 2021 have no CSV header; later files have a header.
The parser explicitly supports and tests both forms.

## Causal aggregation

- Five-minute key: UTC floor of the transaction timestamp.
- `is_buyer_maker=true` means the buyer was passive, so the aggressive side is
  sell (`-1`); false is aggressive buy (`+1`).
- `last_trade_id - first_trade_id + 1` reconstructs the inclusive underlying
  fill-ID span per aggregate event.
- Price/size/side sequence preserves official transaction order; no later
  bin enters an earlier feature.
- The build covers `[2020-01-01, 2024-01-01)` so later outcome windows remain
  unopened during feature construction.

## Emitted five-minute observables

- aggregate-event and underlying-fill counts;
- base volume, quote notional, signed/buy/sell notional and flow coherence;
- first/last trade price and signed intrabin price response;
- event notional mean/std/p50/p90/p99/max, HHI and normalized effective count;
- underlying fills per aggregate event;
- signed event imbalance, sign-flip rate, mean/max same-sign run structure;
- inter-arrival mean/std/burstiness;
- buy-versus-sell average event-size log ratio.

Monthly gzip outputs fix gzip `mtime=0`, making their hashes deterministic.
Resume metadata binds each artifact to the exact requested day list and schema
version. A final manifest records every source archive SHA-256, row and ID
ranges, output hashes and the explicit `outcomes_opened=false` contract.

## Intended next hypothesis (not tested here)

True trade-level fragmentation can distinguish a persistent hidden metaorder
from one-off block flow. A later separately preregistered alpha will test
whether fragmented same-sign execution with rising marginal price response
continues, while equally persistent execution with falling response is being
absorbed. The mapping, thresholds and holds must be frozen only after this
dataset passes kline reconciliation and support audits.
