# Binance cross-collateral quarterly-curve source audit — 2026-07-17

## Claim boundary

This artifact freezes a new source axis for a market-neutral alpha search. It
contains **no strategy return, BTC outcome, PnL, CAGR, MDD, label, or 2024+
row**. The only inspected values are source timestamps, completed-candle OHLC
integrity, common-grid support, and distributional incidence needed to decide
whether a preregistered experiment is executable.

Official source: [Binance continuous-contract klines](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#continuous-contract-kline-candlestick-data).
The endpoint accepts `CURRENT_QUARTER` and, after the documented CM migration,
both USD-M and COIN-M pair values. A candle is usable only after its returned
`close_time`. The frozen panel uses an explicit `open_time` and sets
`available_time = close_time + 1 ms`, exactly the next five-minute boundary;
downstream logic may not join completed OHLC on `open_time`.

## Frozen physical source

| Leg | Pair | Requested range | Returned rows | First row | Last row | Invalid OHLC |
|---|---|---|---:|---|---|---:|
| USD-M | `BTCUSDT` | 2021-01-01..2024-01-01 exclusive | 305,756 | 2021-02-03 08:20 UTC | 2023-12-31 23:55 UTC | 1 |
| COIN-M | `BTCUSD` | 2021-01-01..2024-01-01 exclusive | 315,360 | 2021-01-01 00:00 UTC | 2023-12-31 23:55 UTC | 0 |
| Joint panel | both | inner common five-minute grid | 305,756 | 2021-02-03 08:20 UTC | 2023-12-31 23:55 UTC | 1 quarantined |

USD-M current-quarter history starts when that product becomes available; the
builder does not synthesize earlier rows. The single malformed USD-M candle at
`2021-02-03 08:40 UTC` has open above high. It is retained for provenance but
`source_complete=false`, so it cannot seed a signal or mark a position. Every
other common row is a duplicate-free contiguous five-minute candle whose
`close_time` is exactly open time plus `299,999 ms`.

The continuous source does not return a historical symbol identifier. The
panel therefore freezes the standard UTC quarterly delivery calendar and adds
`delivery_time`, maturity-key `contract_segment`, `bars_to_delivery`,
`is_roll_boundary`, and `is_pre_roll_final_bar` to every row. The current
contract segment changes at 08:00 UTC on the last Friday of March, June,
September, and December. Any strategy must reset its rolling state by
`contract_segment` and reserve an exit strictly before `delivery_time`.

The initial API pull was staged locally before a repeated replay hit Binance
HTTP 429. The builder hashes the staged response bytes, rewrites canonical
deterministic gzip raw snapshots, validates them through the same parser, and
records `source_mode=offline_official_api_snapshot`. This is a transport replay,
not a data substitution. The downloader now also honors HTTP `Retry-After`.

## Frozen hashes

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| joint executable panel | `54addc04b997cfb077197cd845f2aa286a219bdae4a29b49c2086667007046f7` | 11,445,573 |
| USD-M raw snapshot | `259d5c8c627797c8c3651517219e4d2aff52d81a8e2e450985ff6d21a42caed1` | 12,174,061 |
| COIN-M raw snapshot | `7de18164d4367fbcf951779f848a247e5b6d536135cba800bdbb008b27a09397` | 12,197,628 |
| manifest identity | `197755f0ce6823eea7d0fd47e6db5cbec2ddb1a18542fc47b57ab7f02f69321b` | — |

## Why this axis is materially different

The active portfolio is concentrated in directional BTC price action, REX,
OI, perpetual funding/premium, taker flow, and kimchi/FX contexts. This source
measures the relative price of the **same current-quarter maturity** under
stablecoin-margined and coin-margined collateral. A future experiment can hold
equal initial USD face in opposite derivative legs, removing first-order
**derivative-leg** BTC beta and testing collateral-basis convergence rather
than another directional gate. This does not remove the account-equity beta of
BTC collateral posted to the inverse COIN-M leg; an exact collateral ledger or
separate hedge is required before any live promotion.

This distinction does not establish profitability. The next step must freeze
the spread equation, roll exclusion, support threshold, transaction costs,
strict two-leg MDD path, and orthogonality gates before any spread outcome is
opened.

## Reproduction

```bash
python -m training.build_binance_cross_collateral_quarterly_curve_2021_2023
```

The three bounded source files under
`data/binance_cross_collateral_quarterly_curve_2021_2023/` are force-tracked
despite the repository-wide `data/` ignore rule. This keeps clean-checkout hash
verification possible without relying on a mutable future API replay.
The default command reads the two tracked deterministic `.raw.json.gz`
snapshots, rebuilds them byte-for-byte, and reproduces the frozen panel and
manifest identity offline. `--live-api` is an explicit refresh mode and is not
the frozen-artifact reproduction path.
