# Binance Spot 1m→5m microstructure data design — 2026-07-14

## Purpose

The existing alpha inventory is concentrated in BTC perpetual price, OI,
funding, premium, and futures aggressor flow. This dataset adds a separate cash
market observable without using revised third-party history: official Binance
Spot `BTCUSDT` one-minute klines are aggregated into causal five-minute auction
features and later aligned with the already verified USD-M aggTrade topology.

No return, label, action, or future path is computed by the builder.

## Official source contract

- Archive description and checksum convention:
  <https://github.com/binance/binance-public-data>
- Spot aggregate/kline field semantics:
  <https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md>
- Archive root:
  <https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/>

Each monthly ZIP and its official `.CHECKSUM` sidecar are fetched in memory.
Only the deterministic five-minute gzip and metadata are retained, so the build
does not consume the project's WSL disk budget with raw archives.

## Five-minute observables

- exact OHLC, base/quote volume, and trade count;
- aggressive buy/sell base and quote volume;
- signed cash flow and flow coherence;
- buyer and seller execution centroids reconstructed from taker base/quote
  fields;
- centroid ordering, spread, and close displacement;
- average trade notional;
- cash-flow-aligned price response;
- one-minute flow flip rate, flow/price alignment, and price/flow path
  efficiency inside the completed five-minute bar.

`source_complete=true` requires exactly five contiguous one-minute bars,
positive market activity, and finite two-sided execution centroids. Downstream
signals must fail closed on every other row.

## Intended alpha boundary

The first preregistered use will test whether cash-market flow that is accepted
by price while perpetual aggressor flow is marked wrong represents an observable
transfer of adverse selection from spot to leveraged inventory. It must beat
plain price momentum, spot-flow-only, perpetual-flow-only, role-swapped, and
timing-placebo controls before any RLLM integration is allowed.
