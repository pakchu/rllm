# Weekend FX reconciliation alpha — frozen pre-outcome design

## Hypothesis

BTC continues trading while spot FX is closed. The first fully completed FX
hour after the closure can contain information not reflected in BTC's weekend
displacement. The fixed policy trades the standardized disagreement:

`reconciliation_residual = fx_event_z - btc_event_z`

- positive residual: long BTC;
- negative residual: short BTC.

This is a cross-sectional **safe-haven differential**, not a claim that the FX
basket is a pure global risk factor.

## Frozen source and timing contract

- BTC source: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- FX source: Wave Trading one-minute multi-asset cache.
- The BTC file must contain only `tic=BTCUSDT`. Project provenance treats its
  OHLC fields as Binance USD-M perpetual bars; the CSV itself does not carry a
  venue/feed identifier beyond `tic`.
- The FX cache's stored `close` is used as-is. Bid/ask/mid semantics and vendor
  publication/ingestion timestamps are not encoded, so even a statistical pass
  remains research-only until a live source contract is verified.
- All timestamps are parsed with `utc=True` and represented as timezone-naive
  UTC internally.
- Analysis frames are returned strictly before `2024-01-01`.
- An FX hour is valid only when all six fixed pairs have at least 55 one-minute
  rows, include minute 59, and have a source timestamp before the next-hour
  effective boundary.
- Fixed pairs: `EURUSD`, `GBPUSD`, `USDCAD`, `USDCHF`, `USDJPY`, `USDSEK`.
- Pair orientation: EURUSD/GBPUSD are multiplied by -1; the four USD-base
  pairs are multiplied by +1, so positive oriented returns mean USD strength.
- A closure event is the first valid six-pair FX hour after an observed
  45–72-hour gap and must occur Sunday or Monday UTC. Other outages are skipped.
- BTC displacement uses the completed minute-55 closes at the previous and
  current FX effective boundaries.
- FX displacement uses the six minute-59 closes at those same boundaries.
- Both raw gap returns are divided by `sqrt(elapsed_hours)`.
- Each component is z-scored with `shift(1)` using only the previous 52 closure
  events and at least 26 prior observations. No full-fit centering is allowed.
- Signal is formed at minute 00 and executed at the fixed minute-05 open.
- `source_time` is a candle timestamp, not an ingestion timestamp. The fixed
  five-minute delay is conservative for this replay but does not prove live
  publication latency.
- BTC duplicate timestamps keep the last row; a complete five-minute grid and
  finite, positive, internally consistent OHLC are mandatory. FX timestamps
  must be sorted; non-positive/non-finite event closes are rejected. No
  statistical outlier clipping is applied, and structurally complete unchanged
  FX closes are retained because the cache has no quote-staleness field.

## Frozen replay

- Primary hold: 288 five-minute bars / 24 hours.
- Entry is the open of the BTC five-minute bar beginning
  `effective_time + 5min`; exit is the open exactly 288 bars later, i.e.
  `effective_time + 24h + 5min`.
- Leverage: 0.5x.
- Cost: 6 bp per side.
- Strict MDD: favorable-first/adverse-second OHLC high-water convention.
- Fit: `2020-06-01 <= t < 2023-01-01`.
- One-shot internal selection: calendar 2023, with fixed H1/H2 diagnostics.
- Events are assigned by current FX `effective_time`. Entry and exit must both
  remain inside the declared split; boundary-crossing trades are skipped.
- Earlier outcome-blind 2023 closure features update the prior-event z-score
  state used by later 2023 events. Returns never update that state.
- 2024 and later remain unopened.
- Admission requires positive return and CAGR/strict-MDD >= 3 on both fit and
  2023, positive 2023 H1/H2, at least 80 fit trades, 24 2023 trades, 8 per
  2023 half, and minimum long/short support.

## Frozen controls

- same schedule always-long and always-short;
- BTC weekend continuation and reversal;
- FX reopen differential and its opposite;
- exact primary direction flip;
- previous closure's side as a causal placebo;
- cost stress from 0 to 15 bp/side;
- fixed entry diagnostics at minute 05/10/15;
- fixed hold diagnostics at 12/24/48 hours;
- component/residual Spearman and side-agreement audit.

Entry and hold diagnostics are report-only. They cannot replace the primary
after 2023 is opened. Historical perp funding is not included in this first
canonical discovery replay, so even a statistical pass remains blocked from
live promotion until a funding-aware replay is implemented.
