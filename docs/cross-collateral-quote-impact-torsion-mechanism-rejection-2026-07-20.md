# Cross-collateral quote-impact torsion mechanism rejection — 2026-07-20

## Decision

**Do not preregister CQIT.** The proposed cross-collateral quote-impact
torsion is rejected at the source/mechanism review, before event incidence,
BTC outcomes, funding, returns, PnL, equity, CAGR, or MDD are opened.

The proposal compared normalized radial bid/ask quote-impact geometry between
Binance USD-M `BTCUSDT` and COIN-M `BTCUSD_PERP`. A bounded one-day schema
inspection confirmed that both official `bookDepth` archives contain
`timestamp`, `percentage`, `depth`, and `notional`. It did not inspect a market
outcome or establish a predictive effect.

Official source references:

- Binance public-data archive policy and checksum convention:
  <https://github.com/binance/binance-public-data>
- USD-M `BTCUSDT` `bookDepth` archive:
  <https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/>
- COIN-M `BTCUSD_PERP` `bookDepth` archive:
  <https://data.binance.vision/data/futures/cm/daily/bookDepth/BTCUSD_PERP/>
- USD-M live local-order-book reconstruction contract:
  <https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- COIN-M live local-order-book reconstruction contract:
  <https://developers.binance.com/docs/derivatives/coin-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

## Why the mechanism is not independent enough

The candidate's central transform is another radial average-quote geometry.
For the linear book, `notional / depth` is the same cumulative average-quote
object already used by RNCM and RQCI. Reversing the ratio for the inverse book,
normalizing each venue, and taking a cross-venue torsion does not create a new
observable; it cross-collateralizes an already heavily tested derivative.

This repository has already tested cross-collateral depth mass, shell flow,
refill, credibility, persistence, inventory pressure, and positioning recoil,
as well as the RNCM radial translation and RQCI radial curvature families.
CQIT would therefore be closer to a model repair of those rejected families
than a new source axis.

## Unresolved mechanical and causal defects

1. **Moving-band artifact.** The archive reports cumulative depth at percentage
   bands around a changing reference price. Apparent radial motion can be
   produced by the bands moving across a static absolute-price book. CQIT has
   no frozen fixed-absolute-book null that proves invariance to this mechanism.
2. **Inverse-contract dimensional ambiguity.** A contract-size factor can
   cancel after some normalizations, but the public-data README does not define
   `bookDepth.depth` and `bookDepth.notional` precisely enough to treat
   `depth / notional` as a production-grade COIN-M quote-impact measure without
   an independently frozen dimensional proof.
3. **No defensible trading orientation.** A positive cross-venue torsion does
   not imply a stable long or short action without an additional causal story.
   Freezing the side after seeing returns would be outcome-driven selection.
4. **Archive/live mismatch.** The archive is a sampled percentage-band summary.
   Binance's live contract instead requires a REST snapshot plus ordered
   WebSocket depth updates and update-ID continuity. Historical CQIT and live
   CQIT would not be the same observable until an explicit parity layer is
   demonstrated.

## Reconsideration boundary

CQIT may be reconsidered only as an **unsigned context feature**, not as the
same directional candidate, after all of the following exist:

- a checksum-audited dual-venue quote panel;
- an exact USD-M/COIN-M dimensional derivation;
- fixed-absolute-book and moving-reference-price null tests;
- archive/live feature-parity evidence; and
- a causal side rule frozen independently of BTC returns.

No threshold, hold, sign, or event-density repair is allowed under the CQIT
name. The next alpha search must use a materially different observable axis.
