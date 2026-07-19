# BTC alpha source-axis decision — 2026-07-20

## Decision

The next standalone BTC candidate will use the previously unused **USD-M
`bookDepth.notional / bookDepth.depth` radial-centroid geometry**, not another
threshold repair of price, funding, OI, positioning, aggregate-trade topology,
or the already-tested depth-mass features.

The first candidate is provisionally named **RNCM-72** (Reported-Notional
Centroid Migration, six-hour hold).  This file freezes only the source axis and
research boundary.  It does not freeze a trading rule and opens no return.

## Why the options-surface branch was stopped before implementation

Binance's official BTC options `EOHSummary` archive is genuinely richer than
the local BVOL/DVOL series: the CSV contains hourly contract rows with strike,
expiry, call/put type, mark IV, delta, gamma, vega, theta, volume, bid/ask, and
open interest.  However, a direct S3-prefix enumeration on 2026-07-20 found only
147 BTCUSDT ZIPs:

- first: `2023-05-18`;
- last: `2023-10-23`;
- coverage: about five months, not three years.

That source cannot support the user's three-year validation contract.  It is
therefore rejected at the **source-coverage gate**, before any option-surface
return, signal direction, threshold, or holding period is tested.

Official archive root:

- <https://data.binance.vision/?prefix=data/option/daily/EOHSummary/BTCUSDT/>

## Selected source coverage

The official USD-M BTCUSDT `bookDepth` prefix contained 1,292 daily ZIPs when
enumerated on 2026-07-20:

- first: `2023-01-01`;
- last available at inspection: `2026-07-18`;
- companion checksum files are published beside the ZIPs;
- the archive therefore spans more than three and a half years, including the
  recent year.

Official references:

- Binance public-data repository and checksum/update policy:
  <https://github.com/binance/binance-public-data>
- USD-M BTCUSDT `bookDepth` archive:
  <https://data.binance.vision/?prefix=data/futures/um/daily/bookDepth/BTCUSDT/>
- Binance futures live-order-book reconstruction rules, which also make clear
  that a production implementation must use update IDs rather than treating
  snapshots as an event-level feed:
  <https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

Other official long-history candidates were not selected because this branch
has already evaluated their relevant mechanism families: Binance five-minute
OI/top-trader/global/taker metrics through positioning disagreement, simplex,
lifecycle, continual-model, cost-basis, and cross-collateral experiments;
CFTC positioning through transfer, concentration, and release-assimilation
experiments; and BVOL/DVOL through volatility disagreement and
options-perpetual demand experiments.  Reusing those fields with another model
would not constitute a new observable axis.

The archive normally reports repeated snapshots with cumulative `depth` and
`notional` at `-5..-1,+1..+5` percent.  The archive is a sampled percentage-band
summary, not an order-ID feed, and Binance's public-data README does not define
the alpha interpretation below.  Any “centroid,” “migration,” or “support”
language is therefore a falsifiable research transform, not an exchange claim.

## Why this is not another tested book-depth repair

The frozen 2023 book builders and alpha families used:

- cumulative native depth medians;
- depth log-MAD, log-net change, and path activity;
- non-overlapping depth-shell mass, additions, withdrawals, churn, and radial
  wavefront ordering;
- cross-collateral static geometry, persistence, refill, credibility, and
  price/flow rejection.

Repository search confirms that `notional` was previously parsed only for
schema/finite/positive validation.  It was not pivoted into a book feature or
used by a preregistered alpha.  For linear BTCUSDT, the reported cumulative
average quote per depth unit is

```text
average_quote_(side,k) = notional_(side,k) / depth_(side,k)
```

RNCM will compare the **radial movement of those average quotes**, rather than
the amount of displayed depth.  This is distinct from the rejected depth-mass
wave, refill, and flicker hypotheses even though it reuses the same official
archive.

## Frozen research sequence

1. Build and checksum-audit only calendar 2023, without loading BTC price,
   funding, future OHLC, labels, or PnL.
2. Validate the cumulative-average algebra, radial monotonicity, five-minute
   completeness, causal availability, and event support.
3. Commit a separate RNCM preregistration with exact feature, action, scheduler,
   controls, and stopping gates before any post-entry return is read.
4. Commit and hash-freeze the strict evaluator before opening 2023 outcomes.
5. Reject immediately if 2023 fails.  Only a frozen 2023 pass may open 2024;
   later years remain physically unopened until their preceding gate passes.

The branch has broad prior research exposure, so this process can establish a
candidate-level frozen sequence but cannot honestly recreate a pristine global
human holdout.
