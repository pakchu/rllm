# CBFR-72 preregistration — 2026-07-18

## Decision being frozen

`CBFR-72` (Cross-Collateral Book-Validated Flow Rejection) tests whether an
aggressive five-minute futures flow that is rejected by price becomes a useful
reversal signal when both Binance collateral venues show credible liquidity on
the opposite side.

This is not a profitability claim. At this commit, no CBFR post-entry return,
funding, PnL, equity curve, CAGR, strict MDD, hit rate, or payoff has been read.

## Why this is a different hypothesis

- `UMFR-36` obtained direction from taker flow alone. Its pre-2024 effect had
  the expected sign but was too small for the frozen execution cost.
- `PDF-10` obtained direction from displayed depth versus firmness and did not
  use aggressive flow.
- `CRRC-72` obtained direction from radial book additions and withdrawals.
- `CBFR-72` obtains direction from **flow-price rejection**. Book credibility
  is a two-venue absorption/adverse-selection confirmation, not the original
  directional trigger.

The economic claim is narrow: a large aggressive order imbalance that cannot
move a completed bar may have encountered passive absorption. Agreement from
the USD-M and COIN-M book-firmness measurements is required before fading it.

## Frozen source boundary

Selection/support uses calendar 2023 only:

- official Binance USD-M `BTCUSDT` five-minute kline archive, SHA256
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`;
- Binance cross-collateral book-credibility panel, SHA256
  `45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429`.

Only these market columns may be loaded before the event clock is frozen:
`date`, `open`, `close`, `quote_asset_volume`, and `taker_buy_quote`. `open` and
`close` are from the completed signal bar. No `high`, `low`, entry price, later
price, funding, or outcome field may be loaded.

## Causal clock and formula

For the five-minute bar labelled `t`:

1. At `t+5m`, compute
   `flow = 2*taker_buy_quote/quote_asset_volume - 1`.
2. Compare `abs(flow)` with an 8,640-row rolling quantile whose entire baseline
   is shifted one row. The signal bar and every future row are excluded.
3. Require `sign(flow) * log(close/open) <= 0`: the completed bar did not move
   in the aggressive flow direction.
4. Compute per-venue defense as
   `-sign(flow) * book_credibility`. Require positive defense independently on
   USD-M and COIN-M and require their mean to exceed the selected threshold.
5. Enter `-sign(flow)` at the next five-minute open (`t+5m`).
6. Exit at the open 72 bars after entry, exactly six hours later.

Signals are non-overlapping within each UTC calendar quarter, and entry plus
exit must remain inside the same quarter. Missing source rows fail closed.

## Outcome-blind incidence calibration

An incidence-only pilot was inspected before this document. It did not load
post-entry outcomes. The complete disclosed grid is now frozen:

- flow quantiles: `0.75`, `0.80`, `0.85`;
- cross-collateral defense thresholds: `0.00`, `0.25`, `0.50`, `0.75`.

A cell passes only with all of the following:

- at least 120 non-overlapping events;
- at least 50 events in each half-year;
- at least 20 events in every quarter;
- each side is at least 35% of events;
- no quarter exceeds 40% of events.

Among passing cells, select the largest defense threshold first and then the
largest flow quantile. Returns cannot break a tie or select a cell. If no cell
passes, CBFR-72 is retired without opening outcomes.

## Orthogonality gate

Using timestamps and sides only, compare the selected clock against PDF-10,
CRRC-72, CSPR, and UMFR-36. A deterministic one-to-one match allows a tolerance
of 12 five-minute bars. Every comparison must have:

- Jaccard overlap no greater than `0.20`; and
- CBFR clock containment no greater than `0.30`.

Failure retires the candidate before return evaluation.

## Frozen eventual execution and risk accounting

Only after support passes and a separate evaluator is committed and hash-frozen
may calendar-2023 outcomes be opened. The evaluator must use:

- Binance `BTCUSDT` USD-M perpetual only;
- `0.5x` notional leverage;
- 6 bp per notional side (fee plus slippage), 12 bp round trip;
- realized funding cash flows while held;
- full declared wall-clock CAGR, including idle periods;
- strict MDD from the global pre-entry high-water mark through every held OHLC
  path, not close-only trade MDD.

Calendar 2024 and later remain sealed. A failed stage retires this exact policy;
threshold, holding-period, side, or filter repair after seeing outcomes is not
allowed.
