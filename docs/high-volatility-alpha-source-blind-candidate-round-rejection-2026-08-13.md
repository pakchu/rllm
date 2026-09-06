# High-volatility alpha source-blind candidate round rejection — 2026-08-13

## Decision

Reject this candidate round before preregistration and before opening any new
source value, candidate incidence, Gross9 clock, execution price, funding, or
economic outcome.  None of the reviewed ideas is both independent of terminal
repository candidates and supported by a causally reproducible source.

This decision does not authorize a repair, source substitution, threshold
change, clock change, side change, hold change, universe change, or promotion
of a diagnostic control from any prior candidate.

## Rejected axes

### Address activity utilization

`AdrActCnt / AdrBalCnt` is not a new object.  ARCR-864 already froze
`turnover = log(AdrActCnt_t / AdrBalCnt_t)` in
`training/preregister_address_reservoir_capacitance_support.py`.  A high-rank
utilization relay would therefore be a semantic repair of that terminal family.

### Remaining Coin Metrics Community fields

The unused catalog fields reviewed in this round do not supply an independent
high-volatility alpha:

- `ROI30d` and `ROI1yr` are trailing price returns;
- `IssTotUSD` is native issuance multiplied by USD price;
- `SplyExpFut10yr` is the known issuance schedule extrapolation;
- `CapMrktEstUSD` is circulating supply multiplied by USD price;
- `PriceBTC` is a price-numeraire transform; and
- `FlowOutExNtv` is an exchange-flow proxy adjacent to the terminal exchange
  reserve and deposit candidates.  The repository already records that BTC
  exchange net flow lacked sufficiently general return-predictive support.

The inaccessible address-cohort metrics remain rejected by
`docs/address-cohort-residual-transport-source-rejection-2026-07-20.md`, and
the realized-cap ratchet remains rejected by
`docs/realized-cap-ratchet-absorption-source-feasibility-rejection-2026-07-20.md`.

### Binance microstructure proposals

The proposed exhaustion, taker-flow acceleration, and flow-sign-flip rules are
not independent.  The repository already contains frozen aggressive-flow
confirmation, aggressive-flow flip absorption, taker-imbalance persistence,
taker-imbalance seasonal innovation, taker-imbalance concentration, daily flow
acceleration, jump-flow confirmation, and liquidity-vacuum families.  Recasting
those primitives at another threshold or hold would be a prohibited repair.

### Published intraday return mechanisms

The reproducible literature mechanisms found in this round collide with
already frozen candidates:

- Wen, Bouri, Xu, and Zhao's intraday momentum/reversal result is already the
  basis of the high-volatility intraday-hour reversal family;
- large negative-shock rebound is already covered by frozen normalized-price
  spike and bipower-jump reversal families;
- cumulative intraday-return curve forecasting is already covered by the
  frozen functional curve projection relay; and
- short-horizon cross-cryptocurrency return diffusion is already represented
  by frozen cross-alt response, lag-transfer, leadership, and spillover
  families.  Its reported ten-minute horizon is also not suitable for the
  frozen 20 bp mean-gross-move gate without inventing a longer holding-period
  adaptation.

### Deribit option risk reversal and butterfly

The 25-delta 90-day risk-reversal / butterfly literature is directionally
interesting, but an exact 2021 through 2026-08-01 history cannot be reproduced
from the official free unauthenticated Deribit surfaces:

- public ticker greeks and `mark_iv` are current snapshots, not a historical
  delta archive;
- `public/get_instruments` does not document a complete all-expiry instrument
  archive back to 2021;
- `public/get_mark_price_history` covers only a subset of options used in the
  volatility-index calculation; and
- no official public historical order-book or complete option-surface archive
  is documented.

The repository's broader historical option-trade route is independently
terminal at
`docs/deribit-options-trade-ledger-source-rejection-2026-07-24.md`; substituting
a relaxed schema or incomplete option subset would be a source repair.

Official references reviewed for this feasibility boundary:

- <https://docs.deribit.com/api-reference/market-data/public-get_instruments>
- <https://docs.deribit.com/api-reference/market-data/public-get_mark_price_history>
- <https://docs.deribit.com/api-reference/market-data/public-ticker>
- <https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency>
- <https://docs.deribit.com/articles/api-usage-policy>

## Evidence boundary

- Repository source code, preregistrations, result metadata, rejection
  decisions, and CSV headers were inspected.
- No new CSV data row beyond a header was decoded for candidate selection.
- No new network market-data response body was opened.
- No new candidate incidence or event count was computed.
- No Gross9 row, post-entry return, execution price, funding value, PnL, CAGR,
  MDD, sign-flip statistic, stress result, or RV20 audit was opened.

## Stopping rule for this round

This round is terminal unchanged.  Future work must begin from a genuinely
different source object with a documented causal history and must preregister
its singleton formula before opening source values.  Nothing in this document
authorizes reopening any rejected axis above.
