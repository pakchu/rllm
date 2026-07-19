# BitMEX insurance-fund absorption mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **IFAR-288 — Insurance-Fund Absorption
Reversal**. It will use only the daily `XBt` wallet balance published by the
BitMEX insurance endpoint and a completed pre-snapshot BTC move to identify a
directional forced-liquidation episode. The tentative economic action is to
trade against that completed move after the default fund has reported a net
loss from liquidation trades.

This decision opens no complete IFAR source history, event incidence,
post-decision return, PnL, or 2023+ row. The three-row API request used to
verify schema and historical reach exposed no candidate count or outcome.

## Why this is a different data-generating mechanism

BitMEX's current Exchange Rules define the insurance fund as a default fund.
Profit or loss from a liquidation trade is booked to that fund, and an
insufficient allocated balance can trigger auto-deleveraging. The public
`walletBalance` change is therefore a venue-level realized absorption state,
not another estimate derived from Binance price, open interest, funding,
premium, liquidation snapshots, or order-book geometry.

The candidate hypothesis is:

1. a negative `XBt` balance change means liquidation trades generated a net
   loss for the default fund over the completed daily accounting interval;
2. a same-interval, sufficiently large BTC move supplies the otherwise absent
   directional context;
3. the fund loss confirms that forced risk transfer was costly rather than
   merely high-volume trading; and
4. after a deliberately long publication embargo, part of the completed move
   can reverse as the impaired positions have already been absorbed.

The endpoint is exchange-wide and the rules do not state that every balance
change is caused only by `XBTUSD`. IFAR must therefore be described as a
BitMEX `XBt` default-fund stress proxy, not as a complete BTC liquidation tape.
The pre-snapshot BTC move is mandatory; the fund sign alone will never choose
a trade direction.

Official references:

- [BitMEX insurance-fund history endpoint](https://docs.bitmex.com/api-explorer/get-insurances)
- [BitMEX API overview](https://docs.bitmex.com/api-explorer/bitmex-api)
- [BitMEX Exchange Rules, sections 15–17](https://www.bitmex.com/legal/exchange-rules)
- [BitMEX application/API overview](https://www.bitmex.com/app/apiOverview)
- [BitMEX Terms of Service](https://www.bitmex.com/terms)

## Causal source boundary

The official REST endpoint is `GET https://www.bitmex.com/api/v1/insurance`.
Its current contract exposes `currency`, `timestamp`, and `walletBalance`,
supports chronological filters and offset pagination, and caps one request at
500 rows. A source probe returned consecutive `XBt` observations at
`12:00:00Z` in January 2020. The API changelog describes the insurance
WebSocket table as a daily update, while the general API documentation says
table data is available over WebSocket.

The documentation does **not** provide a historical point-in-time SLA for the
first publication of each 12:00 row. IFAR will therefore impose a full
snapshot-day embargo:

- accounting snapshot `D 12:00 UTC`;
- no decision from that row before `D+1 12:00 UTC`;
- one complete five-minute latency bar;
- earliest entry `D+1 12:05 UTC`.

Live code must additionally fail closed until the expected timestamp is
actually observed. This is intentionally slower than assuming the historical
timestamp equals publication time.

Only `XBt` rows strictly before `2023-01-01` may be downloaded for source
support and train/test selection. The pre-decision BTC path may use the already
audited Binance BTCUSDT five-minute prefix, but source-only support may parse
no bar after its embargoed decision time and no funding mark.

## Data-use boundary

BitMEX advertises its APIs for applications and publishes the insurance
history endpoint, but its Terms of Service apply to API use and the Exchange
Rules disclaim the completeness and accuracy of public data. This repository
will not redistribute the raw insurance history:

- raw responses stay ignored and local;
- committed artifacts contain only source URLs, row/range audits, hashes, and
  aggregate support counts;
- the research does not grant a third-party data license; and
- any future redistribution or external commercial data service requires a
  separate terms/legal review.

## Why the other newly checked sources are not selected

- Wikimedia attention is not new here. The frozen 14-policy attention family
  was already rejected before its holdout in commit `0936644`.
- Coinbase Statuspage is causally attractive for forward collection, but its
  current public incident surfaces expose only a shallow recent archive, not a
  reproducible 2020–2022 prefix.
- Deribit has a deterministic 08:00 UTC expiry and official settlement
  endpoints, but the official public interfaces do not establish a complete
  free 2020–2022 pre-expiry microstructure archive. Current membership terms
  also make redistribution/processing a separate permission boundary.

These sources may remain forward-shadow inputs. They cannot replace the
three-year candidate sequence requested here.

## Frozen research sequence

1. Commit this mechanism decision before downloading the complete `XBt`
   prefix or calculating IFAR incidence.
2. Freeze one source downloader, one source-only feature definition, one
   support gate, and synthetic causality tests.
3. Download only pre-2023 insurance rows; hash the private source and commit
   only its manifest and aggregate source audit.
4. Reject without repair if support, calendar dispersion, or long/short
   balance fails. Do not change the embargo, fund threshold, price threshold,
   side, or hold after seeing incidence.
5. If support passes, commit and hash-freeze one strict evaluator before
   loading any post-entry pre-2023 path.
6. Use 2020–2021 as train and 2022 as test. Both must pass absolute-return,
   full-calendar CAGR/strict-MDD, stress-cost, delayed-entry, and statistical
   gates before any 2023+ insurance row is requested.
7. Open 2023, 2024, 2025, and 2026 sequentially. Stop at the first failed
   sealed year and do not repair the policy.

The branch is globally contaminated by prior BTC research. This process can
support only a candidate-level frozen sequence, not a pristine global human
holdout claim.
