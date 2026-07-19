# Deribit expiry hedge-release mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **DEHR-72 — Deribit Expiry Hedge
Release, six-hour hold**. It will use the option `delivery` rows published by
Deribit's public settlement-history endpoint to estimate the terminal delta
that disappears at an expiry. A put-dominant in-the-money release tentatively
maps to a post-expiry BTC long; a call-dominant in-the-money release maps to a
post-expiry BTC short.

This is a source-axis and mechanism decision only. It opens no complete
historical delivery prefix, event incidence, post-entry BTC path, return, PnL,
or later calendar year. A bounded 1,000-row request was used only to verify the
current schema, option naming convention, pagination, positive reported
positions, and the common 08:00 UTC event clock.

## Why this is a new observable axis

The repository has already tested Deribit DVOL levels and their disagreement
with Binance BVOL. Those series contain aggregate implied-volatility state but
not the strike-by-strike positions that disappear at an option expiry.

Deribit's public `delivery` rows report, for each expired instrument:

- event timestamp;
- instrument name, including expiry, strike, and call/put type;
- reported `position`;
- delivery index price and mark price; and
- settlement profit/loss fields.

DEHR will ignore the profit/loss fields. It will parse only expired BTC options
and use the reported position together with strike, option type, and delivery
index price. For a terminally in-the-money call the holder delta is `+1`; for a
terminally in-the-money put it is `-1`; out-of-the-money terminal delta is
zero. The source-only release imbalance is therefore proportional to

```text
put_release  = sum(position for ITM puts)
call_release = sum(position for ITM calls)
net_release  = put_release - call_release
```

The economic hypothesis assumes that option liquidity providers are, in
aggregate, short customer convexity and hedge the opposite option delta before
expiry. Once cash delivery removes the option position, call-heavy terminal
delta implies hedge selling and put-heavy terminal delta implies hedge buying.
This dealer-side interpretation is a falsifiable research hypothesis, not a
claim made by Deribit. Open interest alone does not identify every holder or
dealer, and controls must reject the candidate if the option-type composition
adds no information over an expiry-time clock.

This mechanism is distinct from price momentum, Binance funding/premium/OI,
liquidation, book-depth geometry, Coin Metrics network state, social attention,
and aggregate DVOL/BVOL. Its event clock is an exchange-defined removal of
strike-specific derivative inventory.

## Official source contract

The official endpoint is:

```text
GET https://www.deribit.com/api/v2/public/get_last_settlements_by_currency
    ?currency=BTC&type=delivery&count=1000
```

Deribit documents this endpoint as a public history of settlement, delivery,
and bankruptcy events, with a continuation token for pagination and an
optional `search_start_timestamp` boundary:

- <https://docs.deribit.com/api-reference/market-data/public-get_last_settlements_by_currency>
- <https://docs.deribit.com/>

The documentation example contains a BTC option delivery from February 2019,
while the bounded production probe contained current daily expiries. Complete
coverage and continuity remain unopened until the source-only builder and
support contract are frozen.

The public market-trade endpoint is **not** an eligible substitute. Deribit's
current documentation says `public/get_last_trades_by_currency_and_time`
returns only the latest 24 hours. The separate `historical=true` documentation
applies to authenticated private user trade/order history, not to a free
historical public options tape:

- <https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency_and_time>
- <https://docs.deribit.com/articles/accessing-historical-trades-orders>

DEHR therefore makes no claim to pre-expiry option flow, quote, implied
volatility, or dealer identity.

## Causal availability and live parity

The historical row timestamp denotes the delivery event, not a documented
first-publication SLA. The candidate must not assume that all strike rows were
queryable at exactly 08:00 UTC.

The frozen availability rule will be conservative:

1. expiry event at `E` (normally 08:00 UTC);
2. no decision before `E + 60 minutes`;
3. require two identical canonical delivery sets observed five minutes apart
   in live/shadow collection;
4. use the later observation as `observation_end`; and
5. earliest executable entry is the next completed five-minute open.

For historical evaluation, where first-seen timestamps are absent, the
earliest synthetic observation time is `E + 65 minutes` and entry is one full
five-minute bar later. A later live arrival delays or cancels a trade; it can
never be backdated. This intentionally sacrifices freshness rather than
pretending the event timestamp proves publication.

Raw responses will remain ignored and local because no open redistribution
grant was found. Committed artifacts may contain source URLs, request and
response hashes, aggregate quality/support statistics, and a derived event
clock without instrument-level raw rows. Deribit terms remain binding for any
production use.

## Frozen research sequence

1. Commit this mechanism decision before downloading the complete delivery
   history or calculating DEHR incidence.
2. Implement a source-only downloader/parser with continuation-loop,
   duplicate, schema, option-name, expiry-clock, strike, position, and
   response-hash checks. It may read no Binance price, funding, or future path.
3. Before full incidence is opened, preregister one source-support contract:
   coverage, minimum expiry count, calendar dispersion, both release signs,
   concentration, and synthetic causality controls.
4. Reject without repair if the pre-2023 source cannot support a 2020–2021
   train and a separate 2022 test. Do not change the latency, terminal-delta
   definition, side map, or event family after support is known.
5. If support passes, freeze exactly one threshold, scheduler, six-hour hold,
   base/stress cost contract, strict path accounting, funding boundary, and
   mechanism controls before loading any post-entry BTC path.
6. Open 2020–2021 train first. Only a complete train pass may open 2022 test.
   Later 2023, 2024, 2025, and 2026 windows open sequentially and stop at the
   first failed sealed gate.
7. Required controls include expiry-time-only random side, direction flip,
   equal-position strike ablation, call/put-type ablation, delayed entry, and
   deterministic random side. Controls can falsify DEHR but cannot replace it.

The strict performance target remains positive absolute return, full-calendar
CAGR divided by global/pre-entry-HWM strict MDD of at least `3`, strict MDD no
greater than `15%`, stress-cost survival, adequate trade and weekly-cluster
counts, balanced long/short support, positive contained subperiods, and a
mechanism-control margin. All held five-minute paths, entry cost, exact
funding, virtual adverse exit fee, and actual exit remain part of strict MDD.

The branch is globally contaminated by prior BTC research. This sequence can
support only a candidate-level frozen claim, never a pristine global holdout.

## Rejected operational-announcement source

The Binance API announcement channel linked by Binance's official API
documentation was also checked without market outcomes. Its complete public
Telegram history contains only 46 consecutive messages, IDs 1–46, from
2017-12-07 through 2022-05-23. It cannot cover the recent year or a three-year
sequential validation and is rejected at source feasibility.

- Official API documentation linking the channel:
  <https://github.com/binance/binance-spot-api-docs>
- Public channel archive: <https://t.me/s/binance_api_announcements>

