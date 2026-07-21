# Bitfinex margin warehouse deployment source decision — 2026-07-20

## Decision

The next standalone BTC candidate will use a new observable axis: **public
Bitfinex margin-funding balance-sheet state**, provisionally named
**BFMWD — Bitfinex Funding Margin Warehouse Deployment**.

This source decision opens no BTC price, future return, Binance funding,
portfolio PnL, CAGR, MDD, or 2024+ row. It freezes only the source, causal
interpretation boundary, collection window, and next research sequence.

## Why this is a new axis

The repository has repeatedly tested Binance perpetual funding, premium, OI,
top-trader ratios, taker flow, REX, Kimchi/FX, Coinbase leadership, COIN-M
curve migration, order-book geometry, cross assets, macro releases, chain
activity, and narrative data. It has not used Bitfinex public margin-funding
statistics or Bitfinex's funding-trade direction/tenor.

Bitfinex exposes a distinct financed-spot balance sheet rather than another
perpetual funding-rate estimate:

- `FUNDING_AMOUNT`: total funding provided;
- `FUNDING_AMOUNT_USED`: provided funding currently used in positions;
- `FRR`: the Flash Return Rate;
- `AVG_PERIOD`: average funding period; and
- separate public symbols such as `fUSD` and `fBTC`.

Bitfinex also documents public funding trades. Their signed amount is positive
when funding is **provided** and negative when funding is **taken**, and each
trade carries its realized rate and period. That gives a future extension a
same-schema historical/live event stream without inferring borrower direction
from price.

Official sources:

- funding-stat history:
  <https://docs.bitfinex.com/reference/rest-public-funding-stats>;
- public funding-trade history:
  <https://docs.bitfinex.com/reference/rest-public-trades>;
- live public funding-trade WebSocket:
  <https://docs.bitfinex.com/reference/ws-public-trades>;
- public position/funding/credit statistics:
  <https://docs.bitfinex.com/reference/rest-public-stats>; and
- public API access and rate-limit requirements:
  <https://docs.bitfinex.com/docs/requirements-and-limitations>.

The official API guide says public REST/WebSocket access does not require a
Bitfinex account and explicitly describes the API as supporting customized
trading strategies. Any production use remains subject to Bitfinex's API and
market-data terms.

## Bounded source probes

On 2026-07-20, bounded five-row requests proved that the public historical
interfaces return both ends of the required frozen prefix without downloading
or joining any BTC outcome:

| source | 2020 observation | 2023 year-end observation | fields |
|---|---|---|---|
| `fUSD` funding stats | available | available | MTS, FRR, average period, total, used |
| `fBTC` funding stats | available | available | MTS, FRR, average period, total, used |
| `fUSD` funding trades | available | available | ID, MTS, signed amount, rate, period |
| `fBTC` funding trades | available | available | ID, MTS, signed amount, rate, period |
| `fUSD` credit used on `tBTCUSD` | available | available | MTS, amount |
| `fBTC` credit used on `tBTCUSD` | available | available | MTS, amount |
| `tBTCUSD` long/short position size | available | available | MTS, amount |

The initial frozen source artifact intentionally downloads only hourly `fUSD`
and `fBTC` funding statistics. The minute-level pair-credit/position feeds and
tick-level funding trades are admissible future source extensions only after a
separate pre-outcome amendment; they cannot be silently added to repair a
failed result.

## Causal interpretation boundary

`fUSD` and `fBTC` describe funding currencies, not guaranteed BTC-only intent.
The initial mechanism may therefore claim only:

1. unused provided funding is a venue-level financing warehouse;
2. a later increase in used funding while unused funding contracts is a
   warehouse-to-position deployment transition; and
3. `fUSD` versus `fBTC` supplies a directional hypothesis because borrowing
   quote currency can finance asset purchases while borrowing BTC can finance
   BTC sales.

It may **not** claim that every `fUSD` change is a BTC long, every `fBTC`
change is a BTC short, or that Bitfinex represents the whole market. Those are
falsifiable hypotheses and require source-only controls and economic testing.

## Frozen source clock

- physical source window: `[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`;
- symbols: exactly `fUSD` and `fBTC`;
- endpoint page size: 250;
- rate limit: at most 15 requests per minute;
- raw observation time: official `MTS`;
- conservative research availability: `floor(MTS, 1h) + 15 minutes`;
- 2024 and later are physically forbidden in this source artifact; and
- no BTC market, return, funding-payment, label, portfolio, or PnL loader is
  allowed in the builder.

## Frozen sequence

1. Commit the source decision, downloader, and unit tests.
2. Download/checksum only 2020–2023 `fUSD`/`fBTC` funding statistics.
3. Audit coverage, gaps, finite values, utilization identities, source-only
   event incidence, calendar concentration, and novelty against source clocks.
4. Commit exact feature formula, direction, thresholds, hold, controls,
   multiple-testing procedure, costs, and stopping gates before reading BTC
   outcomes.
5. Open 2021–2022 train only. A frozen train pass may open 2023 selection.
   No 2024+ window may be accessed until both pre-2024 stages pass.

No side inversion, threshold repair, hold repair, LLM rescue, or post-failure
source extension is permitted under the BFMWD candidate identifier.
