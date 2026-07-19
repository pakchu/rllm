# Binance COIN-M BTC liquidation-snapshot source design — 2026-07-19

## Purpose

Create a checksum-verified, causal five-minute source from Binance's official
`BTCUSD_PERP` force-order snapshots.  This stage opens no BTC market outcomes,
returns, funding, strategy labels, or PnL.

## Official semantics

- Binance calls `{symbol}@forceOrder` a **liquidation order snapshot** stream.
  It publishes at most the latest snapshot for a symbol in each 1,000ms
  interval, and emits nothing when no liquidation occurs.  It is therefore a
  censored snapshot feed, not a complete liquidation-fill tape:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-coin-m-futures/api/ws-streams/~/>.
- The official daily archive and adjacent `.CHECKSUM` files are listed at
  <https://data.binance.vision/?prefix=data%2Ffutures%2Fcm%2Fdaily%2FliquidationSnapshot%2FBTCUSD_PERP%2F>.
- COIN-M exchange information publishes `contractSize`; the current
  `BTCUSD_PERP` contract size is 100 USD:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-coin-m-futures/api/rest-api/market-data>.
- `SELL` is interpreted as a forced sale closing a long and `BUY` as a forced
  purchase closing a short.  This is standard order-side semantics; the public
  liquidation-stream page does not spell out the long/short mapping itself.
- The user force-order REST endpoint exposes only the past 90 days, so it cannot
  reconstruct a long complete history:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-coin-m-futures/api/rest-api/trade#users-force-orders>.

## Outcome-blind archive audit

- first published BTC archive: `2023-06-25`;
- last published BTC archive: `2024-10-14`;
- 472 daily ZIP/checksum pairs over 478 calendar days;
- missing archive dates: `2023-09-09`, `2023-09-23`, `2023-09-25`,
  `2024-06-01`, `2024-06-11`, `2024-06-12`;
- total ZIP payload is about 1.13 MiB;
- the full checksum-verified build contains 106,822 raw rows and 53,398 unique
  snapshots after removing 53,424 exact duplicates;
- 471 available days contain exactly two copies of every row.  `2023-09-21`
  contains 196 raw rows but only 85 unique rows, so some snapshots occur more
  than twice.  The builder removes exact whole-row duplicates only;
  same-millisecond rows with different contents remain distinct snapshots.

The builder must audit the entire range before this observation is treated as a
global property.  Missing days are explicit invalid five-minute bars, never
zero-liquidation imputations.

Frozen source artifacts:

- five-minute panel: 137,664 rows, of which 135,936 are source-valid and
  18,897 contain at least one snapshot;
- panel SHA-256:
  `a23b93d8567a589e9f045ae4a56393e493a8da2748c5a051804c9bdf9388ccc3`;
- manifest SHA-256:
  `5d78686e7c40d69261f09bc77e27ff734f682abba4abb95c2291e8282380053e`.

## Causal aggregation contract

- each filled snapshot contributes `accumulated_fill_quantity * 100 USD` face
  notional;
- `SELL` contributes long-liquidation notional and `BUY` contributes
  short-liquidation notional;
- first/last/min/max snapshot average price, within-bar log return, price range,
  and closing location are retained so a later preregistration can distinguish
  an unresolved cascade from an already-rejected burst without reading a future
  market bar;
- bars with a published archive but no snapshot are valid zero-event bars;
- a completed five-minute feature becomes available at bar end plus one second;
- executable research must use the first market open at or after that time,
  which normally imposes one full five-minute delay;
- raw ZIPs stay in memory and are not retained on disk;
- every ZIP is verified against its official checksum before parsing.

## Fixed research split allowed by the archive

The source is too short for the older pre-2024/2024+/2026 protocol.  If the
source passes integrity checks, the only admissible staged split is:

| stage | start inclusive | end exclusive | length |
|---|---|---|---:|
| train | 2023-06-25 | 2023-10-15 | 112 days |
| test | 2023-10-15 | 2024-04-15 | 183 days |
| eval | 2024-04-15 | 2024-10-15 | 183 days |

Test and eval are each six months.  Train is necessarily short, so candidate
complexity must be very small and promotion requires forward live-shadow
evidence.  No result from this archive alone can be called a production alpha.

## Stop conditions

Reject before market outcomes if checksum coverage, duplicate semantics,
timestamp bounds, contract quantities, side semantics, or six-month test/eval
boundaries cannot be enforced.  After outcomes open, reject on test failure and
keep eval sealed; do not repair thresholds from test or eval.
