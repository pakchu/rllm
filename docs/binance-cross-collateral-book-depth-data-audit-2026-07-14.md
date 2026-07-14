# Binance BTC cross-collateral book-depth audit — 2026-07-14

## Purpose and claim boundary

This work adds a new, outcome-blind data source for later alpha research. It
does **not** inspect future BTC returns, strategy PnL, CAGR, MDD, or labels.

The source is Binance's public daily `bookDepth` archive for two perpetual
contracts with different collateral structures:

- USD-M linear `BTCUSDT`;
- COIN-M inverse `BTCUSD_PERP`.

No prior repository alpha script or document referenced `bookDepth`. This
therefore adds direct displayed-liquidity information rather than another
transformation of the already mined price, funding, OI, or aggregate-trade
features.

Official source references:

- Binance public-data repository:
  <https://github.com/binance/binance-public-data>
- USD-M archive root:
  <https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/>
- COIN-M archive root:
  <https://data.binance.vision/data/futures/cm/daily/bookDepth/BTCUSD_PERP/>
- Binance COIN-M local-order-book semantics:
  <https://developers.binance.com/legacy-docs/derivatives/coin-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

The public-data repository is MIT-licensed and states that archived files have
companion checksum files. Every retained day was verified against that official
SHA256 sidecar. The build manifest records the hash of each source archive, so
an upstream replacement is detectable.

## Physical time boundary

- requested start: 2023-01-01 00:00 UTC
- exclusive end: 2024-01-01 00:00 UTC
- full output grid: 105,120 five-minute timestamps
- post-2023 rows requested: **false**
- outcomes opened: **false**

The builder rejects any start before 2023 or end after 2024 before making a
network request. Raw ZIPs are verified in memory and discarded; only a compact
five-minute panel remains.

## Raw archive audit

Each complete raw snapshot has exactly ten cumulative levels:
`-5,-4,-3,-2,-1,+1,+2,+3,+4,+5` percent. Duplicate levels, missing levels,
non-positive values, non-finite values, or cumulative depth that decreases as
distance increases fail closed.

| market | expected days | verified archives | raw rows | raw snapshots | accepted 5m bars |
|---|---:|---:|---:|---:|---:|
| USD-M `BTCUSDT` | 365 | 363 | 10,211,110 | 1,021,111 | 101,964 |
| COIN-M `BTCUSD_PERP` | 365 | 364 | 10,275,120 | 1,027,512 | 102,645 |

Officially absent archive dates are preserved as gaps rather than filled:

- USD-M: 2023-02-08 and 2023-02-09;
- COIN-M: 2023-09-25.

Some published days contain only a partial session. For example, both markets
have just a few late-day snapshots on January 9, 15, and 21; those days produce
zero accepted bars. Published-but-partial days are not treated as complete.

## Five-minute availability rule

Raw snapshots normally arrive about every 30 seconds, but the two markets are
not timestamp-synchronous. They are therefore aggregated independently over
the half-open interval `[bar_open, bar_open + 5m)`.

A venue bar is retained only when it has:

1. at least eight complete snapshots;
2. a first snapshot no later than 60 seconds after bar open;
3. a last snapshot at least 240 seconds after bar open.

For every retained venue bar, the median native `depth` is stored separately at
all ten distance levels. A combined row is marked `source_complete=true` only
when both venues pass. This produces 101,649 joint-complete rows, 96.70% of the
calendar grid.

| quarter | joint-complete 5m rows |
|---|---:|
| 2023 Q1 | 23,192 |
| 2023 Q2 | 25,799 |
| 2023 Q3 | 26,188 |
| 2023 Q4 | 26,470 |

No interpolation, forward fill, backward fill, nearest-time join, or compressed
event clock is used. A later signal must treat incomplete rows and any required
lookback crossing them as quarantined.

## Frozen output

- data:
  `data/binance_cross_collateral_book_depth_btc_2023/BTC_cross_collateral_book_depth_5m_2023.csv.gz`
- data SHA256:
  `53e16cf71581f03c7b1cc3da6a13222923ce68aa9e869d89f02078221bb4eee4`
- rows / columns: 105,120 / 28
- manifest:
  `results/binance_cross_collateral_book_depth_btc_2023_manifest.json`
- manifest SHA256:
  `95ec6e133dfcc7ed3c058538f380d24d98552c0a921fc24a679d247159a4f080`

The output stores dimensionless-useful bid/ask and near/far depth components
without pretending that USD-M and COIN-M native depth units are directly
comparable. A later experiment must compare within-market ratios or changes
before taking a cross-market difference.
