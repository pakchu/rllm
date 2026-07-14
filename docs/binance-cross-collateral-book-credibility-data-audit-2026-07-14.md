# Binance cross-collateral book-credibility audit — 2026-07-14

## Purpose and claim boundary

This work adds an outcome-blind measure of whether displayed cumulative depth
is stable, replenishing, or flickering inside each five-minute bar. It does
**not** inspect future BTC returns, labels, strategy PnL, CAGR, MDD, or a trading
action.

The prior panel retained only median depth. That could not distinguish depth
visible throughout a bar from a similar median produced by rapid cancellation
and replacement. The new panel replays the exact same official archives and
adds three scale-free path statistics at every `-5..-1,+1..+5` percent level
for both:

- USD-M linear `BTCUSDT`;
- COIN-M inverse `BTCUSD_PERP`.

Official source references:

- Binance public-data repository:
  <https://github.com/binance/binance-public-data>
- Binance maintainer confirmation that USD-M `bookDepth` is sampled every
  30 seconds:
  <https://github.com/binance/binance-public-data/issues/437>
- Binance maintainer confirmation of the `-5..-1,+1..+5` bands and
  `depth`/`notional` fields:
  <https://github.com/binance/binance-public-data/issues/447>
- USD-M archive root:
  <https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/>
- COIN-M archive root:
  <https://data.binance.vision/data/futures/cm/daily/bookDepth/BTCUSD_PERP/>
- live order-book update-ID semantics, which are **not** present in this
  snapshot archive:
  <https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

Primary research supports studying resiliency and relative additions versus
cancellations, but does not establish this factor's profitability:

- limit-order-book resiliency after market-order shocks:
  <https://arxiv.org/abs/1602.00731>
- meso-scale order flow and resiliency:
  <https://arxiv.org/abs/1708.02715>
- quote imbalance and next-price-move dynamics:
  <https://arxiv.org/abs/1312.0514>

## Physical time and archive boundary

- requested start: 2023-01-01 00:00 UTC
- exclusive end: 2024-01-01 00:00 UTC
- output grid: 105,120 five-minute timestamps
- post-2023 rows requested: **false**
- outcomes opened: **false**

The builder rejects any request before 2023 or after 2023 before a network
call. It downloaded 730 venue-days, checksum-verified every available ZIP in
memory, and retained no raw archive. The same 727 archives and three official
gaps as the frozen median-depth build were reproduced:

- USD-M missing: 2023-02-08 and 2023-02-09;
- COIN-M missing: 2023-09-25.

Every available-day archive SHA256, raw-row count, and snapshot count had to
match the frozen base manifest. A changed or newly appearing archive fails the
build instead of silently changing history.

## Five-minute availability and statistics

The acceptance rule is unchanged. A contract bar in
`[bar open, bar open+5m)` needs at least eight complete snapshots, first
snapshot no later than 60 seconds, last snapshot no earlier than 240 seconds,
and all ten cumulative levels per snapshot. Markets are aggregated
independently and then joined; there is no interpolation or nearest-time join.

For positive cumulative depth `X_i` ordered by snapshot time at one fixed
contract/side/distance level, the new statistics are:

- `log_mad = median(abs(log(X_i) - median(log(X))))`;
- `log_net = log(X_last) - log(X_first)`;
- `log_step = mean(abs(log(X_i) - log(X_(i-1))))`.

`log_mad` is within-bar dispersion, `log_net` is net displayed-depth change,
and `log_step` is average observed path activity. They are dimensionless and
can be compared only after product-local normalization. They are **net
snapshot proxies**, not counts of order submissions or cancellations.

All 60 credibility fields are finite on every joint-complete row. Observed
complete-row distributions across every market/side/level are:

| statistic | minimum | p01 | median | p99 | maximum |
|---|---:|---:|---:|---:|---:|
| `log_mad` | 0.0000 | 0.000715 | 0.006079 | 0.059523 | 1.488819 |
| `log_net` | -15.266985 | -0.152407 | 0.001614 | 0.145423 | 3.386476 |
| `log_step` | 0.0000 | 0.001372 | 0.009322 | 0.071386 | 3.408473 |

Rare extreme net/path values are preserved rather than outcome-conditionally
deleted. A downstream experiment must use causal robust ranks/z-scores and
fail closed on non-finite values. It may not choose clipping from future return
behavior.

## Exact base-panel replay

The original 28-column depth/timing panel was rebuilt from the new downloads
and compared against the frozen gzip. Dates, missingness, booleans, and all
values had to match; numeric tolerance was limited to absolute `1e-10` solely
for decimal CSV round-trip representations such as `17305.4065` versus
`17305.406500000003`. A `1e-6` perturbation is covered by a failing regression
test.

- USD-M accepted bars: 101,964
- COIN-M accepted bars: 102,645
- joint-complete bars: 101,649
- base depth replay: **passed**

## Frozen output

- builder commit: `0829399`
- builder SHA256:
  `c08d810b4197a464e00f78b7bd145cbead1e6bffa862f698f3d0caee9e5f043b`
- data:
  `data/binance_cross_collateral_book_credibility_btc_2023/BTC_cross_collateral_book_credibility_5m_2023.csv.gz`
- data SHA256:
  `45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429`
- data size: 53,759,901 bytes
- rows / columns: 105,120 / 88
- manifest:
  `results/binance_cross_collateral_book_credibility_btc_2023_manifest.json`
- manifest SHA256:
  `f530f472765c8cb56bf564efd346c734e2404e072b87e2ff8dc3b84e303c30f7`

The WSL filesystem remained at 294 GB used, below the user's 300 GB ceiling.

## Limitations and safe next use

This is a nominal 30-second snapshot archive, not event-level L2. It cannot
recover individual order lifetime, exact cancel/replace messages, queue
position, or sub-30-second churn. Snapshot differences may aggregate many
hidden events. Consequently, the next alpha must call additions/cancellations
**net proxies**, use a minutes-scale horizon, execute no earlier than the next
five-minute open, and compare UM versus CM only after dimensionless/product-
local standardization.
