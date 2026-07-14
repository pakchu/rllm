# Binance Spot↔USD-M minute-order data design — 2026-07-14

## Purpose

CSPR and RIFT were reproducible but failed out of sample because five-minute
aggregation erased the ordering needed to distinguish cash-market discovery
from a simultaneous common shock. This source preserves only the causal
ordering observable inside each **completed** UTC five-minute bar by aligning
official Binance Spot and USD-M `BTCUSDT` one-minute klines.

The builder creates descriptors, not labels. It never computes a future return,
trade action, reward, or outcome. “Leadership” below means a lagged descriptive
asymmetry; it does not establish economic causation.

## Official source contract

- Binance archive layouts, kline schemas, and checksum convention:
  <https://github.com/binance/binance-public-data>
- Binance Spot kline endpoint and open-time identity:
  <https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data>
- Spot archive root:
  <https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/>
- USD-M archive root:
  <https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/>
- USD-M kline endpoint:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>

Each monthly ZIP and adjacent `.CHECKSUM` are fetched and SHA-256 verified in
memory. Raw archives are discarded. Only deterministic gzip features and JSON
provenance are retained. The initial build is hard-sealed to `[2020-01-01,
2024-01-01)`; no 2024+ price is opened during source or candidate selection.

## Availability and alignment contract

For a five-minute interval `[T, T+5m)`:

1. Spot and USD-M are joined on exact UTC one-minute **open time**.
2. A source is complete only with `{T, T+1m, ..., T+4m}`, valid OHLC, positive
   quote activity, and an exact one-minute close boundary.
3. Lagged pairs are only `0→1`, `1→2`, `2→3`, and `3→4` inside that interval.
   The forbidden `4→next-bar 0` pair is never read by the current feature row.
4. `feature_available_time_utc = trade_earliest_time_utc = T+5m`.
5. Any missing source minute, invalid denominator, or non-finite descriptor is
   quarantined with `cross_venue_feature_valid=false`.

A downstream signal indexed at `T` may therefore enter no earlier than the
open at `T+5m`.

## Normalized minute primitives

For venue `v`, minute `i`, quote notional `Q`, taker-buy quote `B`, and minute
path return `r`:

```text
flow_frac[v,i] = (2 * B[v,i] - Q[v,i]) / Q[v,i]
r[v,0]         = log(close[v,0] / open[v,0])
r[v,i>0]       = log(close[v,i] / close[v,i-1])
```

Raw Spot and USD-M notionals are never directly subtracted because venue scale
is structurally different. The output retains raw activity only for auditing.

## Five-minute descriptors

- Per venue: total quote activity, trade count, aggregate signed flow fraction,
  flow coherence, five-minute log return, absolute path return, quote-activity
  timing centroid, absolute-flow timing centroid, and absolute-return timing
  centroid.
- Lagged cross-venue response:

```text
S→P = sum(flow_frac[spot,i] * r[perp,i+1])
      / sum(abs(flow_frac[spot,i]))
P→S = sum(flow_frac[perp,i] * r[spot,i+1])
      / sum(abs(flow_frac[perp,i]))
```

  The builder emits both responses in basis points and their difference.
- Scale-free flow-transfer asymmetry:

```text
(sum(flow_spot[i] * r_perp[i+1]) - sum(flow_perp[i] * r_spot[i+1]))
/ sum(abs(each contribution from both arrows))
```

- Price-order asymmetry uses the same antisymmetric construction with lagged
  minute returns instead of flow fractions.
- Simultaneous flow-sign and return-sign agreement are controls, not evidence
  of leadership.
- Open/close Spot–perpetual basis and basis change are retained to distinguish
  transfer from simple convergence.

Timing centroids are normalized to `[0,1]`; smaller means earlier in the
completed bar. Cross-venue timing differences are `perp - spot`, so positive
values mean Spot activity occurred earlier.

## Required invariance tests

1. Mutating the next five-minute bar cannot change the current row.
2. Swapping Spot and USD-M negates antisymmetric arrows and timing differences
   while preserving simultaneous agreement.
3. Multiplying either venue's prices or activity by a positive constant cannot
   change normalized ordering features.
4. A missing or misaligned minute fails closed.
5. Flat price, zero flow support, and zero quote denominators never produce an
   accepted infinity/NaN.
6. Monthly rebuilds are checksum-aware and byte deterministic.

## Candidate boundary

Only after the full 2020–2023 source is built and independently audited may a
directional alpha be preregistered. Support/novelty selection must remain blind
to returns. Required controls include venue swap, reversed minute order,
simultaneous-only agreement, aggregate-only features without ordering, 1h/24h
staleness, exact direction flip, and overlap with CSPR/RIFT. The 2024+ seal may
be reconsidered only after a frozen pre-2024 evaluator passes.
