# Binance USDT collateral-breadth source audit — 2026-07-20

## Verdict

The initial UCBR source prefix passed its integrity and live-parity gate.
Twenty official Binance Spot monthly `1h` archives were verified against both
their published checksums and the hashes frozen before the complete build.
They produced 3,672 exact common hours in `[2023-08-01, 2024-01-01)`.

No BTC price, perpetual OHLC, funding, future return, label, PnL, CAGR, MDD, or
real UCBR event incidence was opened. Source distributions are not evidence of
profitability.

## Frozen source

| Item | Value |
|---|---|
| Symbols | `USDCUSDT`, `TUSDUSDT`, `USDPUSDT`, `FDUSDUSDT` |
| Interval | completed UTC `1h` Spot klines |
| Range | `[2023-08-01, 2024-01-01)` |
| Official archives | 20 monthly ZIPs plus published SHA-256 files |
| Exact common rows | 3,672 |
| Missing / duplicate common hours | 0 / 0 |
| Source availability | hour start plus exactly one hour |
| Rows with four valid books | 3,626 |
| Rows with three valid books | 46 |
| Rows with fewer than three valid books | 0 |

Official references:

- Binance public-data schema and checksum policy:
  <https://github.com/binance/binance-public-data>
- official Spot monthly-kline archive root:
  <https://data.binance.vision/?prefix=data/spot/monthly/klines/>
- live symbol metadata:
  <https://data-api.binance.vision/api/v3/exchangeInfo>

All four symbols were live and Spot-enabled at the audit time. Production must
use final live klines rather than treating later archive publication as the
historical decision timestamp.

## Persisted schema

For each symbol, the builder persists only:

```text
log_close = log(direct stablecoin / USDT hourly close)
valid = current hour has positive base volume, quote notional, and trade count
```

It also stores `valid_breadth`, `source_complete = valid_breadth >= 3`, exact
source time, and availability time. Raw open/high/low/close, volume, quote
notional, trade count, and taker fields are discarded after validation. The
46 three-book rows are all caused by `USDPUSDT` inactivity; they remain usable
only under a later preregistered three-of-four breadth rule and cannot be
imputed as active USDP evidence.

## Source-only distributions

| Direct log price | Min | 1% | Median | 99% | Max | Valid hours |
|---|---:|---:|---:|---:|---:|---:|
| `USDCUSDT` | -0.001101 | -0.000700 | -0.000100 | +0.001399 | +0.002098 | 3,672 |
| `TUSDUSDT` | -0.004510 | -0.003105 | -0.001201 | +0.001299 | +0.004291 | 3,672 |
| `USDPUSDT` | -0.003105 | -0.001001 | -0.000100 | +0.002397 | +0.012225 | 3,626 |
| `FDUSDUSDT` | -0.002403 | -0.001601 | -0.000300 | +0.002725 | +0.009950 | 3,672 |

The different issuer offsets and tail widths prohibit pooling raw levels.
Every later z-score must normalize each issuer separately with strictly prior
history. Cross-issuer breadth must use only members valid in the current hour.

## Integrity anchors

- builder commit: `0c59675`
- builder SHA-256:
  `a962ae5c774a837da481403cba2a6061f93bbdcc25d08451fb487e3c42f09ef7`
- source panel:
  `data/binance_usdt_collateral_breadth_2023/stablecoin_usdt_breadth_1h_2023-08-01T00_2023-12-31T23.csv.gz`
  - SHA-256:
    `e96fae39c869f6db0dc30bccc5b2fa72f5e7f717c2528038afede18dd5b9892d`
- source manifest:
  `data/binance_usdt_collateral_breadth_2023/build_manifest.json`
  - SHA-256:
    `26e142b818306275d48690711b7adca00b43750041d104e5c27a65b355c424f2`

Two complete network builds were byte-identical. The first measured build used
about 127 MB maximum resident memory and 1.46 seconds wall time on this host.

The next work unit must freeze one UCBR breadth clock, source-only controls,
support gates, and SDDR/SQFD novelty limits before calculating real incidence.
