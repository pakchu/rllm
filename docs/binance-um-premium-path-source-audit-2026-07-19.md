# Binance USD-M premium path source audit — 2026-07-19

## Frozen result

The official Binance Vision monthly `BTCUSDT` USD-M
`premiumIndexKlines/1m` archives were downloaded for
`[2020-01-01, 2026-07-01)`. All 78 published ZIPs matched their adjacent
checksums. No BTC execution OHLC, return, funding or PnL was opened.

| item | value |
|---|---:|
| complete UTC one-minute grid | 3,417,120 rows |
| source-valid premium rows | 3,405,399 |
| explicitly invalid missing rows | 11,721 |
| published monthly archives | 78 / 78 |
| data SHA-256 | `7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9` |
| manifest SHA-256 | `821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8` |

## Exchange gaps preserved

The published archives are not a perfectly complete one-minute stream. The
builder preserved every absent row as `source_valid=false`; no value was
forward-filled or replaced by zero.

| month | missing rows |
|---|---:|
| 2020-01 | 29 |
| 2020-12 | 106 |
| 2021-07 | 7,200 |
| 2022-07 | 50 |
| 2022-10 | 1,440 |
| 2023-02 | 1,440 |
| 2023-11 | 14 |
| 2024-08 | 2 |
| 2026-06 | 1,440 |

Any candidate path containing an invalid minute is invalid. Rolling reference
windows must enforce their preregistered completeness threshold rather than
silently treating an exchange outage as market information.

## Causal boundary

- Source close time is exactly bar open plus 59.999 seconds after normalizing
  Binance's millisecond/microsecond archive transition.
- Conservative feature availability is bar open plus 61 seconds.
- Only premium-index OHLC is retained. It is derivatives state, not an
  exogenous variable, but it is not BTC execution-price outcome data.
- This artifact authorizes source-only clock construction. It does not
  authorize opening a BTC price, return, funding or PnL file.
