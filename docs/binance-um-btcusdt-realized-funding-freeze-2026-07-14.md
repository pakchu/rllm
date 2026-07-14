# Binance BTCUSDT USD-M realized funding freeze — 2026-07-14

## Frozen source

The public Binance USD-M funding-history endpoint was queried only for the
inclusive interval `2020-01-01T00:00:00.000Z` through
`2023-12-31T23:59:59.999Z`. No 2024+ funding row was requested or serialized.

- source-builder commit:
  `86d0e71ea516f060ac1461888f54f7908f2b8301`
- endpoint: `GET https://fapi.binance.com/fapi/v1/fundingRate`
- official endpoint documentation:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history>
- official funding direction explanation:
  <https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets>
- frozen data:
  `results/binance_um_btcusdt_realized_funding_2020_2023.csv`
- data SHA-256:
  `c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7`
- manifest:
  `results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json`
- manifest SHA-256:
  `c70280e46bcbc2410cc59c2bcc93780c40997dbc5d0edb82d82127b59593250c`

Binance documents `startTime` and `endTime` as inclusive, returns rows in
ascending order, and defines `fundingTime`, `fundingRate`, and the mark price
associated with the charge. Pagination advanced from the final returned
millisecond plus one, preventing duplicated inclusive boundary rows.

## Data facts

| Year | Settlements |
|---|---:|
| 2020 | 1,098 |
| 2021 | 1,095 |
| 2022 | 1,095 |
| 2023 | 1,095 |
| **Total** | **4,383** |

- first settlement: `2020-01-01T00:00:00.000Z`
- last settlement: `2023-12-31T16:00:00.000Z`
- timestamps are strictly increasing and unique
- symbol is uniformly `BTCUSDT`
- decimal funding-rate strings are retained exactly as returned
- the exact returned timestamps are used; an eight-hour cadence is not assumed
- two consecutive official downloads produced byte-identical data and manifest

The frozen rows were independently compared with the existing local Binance
archive
`data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz`
(SHA-256
`4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7`).
All 4,383 timestamps matched exactly; every funding rate and available mark
price matched numerically. The archive comparison is corroboration, not the
frozen evaluator input.

## LURI application boundary

Official Binance material states that positive funding is paid by longs to
shorts and negative funding reverses that transfer. The preregistered LURI
account factor therefore remains:

```text
product(1 - 0.5 * side * funding_rate)
```

for every exact settlement satisfying the conservatively inclusive evaluator
condition `entry_time <= funding_time <= exit_time`. This endpoint inclusion
condition and the `0.5x` account approximation were frozen by LURI before any
outcome was opened; they are not inferred from an assumed settlement cadence.

The funding source is now immutable. The next commit must freeze the evaluator
and all primary/control schedules before loading either this file or the USD-M
post-entry OHLC path. Calendar 2024 and later remains sealed.

## Version note

The official API page was read on 2026-07-14 from Binance's current developer
portal. Binance's current explanatory article discusses interval changes after
2025; those changes are irrelevant here because the dataset ends in 2023 and
the evaluator consumes explicit historical settlement timestamps rather than a
hard-coded interval.
