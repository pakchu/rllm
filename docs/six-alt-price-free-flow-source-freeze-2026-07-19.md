# Six-alt price-free flow source freeze — 2026-07-19

## Decision

A new source axis is frozen for the next BTC alpha attempt: completed-hour
aggressive quote-flow from six Binance USD-M alt perpetuals. This work unit
opens **no BTC market, funding, return, excursion, PnL, or equity outcome**.
It is a source transformation, not evidence that the candidate is profitable.

## Frozen universe and range

- Symbols: `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`
- Raw cadence: exact UTC five-minute grid
- Raw range: `2023-01-01 00:00` through `2026-05-31 23:55`
- Output cadence: completed UTC hour
- Feature availability: the right edge of the completed hour
- Intended earliest BTC entry: five minutes after feature availability; this
  entry rule is not applied in the source builder.

## Physical evidence boundary

The builder physically reads only:

1. `date`
2. `quote_asset_volume`
3. `number_of_trades`
4. `taker_buy_quote`

The repository-local inputs also contain OHLC and base-volume fields, but the
transform does not read their values. It does not read BTC data. The normalized
hourly imbalance is:

```text
(2 * taker_buy_quote_usdt - quote_volume_usdt) / quote_volume_usdt
```

Therefore “price-free” here means **no OHLC or return fields**, not that quote
notional is economically independent of the traded price level.

## Exact-grid and invalid-hour handling

- Every symbol matched all `359,136` expected five-minute timestamps.
- Every output hour contains exactly twelve source rows.
- The panel has `179,568` hour-symbol rows (`29,928 × 6`).
- `179,532` rows are feature-valid.
- Six exchange-wide hour boundaries contain at least one zero-activity source
  bar for every symbol, producing `36` invalid hour-symbol rows.
- Invalid hours retain aggregate diagnostics but blank `taker_flow_fraction`
  and `mean_ticket_usdt`; there is no fill, stale carry, or interpolation.

## Candidate enabled by this source

The next bounded candidate is **FCDR-1 (Flow-Centrality Dominance Relay)**:

1. estimate a directed lag-one network of normalized alt taker imbalance using
   strictly prior completed hours;
2. use the prior network to weight the current completed-hour imbalances;
3. require disagreement between influential-flow direction and equal-weight
   crowd breadth;
4. follow the influential-flow direction in BTC;
5. trigger only on false-to-true onset, enter after five minutes, and hold a
   fixed twelve hours.

This differs from CLD-72 because it uses neither alt returns, residual HHI, nor
BTC lag. It differs from SQFD because its network is cross-alt USD-M flow rather
than BTC spot quote-book diffusion. It is still adjacent to prior breadth and
topology families, so novelty must be measured against committed clocks before
any BTC outcome is opened.

## Known limitations before support selection

- The current six-symbol universe has deployable-universe/survivorship risk.
- This transform freezes exact local input bytes but does not newly prove the
  upstream monthly archive checksum chain.
- Historical completed klines do not prove that a live collector can finalize
  all six symbols before `hour + 5m`; live parity remains a production gate.
- Centrality, threshold, side, invalid-window, tie, and negative-edge contracts
  must be fixed before train outcomes are opened.

## Frozen hashes

- Builder SHA-256: `7e6a212ab2eeb30ef69e4f9ea5772b757c46fa46dae5f1ea52c0732289b28506`
- Panel SHA-256: `bf4d67ee02948444712a6ff7862a0d4f4ae4ae2a704c9d0586538043c169f6b9`
- Manifest SHA-256: `eab61cbc7f5fc51e78f574e8bef163b3a3b91bd027136cae8efd7aaf26edc0f1`
- Input summary SHA-256: `9010bf1110e2555581942d6a0400d98466b64a5bee355c05ce5b7d2c85af173a`

The output and manifest were rebuilt twice and were byte-identical.
