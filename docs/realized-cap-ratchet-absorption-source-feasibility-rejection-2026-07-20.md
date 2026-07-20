# Realized-Cap Ratchet Absorption source-feasibility rejection — 2026-07-20

## Decision: reject before implementation

**RCRA — Realized-Cap Ratchet Absorption** is rejected at the production
source-entitlement gate. The proposed observable used the keyless Coin Metrics
Community API, but Coin Metrics states that Community data is free only for
non-commercial use. The repository has no Coin Metrics Pro API entitlement,
and the project objective is profit-seeking live trading. Building a production
alpha on the Community source would therefore violate the deployment contract.

Official license/access references:

- [Coin Metrics API conventions](https://docs.coinmetrics.io/api)
- [Coin Metrics API v4](https://docs.coinmetrics.io/api/v4/)

The first reference explicitly distinguishes the free Community API for
non-commercial use from the paid Pro API for institutions. This is a source
entitlement failure, not evidence for or against the proposed algebra. No
complete source prefix, feature incidence, BTC market row, funding row,
post-entry return, PnL, or 2024+ source row was opened.

## Rejected observable concept

Coin Metrics defines:

- `CapMrktCurUSD` as current-supply market capitalization, computed from
  current supply and the daily closing price;
- `CapMVRVCur` as current-supply market capitalization divided by realized
  capitalization; and
- realized capitalization as the sum of each native unit valued at the USD
  closing price on the day that unit last moved. Coin Metrics describes it as
  a gross approximation of aggregate holder cost basis.

The source would disclose the exact identity

```text
realized_cap_t = CapMrktCurUSD_t / CapMVRVCur_t
```

Official metric definitions and source surfaces used only for this bounded
feasibility review:

- [Market capitalization, MVRV, and realized capitalization](https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/market/market-capitalization)
- [Asset end-of-day completion time](https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/availability/asseteodcompletiontime)
- [Community metric catalog](https://community-api.coinmetrics.io/v4/catalog-all-v2/asset-metrics?assets=btc&metrics=CapMVRVCur%2CCapMrktCurUSD%2CAssetEODCompletionTime)

The official catalog reported daily community coverage on 2026-07-20 from
2010-07-18 through 2026-07-18 for both capitalization fields. Repository-wide
search found no use of either metric and no MVRV/realized-cap implementation.
The earlier OI cost-basis experiment estimated derivative cohorts from Binance
open interest; it did not observe UTXO-based aggregate holder cost basis.

The concept is not price-independent. `CapMrktCurUSD` contains the completed
daily close, and historical prices also enter realized capitalization when
UTXOs last moved. It would be a cost-basis/price relation, not a pure
network-flow signal, and any eventual controls would need to establish that
realized-cap movement adds information beyond completed market-cap movement.

## Unopened mechanism sketch

Had a production-entitled source existed, the state would have started from:

```text
realized_cap_t    = CapMrktCurUSD_t / CapMVRVCur_t
realized_flux_7d  = log(realized_cap_t / realized_cap_t-7)
market_flux_7d    = log(CapMrktCurUSD_t / CapMrktCurUSD_t-7)
```

The falsifiable hypothesis was:

- positive realized-cap flux while market-cap flux is negative could indicate
  new holders absorbing older supply and would tentatively map long;
- negative realized-cap flux while market-cap flux is positive could indicate
  loss-bearing supply transferring out into strength and would tentatively map
  short.

UTXO movement is not identical to beneficial-owner transfer, internal wallet
reshuffling can change last-movement value, and the aggregate metric cannot
identify buyer or seller intent. No threshold, prior-window length, scheduler,
support minimum, event count, side result, or hold was implemented or tested.
The sketch is retained only so the same unentitled source proposal is not
rediscovered and accidentally deployed.

## Bounded schema and causal probe

A bounded source-only probe read three rows each around 2019-01-01,
2021-01-01, and 2023-01-01. It confirmed positive decimal capitalization
fields and the exact response columns `asset`, `time`, `CapMVRVCur`,
`CapMrktCurUSD`, and `AssetEODCompletionTime`. It also exposed a conservative
backfill boundary: the sampled 2019 and early-2021 observations report
completion on 2021-02-26, whereas the sampled 2023 observations completed the
following UTC day.

`AssetEODCompletionTime` is an operational calculation-completion field, not
an archive of every first-seen value. Coin Metrics may recalculate historical
reference prices, and the Community API does not provide the needed historical
value vintages here. The probe calculated no realized cap, flux, z-score,
event incidence, or market outcome.

## Reopening conditions

RCRA may be reconsidered only after one of these production-valid source paths
is established **before** any incidence or outcome is read:

1. a Coin Metrics Pro agreement/API key whose terms permit this live trading
   use; or
2. a self-computed realized-cap pipeline from Bitcoin ledger data and a
   commercially usable historical price source, with practical disk/runtime
   bounds and a causal revision contract.

Reopening would require a new source decision and exact support
preregistration. The current Community probe cannot be silently relabeled as a
production source. The same restriction applies to earlier research artifacts
that consumed Coin Metrics Community data: a statistical result alone does not
grant live-deployment rights.

## Rejected high-frequency source alternative

The official Binance USD-M `bookTicker` daily archive was checked without
opening any market outcome. Its S3 prefix contained 320 ZIP files on
2026-07-20, from 2023-05-16 through 2024-03-30, after which the archive stops.
That ten-and-a-half-month history cannot satisfy a three-year validation and is
rejected at source feasibility before any spread/queue feature or return is
calculated.

- [Official Binance public-data repository](https://github.com/binance/binance-public-data)
- [Official BTCUSDT USD-M bookTicker archive](https://data.binance.vision/?prefix=data/futures/um/daily/bookTicker/BTCUSDT/)
