# Address Reservoir Capacitance mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **ARCR-864 — Address Reservoir
Capacitance Release, three-day hold**. It will use the previously unused Coin
Metrics `AdrBalCnt` stock together with `AdrActCnt` flow to detect opposing
changes in funded-address inventory and daily address turnover.

This file freezes the source axis and economic direction before downloading
the complete source prefix or calculating event incidence. It opens no BTC
market row, funding row, post-entry return, PnL, or 2024+ source row.

## New observable axis

Coin Metrics defines:

- `AdrBalCnt` as the number of unique addresses holding any positive native
  balance at the end of an interval; and
- `AdrActCnt` as the number of unique addresses active as an originator or
  recipient of a ledger change during the interval.

Official metadata and API surfaces:

- [Address Balances](https://docs.coinmetrics.io/network-data/network-data-overview/addresses/address-balances)
- [Active Addresses](https://docs.coinmetrics.io/network-data/network-data-overview/addresses/active-addresses)
- [Availability](https://docs.coinmetrics.io/network-data/network-data-overview/availability/)
- [Coin Metrics API v4](https://docs.coinmetrics.io/api/v4/)
- [Community metric catalog](https://community-api.coinmetrics.io/v4/catalog-all-v2/asset-metrics?assets=btc&metrics=AdrBalCnt%2CAdrActCnt%2CAssetEODCompletionTime)

The official catalog reports daily community coverage for all three BTC
metrics from 2009 through the current source date. Repository search found no
prior use of `AdrBalCnt`. Existing NTB/NWE network experiments used active
addresses per transfer, transfers per transaction, fees, issuance, block
count, and transaction count; they did not model the stock of all funded
addresses.

ARCR interprets `AdrBalCnt` only as an address-level reservoir proxy. Addresses
are not people, one owner may control many addresses, one custodian may
represent many owners, and UTXO consolidation or dust can move the count. The
mechanism must survive component and stale controls; it is not an entity-flow
claim.

## Mechanism

The source-only state will start from:

```text
reservoir_flux_7d = log(AdrBalCnt_t / AdrBalCnt_t-7)
turnover_t         = log(AdrActCnt_t / AdrBalCnt_t)
turnover_shift_7d  = turnover_t - turnover_t-7
```

The singleton rule will standardize both changes against a strictly earlier
calendar reference and require an opposing dislocation:

- unusually positive reservoir flux with unusually negative turnover shift is
  tentatively **long**: funded-address stock expanded while a smaller fraction
  was active, consistent with address-level sequestration/cold dispersion;
- unusually negative reservoir flux with unusually positive turnover shift is
  tentatively **short**: the funded-address reservoir contracted while
  turnover rose, consistent with consolidation or mobilization of stored
  inventory.

The term “capacitance” denotes stock accumulation relative to contemporaneous
turnover. It is a falsifiable analogy, not a property asserted by Coin Metrics
or Bitcoin.

Thresholds, prior-window length, transition scheduler, support minima, and
controls must be committed before the complete 2019–2023 prefix is downloaded.
There will be no sign, threshold, cadence, or hold grid. The hold is fixed at
864 five-minute bars (three days) because the source is daily and the mechanism
concerns inventory digestion rather than intraday execution noise.

## Causal availability and bounded schema probe

Every source observation is unavailable until its recorded
`AssetEODCompletionTime`, with an additional invariant that availability cannot
precede observation time plus one full day. Entry is one completed five-minute
latency bar after the first five-minute open at or after availability.

A bounded source-only probe read three rows each around 2019-01-01,
2021-06-01, and 2023-12-28. It confirmed positive numeric fields and no metric
status/review columns in those responses. It also exposed the causal backfill
boundary: the three 2019 observations report availability on 2021-02-26, while
2021 and 2023 observations were completed the following UTC day. Backfilled
rows may seed a reference only after their recorded availability; they may not
emit backdated events.

The probe did not calculate either feature, threshold, event incidence, or any
market outcome. The source remains a frozen current vintage rather than a full
archive of historical revisions. Live promotion therefore requires a forward
vintage-parity audit and must delay or cancel a signal when source publication
is late.

## Frozen research sequence

1. Implement a source-only downloader for `[2019-01-01, 2024-01-01)` with
   pagination, exact columns, positive/integer checks, UTC daily continuity,
   availability, duplicate, deterministic gzip, response-hash, and manifest
   validation.
2. Before opening complete incidence, commit one exact signal/support
   preregistration. Train-support is 2021H2–2022 and separate test-support is
   calendar 2023.
3. Reject without repair if coverage, both signs, calendar dispersion, or event
   counts fail. No BTC market source may be opened for a support failure.
4. If support passes, commit a strict evaluator before loading any post-entry
   path. Open train first and 2023 only after a complete train pass.
5. Open 2024, 2025, and 2026 YTD sequentially, stopping at the first failed
   sealed gate.

The later evaluator must preserve full-calendar CAGR, global/pre-entry-HWM
strict MDD, every held five-minute path, entry cost, exact funding, virtual
adverse exit cost, actual exit, base/stress cost, and non-overlap. Required
controls include exact side flip, reservoir-only, turnover-only, seven-day
stale source, one-bar delay, constant three-day long/short clocks, and
deterministic within-year source permutation.

The target remains positive absolute return, CAGR/strict-MDD at least 3,
strict MDD no greater than 15%, stress-cost survival, statistically meaningful
trades and weekly clusters, balanced directions, and positive contained
subperiods. The branch has broad prior BTC exposure, so this can establish only
a candidate-level frozen sequence, not a pristine global holdout.
