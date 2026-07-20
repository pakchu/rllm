# Address Cohort Residual Transport mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **ACRT-576 — Address Cohort Residual
Transport, two-day hold**. It will test whether native-unit supply moves across
large-address balance cohorts faster than the population of addresses in those
cohorts changes.

This file freezes the source axis, feature family, directional hypothesis, and
hold before any complete time-series prefix, feature value, event incidence,
BTC market row, funding row, post-entry return, or PnL is opened. Exact
thresholds, reference windows, support floors, and controls must be committed
after the immutable source loader but before complete source incidence.

## New observable axis

ACRT uses six previously unused Coin Metrics community metrics:

- `AdrBalNtv10Cnt`, `AdrBalNtv100Cnt`, and `AdrBalNtv1KCnt`: counts of
  addresses holding at least 10, 100, and 1,000 native units at interval end;
- `SplyAdrBalNtv10`, `SplyAdrBalNtv100`, and `SplyAdrBalNtv1K`: native-unit
  supply held by those same cumulative address cohorts; and
- `AssetEODCompletionTime` only as the causal publication clock.

Official surfaces:

- [Coin Metrics Address Balances](https://docs.coinmetrics.io/network-data/network-data-overview/addresses/address-balances)
- [Coin Metrics Current Supply](https://docs.coinmetrics.io/asset-metrics/supply/ser)
- [Coin Metrics API v4](https://docs.coinmetrics.io/api/v4/)
- [Community BTC metric catalog](https://community-api.coinmetrics.io/v4/catalog-all-v2/asset-metrics?assets=btc)
- [Metric reference data](https://community-api.coinmetrics.io/v4/reference-data/asset-metrics?metrics=AdrBalNtv10Cnt%2CAdrBalNtv100Cnt%2CAdrBalNtv1KCnt%2CSplyAdrBalNtv10%2CSplyAdrBalNtv100%2CSplyAdrBalNtv1K%2CAssetEODCompletionTime)

A catalog-only probe on 2026-07-20 confirmed daily community coverage from
Bitcoin genesis through 2026-07-19 for all six balance-cohort metrics. It read
metric identifiers, definitions, units, frequencies, and coverage boundaries
only; no time-series value was requested.

Repository search found no prior use of any of the six cohort metrics. ARCR
used only total funded-address stock (`AdrBalCnt`) and daily active-address
flow (`AdrActCnt`). NTB used active addresses, transfers, and transactions.
Neither reconstructed native-balance cohorts or compared cohort supply motion
with cohort address-count motion.

## Frozen cohort topology

For each daily source row, cumulative metrics define three exact bands:

```text
C_10_100   = AdrBalNtv10Cnt  - AdrBalNtv100Cnt
C_100_1K   = AdrBalNtv100Cnt - AdrBalNtv1KCnt
C_1K_plus  = AdrBalNtv1KCnt

S_10_100   = SplyAdrBalNtv10  - SplyAdrBalNtv100
S_100_1K   = SplyAdrBalNtv100 - SplyAdrBalNtv1K
S_1K_plus  = SplyAdrBalNtv1K
```

All exact counts and supplies must be strictly positive. Cumulative address
counts and supplies must be monotonically non-increasing as the native-balance
threshold rises. A violation rejects the source snapshot rather than being
clipped or repaired.

Normalize exact-band supply by `SplyAdrBalNtv10` and exact-band address count
by `AdrBalNtv10Cnt`. Assign ordinal locations `0, 1, 2` to the three bands:

```text
supply_center_t  = sum(location_i * supply_share_i,t)
address_center_t = sum(location_i * address_count_share_i,t)

supply_transport_14d  = supply_center_t  - supply_center_t-14
address_transport_14d = address_center_t - address_center_t-14
residual_transport_14d = supply_transport_14d - address_transport_14d

upper_intensity_14d =
    log(S_1K_plus / C_1K_plus)_t
  - log(S_1K_plus / C_1K_plus)_t-14
```

This is a distribution-shape feature, not a level of total addresses, total
supply, price, return, volume, funding, open interest, or exchange inventory.
Native-unit thresholds avoid the direct price-driven reclassification present
in USD balance bands.

## Frozen economic direction

The singleton direction is fixed before source values:

- unusually positive residual transport, confirmed by positive upper-cohort
  intensity, is tentatively **long**: large-address cohorts absorbed supply
  faster than their address population broadened; and
- unusually negative residual transport, confirmed by negative upper-cohort
  intensity, is tentatively **short**: supply dispersed toward lower cohorts
  faster than the address distribution alone explains.

The two-day hold is fixed at 576 five-minute bars. The source is daily and the
mechanism concerns inventory redistribution; an intraday hold would mostly
measure publication/execution noise, while a longer fixed hold would reduce
independent event support before incidence is known. There will be no hold,
sign, metric, cohort, or native-threshold grid.

## Interpretation limits

Addresses are not entities. One owner may control many addresses, custodians
and exchanges may aggregate many owners, change-address behavior can alter the
distribution, and movement between address cohorts does not prove exchange
inflow, accumulation, or sale. “Transport” describes movement in aggregate
distribution mass; it does not track individual addresses across days.

The direction must therefore survive exact side-flip and component controls.
Live promotion additionally requires forward comparison against newly
published rows. The downloaded history is a current source vintage, not a
complete archive of every historical revision.

## Causal availability

Every row is unavailable until its recorded `AssetEODCompletionTime`, with an
additional lower bound of source day plus one full UTC day. Entry will be one
completed five-minute latency bar after the first five-minute open at or after
that availability. Late or backfilled source rows may seed future references
only after their recorded availability; they may not emit backdated events.

The immutable downloader must prove exact schema, pagination, unique daily UTC
rows, numeric types, nesting, positivity, causal completion times,
deterministic compression, exact response hashes, and no unexpected source
columns. It will first acquire only `[2019-01-01, 2024-01-01)`.

## Frozen research sequence

1. Implement and test a source-only immutable downloader. It may use bounded
   schema fixtures but no market or outcome input.
2. Acquire and freeze the exact 2019–2023 source prefix.
3. Before calculating complete incidence, commit one machine-readable
   singleton preregistration with strict-prior ranks, support floors, calendar
   splits, and controls. No threshold repair follows.
4. Build source-only train/test support. Reject without opening BTC outcomes if
   coverage, both directions, event count, or calendar dispersion fails.
5. Only after support passes, freeze a strict evaluator and open market
   outcomes sequentially: development train first, then sealed calendar test,
   then later annual evaluations. Stop at the first failed sealed gate.

Any later evaluator must use next-open execution, full-calendar CAGR,
global/pre-entry-high-water strict MDD over every held five-minute path, exact
funding, entry and exit costs, virtual adverse exit cost, chronological
non-overlap, and split-contained holds. It must report absolute return, CAGR,
strict MDD, CAGR/strict-MDD, trades, directions, and calendar clusters.

The target remains positive absolute return, CAGR/strict-MDD at least 3,
strict MDD no greater than 15%, stress-cost survival, statistically meaningful
trade support, both directions, and positive contained subperiods. The branch
has broad prior BTC exposure, so a pass can establish only a candidate-level
frozen sequence, not a pristine global holdout.
