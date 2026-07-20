# Address Cohort Residual Transport source rejection — 2026-07-20

## Decision

Reject **ACRT-576** before source acquisition. The six balance-cohort metrics
selected in the mechanism decision are present in the Coin Metrics full catalog
but are not available through the unauthenticated Community API entitlement.
No usable paid Coin Metrics API credential exists in the repository, sibling
Wave Trading environment, or current process environment.

Delete the uncommitted downloader prototype and do not preregister thresholds,
calculate source incidence, open BTC market/funding rows, or report a
performance statistic for ACRT-576.

## Evidence

Official surfaces queried on 2026-07-20:

- [Full BTC asset-metric catalog](https://community-api.coinmetrics.io/v4/catalog-all-v2/asset-metrics?assets=btc)
- [Asset-metric timeseries API](https://community-api.coinmetrics.io/v4/timeseries/asset-metrics)
- [Coin Metrics API v4 documentation](https://docs.coinmetrics.io/api/v4/)

The catalog advertises daily BTC support through 2026-07-19 for each proposed
metric, but `catalog-all-v2` describes Coin Metrics' full supported universe;
it does not prove the caller's data entitlement. A bounded access probe made
four-day requests and inspected only HTTP status and structured error fields:

| Requested metric set | HTTP status | Result |
|---|---:|---|
| `AdrBalCnt,AdrActCnt,AssetEODCompletionTime` | 200 | existing Community metrics remain accessible |
| `AdrBalNtv10Cnt` | 403 | unavailable with supplied credentials |
| `SplyAdrBalNtv10` | 403 | unavailable with supplied credentials |
| each cohort metric plus `AssetEODCompletionTime` | 403 | cohort metric remains forbidden |
| all six proposed cohort metrics together | 403 | source acquisition impossible |

The API's exact error type was `forbidden`; its message stated that the
requested metric/frequency/asset was not available with supplied credentials.
The probe did not print, persist, or calculate any forbidden cohort value,
feature, signal, incidence, market outcome, return, funding, or PnL.

## Why this cannot be repaired

Substituting USD balance bands, total funded addresses, or another freely
available metric after seeing the entitlement failure would define a different
mechanism. Using an undocumented scrape would weaken provenance and live
reproducibility. Purchasing or adding a credential is an external authority and
scope change, not an automatic local repair.

The mechanism may be revisited only with explicit access to all six historical
and live metrics under a stable license. Until then, its performance is
**unmeasured**, not zero, and it is excluded from alpha and portfolio claims.
