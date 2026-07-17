# Federal Reserve H.4.1 net-liquidity source audit — 2026-07-17

## Decision

The source panel is accepted for **source-only alpha preregistration**. It does
not establish profitability. No BTC price, return, funding cash flow, trade,
CAGR, or drawdown was opened while selecting or validating this source.

The frozen source axis is deliberately different from the live exchange-data
families:

```text
H41_net_liquidity[t]
  = Federal_Reserve_total_assets[t]
  - U.S._Treasury_General_Account[t]
  - Federal_Reserve_reverse_repurchase_agreements[t]
```

All three values come from the **same consolidated statement in the same
archived weekly release**. This avoids mixing a weekly Fed series with daily
Treasury and New York Fed histories whose present-day APIs do not provide a
clean historical-vintage interface.

## Official source contract

- Release archive and cadence: <https://www.federalreserve.gov/releases/h41/>
- Example archived release: <https://www.federalreserve.gov/releases/h41/20231228/>
- H.4.1 Data Download Program: <https://www.federalreserve.gov/datadownload/choose.aspx?rel=h41>

The Federal Reserve states that H.4.1 is generally released each Thursday at
16:30 Eastern, with holiday shifts. The panel uses the archive's actual release
date and a conservative availability time of **16:35 America/New_York**. It
uses the Wednesday/as-of observation date printed in the same release.

The archived 2023-12-28 consolidated statement provides an externally
checkable terminal row:

| Field | USD millions |
|---|---:|
| Total assets | 7,712,781 |
| U.S. Treasury, General Account | 713,064 |
| Reverse repurchase agreements | 1,165,574 |
| Derived net liquidity | 5,834,143 |

## Frozen artifact

- Builder: `training/build_federal_reserve_h41_net_liquidity.py`
- Tests: `tests/test_build_federal_reserve_h41_net_liquidity.py`
- Output: `data/federal_reserve_h41_net_liquidity_2018_2023/federal_reserve_h41_net_liquidity_2018-01-04_2023-12-28.csv.gz`
- Source manifest: `data/federal_reserve_h41_net_liquidity_2018_2023/source_manifest.json`
- Build manifest: `data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json`
- Audit JSON: `results/federal_reserve_h41_net_liquidity_source_freeze_2026-07-17.json`

| Check | Result |
|---|---:|
| Weekly rows | 313 |
| Coverage | 2018-01-04 through 2023-12-28 |
| Legacy PRE pages | 189 |
| Modern HTML-table pages | 124 |
| Archived response snapshots | 313 |
| Snapshot directory size | 7.3 MiB |
| Duplicate release dates | 0 |
| Invalid observation/release ordering | 0 |
| Net-liquidity identity failures | 0 |
| 2024+ releases opened | 0 |
| Crypto outcome fields opened | 0 |

Release-day distribution exactly reflects the archive index: 305 Thursdays,
7 holiday-shift Fridays, and the special Monday 2020-12-28 release. Printed
observation lags are 305 one-day, 7 two-day, and 1 five-day cases.

## Integrity and replay

| Artifact | SHA-256 |
|---|---|
| Output CSV gzip | `224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621` |
| Source manifest | `61dca0ae9e29c2c96307a3442037e43aedae15e21d3aedc9ee209c7ebbcac271` |
| Build manifest | `1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b` |

An offline `--from-snapshot` rebuild reproduced all three hashes byte for byte.
The builder rejects a changed archive index, a changed release payload, a
corrupt gzip snapshot, duplicate dates, source coverage drift, noncausal
observation dates, invalid balance values, and any requested 2024+ year.

## Research boundary

This is a source freeze, not alpha evidence. The next work unit may inspect only
source-derived feature density while defining one bounded candidate family.
BTC outcomes for the candidate's final OOS window remain sealed until the
clock, controls, selection correction, costs, and strict-MDD evaluator are
committed.
