# EIA petroleum stock breadth source audit — 2026-07-17

## Decision

**ACCEPT SOURCE; QUARANTINE ONE INCONSISTENT ISSUE; PREREGISTER EPSB-1.**

This work opens no BTC price, return, funding, FX, on-chain, options, or
existing-alpha outcome. It freezes a new official-data family whose source is
orthogonal to the current crypto and market-microstructure families.

## Official source contract

- The [EIA WPSR archive](https://www.eia.gov/petroleum/supply/weekly/archive/)
  retains issue-specific pages and Table 1 CSV files by release date.
- EIA describes the
  [Weekly Petroleum Status Report](https://www.eia.gov/petroleum/supply/weekly/)
  as a weekly view of petroleum supply and inventories; Table 1 is the U.S.
  petroleum balance sheet.
- The [official schedule](https://www.eia.gov/petroleum/supply/weekly/schedule.php)
  states that the summary and Tables 1–14 are normally released after 10:30
  a.m. Eastern on Wednesday and lists holiday exceptions.
- The downloader uses a contact-bearing user agent and no more than four
  workers, consistent with EIA's warning against excessive automated retrieval
  in its [automation policy](https://www.eia.gov/about/privacy_security_policy.php#robot).

## Point-in-time fields

For every archived issue, the panel retains that issue's own current value,
prior-week value, and published `Difference` for:

1. commercial crude oil stocks excluding SPR;
2. total motor gasoline stocks;
3. distillate fuel oil stocks.

The candidate will use only the issue-local published differences. It never
substitutes a later revised time series. Raw archive-index HTML and all 259
Table 1 payloads are retained in deterministic gzip snapshots.

## Conservative availability

Archive pages bind the release **date**, while historical holiday release
times are not exposed as a uniform machine-readable field. Rather than infer a
10:30 clock for every issue, the panel marks each row available at **13:00 UTC
on the next calendar day**. This is deliberately after the complete U.S.
Eastern release date and removes early-entry ambiguity at the cost of latency.

## Integrity results

| Check | Result |
|---|---:|
| Archived issues, 2019–2023 | 259 |
| Complete rows | 258 |
| Quarantined rows | 1 |
| Market/funding rows read | 0 |
| Output size | 25,274 bytes |
| Output SHA-256 | `26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b` |
| Source manifest SHA-256 | `3969288900528d103016cdb0870a11269c1b352b9077faffdc61427f7fce29fb` |

The 2023-12-28 issue is quarantined because all three displayed current-minus-
prior arithmetic changes disagree with its published `Difference` values
(crude discrepancy -0.203 million barrels; gasoline -0.094; distillate
-0.092). No repair or value preference is inferred.

## Outcome-blind EPSB-1 density

The frozen candidate clock will emit a signal only when all three complete
published stock changes have the same nonzero sign:

- all build: LONG BTC (broad physical glut/disinflation impulse);
- all draw: SHORT BTC (broad scarcity/reflation impulse);
- mixed, zero, or quarantined: no trade.

| Source-only window | Events | Long | Short |
|---|---:|---:|---:|
| 2019 history | 19 | 8 | 11 |
| 2020 | 12 | 6 | 6 |
| 2021 | 11 | 3 | 8 |
| 2022 | 14 | 4 | 10 |
| Stage1 2020–2022 | 37 | 13 | 24 |
| Sealed Stage2 2023 | 13 | 6 | 7 |

The density is sufficient to preregister without weakening the breadth rule.
No return, direction, holding-period, or threshold scan has been run.
