# CITA-1 CFTC source audit — 2026-07-17

## Scope

CITA-1 uses only the CFTC Traders in Financial Futures futures-only report for
`BITCOIN - CHICAGO MERCANTILE EXCHANGE`, contract code `133741`. No BTC price,
return, funding, existing-alpha score, or portfolio outcome was read while the
source was built.

Official references:

- historical annual archives: <https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm>
- TFF field names: <https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/cotvariablestfm.html>
- TFF category definitions: <https://www.cftc.gov/idc/groups/public/%40commitmentsoftraders/documents/file/tfmexplanatorynotes.pdf>
- normal and holiday publication rule: <https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm>
- 2023 ION backlog publication record: <https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm>

## Frozen rows

| Report year | Rows |
|---|---:|
| 2018 | 39 |
| 2019 | 52 |
| 2020 | 52 |
| 2021 | 52 |
| 2022 | 52 |
| 2023 | 52 |
| **Total** | **299** |

The first available Bitcoin report, `2018-04-10`, has no preceding row and is
quarantined. The remaining 298 rows exactly reconcile every published dealer,
asset-manager, and leveraged-money long/short change against adjacent report
levels.

## Availability clock

The CFTC states that COT normally publishes Friday at 15:30 Eastern for the
preceding Tuesday and that federal holidays can delay it one or two days. The
backtest therefore uses the deliberately later `report date + 8 days 00:00 UTC`
clock instead of reconstructing historical DST and holiday release minutes.

Seven reports affected by the 2023 ION outage do **not** use that shortcut.
Their actual publication dates are taken from the CFTC special-announcement
ledger, then moved to the next UTC midnight:

| Report date | Actual publication date |
|---|---|
| 2023-01-31 | 2023-02-24 |
| 2023-02-07 | 2023-03-03 |
| 2023-02-14 | 2023-03-08 |
| 2023-02-21 | 2023-03-10 |
| 2023-02-28 | 2023-03-14 |
| 2023-03-07 | 2023-03-16 |
| 2023-03-14 | 2023-03-21 |

This explicitly removes the false `report date + 8 days` availability used by
older exploratory CFTC scripts during the outage.

## Revision boundary

The annual compressed files are official consolidated archives rather than
captured browser bytes from every original Friday. CFTC special notices were
checked for the source horizon; no Bitcoin `133741` correction was identified.
Published change fields also reconcile exactly to adjacent archived positions.
This does not prove that no unannounced historical correction ever occurred, so
the source manifest records the residual archive-revision limitation. It does
remove known publication-delay leakage and any arithmetic reconstruction drift.

## Identity

- panel SHA-256: `064eed3fa340b1701f4686d1176de2a10f39128abc5ebf846e8b6319b8144ee6`
- source manifest SHA-256: `a594b02d1191c32f905c13be3faaa74ec2f3f0e04723d3b11b76ee8b454d6897`
- market/funding rows read: `0`
