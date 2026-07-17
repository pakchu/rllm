# New York Fed SOFR distribution source audit (2026-07-17)

## Decision

Freeze a source-only panel for a new alpha family based on stress in the U.S.
Treasury repo funding distribution. This input axis is structurally separate
from the repository's currently occupied crypto price/taker, open-interest,
perpetual funding/premium, Kimchi/FX, and REX-prediction families.

This work unit opens **no BTC price, return, forward label, trade, funding,
open interest, FX, REX, portfolio, or 2024+ observation**. Source continuity is
not evidence of profitability. Absolute return, CAGR, strict MDD, and
CAGR/strict-MDD are therefore intentionally **N/A** here.

## Official sources

- Federal Reserve Bank of New York, SOFR data and definition:
  <https://www.newyorkfed.org/markets/reference-rates/sofr>
- New York Fed Markets Data API documentation:
  <https://markets.newyorkfed.org/static/docs/markets-api.html>
- New York Fed Markets Data API OpenAPI specification:
  <https://markets.newyorkfed.org/static/docs/markets-api.yml>
- New York Fed publication and revision policy:
  <https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates>

The New York Fed defines SOFR as a broad measure of overnight cash borrowing
secured by Treasury securities. The official daily table provides the
volume-weighted median rate, 1st/25th/75th/99th percentiles, and transaction
volume in billions of dollars.

## Frozen source contract

| Item | Frozen value |
|---|---|
| Endpoint | `/api/rates/secured/sofr/search.json` |
| Query partition | one inclusive calendar-year request per year |
| Fetched years | 2018 through 2023 |
| Hard future guard | builder rejects every `end_year > 2023` |
| Source dates fetched | 2018-04-02 through 2023-12-29 |
| Rows fetched | 1,437 |
| Rows emitted | 1,436 |
| Last fetched row | provenance only; not emitted because its next publication day lies outside the bounded panel |
| Persisted observables | median SOFR, four distribution percentiles, USD-billion volume, revision indicator |
| Derived market fields | none |
| Missing distribution policy | retain with `source_complete=false`; never fill or carry forward |
| Invalid finite distribution | fail if quantiles are out of order |
| Duplicate or unexpected gap | fail closed |
| Raw payload policy | persist the exact six annual JSON responses and record each SHA-256 |
| Snapshot mutation policy | fail if a live response hash differs from the frozen 2026-07-17 payload hash |
| Outcome policy | no market or performance outcome is opened |

Annual coverage is frozen in code so a truncated or changed upstream response
cannot silently replace the research source:

| Year | Rows | First effective date | Last effective date |
|---:|---:|---|---|
| 2018 | 188 | 2018-04-02 | 2018-12-31 |
| 2019 | 250 | 2019-01-02 | 2019-12-31 |
| 2020 | 251 | 2020-01-02 | 2020-12-31 |
| 2021 | 250 | 2021-01-04 | 2021-12-31 |
| 2022 | 249 | 2022-01-03 | 2022-12-30 |
| 2023 | 249 | 2023-01-03 | 2023-12-29 |

## Temporal availability contract

The API's `effectiveDate` is the date of the underlying repo transactions, not
the date at which a strategy could know the value. The official documentation
states that SOFR is normally published at approximately 08:00 ET on the next
business day. It also states that qualifying corrections can be published at
approximately 14:30 ET on that same publication day.

The official policy additionally states that updated percentile and volume
summary statistics for the Treasury repo rates are released on a lagged basis
shortly after quarter-end and may differ from the initially published values.
The current historical API is not a point-in-time archive of those initial
daily summaries. Giving every field the daily-rate timestamp would therefore
create look-ahead bias.

The frozen panel uses two non-interchangeable clocks:

1. sort the official SOFR effective dates;
2. assign each observation's `publication_date` to the next observed SOFR
   effective date;
3. expose only the median `sofr_percent` at **15:00 America/New_York** on that
   publication date, yielding 19:00 UTC in EDT and 20:00 UTC in EST;
4. expose the four percentiles and volume only at 21:00 UTC on the first day of
   the **second quarter after the observation's effective quarter**;
5. never permit a downstream evaluator to substitute the earlier
   `sofr_available_at_utc` for `summary_available_at_utc`;
6. drop the final fetched observation because the bounded panel has no next
   observed publication date;
7. require any later strategy evaluator to enter only after the clock for every
   field it actually uses.

The summary-statistics timestamp is deliberately much later than the stated
quarter-end update. It consumes a complete additional quarter as a vintage
safety margin because the daily API does not expose the exact original/final
summary vintage. This sacrifices timeliness to preserve causal historical/live
parity. The two dates whose official distribution percentiles are `NA` remain
present but unusable for a distribution-based rule:

- 2019-05-31, published 2019-06-03;
- 2021-08-05, published 2021-08-06.

The median and volume are present on those dates, but a distribution-based
alpha must fail closed because `source_complete=false`.

## Integrity result

Builder:
`training/build_new_york_fed_sofr_distribution.py`

Artifact:
`data/new_york_fed_sofr_distribution_2018_2023/new_york_fed_sofr_distribution_2018-04-02_2023-12-28.csv.gz`

Manifest:
`data/new_york_fed_sofr_distribution_2018_2023/build_manifest.json`

| Check | Result |
|---|---:|
| Fetched rows | 1,437 |
| Emitted rows | 1,436 |
| Complete distribution rows | 1,434 |
| Incomplete distribution rows | 2 |
| Duplicate effective dates | 0 |
| Finite quantile-order violations | 0 |
| Non-empty revision indicators in emitted rows | 0 |
| Observed inter-date gaps | 1 to 4 calendar days |
| First median-rate availability | 2018-04-03 19:00 UTC |
| Last median-rate availability | 2023-12-29 20:00 UTC |
| First summary-stat availability | 2018-10-01 21:00 UTC |
| Last summary-stat availability | 2024-04-01 21:00 UTC |
| Exact annual raw snapshots | 6 |
| Panel SHA-256 | `4993eda2b659e346b4d7b6e3aa0e2ff31cacf868f0e1fe2e1a5a76a03d1b5852` |
| Manifest SHA-256 | `873afb5234fd013e3bc454a83713abf34d9f4a4bffc9895683add7891c636598` |
| Builder SHA-256 | `353dbd99a5b546655532f89ce8df99425d5fe833398bf1477599e0150d160b45` |

Two full network builds produced byte-identical panel and manifest files. The
manifest records the fixed 2026-07-17 snapshot date, builder hash, request URL,
raw snapshot path, row count, date range, and raw response SHA-256 separately
for all six annual requests.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m training.build_new_york_fed_sofr_distribution

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_build_new_york_fed_sofr_distribution.py

ruff check \
  training/build_new_york_fed_sofr_distribution.py \
  tests/test_build_new_york_fed_sofr_distribution.py
```

## Next work-unit boundary

Before any BTC outcome is opened, exactly one SOFR funding-stress alpha
must preregister:

- whether it uses same-day median-rate dislocation, safely lagged summary
  statistics, or both;
- source-only feature equations and strictly-prior normalizers;
- event threshold and episode de-duplication;
- side, execution clock, hold, and overlap rule;
- incomplete-source behavior;
- a field-level assertion that blocks use before the corresponding frozen
  availability timestamp;
- transaction-cost and strict-MDD assumptions;
- controls and pass/reject gates;
- 2021-2022 Stage 1 and sealed conditional 2023 Stage 2.

Orthogonality will be measured only if the candidate first passes standalone
economics. No threshold or direction may be repaired from a viewed BTC outcome.
