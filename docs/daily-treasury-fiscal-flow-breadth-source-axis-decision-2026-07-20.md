# Daily Treasury fiscal-flow breadth source-axis decision — 2026-07-20

## Decision

The next standalone BTC research family will use the previously unused
**category composition and breadth of the U.S. Treasury Daily Treasury
Statement (DTS)**. The provisional family name is **DFFB — Daily Fiscal Flow
Breadth**.

This work unit freezes only the source, vintage, causal availability, research
splits, source-quality gates, and outcome-blind boundary. It does **not** choose
a trading rule, feature threshold, sign, holding period, leverage, or model,
and it opens no BTC market value or strategy outcome. Those choices may be
frozen only after the source-only schema and coverage audit passes.

## Why this is a new observable axis

The DTS reports the federal government's daily cash and debt operations on a
modified cash basis. In particular, it contains category-level operating-cash
deposits and withdrawals and public-debt issues and redemptions. This allows a
later hypothesis about the *composition, breadth, and concordance* of fiscal
cash flows rather than another transform of an asset price.

This is not a repair of the existing Federal Reserve H.4.1 or Treasury-auction
families:

- H.4.1 reports weekly balance-sheet stocks, including the Treasury General
  Account balance. DFFB will not use H.4.1.
- Treasury-auction candidates used auction demand and allocation fields. DFFB
  will not use auction results.
- A plain DTS operating-cash/TGA level, daily net change, or total-only rule is
  prohibited as DFFB's primary signal. The new observable is the cross-category
  shape of deposits, withdrawals, issues, and redemptions.

The economic interpretation remains provisional. A category-level fiscal
injection/drain breadth state may transmit to dollar liquidity and risk
appetite, but the source does not identify a BTC trader, intent, or causal
effect. Direction must be preregistered after the source audit and before any
BTC outcome is opened.

## Official sources

The authoritative source bytes for historical research are an **official
historical PDF archive snapshot acquired now** from the U.S. Department of the
Treasury, Bureau of the Fiscal Service:

- DTS description and report archive:
  <https://fiscal.treasury.gov/accounting/daily-treasury-statement/>
- Fiscal Data DTS dataset page:
  <https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/>
- Fiscal Data API documentation and base service:
  <https://fiscaldata.treasury.gov/api-documentation/>
  and
  <https://api.fiscaldata.treasury.gov/services/api/fiscal_service/>
- Historical-report URL pattern:
  `https://fiscaldata.treasury.gov/static-data/published-reports/dts/`
  `DailyTreasuryStatement_YYYYMMDD.pdf`
- Official historic-announcement workbook:
  <https://fiscaldata.treasury.gov/static-data/published-reports/dts/DailyTreasuryStatement_Announcements.xlsx>
- Fiscal Data dataset metadata used to bind documented coverage and refresh
  notes:
  <https://fiscaldata.treasury.gov/page-data/datasets/daily-treasury-statement/page-data.json>

The following API tables are allowed only for schema discovery, coverage
diagnostics, and a revision comparison against the frozen PDFs:

```text
v1/accounting/dts/operating_cash_balance
v1/accounting/dts/deposits_withdrawals_operating_cash
v1/accounting/dts/public_debt_transactions
```

The current API is not the historical point-in-time source of truth and may
not overwrite, fill, normalize, or select values from a historical PDF. A PDF
versus API mismatch must be retained as a revision diagnostic. It does not
invalidate the authoritative PDF row unless a source-only parser review proves
that the mismatch came from extraction rather than a current-API revision.

## Frozen historical horizon and research boundary

The first immutable source build may acquire report dates only from
`2019-01-02` through `2023-12-29`. Research stages are assigned by causal
`earliest_execution_time`, not by report date:

| Purpose | `earliest_execution_time` in UTC |
|---|---|
| source warm-up only | before `2021-01-01 00:00:00` |
| development train source | `[2021-01-01, 2023-01-01)` |
| outcome-blind selection/support source | `[2023-01-01, 2024-01-01)` |
| parsed boundary quarantine | at or after `2024-01-01 00:00:00` |

The report dated `2022-12-30`, for example, is first available in 2023 and is
therefore selection source, not train source. The report dated `2023-12-29` is
first available in 2024: it may be parsed only for the pre-capped schema audit,
must be labeled `boundary_quarantine`, and can never emit a 2023 support event
or enter a pre-2024 rolling feature.

These are report-date bounds, not assumed row counts. The expected set is the
official Fiscal Data `publishedReports` index filtered to these bounds. A
second deterministic calendar audit must account explicitly for every weekday
not in that set as a U.S. federal holiday or an officially documented
publication exception. The index cannot self-justify an unexplained missing
weekday. The downloader may not silently drop a URL, backfill a missing PDF
from the current API, or infer a value from adjacent days.

No DTS report PDF or DTS API data row dated `2024-01-01` or later may be
requested, opened, hashed, or used during this source stage. Current
documentation metadata is the explicit non-value exception: `page-data.json`
and the announcement workbook may disclose later dates because they are
current documents. Their full response bytes are archived for provenance, but
rows effective after `2023-12-29` must be excluded before expected-date,
schema, taxonomy, feature, and gate logic. No post-cap report URL may be
fetched. No BTC OHLC, trade, order-book, funding, premium, open-interest,
liquidation, return, label, position, PnL, equity, CAGR, MDD, existing-alpha
signal, or portfolio artifact may be opened. Filename, directory, database,
and network allowlists must enforce this boundary; an assertion in a report is
insufficient.

## Historical vintage and deterministic acquisition

Each PDF is the Treasury-published historical report-form archive snapshot
available at acquisition time. The archive does not by itself prove that its
bytes have remained unchanged since the report date; PDF creation metadata is
evidence, not a first-seen guarantee. The builder therefore hash-freezes the
first acquired bytes, refuses silent replacement, and discloses this residual
vintage risk. It must preserve the exact response bytes and two manifests.
The deterministic content manifest contains at least:

```text
record_date
pdf_url
byte_length
sha256
pdf_creation_metadata_raw
http_etag
http_last_modified
parser_version
table_ids_found
extraction_status
```

A separate append-only receipt log contains URL, HTTP status, redirect chain,
ETag, Last-Modified, TLS retrieval result, and `retrieved_at_utc`. It is
provenance and is excluded from content-artifact hashing. ETag, Last-Modified,
and PDF metadata are diagnostics only; none establishes original publication
time. The content manifest, normalized rows, and rerun report must be
byte-identical when the same frozen PDFs and parser version are supplied.

The announcement workbook and dataset page-data JSON must also be archived and
hash-bound, while only their pre-cap rows may enter logic. An HTTP redirect is
retained in the receipt log and accepted only when the terminal HTTPS host is
still `fiscaldata.treasury.gov`; a cross-host or downgrade redirect fails.
HTML error pages, encrypted PDFs, unsupported PDF filters, duplicate report
dates, changed bytes at an already bound URL, and a record date inconsistent
with the filename all fail closed.

The parser is a repository-owned, standard-library implementation rather than
an ambient PDF/OCR tool. Every build binds the parser source SHA-256, exact
CPython and zlib versions, Unicode database version, `America/New_York` tzfile
SHA-256, holiday-calendar source SHA-256, `LC_ALL=C`, and label-map SHA-256.
The parser contract fixes PDF literal-string escape/octal decoding, supported
stream filters and text operators, Unicode normalization to NFC, integer-only
million-dollar parsing, and explicit null-token grammar. Unsupported syntax
fails closed; OCR, font-shape guessing, locale-aware number conversion, and
unversioned external PDF engines are prohibited. Rows sort by record date,
table, section, source order, and raw label. JSON uses UTF-8, LF, sorted keys,
and compact separators; CSV uses UTF-8, LF, RFC 4180 quoting, and fixed column
order.

## Causal availability and live parity

The publication notice printed in historical DTS reports states that the DTS
is available by 4:00 p.m. on the following business day; for example:
<https://fiscaldata.treasury.gov/static-data/published-reports/dts/DailyTreasuryStatement_20210104.pdf>.
Historical PDF creation metadata is not a trustworthy universal first-seen
timestamp and may never advance a decision.

For every report date `d`, DFFB therefore uses the conservative fixed clock:

```text
source_available_not_before[d]
  = 16:00:00 America/New_York on the next U.S. federal business day after d

earliest_decision_time[d]
  = source_available_not_before[d]

earliest_execution_time[d]
  = the five-minute bar open after one full computation/transport interval
  = normally 16:05:00 America/New_York
```

The calendar must be timezone-aware and honor daylight-saving transitions and
federal holidays. Weekend and holiday gaps are not stale values: the most
recent report may remain the latest known state, but it may not emit a new
event or be presented as newly published.

Production must poll only the official DTS endpoint/report archive, match the
expected report date, persist response bytes and receipt time, and fail closed
if the report is absent, late, revised, unparsable, or schema-incompatible.
Backtest eligibility at 16:05 is a conservative causal simulation, not proof
that live receipt was successful; promotion requires forward shadow logs that
show receipt and parse completion before each admitted decision.

## Frozen raw tables and units

The source audit may extract only these report sections:

1. operating cash balance, for table-level arithmetic and provenance controls;
2. operating-cash deposits and withdrawals by literal category; and
3. public-debt issues and redemptions by literal security category.

All report figures are retained in their published unit of millions of U.S.
dollars. Parentheses, minus signs, footnote markers, blank cells, dashes,
asterisks, and totals must be parsed explicitly rather than coerced to zero.
Both the literal printed label and a separately versioned normalized label may
be stored. Raw labels may never be rewritten in place.

The source parser must retain, at minimum:

```text
record_date
source_available_not_before
table_id
section
raw_category_label
normalized_category_label
today_amount_usd_millions
month_to_date_amount_usd_millions        # when printed
fiscal_year_to_date_amount_usd_millions  # when printed
footnote_markers
source_pdf_sha256
```

Only `today` values may later enter a daily DFFB feature. Month-to-date and
fiscal-year-to-date fields are retained solely to audit parsing and accounting
relationships; they are prohibited signal inputs.

## Source-only schema and quality gates

The source build is rejected before rule design unless every gate below passes:

1. every expected report date is represented by exactly one hash-bound PDF, or
   by an explicitly named official exception with no fabricated row;
2. every retained PDF contains an unambiguous report date and the required
   operating-cash and public-debt tables;
3. every non-missing extracted numeric cell round-trips from its literal PDF
   token and is finite; each missing token remains typed null with its literal
   token retained and is never coerced to zero;
4. printed table totals reconcile to the report-defined category/subtotal
   hierarchy. Because each displayed number is rounded to the nearest million,
   the absolute residual for a total of `N` displayed components may not
   exceed `0.5 * (N + 1)` million dollars;
5. operating-cash opening balance, deposits, withdrawals, net change, and
   closing balance satisfy the identities printed by that report version,
   again allowing only the explicit nearest-million rounding tolerance;
6. duplicate normalized labels within one table/date are rejected unless the
   raw labels and official schema note establish separate sections;
7. category-label births, deaths, renames, and table moves are emitted as a
   deterministic schema-transition artifact rather than silently merged;
8. every announcement-workbook row with an effective date on or before
   2023-12-29 is reconciled against the report bytes, together with every
   additional label/table transition detected from PDFs. The API note dated
   2022-04-18 and the report/workbook transition dated 2022-04-19 are retained
   as distinct claims until the source audit resolves their exact scope; no
   hand-written three-date allowlist is sufficient;
9. the same frozen inputs produce byte-identical normalized rows, manifests,
   hashes, and audit decisions on two clean runs; and
10. the complete source build stays physically capped at 2023-12-29 and its
    protocol report proves that no prohibited market or outcome source opened.

No minimum event count, threshold, direction balance, BTC-return correlation,
or performance condition belongs in this source audit because no DFFB trading
event exists yet.

## Post-audit mechanism-design constraints

If and only if every source-quality gate passes, a separate committed
preregistration may select one singleton DFFB mechanism. It must:

- use strict-prior normalization and the causal availability clock above;
- operate on category composition/breadth or issue-redemption concordance, not
  a total TGA/net-cash threshold in disguise;
- define a fixed label taxonomy or a source-only rule for handling category
  transitions without consulting BTC returns;
- freeze direction, threshold, event definition, hold, controls, and support
  floors before any BTC outcome is opened;
- build source-only comparator clocks without opening any comparator return,
  PnL, equity, CAGR, or MDD column: the existing H.4.1/TGA primary clock, the
  existing Treasury-auction primary clock, a Treasury auction/settlement-date
  calendar, and a same-window/same-tail control using only DTS total net cash;
- reject before outcomes if, against any comparator, New York decision-date
  Jaccard exceeds `0.30` or more than `50%` of DFFB events fall within plus or
  minus one U.S. business day of a comparator event. An empty comparator fails
  rather than vacuously passing, and all comparator definitions and hashes
  must be frozen before DFFB incidence is opened;
- after the hold is frozen, reject before outcomes if absolute signed occupied-
  exposure correlation against either prior primary strategy exceeds `0.40`;
  and
- allow no sign flip, threshold search, label regrouping, hold grid, or model
  selection after an outcome is observed.

The eventual strict evaluator, if authorized, must use next-open execution,
full-calendar absolute return and CAGR, global/pre-entry-high-water strict MDD
over every held five-minute bar, exact funding, costs at entry/exit, virtual
adverse exit cost, chronological non-overlap, and split-contained holds. Each
opened split must report absolute return, CAGR, strict MDD, CAGR/strict-MDD,
trade count, direction count, and calendar concentration.

## Stop conditions

DFFB is rejected without a BTC backtest if source coverage, deterministic
extraction, accounting reconciliation, schema transitions, causal timing, or
the 2023 physical cap fails. After a source pass, failure of the separately
frozen support/novelty stage also rejects it without opening outcomes. Once an
outcome stage is opened, the first failed frozen gate ends this candidate; it
may not be repaired in place.

The branch already contains broad prior BTC research exposure. A successful
sequence can establish a candidate under this frozen protocol, but cannot
recreate a pristine global human holdout.
