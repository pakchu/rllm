# Circle Reserve schema-bridge source-axis decision — 2026-07-30

## Decision

Open one new source-only BTC research axis:

**CRF-NMFP-SB — Circle Reserve Fund SEC form-schema bridge.**

This is a new identity after `CRLC-336` was retired without production source
access. It does not alter or reuse that candidate's homogeneous
`liquidAssetsDetails` contract.

CRF-NMFP-SB reads the original structured monthly Form N-MFP filing for the
Circle Reserve Fund and preserves the official form transition:

- N-MFP2 discloses ordered Friday-slot daily/weekly liquidity percentages;
- N-MFP3 discloses dated `liquidAssetsDetails` rows.

The bridge produces one source-native ordered liquidity path per filing
without pretending that Friday slots are dated N-MFP3 rows. This decision
authorizes only source retrieval, byte sealing, identity reconciliation,
schema-specific exact parsing, causal availability, and source support. It
does not authorize a signal side, BTC row, comparator, return, PnL, CAGR,
strict MDD, or portfolio weight.

No production Circle filing, source incidence, liquidity percentage, WAM,
WAL, BTC outcome, comparator row, or Gross9 value was opened while selecting
this axis.

## Official schema evidence

The bridge is fixed from the disclosed SEC technical-specification packages:

```text
N-MFP2 ZIP SHA-256
  0f055d8f3660ad0d328f6abf973995734a9ac7a96fda17f95ada17ea405e0b4e
N-MFP3 ZIP SHA-256
  4a8daf4801d79e8a2f0484bc41a22e098771c1fec822045a3e96c6fb82b82ec7
N-MFP2 filer XSD SHA-256
  372808e0b8a348047d97d7631df505f627abb62d1bd8d3069172f9c12339d84a
N-MFP2 common XSD SHA-256
  b6fa1b1a545845f1c17daf078ac610ea6a27164108c2d328c3dbbe94b92aeda7
N-MFP3 filer XSD SHA-256
  bf6fb5e217caac77b85ef90567da94f32fa41198c1b254ef132b7a2ee4b89e08
```

Official authorities:

- <https://www.sec.gov/info/edgar/specifications/form-n-mfp2-xml-tech-specs.htm>
- <https://www.sec.gov/edgar/filer-information/specifications/form-n-mfp3-xml-tech-specs>
- <https://www.sec.gov/submit-filings/technical-specifications>
- <https://www.sec.gov/files/formn-mfp.pdf>
- <https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data>

Circle and BlackRock pages remain mechanism context only:

- <https://www.circle.com/usdc>
- <https://www.blackrock.com/cash/en-us/products/329365/circle-reserve-fund-institutional-shares>

Their mutable values, charts, factsheets, holdings, flows, and timestamps are
forbidden.

## Frozen SEC identity

The target series is:

| Field | Exact value |
|---|---|
| Daily-index company name | `BlackRock Funds` |
| Registrant CIK | `0000844779` |
| Registrant numeric CIK | `844779` |
| EDGAR series ID | `S000077205` |
| Registrant LEI | `549300OZUEVJZHOBFP42` |
| Series name, N-MFP3 only | `Circle Reserve Fund` |
| Series LEI, N-MFP3 only | `549300X6KEJFVQHDAG85` |
| Context-only ticker | `USDXX` |

For N-MFP2, exact routing requires the daily-index company name plus XML CIK,
series ID, and registrant LEI. The N-MFP2 schema does not contain
`registrantFullName`, `nameOfSeries`, or `leiOfSeries`; none may be fabricated
or imported from a current page.

For N-MFP3, exact routing additionally requires XML registrant name, series
name, and series LEI. Ticker never participates in routing.

## Frozen first-dissemination transport

Membership comes from SEC static daily indexes. Candidate complete-submission
bytes come from the matching first-dissemination Feed archive. Current archive
objects are parity checks only.

The builder must use HTTPS, a declared contact `User-Agent`,
`Accept-Encoding: identity`, no authentication/cookies, zero redirects, one
attempt per URL, and a minimum elapsed request interval of `0.20` seconds.
Only these path templates are allowed:

```text
/Archives/edgar/daily-index/<YYYY>/QTR<n>/master.<YYYYMMDD>.idx
/Archives/edgar/Feed/<YYYY>/QTR<n>/<YYYYMMDD>.nc.tar.gz
/Archives/edgar/data/844779/<accession_without_dashes>/index.json
/Archives/edgar/data/844779/<accession_without_dashes>/primary_doc.xml
/Archives/edgar/data/844779/<accession_without_dashes>/<accession_with_dashes>.txt
```

Daily-index HTTP 404 is the only expected non-200 receipt. Every other
response must be HTTP 200, nonempty, exact host/path, and zero redirects.
Search, mirrors, current full/quarterly indexes, bulk N-MFP data, BlackRock
files, Circle APIs, and browser automation are forbidden.

Every response gets an ordered ledger row with URL, request number, UTC
receipt, status, content type, byte count, SHA-256, and redirect count. The
contact string is never serialized.

## Frozen discovery universe

For every report month `2022-11` through `2026-04`, request calendar days
`01..15` of the following month. This is exactly 42 discovery months and 630
daily-index paths.

Successful indexes are strict ASCII SEC master indexes. Candidate rows require
numeric CIK `844779`, exact company name `BlackRock Funds`, form in
`N-MFP2|N-MFP2/A|N-MFP3|N-MFP3/A`, filing date equal to the index date, and
archive filename:

```text
edgar/data/844779/<##########-##-######>.txt
```

A Feed is fetched only for an index day with at least one candidate. It is
opened as gzip tar in memory with no extraction. Members must be unique safe
regular relative POSIX files: no links, devices, sparse entries, absolute
paths, backslashes, NUL, or dot components.

Every safe regular Feed member is inventoried in tar order by member name,
declared size, actual size, and SHA-256. A candidate accession resolves to the
unique member whose first ASCII line is exactly:

```text
<SEC-DOCUMENT><accession>.txt : <YYYYMMDD>
```

where `<accession>` is the indexed dashed accession and `<YYYYMMDD>` is the
index date. Zero or multiple matching members are terminal. Candidate
accessions must map one-to-one to distinct members. No member-name convention
is used to repair or infer membership.

Every resolved submission inventories all `DOCUMENT` sections in source order
by sequence, type, filename, description, byte count, and SHA-256. Exactly one
document must have the indexed form and filename `primary_doc.xml`.

For each routed target accession, current `index.json`, `primary_doc.xml`, and
the dashed-accession `.txt` are fetched. Current XML and complete-submission
bytes must equal first-dissemination bytes exactly. Mismatch, removal,
missing object, or changed identity is terminal.

`index.json` is UTF-8 JSON decoded with duplicate-object-key rejection. Its
root key set is exactly `{"directory"}`. The directory key set is exactly
`{"name","parent-dir","item"}`; `name` must be the requested archive
directory and `item` must be a list. Every item has exactly
`{"last-modified","name","size","type"}`. Names are unique safe basenames;
size is a canonical nonnegative base-10 integer string. The entire ordered
item inventory is retained. Exactly one `primary_doc.xml` and one dashed
accession `.txt` must exist, and each declared size must equal its fetched
byte count. Unknown items are inventoried but never opened.

## Filing envelope and amendments

Original forms are exactly `N-MFP2` and `N-MFP3`; amendments
`N-MFP2/A` and `N-MFP3/A` are audit-only.

```text
acceptance local date < 2024-06-11  => original form N-MFP2
acceptance local date >= 2024-06-11 => original form N-MFP3
```

There must be exactly one original for every report month `2022-11` through
`2026-04`. Report dates must be in the declared month and its last seven
calendar days. The first two report months are warm-up.

Amendments inside the 630-path universe are sealed and inventoried but no
candidate field is parsed. They never replace, retime, delete, or alter an
original. Outside-universe amendments are outside the claim.

## Acceptance reconciliation

Every routed submission must have:

1. dashed accession grammar `^[0-9]{10}-[0-9]{2}-[0-9]{6}$`;
2. exact CIK path `844779`;
3. indexed form equal to XML `submissionType`;
4. exactly one `<ACCEPTANCE-DATETIME>YYYYMMDDhhmmss`;
5. exactly one `<FILED-AS-OF-DATE>YYYYMMDD` equal to the index date; and
6. current and first-dissemination XML/complete-submission byte parity.

The zone-less acceptance timestamp is interpreted in
`America/New_York`; ambiguous or nonexistent local time is terminal.

## N-MFP2 parser

The only accepted namespaces are exactly:

```text
http://www.sec.gov/edgar/nmfp2
http://www.sec.gov/edgar/nmfp2common
```

Let `N2={http://www.sec.gov/edgar/nmfp2}` and
`C2={http://www.sec.gov/edgar/nmfp2common}` in Clark notation. The document
root must be exactly `N2edgarSubmission`. Required selectors are absolute,
direct-child paths and each terminal must occur exactly once:

```text
N2edgarSubmission/N2headerData/N2submissionType
N2edgarSubmission/N2formData/N2generalInfo/N2reportDate
N2edgarSubmission/N2formData/N2generalInfo/N2cik
N2edgarSubmission/N2formData/N2generalInfo/N2registrantLEIId
N2edgarSubmission/N2formData/N2generalInfo/N2seriesId
N2edgarSubmission/N2formData/N2seriesLevelInfo/N2averagePortfolioMaturity
N2edgarSubmission/N2formData/N2seriesLevelInfo/N2averageLifeMaturity
N2edgarSubmission/N2formData/N2seriesLevelInfo/N2percentageDailyLiquidAssets
N2edgarSubmission/N2formData/N2seriesLevelInfo/N2percentageWeeklyLiquidAssets
N2edgarSubmission/N2formData/N2seriesLevelInfo/N2totalValueDailyLiquidAssets
N2edgarSubmission/N2formData/N2seriesLevelInfo/N2totalValueWeeklyLiquidAssets
```

At the same `seriesLevelInfo` parent,
`N2moneyMarketFundCategory` must be a nonempty, duplicate-free, source-ordered
subset of `Treasury`, `Government/Agency`, and `Exempt Government`; any other
value is terminal.

The percentage containers are parsed only through matching numbered children:

```text
daily:  C2fridayDay1 ... C2fridayDay5
weekly: C2fridayWeek1 ... C2fridayWeek5
```

Slots `1..4` must exist in both containers. Slot 5 must be present in both or
absent in both. The retained path is the four or five paired slots in numeric
order. Child names, positions, missingness, duplicates, namespaces, and order
must match exactly. No main-namespace or unqualified Friday child is accepted.
The two N-MFP2 dollar-valued elements are complex containers, not scalar
leaves. Their direct children must use the corresponding `C2fridayDay1..5`
or `C2fridayWeek1..5` QNames in official numeric order, with no unknown or
duplicate child. Daily dollar children are any subset, including empty, of
slots `1..5`, each at most once. Weekly dollar slots `1..4` occur exactly once
each and slot 5 occurs zero or one time. Each dollar child is leaf-only and
not nil. The builder does not access any dollar child `.text`; dollar slots
and values are never serialized, compared to percentage-slot presence, or
used.

The official N-MFP2 XSD makes registrant LEI and daily-liquidity fields
optional. CRF-NMFP-SB deliberately imposes their presence as a stricter target
admission gate because the candidate cannot be constructed without them. It
does not describe that gate as general N-MFP2 schema validity.

## N-MFP3 parser

The only accepted namespaces are exactly:

```text
http://www.sec.gov/edgar/nmfp3
http://www.sec.gov/edgar/nmfp3common
```

Let `N3={http://www.sec.gov/edgar/nmfp3}` and
`C3={http://www.sec.gov/edgar/nmfp3common}`. The root is exactly
`N3edgarSubmission`. Required scalar paths are direct-child absolute paths,
each with exactly one terminal:

```text
N3edgarSubmission/N3headerData/N3submissionType
N3edgarSubmission/N3formData/N3generalInfo/N3reportDate
N3edgarSubmission/N3formData/N3generalInfo/N3registrantFullName
N3edgarSubmission/N3formData/N3generalInfo/N3cik
N3edgarSubmission/N3formData/N3generalInfo/N3registrantLEIId
N3edgarSubmission/N3formData/N3generalInfo/N3nameOfSeries
N3edgarSubmission/N3formData/N3generalInfo/N3leiOfSeries
N3edgarSubmission/N3formData/N3generalInfo/N3seriesId
N3edgarSubmission/N3formData/N3seriesLevelInfo/N3moneyMarketFundCategory
N3edgarSubmission/N3formData/N3seriesLevelInfo/N3govMoneyMrktFundFlag
N3edgarSubmission/N3formData/N3seriesLevelInfo/N3averagePortfolioMaturity
N3edgarSubmission/N3formData/N3seriesLevelInfo/N3averageLifeMaturity
```

Category must be `Government`; government flag must be `Y`.

There must be 15–31 direct
`N3formData/N3seriesLevelInfo/N3liquidAssetsDetails` rows in source order.
Each retained row must contain exactly one direct child:

```text
N3totalValueDailyLiquidAssets
N3totalValueWeeklyLiquidAssets
N3percentageDailyLiquidAssets
N3percentageWeeklyLiquidAssets
N3totalLiquidAssetsNearPercentDate
```

Dollar values are presence-validated only. Retained rows are ordered by their
exact source dates. No `C3` or unqualified detail child is accepted.

The official N-MFP3 XSD permits 1–31 rows and makes both daily-value children
optional. CRF-NMFP-SB deliberately requires 15–31 rows and daily/weekly pairs
in every row as a stricter target-admission gate. It does not claim those
stricter requirements are general N-MFP3 schema validity.

Every date is exact ASCII `YYYY-MM-DD`, valid Gregorian, inside the report
month, and not later than `reportDate`. The first date is no later than day 4;
adjacent dates differ by one to four calendar days; the final date is zero to
three days before or equal to `reportDate`. Source order must already be
strictly increasing and unique. The builder never sorts, deduplicates, or
tie-breaks detail rows.

## XML validation boundary

The pinned XSD packages define QName and cardinality metadata only. The
production builder does not perform general validating-XSD execution and does
not claim whole-document XSD validity. It performs the exact selected-field,
identity, namespace, lexical, and stricter target-admission gates in this
decision.

Every selected scalar terminal is leaf-only: zero child elements, zero
attributes, no mixed content, and exactly one nonempty decoded text value.
There is no `xs:token` whitespace collapse. The decoded value must contain no
tab, CR, LF, leading/trailing whitespace, or NUL; its Unicode code points are
compared exactly without Unicode normalization or case folding.

CIK is exactly ten ASCII digits, series ID is `S` plus nine ASCII digits, and
LEI is exactly 20 uppercase ASCII alphanumeric characters. Form, category,
flag, registrant name, and series name are exact displayed ASCII literals.
Dates, integers, and decimals use their separately frozen lexical grammars.

CDATA sections are globally forbidden. A selected or retained text span may
contain no `&` byte, so predefined and numeric character references are
forbidden there; the preserved lexical value is exactly the raw UTF-8 text
bytes decoded as ASCII. Predefined and numeric character references are
permitted only in ignored nonselected text, whose decoded value is never
accessed. A DTD, external entity, parameter entity, custom general entity
declaration/reference, or unresolved entity is terminal. The initial exact declaration
`<?xml version="1.0" encoding="UTF-8"?>` is allowed; every other processing
instruction is forbidden.

Element QNames anywhere in the XML may use only the applicable main and common
namespaces. The only allowed namespaced attributes are
`{http://www.w3.org/2001/XMLSchema-instance}schemaLocation` on the root and
`{http://www.w3.org/2001/XMLSchema-instance}nil` where the official schema
permits it. Root `schemaLocation`, when present, must be exactly the main
namespace followed by one ASCII space and the applicable official filer-XSD
filename. A retained or required element may not be nil. Unqualified
attributes are allowed only where ignored by the selected parser and are
inventoried by QName and value; namespaced attributes outside XSI are
terminal.

Namespace declarations are validated from parser `start-ns` events, not as
ordinary attributes. Exactly three root declarations and no descendant
redeclaration are allowed:

```text
default prefix -> applicable main namespace
ns3            -> applicable common namespace
xsi            -> http://www.w3.org/2001/XMLSchema-instance
```

No alias prefix or unused additional namespace is accepted.

“Value opened” has an operational definition. The parser necessarily reads,
well-forms, and hashes the complete XML bytes and scans every element and
attribute QName plus namespace event. It may read `.text` only for:

- the exact selected scalar paths;
- Friday percentage children;
- N-MFP3 retained daily/weekly percentage and date children, excluding both
  dollar-valued children.

For N-MFP3 dollar-valued liquidity scalar children it checks only QName,
cardinality, leaf-only shape, nil state, and presence; it never reads `.text`.
For N-MFP2 dollar complex containers it applies the separate child-shape rule
above without reading child text. For every other XML element it never
accesses, serializes, separately hashes, logs, or branches on `.text` or
`.tail`. QName/shape scanning and the whole-byte source hash do not count as
opening a forbidden value. Complete-submission header text access is counted
separately from XML. The source manifest records both exact access maps and
aggregates plus `forbidden_xml_text_access_count=0`; any instrumentation
mismatch is terminal.

## Common exact values and canonical row

WAM and WAL are plain nonnegative integers with:

```text
0 <= WAM <= 60
0 <= WAL <= 120
WAM <= WAL
```

Percentages use exact ASCII grammar
`^(?:0|[1-9][0-9]*)(?:[.][0-9]+)?$`: no sign, exponent, comma, leading zero,
surrounding whitespace, NaN, or infinity. Their source lexical string is
preserved byte-for-byte after validation, never numerically normalized.
Every exact rational value is in `[0,100]`, and daily is at most weekly for
each paired observation.

The canonical source row is:

```text
accession
form
report_date
acceptance_datetime_et
source_available_at_utc
registrant_cik
registrant_lei
series_id
schema_path_kind
wam_days
wal_days
liquidity_path_json
```

`schema_path_kind` is exactly `nmfp2_friday_slots` or
`nmfp3_dated_details`.

`liquidity_path_json` is compact ordered JSON:

```json
[
  {
    "position": 1,
    "source_label": "friday1 or YYYY-MM-DD",
    "daily_pct": "plain exact decimal",
    "weekly_pct": "plain exact decimal"
  }
]
```

N-MFP2 labels are exactly `friday1` through `friday4|5`. N-MFP3 labels are
exact dates. No inferred N-MFP2 dates exist.

`liquidity_path_json` is UTF-8 compact JSON with `ensure_ascii=true`,
separators `(',',':')`, array order preserved, and each object key order
exactly `position,source_label,daily_pct,weekly_pct`.

DTD, custom entities, retained-text character references, comments,
processing instructions, XInclude, XSLT, network
resolution, disallowed namespaces/attributes, duplicate required elements,
unknown children inside retained containers, and forbidden-field
serialization are terminal.

## Causal availability

```text
acceptance_local_date = America/New_York date of acceptance
source_available_at   = 12:00:00Z on acceptance_local_date + 5 calendar days
```

The historical timestamp is five-minute aligned. Any later trading protocol
must wait one more complete five-minute bar. Live availability is the later
of this floor and durable local receipt, validation, hashing, artifact write,
manifest commit, and source-support commit.

## Source-only gates

Before mechanism incidence or external clocks:

1. all 630 daily-index receipts exist;
2. Feed membership and complete submissions reconcile exactly;
3. current archive parity is exact;
4. exactly 42 consecutive originals exist;
5. the form transition is exact;
6. each row passes the exact selected-field and stricter target-admission
   contract for its form;
7. the canonical bridge path is deterministic;
8. a sealed-input rebuild is byte-identical; and
9. no amendment values, forbidden fields, market, funding, comparator,
   return, PnL, CAGR, or MDD data was opened.

The source artifact is one deterministic gzip CSV plus manifest-last JSON.
The first failed gate is `REJECT_NO_REPAIR`.

Rows are sorted only by `(source_available_at_utc,accession)` and must then
have strictly consecutive unique report months. Date fields use `YYYY-MM-DD`.
`acceptance_datetime_et` is whole-second ISO ASCII with explicit `-05:00` or
`-04:00`; `source_available_at_utc` is exact
`YYYY-MM-DDTHH:MM:SSZ`. Integers are canonical base 10 with no plus or leading
zero.

CSV uses the displayed header order, UTF-8 without BOM, comma delimiter,
double-quote quote character, doubled embedded quotes, minimal quoting, and
LF line endings. Nulls and extra columns are forbidden. Gzip uses level 9,
empty filename and `mtime=0`.

The exact gzip writer is manual RFC 1952 framing:

```text
header hex: 1f8b08000000000002ff
raw DEFLATE: zlib.compressobj(
  level=9, method=DEFLATED, wbits=-15, memLevel=8,
  strategy=Z_DEFAULT_STRATEGY
)
trailer: little-endian CRC32 then input-size modulo 2^32
required zlib runtime version: 1.3
```

The empty-input golden vector is:

```text
1f8b08000000000002ff03000000000000000000
```

The source-access claim binds Python/platform ABI and zlib compile/runtime
versions. Any mismatch is terminal; no alternate compressor is allowed.

JSON reports use sorted keys, two-space indentation, ASCII escaping,
`allow_nan=false`, and one trailing LF. Each internal `manifest_hash` is
SHA-256 of compact sorted-key JSON with that field excluded. Small sealed
source objects use the same deterministic gzip settings; already-compressed
Feed archives retain original bytes.

## Exact sealed-object paths

The evidence root is:

```text
data/circle_reserve_schema_bridge_raw_2026-07-30
```

Each HTTP-200 object uses:

```text
<six_digit_request_number>_<SHA256(UTF8 exact URL)>.<suffix>
```

Feed suffix is `nc.tar.gz` and stores original bytes. Every other successful
object uses suffix `gz` and stores the exact response bytes under the frozen
gzip writer. A daily-index 404 has empty body SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
byte count zero, and empty sealed path. No overwrite or alias is allowed.

## Exact source manifest schema

The successful manifest has exactly these top-level keys and types:

```text
protocol_version:str
source_id:str
status:str
terminal:bool
artifact_eligible:bool
registration:object
source_access_claim:object
discovery_contract:object
request_ledger:array[object]
daily_index_receipts:object
feed_receipts:object
current_archive_receipts:object
candidate_inventory:array[object]
amendment_inventory:array[object]
original_rows:array[object]
source_csv:object
sealed_objects:array[object]
source_checks:object[bool]
evidence_boundary:object
reproduction:object
manifest_hash:str
```

Unknown or missing keys at any specified level are terminal.

`registration` is exactly
`path,sha256,manifest_hash,policy_id`; `source_access_claim` is exactly
`path,sha256,manifest_hash,claim_commit`. `discovery_contract` is exactly
`report_months,discovery_dates,daily_index_path_count,allowed_forms`.

Every request-ledger row is exactly:

```text
request_number
role
url
requested_at_utc
received_at_utc
status
content_type
redirect_count
response_bytes
response_sha256
sealed_path
stored_bytes
stored_sha256
```

Request numbers are contiguous from one; ledger order is request order. Role
is `daily_index`, `feed`, `current_index`, `current_xml`, or
`current_submission`. Receipt summary objects are exactly
`requested,ok,not_found,total_response_bytes,ordered_ledger_sha256`.

`ordered_ledger_sha256` is SHA-256 of compact sorted-key ASCII JSON
`{"rows":<the exact request-ledger row array in request order>}` with no
trailing LF.

Every candidate-inventory row is exactly:

```text
accession,form,filing_date,index_url,feed_url,feed_member_name,
feed_member_sha256,routed_target,is_amendment,document_inventory_sha256
```

`routed_target` is exactly `target` or `other_series` for originals and
exactly `not_evaluated_amendment` for amendments. Amendment identity fields
are never opened to produce that explicit state.

Each complete-submission document inventory row is exactly
`sequence,type,filename,description,byte_count,sha256`, in source DOCUMENT
order. `document_inventory_sha256` is SHA-256 of compact sorted-key ASCII JSON
`{"documents":<that exact row array>}` with no trailing LF.

Amendment inventory contains exactly
`accession,form,filing_date,feed_member_sha256` and no parsed candidate value.
Original-row inventory contains exactly
`accession,form,report_date,source_available_at_utc,schema_path_kind,
canonical_csv_row_sha256`.

`canonical_csv_row_sha256` is SHA-256 of the exact UTF-8 CSV record bytes for
that row under the frozen dialect, excluding the header and including its
single terminal LF.

`source_csv` is exactly
`path,sha256,stored_bytes,uncompressed_bytes,row_count,header`.
`sealed_objects` is exactly the ordered projection
`request_number,url,sealed_path,response_sha256,stored_sha256,stored_bytes`.

`source_checks` has exactly these boolean keys:

```text
all_630_daily_index_receipts
feed_membership_exact
current_archive_inventory_exact
first_dissemination_current_parity
exact_42_originals
consecutive_report_months
form_transition_exact
identity_exact
selected_field_contract_exact
target_admission_exact
causal_clocks_exact
forbidden_value_access_zero
deterministic_rebuild_exact
source_manifest_published_last
```

Every value must be true for `status=passed`, `terminal=false`, and
`artifact_eligible=true`.

`evidence_boundary` is exactly:

```text
production_source_urls_requested:int
production_response_bytes_opened:int
selected_xml_text_access_count:int
selected_xml_text_access_by_selector:object[int]
complete_submission_text_access_count:int
complete_submission_text_access_by_selector:object[int]
forbidden_xml_text_access_count:int
amendment_candidate_text_access_count:int
btc_market_rows_opened:int
funding_rows_opened:int
comparator_rows_opened:int
gross9_rows_opened:int
return_or_pnl_values_opened:int
```

`selected_xml_text_access_by_selector` has exactly these nonnegative-integer
keys:

```text
nmfp2_identity_and_form
nmfp2_report_and_maturity
nmfp2_category
nmfp2_percentage_slots
nmfp3_identity_and_form
nmfp3_report_and_maturity
nmfp3_category_and_flag
nmfp3_percentage_values
nmfp3_date_values
```

Its values sum exactly to `selected_xml_text_access_count`. Dollar-valued
elements have no selector key because their text is never accessed.

`complete_submission_text_access_by_selector` has exactly
`sec_document_line,acceptance_datetime,filed_as_of_date,document_headers`;
its nonnegative integer values sum exactly to
`complete_submission_text_access_count`.

`forbidden_xml_text_access_count`,
`amendment_candidate_text_access_count`, `btc_market_rows_opened`,
`funding_rows_opened`, `comparator_rows_opened`, `gross9_rows_opened`, and
`return_or_pnl_values_opened` are all zero. Both selected-text counts are
positive by construction. `reproduction` is exactly
`sealed_input_manifest_sha256,independent_csv_sha256,
independent_manifest_core_sha256,byte_identical`.

Before reproduction, the builder writes:

```text
data/circle_reserve_schema_bridge_raw_2026-07-30/sealed-input-manifest.json
```

That JSON has exactly
`protocol_version,source_id,objects,manifest_hash`. `objects` is the
request-ordered list of the complete exact request-ledger rows, including
request/receipt times, role, status, content type, redirects, response
identity, and stored-object identity. Daily-index 404 rows are included with
empty path and zero stored bytes. Serialization and internal hashing use the
frozen JSON rules. `sealed_input_manifest_sha256` is SHA-256 of those complete
pretty-JSON file bytes.

`independent_csv_sha256` is SHA-256 of the independently rebuilt compressed
`.csv.gz` bytes, not the uncompressed CSV. `byte_identical=true` means both:

- independent compressed CSV bytes equal the published source CSV bytes; and
- independently rebuilt canonical `reproducible_core` bytes equal the first
  build's canonical `reproducible_core` bytes.

The manifest is written only after every sealed object and source CSV is
durable. Hashing is two-stage:

1. `reproducible_core` is the full top-level object with both `reproduction`
   and `manifest_hash` omitted.
2. The independent rebuild computes canonical compact sorted-key SHA-256 of
   that exact `reproducible_core`, seals it as
   `independent_manifest_core_sha256`, and compares independently rebuilt CSV
   bytes.
3. `reproduction` is inserted.
4. `manifest_hash` is SHA-256 of the canonical compact sorted-key object with
   only `manifest_hash` omitted, so it includes the already fixed
   `reproduction` object.

This order has no self-reference.

The independent rebuild takes only the committed registration and
source-access claim plus the sealed-input manifest and its referenced sealed
object bytes. It reuses sealed request metadata verbatim; it never regenerates
receipt timestamps or content types. Every request-ledger and receipt-summary
field in `reproducible_core` is therefore derivable from the named inputs.

Every compact hash preimage in this decision uses UTF-8, sorted JSON object
keys, separators `(',',':')`, `ensure_ascii=true`, `allow_nan=false`, and no
trailing LF unless a different exact byte preimage is stated explicitly.

## Stop rule

No production daily index or Feed may be opened before a fresh CRSB mechanism,
preregistration, builder, source-support, novelty, economics, synthetic tests,
and source-access claim are committed and pushed.

After source incidence opens, no form adapter, slot rule, identity, field,
period, threshold, or source path may change. A schema or source failure
permanently retires the new candidate identity.
