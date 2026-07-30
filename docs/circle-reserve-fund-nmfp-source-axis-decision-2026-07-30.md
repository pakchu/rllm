# Circle Reserve Fund N-MFP source-axis decision — 2026-07-30

## Decision

Open one new source-only BTC research axis:

**CRF-NMFP — Circle Reserve Fund SEC liquidity ledger.**

The source is the original, structured monthly Form N-MFP filing for the
Circle Reserve Fund, a series of BlackRock Funds. It exposes a first-party SEC
acceptance clock and fund-level liquidity fields without an exchange feed,
blockchain RPC, current factsheet snapshot, or reconstructed release calendar.

This decision authorizes only source retrieval, byte sealing, identity
reconciliation, exact XML parsing, causal-availability construction, and
source-quality evaluation. It does **not** authorize a signal side, threshold,
hold, BTC market row, funding row, comparator clock, return, PnL, CAGR, strict
MDD, portfolio weight, model, or LLM policy.

No in-window liquidity percentage, WAM, WAL, candidate incidence, BTC outcome,
or Gross9 value was opened while selecting this axis.

## Why this is a materially different axis

CRF-NMFP is not a repair or substitute transport for any retired 2026-07-30
candidate:

- it does not use TRON, Ethereum, an on-chain log, mint/redeem incidence, or
  stablecoin contract supply;
- it does not use ETF shares, holdings, factsheets, or primary-market flow;
- it does not use BVOL, DVOL, options, perpetuals, or cross-venue volatility;
- it does not reuse the failed SOMA securities-lending maturity parser; and
- it does not depend on a large one-shot JSON-RPC replay.

The observable is an off-chain regulated-fund disclosure: the liquidity
composition and maturity limits of the government money-market fund holding
the majority of the USDC reserve. Circle states that the majority of the USDC
reserve is invested in the Circle Reserve Fund, an SEC-registered Rule 2a-7
government money-market fund managed by BlackRock. BlackRock identifies the
fund as USDXX, with inception on 2022-11-03.

Those statements motivate source relevance only. They do not establish a BTC
direction or economic effect.

## Frozen identity

The only eligible series is:

| Field | Exact value |
|---|---|
| Registrant | `BlackRock Funds` |
| Registrant CIK | `0000844779` |
| Series name | `Circle Reserve Fund` |
| EDGAR series ID | `S000077205` |
| Registrant LEI | `549300OZUEVJZHOBFP42` |
| Series LEI | `549300X6KEJFVQHDAG85` |
| Fund ticker, context only | `USDXX` |

Ticker is never source identity. Every retained original XML must itself
contain all six non-ticker identity values exactly. Case folding, aliases,
prefix matches, successor mapping, and human name repair are forbidden.

## Official authorities

Only SEC filing bytes establish historical source membership and values:

- SEC Form N-MFP:
  <https://www.sec.gov/files/formn-mfp.pdf>
- N-MFP2 XML technical specification:
  <https://www.sec.gov/info/edgar/specifications/form-n-mfp2-xml-tech-specs.htm>
- N-MFP3 XML technical specification:
  <https://www.sec.gov/edgar/filer-information/specifications/form-n-mfp3-xml-tech-specs>
- SEC EDGAR Release 23.3 form-transition notice:
  <https://www.sec.gov/filergroup/announcements-old/edgar-release-23-3>
- SEC N-MFP flat-data documentation:
  <https://www.sec.gov/data-research/sec-markets-data/dera-form-n-mfp-data-sets>
- EDGAR programmatic-access policy:
  <https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data>
- Circle USDC reserve description:
  <https://www.circle.com/usdc>
- BlackRock fund page:
  <https://www.blackrock.com/cash/en-us/products/329365/circle-reserve-fund-institutional-shares>

The Circle and BlackRock pages are mechanism context only. Their mutable
current values, charts, factsheets, holdings, flows, and timestamps are
forbidden inputs.

The SEC states that Form N-MFP reports the preceding month, is filed by the
fifth business day, and is supplied as structured XML. SEC also states that
filings are often available on `sec.gov` within one to three minutes of the
EDGAR acceptance timestamp. CRF-NMFP uses a longer fixed delay below.

## Frozen first-dissemination transport

Current EDGAR archive bytes are not assumed to prove what the public first
received. SEC documents that post-acceptance corrections or removals can alter
current full/quarterly indexes, while prior daily, Feed, and Oldloads artifacts
are not retroactively rebuilt for later removals. CRF-NMFP therefore discovers
membership from static daily indexes and parses candidate bytes from the
corresponding first-dissemination Feed archive.

The source builder must use a declared contact `User-Agent`,
`Accept-Encoding: identity`, HTTPS, a minimum elapsed request interval of
`0.20` seconds, and at most one attempt per URL. It may request only:

```text
https://www.sec.gov/Archives/edgar/daily-index/
  <YYYY>/QTR<1-4>/master.<YYYYMMDD>.idx

https://www.sec.gov/Archives/edgar/Feed/
  <YYYY>/QTR<1-4>/<YYYYMMDD>.nc.tar.gz

https://www.sec.gov/Archives/edgar/data/844779/
  <accession_without_dashes>/index.json

https://www.sec.gov/Archives/edgar/data/844779/
  <accession_without_dashes>/primary_doc.xml

https://www.sec.gov/Archives/edgar/data/844779/
  <accession_without_dashes>/<accession_with_dashes>.txt
```

Redirects, HTTP authentication, cookies, JavaScript, search engines, mirrors,
SEC current full/quarterly indexes, SEC bulk N-MFP flat-data ZIPs, BlackRock
files, and Circle APIs are forbidden. A response must come from the requested
SEC host and exact path with zero redirects.

For daily-index requests only, HTTP 404 is an expected text-free non-filing-day
receipt. Every other response must be HTTP 200 and nonempty. A fetch ledger
records URL, ordered request number, UTC receipt, status, content type, bytes,
hash, and zero redirects without storing the declared contact string.

Every successful daily index, Feed archive, selected current directory index,
selected current XML, and selected current complete-submission text file is
retained byte-for-byte. Small non-Feed objects are wrapped as deterministic
gzip with `mtime=0`; the already compressed Feed archive is retained in its
original bytes.

## Frozen bounded-discovery envelope

For each expected report month `M` from 2022-11 through 2026-04, the builder
requests every daily master index for calendar days `1..15` of month `M+1`.
The resulting 630-path index calendar is the complete CRF-NMFP-v1 discovery
universe and is fixed before source incidence. The source does not claim
membership outside those exact dates.

Each successful index is parsed as strict ASCII using the SEC master-index
header and exact pipe-delimited rows. Candidate rows require:

```text
CIK numeric value: 844779
form: N-MFP2 | N-MFP2/A | N-MFP3 | N-MFP3/A
filing date: exact daily-index date
archive filename:
  edgar/data/844779/<accession_with_dashes>.txt
```

Duplicate or conflicting index rows are fatal. If a daily index contains at
least one candidate row, its exact first-dissemination Feed archive is fetched
once. No Feed is fetched for a day with zero candidate rows.

The Feed is opened as gzip-compressed tar with no path extraction. Every member
must be a regular file with a safe relative POSIX name, unique name, finite
nonnegative size, and no links, devices, sparse members, absolute paths,
backslashes, NULs, or `.`/`..` components. Each candidate index accession must
resolve to exactly one Feed member containing its complete dissemination
submission. Missing or extra indexed candidate submissions are fatal.

The complete submission is parsed in memory. Every `DOCUMENT` section is
inventoried by sequence, filename, description, type, byte count, and hash.
Exactly one XML document must have the indexed N-MFP form and filename
`primary_doc.xml`. Only that XML is eligible for identity routing.

The builder routes a candidate to CRF-NMFP only when the first-dissemination
XML exactly contains the six frozen identities. Other BlackRock Funds series
are counted and discarded without parsing any liquidity, maturity, holdings,
flow, or value field.

For each routed Circle accession, the current archive `index.json`,
`primary_doc.xml`, and complete-submission text are fetched only as a parity
check. Their exact primary XML and complete-submission bytes must equal the
first-dissemination Feed bytes. A mismatch, missing current object, or changed
identity is terminal because original-byte provenance can no longer be
asserted safely.

## Frozen filing envelope

Eligible form strings are exactly:

```text
original: N-MFP2, N-MFP3
audit-only amendment: N-MFP2/A, N-MFP3/A
```

The form transition is exact:

```text
acceptance local date < 2024-06-11  => original must be N-MFP2
acceptance local date >= 2024-06-11 => original must be N-MFP3
```

`N-MFP2/A` and `N-MFP3/A` remain amendment-only according to their indexed
form. An original submitted under the wrong side of the boundary is terminal.

The physical source envelope is the exact union of these daily-index dates:

```text
days 01..15 of 2022-12, 2023-01, ..., 2026-05
```

An original is retained only when its XML `reportDate` belongs to calendar
months `2022-11` through `2026-04`, inclusive. The two 2022 months are
source warm-up. Economic use, if later authorized, begins no earlier than
2023-06-01.

Exactly one original filing must exist for every expected report month from
2022-11 through 2026-04. The report date must be a valid date in its declared
month and fall in that month's final seven calendar days. Duplicate originals,
an omitted month, a form-family gap, a report outside the envelope, or an
unexpected original form is terminal.

Amendments found inside the exact 630-date discovery universe are discovered
and sealed from first-dissemination Feed bytes but are audit-only. Amendments
outside that universe are outside CRF-NMFP-v1 and are neither requested nor
claimed complete. After exact identity routing, amendment candidate fields are
not parsed. An amendment may never replace, delete, retime, or alter an
original source row. A post-acceptance correction, removal, or current-archive
byte mismatch is terminal rather than a request to substitute an amendment.

## Exact archive and acceptance reconciliation

For each routed daily-index entry:

1. the accession must match `^[0-9]{10}-[0-9]{2}-[0-9]{6}$`;
2. the indexed archive filename and Feed identity must be under CIK `844779`;
3. current `index.json` must list exactly one safe filename `primary_doc.xml`;
4. the first-dissemination and current XML `submissionType` values must equal
   the indexed form;
5. the complete-submission text filename is the accession with dashes plus
   `.txt`;
6. the complete text must contain exactly one
   `<ACCEPTANCE-DATETIME>YYYYMMDDhhmmss`; and
7. its exact `<FILED-AS-OF-DATE>YYYYMMDD` must equal the daily-index filing
   date.

Every accession, form, filing date, report date, index identity, XML identity,
and acceptance timestamp is hash-bound. Unknown directory entries are
inventoried by filename, byte size, and type but never opened.

## Frozen XML boundary

The source accepts exactly the official N-MFP2 and N-MFP3 namespaces declared
by their SEC schemas. Namespace stripping is forbidden except for comparing
the exact registered namespace plus exact local name.

Required identity and series-level elements are:

```text
reportDate
registrantFullName
cik
registrantLEIId
nameOfSeries
leiOfSeries
seriesId
moneyMarketFundCategory
govMoneyMrktFundFlag
averagePortfolioMaturity
averageLifeMaturity
liquidAssetsDetails
```

Each `liquidAssetsDetails` member must contain:

```text
totalValueDailyLiquidAssets
totalValueWeeklyLiquidAssets
percentageDailyLiquidAssets
percentageWeeklyLiquidAssets
totalLiquidAssetsNearPercentDate
```

The builder may retain only:

- accession, form, report date, acceptance timestamp, and causal timestamp;
- the six frozen identity values;
- integer `averagePortfolioMaturity` (WAM);
- integer `averageLifeMaturity` (WAL);
- the ordered daily liquidity-detail dates; and
- exact decimal strings for daily and weekly liquid-asset percentages.

Dollar values, net assets, shares, shareholder subscriptions/redemptions,
yields, NAVs, class-level data, security rows, holdings, issuers,
counterparties, CUSIPs, maturity dates, and explanatory text are forbidden.
The dollar-valued daily/weekly elements are presence-validated but their text
is not serialized or used.

XML comments, processing instructions, DTDs, entities, XInclude, XSLT, network
resolution, duplicate required elements, and unknown namespaces fail closed.

## Exact source validation

For every original:

- category must be exactly `Government`;
- government-fund flag must be exactly `Y`;
- WAM and WAL must be plain nonnegative integers;
- `0 <= WAM <= 60`, `0 <= WAL <= 120`, and `WAM <= WAL`;
- liquidity-detail dates must be unique, strictly increasing, inside the
  report month, and not later than the report date;
- there must be at least 15 and at most 31 daily details;
- the first detail date must be no later than calendar day 4;
- adjacent detail dates may differ by at most four calendar days;
- the final detail date must be the unique maximum and must be zero to three
  calendar days before or equal to the report date;
- percentages must be plain nonnegative decimals with no sign, exponent,
  comma, surrounding whitespace, NaN, or infinity;
- every percentage must be in `[0, 100]`;
- daily liquidity may not exceed weekly liquidity on the same date; and
- the final retained source values are exactly the final source-native detail
  date's daily and weekly percentages plus WAM and WAL.

No row deletion, clipping, forward fill, interpolation, non-final detail
choice, factsheet substitution, bulk-data repair, or amendment repair is
allowed.

## Conservative causal availability

The zone-less complete-submission acceptance timestamp is interpreted in
`America/New_York`. A nonexistent or ambiguous local time is terminal. It is
then converted explicitly to UTC.

To remain later than ordinary dissemination and any submission that SEC may
disseminate on the next business day, CRF-NMFP deliberately uses a five-day
calendar floor:

```text
acceptance_local_date = date in America/New_York
source_available_at =
  12:00:00Z on acceptance_local_date + 5 calendar days
```

The resulting timestamp is already five-minute aligned. A later trading
protocol must wait at least one additional complete five-minute bar.

Filing date, report date, month end, daily-index receipt, Feed receipt, SEC
nightly index time, BlackRock publication time, and current page-update time
are never substituted for acceptance. Live availability is the later of the
historical floor and durable local receipt, validation, hashing, artifact
write, and manifest commit.

## Source-only gates

Before any mechanism incidence or outside clock is opened, the complete source
must pass:

1. exact transport and response replay;
2. all 630 daily-index receipts plus exact first-dissemination Feed
   reconciliation;
3. one original per required report month;
4. exact N-MFP2/N-MFP3 acceptance-date transition;
5. exact acceptance-header agreement;
6. 100% required-field and frozen daily-detail structural validation;
7. 100% numeric and cross-field validity;
8. exact first-dissemination/current-archive byte parity;
9. deterministic rebuild from sealed SEC bytes; and
10. zero access to amendment candidate fields, forbidden XML fields, market,
   funding, comparator, return, PnL, CAGR, or MDD data.

The canonical source artifact is one deterministic gzip CSV plus a
manifest-last JSON report. No source CSV may be published unless every gate
passes.

## Disclosed metadata-only probes

Selection used official schema documents and synthetic SEC sample XML only.
Their example numbers are documentation fixtures, not Circle source values.

One bounded external research probe opened identity metadata from a 2026
Circle Reserve Fund N-MFP3 XML and BlackRock-family submissions metadata. It
verified the six frozen identities, form family, monthly cadence, and presence
of EDGAR acceptance timestamps in the submissions metadata. It did not inspect
or report Circle liquidity
percentages, WAM, WAL, holdings, net assets, flows, yields, candidate
incidence, or market outcomes.

No production source row has been built or opened.

## Stop and anti-repair rule

The next work unit may commit a source builder and source-support evaluator,
but it may not open the production daily indexes or Feed archives before those
files, tests, and their exact protocol hashes are committed and pushed.

Any first failed frozen gate is `REJECT_NO_REPAIR`. The branch may not change
the series, interval, form family, transport, identity, amendment policy,
required fields, last-date rule, availability delay, or source thresholds
after production source incidence is opened. A rejection closes CRF-NMFP for
the current alpha search.
