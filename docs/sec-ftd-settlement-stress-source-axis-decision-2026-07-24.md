# SEC fails-to-deliver settlement-stress source-axis decision — 2026-07-24

## Decision

Open one new **source-only** research axis:

```text
SFTD-v1 — SEC/NSCC Fails-to-Deliver Settlement-Stress Ledger
```

SFTD-v1 asks only whether the official SEC fails-to-deliver archive can
reproduce a causal, live-readable ledger of aggregate NSCC Continuous Net
Settlement delivery failures over a fixed recent history.

This decision authorizes a source builder and one full source audit for:

```text
2022-01-01 through 2026-06-30
```

It does **not** authorize:

- a crypto-equity, ETF, control, or other symbol universe;
- a candidate clock, direction, threshold, hold, action, or position;
- a rank, z-score, breadth, persistence, transition, or topology feature;
- BTC, equity, funding, premium, open-interest, return, PnL, CAGR, MDD, or
  model access; or
- use of the SEC `PRICE` field as either a feature or outcome.

No official FTD ZIP body, text row, symbol incidence, CUSIP incidence,
row-level quantity, price, candidate count, market value, or outcome was
opened before this decision. Only the official landing-page metadata, field
contract, update description, file labels, file sizes, and site policies were
intentionally inspected.

One search-result snippet incidentally exposed archive-wide quantity examples
copied into a third-party petition hosted on sec.gov. They were neither queried
nor used to select this source. To prevent that accidental aggregate exposure
from influencing a later mechanism, SFTD-v1 forbids archive-total fail
quantities, notional totals, and level thresholds from every candidate and
model input. The source audit may count rows and dates but may not sum
`QUANTITY (FAILS)`.

A separate committed mechanism decision must precede the first
candidate-specific source query or transformation.

## Why this source is genuinely new

Repository-wide searches before this decision found no prior use of:

- SEC fails-to-deliver records;
- NSCC CNS delivery-failure balances;
- settlement-failure breadth; or
- a symbol-level delivery-stress ledger.

Prior SEC work in this repository used EDGAR filing text and filing clocks.
SFTD-v1 is an equity-clearing and settlement observable, not another filing,
news, price, volume, funding, options, blockchain, custody, or macro series.

The following source families remain closed and may not be reconstructed
through SFTD-v1:

- exchange or off-exchange price, trade, volume, or short-volume signals;
- funding, premium, open-interest, liquidation, or options signals;
- SEC EDGAR filing families;
- Bitcoin or Ethereum chain-state families;
- stablecoin, bridge, custody, and exchange-inventory families; and
- the rejected Aave V3 borrow-rate ledger.

## Official economic object

The SEC describes each record as the aggregate net balance of shares that
failed to be delivered as of a settlement date in NSCC's Continuous Net
Settlement system, aggregated over all NSCC members:

<https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data>

The SEC explicitly states that:

- the value is a balance level, not the number of new daily fails;
- a current balance combines newly created fails, previously outstanding
  fails, and fails settled that day;
- the age and underlying source of fails cannot be inferred from adjacent
  records;
- a zero balance is omitted on or after 2008-09-16;
- fails can arise from long or short sales for multiple reasons; and
- the data are not evidence of abusive or naked short selling.

SFTD-v1 therefore names the object **settlement stress**. Any later mechanism
that relabels the quantity as short interest, naked shorts, directional
selling, or fail age is invalid.

The official page states that the archive spans February 2004 through June
2026 as inspected on 2026-07-24. The fixed 2022-01 through 2026-06 audit
window is long enough for train/test/eval separation and avoids the different
pre-2009 reporting threshold.

## Official publication and schema contract

For data starting July 2009, the SEC publishes two ZIP files per month:

- the first half of a month is normally available at the end of that month;
- the second half is normally available at about the fifteenth of the next
  month.

The SEC does not guarantee posting on a particular date and does not
guarantee data accuracy. Settlement date is therefore **never** an availability
timestamp.

The official file is pipe-delimited with this six-field layout:

```text
SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
```

The documented meanings are:

| field | meaning |
|---|---|
| `SETTLEMENT DATE` | eight-digit settlement date |
| `CUSIP` | security identifier, at most 9 characters |
| `SYMBOL` | ticker symbol, at most 10 characters |
| `QUANTITY (FAILS)` | aggregate shares failing to deliver |
| `DESCRIPTION` | company/security name, at most 30 characters |
| `PRICE` | previous-day closing price when available and above one cent |

The SEC warns that `PRICE` may be `"."` and may disagree with prices from
other sources. SFTD-v1 validates that field only as source syntax and then
excludes it physically from normalized candidate inputs.

## Frozen retrieval contract

### Index

The sole archive index is:

```text
GET https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data
```

The builder must identify exactly two links for every month from January 2022
through June 2026. Labels must map one-to-one to first or second half and to
the expected calendar month.

### Archive links

Every accepted archive URL must:

- use `https`;
- have host `www.sec.gov`;
- have no credentials, query, or fragment;
- remain under `/files/data/fails-deliver-data/`; and
- end in exactly `cnsfailsYYYYMMa.zip` or `cnsfailsYYYYMMb.zip`, where the
  date and half match the index label.

URL validation occurs after a single strict percent-decoding pass and RFC 3986
dot-segment normalization. An encoded slash or backslash, literal backslash,
control character, invalid percent escape, user-info component, mixed-case or
trailing-dot host variant, non-default port, second decoding change, or any
normalized path outside the frozen directory rejects the source.

No guessed URL, mirror, search-engine cache, alternate SEC path, CDN rewrite,
Internet Archive copy, vendor copy, or user-supplied replacement is allowed.
Redirects are accepted only when every hop remains HTTPS on `www.sec.gov` and
the final path still satisfies the exact archive rule.

### HTTP behavior

The builder must:

1. use a declared contact-bearing `User-Agent`;
2. request at most one SEC resource per second;
3. use no cookie, browser session, API key, private account, proxy, or
   authentication;
4. cap the index body at 4 MiB and each ZIP body at 16 MiB;
5. send `Accept-Encoding: identity`;
6. require HTTP 200, a non-empty body, a finite `Content-Length` when
   supplied, and a valid RFC 7231 `Last-Modified` archive header;
7. accept `text/html` for the index and only `application/zip`,
   `application/x-zip-compressed`, or `application/octet-stream` for an
   archive, ignoring only an optional media-type parameter;
8. reject partial content, a `Content-Encoding` other than absent or
   `identity`, duplicate links, any other content type, or a body larger than
   its cap;
9. hash the exact index and archive response bytes with SHA-256; and
10. persist retrieval UTC, request URL, final URL, redirect ledger, status,
   selected response headers, byte count, and hash.

SEC site policy currently limits automated access to no more than ten requests
per second and disallows unclassified bots. SFTD-v1 deliberately uses the
stricter one-request-per-second boundary:

<https://www.sec.gov/about/privacy-information>

## ZIP and text safety boundary

Each archive must:

- be a valid, unencrypted ZIP;
- contain exactly one regular, non-symlink text member;
- contain no directory, absolute path, parent traversal, duplicate name,
  nested archive, or unsupported compression method;
- have a declared and realized uncompressed size no greater than 128 MiB;
- have a compression ratio no greater than 100:1;
- pass CRC verification; and
- produce no bytes outside an isolated temporary directory.

The text member must decode as ASCII or UTF-8 without replacement characters.
It must contain exactly one header matching the frozen six-field layout.
Blank records, embedded NULs, surplus delimiters, short or long rows, and
duplicate headers reject the archive.

The normalized source parser must require:

- an eight-digit valid calendar settlement date;
- a non-empty CUSIP of at most 9 printable ASCII characters;
- a non-empty symbol of at most 10 printable ASCII characters;
- a strictly positive integer `QUANTITY (FAILS)`;
- a description of at most 30 printable characters; and
- `PRICE` equal to `"."` or a finite positive decimal.

Every settlement date must lie in the calendar half encoded by its archive.
The identity `(SETTLEMENT DATE, CUSIP)` must be unique within the full frozen
ledger. A missing row on or after 2008-09-16 is not materialized as zero
during the source audit; the SEC's zero-omission semantics may be used only
after a later mechanism decision freezes a symbol universe and absence rule.

## Conservative historical availability

The SEC provides a normal schedule but no guaranteed historical posting
timestamp. The builder must therefore derive a deliberately late clock:

```text
first-half nominal availability
  = day 8 of the following calendar month at 00:00:00 UTC

second-half nominal availability
  = day 23 of the following calendar month at 00:00:00 UTC

archive available_at_utc
  = max(nominal availability, HTTP Last-Modified)
```

The extra seven or eight calendar days are a fixed safety delay beyond the
normal SEC schedule. The `Last-Modified` requirement moves a late publication
or replacement forward rather than backward.

This is still a current-vintage reconstruction, not proof that every
historical revision timestamp was preserved. The source audit must report that
residual limitation. Live collection must additionally record first successful
observation and use:

```text
live_available_at_utc
  = max(nominal availability, HTTP Last-Modified, first_successful_fetch_utc)
```

A later mechanism may use SFTD-v1 only after a separately committed
candidate-clock decision freezes the target market clock. No candidate may
treat any SFTD record as available before `available_at_utc` or backdate it to
a settlement date, file period end, SEC page footer date, or expected posting
date.

If any archive lacks a valid `Last-Modified` header, if that header precedes
its own settlement month by an impossible amount, or if effective availability
leaves less than three years of usable history, SFTD-v1 is rejected without
repair.

## Source-audit outputs

The one-shot audit may report only:

- index and archive hashes;
- file count, total byte count, total row count, and distinct settlement-date
  count;
- earliest and latest settlement and availability timestamps;
- schema, ZIP, duplicate, range, and missingness checks;
- aggregate archive lag distributions;
- whether all fixed gates passed; and
- a generic failure stage and exception class on rejection.

It may not report:

- any symbol, CUSIP, description, quantity, or price;
- per-symbol or candidate-specific incidence;
- top/bottom records, ranks, ratios, breadth, persistence, or transitions;
- any market, funding, return, PnL, portfolio, or model value; or
- which archive contained a value-level failure.

Raw archives and extracted text stay in ignored local source storage. Only the
small audit manifest and gate report may be committed.

## Revision and no-repair policy

The first successful hash of each archive becomes the immutable research
specimen. A later byte change is a new source version with a later observed
availability; it may not silently overwrite prior bytes or revise a historical
candidate backward.

The source identity is terminal:

- one full source-audit attempt is permitted after the tested builder is
  committed;
- transport, schema, timestamp, or coverage failure retires SFTD-v1;
- no alternate endpoint, user-agent tuning, link guessing, mirror, range
  reduction, parser relaxation, timestamp substitution, or manual file repair
  is allowed after source bytes are opened; and
- a rejected source may not proceed to a mechanism, synthetic gate, market
  join, or model stage.

The one-shot attempt begins before the first production archive byte is
requested. Before that request, the committed builder commit hash, decision
file hash, run ID, ordered list of all 108 expected archive labels, and an
empty append-only run manifest must be recorded atomically. Any production
archive-byte request, successful or failed, consumes the single attempt.
Index-only fixture tests and wholly synthetic archive tests do not consume it;
no test or dry-run mode may request a production archive URL.

Before minting the one-shot audit capability, the runner must:

1. verify that this decision, the builder, and its tests are committed;
2. verify this decision's SHA-256 against a constant embedded in the committed
   runner and verify that the exact builder and test paths match `HEAD`;
3. require a clean worktree;
4. require the actual filesystem usage to remain below 300 GiB with at least
   2 GiB free;
5. atomically create an exclusive start sentinel that records the commit,
   decision hash, runner Git blob identity, and UTC start time; and
6. re-run the same commit, path, boundary-hash, cleanliness, and disk checks
   immediately before the first SEC archive request.

The start sentinel is never deleted or overwritten. Unit tests and fixture
tests must not create it. Once it exists, no retry, resume, replacement URL,
parser repair, narrower date range, or second full audit is allowed. The
runner must suppress archive-level progress, URL, filename, label, and value
output so a terminal rejection reveals only a generic stage and exception
class.

## Why FINRA daily short volume is not selected

FINRA daily short-sale volume is more timely and has deeper daily support, but
it is not an admissible live-trading research source for this repository.

FINRA states that the data are free for non-commercial use. Its current Terms
of Use restrict website content to non-commercial personal or professional
use and expressly prohibit use of any portion of the FINRA website with
machine learning, neural networks, predictive analytics, or another process
designed to predict trades for a portfolio, fund, or investment vehicle:

- <https://www.finra.org/finra-data/browse-catalog/short-sale-volume>
- <https://www.finra.org/terms-of-use>

The FINRA source is therefore closed unless a separately documented written
commercial license explicitly authorizes the intended trading and model use.
No FINRA data file was opened.

## Reuse and production boundary

The SEC states that information on sec.gov is public information that users
may copy or further distribute without permission, subject to attribution and
trademark limitations:

<https://www.sec.gov/about/privacy-information>

That policy makes SFTD-v1 operationally preferable to FINRA for this research.
It is not a warranty of data accuracy or investment fitness.

SEC public-information reuse does not by itself authorize real-order or
commercial deployment. Real-order use requires a separate documented
legal/compliance approval covering attribution, trademark limits, the intended
commercial and model use, and any then-current SEC policy changes.

Before live-shadow or real-order use, a separate committed
`SFTD-LIVE-v1` boundary must prove:

- prospective twice-monthly polling with the declared user agent;
- first-seen timestamps and immutable versioned raw bytes;
- hash overlap with the frozen research ledger;
- stale-page, late-file, replacement-file, and partial-month fail-flat
  behavior;
- no source use before the separately frozen target-market clock permits it;
- no use of `PRICE`; and
- forced `ABSTAIN` on missing, changed, delayed, or ambiguous source state.

Until that later boundary passes, all SFTD-derived actions are research-only
and live execution must abstain.

## Stop conditions

Retire SFTD-v1 before mechanism work if any of the following occurs:

- the official index does not expose exactly 108 expected archive links;
- any accepted URL violates the frozen host/path/name rule;
- any required archive cannot be retrieved under the one-request-per-second
  policy;
- a ZIP, text, schema, date-half, numeric, duplicate, or timestamp gate fails;
- the frozen ledger cannot provide at least three years of effective causal
  availability;
- the audit opens or emits a forbidden record value or candidate incidence;
  or
- any market, funding, outcome, portfolio, or model source is accessed.

The immediate next work unit is an independently reviewed, tested source
builder. It must be committed before the single full 2022-01 through 2026-06
source audit is run.
