# Treasury TIC release-vintage source-axis decision — 2026-07-24

## Decision

Select:

```text
TICRV-v1 — Treasury International Capital Release-Vintage Ledger
```

as the next independent source-only BTC alpha research axis.

TICRV-v1 asks only whether official U.S. Treasury monthly release archives can
reproduce a causal, point-in-time ledger of the cross-border securities and
financial-firm data that were publicly available at each release.

The fixed source window contains the 54 monthly release archives whose
official index labels run from:

```text
2022-01-18 through 2026-06-18
```

This decision authorizes one tested source builder and one full source audit.
It does **not** authorize:

- a TIC table, line, field, country, sector, instrument, or flow selection;
- a revision, level, change, breadth, rank, disagreement, or topology feature;
- a candidate clock beyond the source availability boundary;
- a target market, bar size, side, action, hold, threshold, or position;
- market, funding, premium, open-interest, return, PnL, CAGR, MDD, portfolio,
  reward, checkpoint, or model access; or
- any claim that foreign flows predict BTC.

No TIC ZIP body, member body, table header, row, field value, country value,
position, transaction, revision, flow, candidate incidence, market value, or
outcome was opened before this decision. Only official documentation, page
metadata, release labels, link URLs, HTTP status/header metadata, and ZIP byte
lengths were inspected.

A separate committed mechanism decision must precede the first table-specific
member read or candidate transformation.

## Why this source is genuinely new

Repository-wide searches found no prior Treasury International Capital source,
TIC release-vintage ledger, or cross-border portfolio-flow alpha.

TICRV-v1 is not a relabeling of previously studied:

- Federal Reserve H.4.1 balance-sheet components;
- Treasury operating cash, debt, or fiscal-flow data;
- New York Fed repo, securities-lending, or SOMA data;
- CFTC positioning;
- SEC EDGAR filings;
- equity, futures, options, funding, premium, or open-interest data;
- Bitcoin Core development activity; or
- chain, stablecoin, custody, bridge, and exchange-inventory data.

The official Treasury description identifies TIC as the U.S. government's
source of data on capital flows into and out of the United States, excluding
direct investment, and the resulting cross-border claims and liabilities:

<https://home.treasury.gov/data/treasury-international-capital-tic-system-home-page/description-of-TIC-system>

This is a cross-border balance-sheet and transaction source, not another
domestic-liquidity level.

## Official source contract

### System and release semantics

Official TIC system page:

<https://home.treasury.gov/data/treasury-international-capital-tic-system>

Official monthly archive:

<https://home.treasury.gov/archives-of-tic-monthly-data-releases>

The archive states:

> Each archive is a ".zip" file containing the TIC data released on the date
> shown in the filename.

That release-vintage statement is the source's central causal property. A
later release may revise prior months, but TICRV-v1 never substitutes a current
historical table for an earlier archive. When Treasury explicitly annotates a
later public-release date, that annotation overrides an earlier label or
filename date for availability.

Official release-date and revision policy:

<https://home.treasury.gov/data/treasury-international-capital-tic-system/release-dates-of-tic-data>

Treasury states that:

- monthly data are normally released on the eleventh business day plus zero
  to three days;
- releases occur at 4:00 p.m. Washington, D.C. time beginning 2014-09-16;
- data are posted with or soon after the press release;
- Washington federal-office closures can delay a release;
- January, April, July, and October releases normally revise the past year;
- other monthly releases normally revise the prior three months; and
- significant revisions can reach farther back.

The source must therefore preserve each complete release archive as a separate
immutable vintage.

### What TIC covers

Treasury describes monthly and quarterly TIC reports as covering:

- transactions in and holdings of long-term U.S. and foreign securities;
- cross-border claims and liabilities reported by banks and other financial
  firms;
- positions reported by nonfinancial firms;
- derivatives; and
- portfolio claims, liabilities, and holdings.

TIC Form S was discontinued in February 2023 after transaction reporting was
added to expanded Form SLT. That semantic break is not repaired or bridged in
the source audit:

- <https://home.treasury.gov/data/treasury-international-capital-tic-system/tic-forms-instructions>;
- <https://home.treasury.gov/data/treasury-international-capital-tic-system-home-page/frequently-asked-questions-regarding/ticfaq2>.

A later mechanism must either select a source family with documented
continuity across that boundary or treat the boundary as a frozen structural
break. No learned imputation may erase it.

## Pre-boundary transport evidence

Only metadata-level requests were made.

### Archive index

A direct HTTPS `GET` to the official archive index returned:

```text
status: 200
content-type: text/html; charset=UTF-8
bytes: 172547
SHA-256: 26a3e05c77c19aac3d3a52cf15bcc92a3716e8e10107e05c395977b7ee43c0e8
```

The page exposed 284 distinct ZIP links. Exactly 54 distinct labels and URLs
fell in the fixed 2022-01-01 through 2026-06-30 release-label range:

- earliest label: `01/18/2022`;
- latest label: `06/18/2026`.

Within that fixed range the archive description also contains exactly one
explicit delayed-public-release annotation:

```text
index label and filename date: 10/17/2025
archive description:           TIC Data for August 2025 (released 11-18-2025)
```

This annotation is source metadata, not a table value. It prevents the October
archive from being backdated across the federal-office closure.

### Current release index

A direct HTTPS `GET` to:

<https://home.treasury.gov/data/treasury-international-capital-tic-system/tic-press-releases-by-topic>

returned HTTP 200 and contained both:

- the latest release label `07/14/2026`; and
- the next scheduled release label `8/17/2026`.

It also records a single `11/18/2025` release for both August and September
2025 and warns that releases are rescheduled when Washington federal offices
are closed.

This page is live-source evidence only. The frozen research audit ends at the
June 2026 archive and may not add July after seeing source contents.

### Archive transport

The index links begin at `www.treasury.gov` and redirect to
`ticdata.treasury.gov` with the same path. Metadata-only `HEAD` probes for
official indexed releases in 2022, 2023, 2024, and 2026 reached HTTP 200 at
the final official host with `application/zip`.

Observed compressed lengths ranged from roughly 2 MiB to 11 MiB. No archive
body was requested.

One pre-boundary metadata probe guessed `ticrel_20240118.zip`; it returned 404.
The official index correctly labels the January 2024 release `01/19/2024` and
links `ticrel_20240119.zip`, which returned 200. The guessed URL is excluded
from the source identity and may never be retried.

## Frozen release identity

The index label is an ASCII date in `MM/DD/YYYY`. The linked filename normally
ends in:

```text
ticrel_YYYYMMDD.zip
```

For 53 of the 54 fixed releases, label date and filename date agree.

The sole frozen exception is:

```text
index label: 02/15/2022
filename:    ticrel_20220222.zip
```

For every release, parse only an official archive-description parenthetical
with the exact ASCII form:

```text
(released MM-DD-YYYY)
```

as:

```text
explicit_public_release_date
```

If that exact parenthetical is absent, the value is null. Any other text,
date order, punctuation, or inferred closure date is not parsed.

For every release:

```text
release_identity_date
  = max(
      index_label_date,
      filename_date,
      explicit_public_release_date when non-null
    )
```

The builder must reproduce exactly:

1. one label/filename mismatch, the frozen February 2022 exception; and
2. one delayed-public-release annotation:

```text
index label:                  10/17/2025
filename:                     ticrel_20251017.zip
explicit public release:     11/18/2025
```

A missing, additional, changed, or differently parsed exception, a changed
URL, or a different annotation rejects TICRV-v1. This rule uses the latest
official public identity and never moves information backward.

The normalized ordered identity is:

```text
(release_identity_date,
 index_label_date,
 filename_date,
 explicit_public_release_date,
 starting_url)
```

It must contain exactly 54 unique rows ordered by the full tuple above. The
October-label and November-label 2025 archives share the same
`release_identity_date`; their tuple order is deterministic but does not imply
an intra-day public order.

## Frozen retrieval contract

### Index URLs

Research archive index:

```text
GET https://home.treasury.gov/archives-of-tic-monthly-data-releases
```

Live metadata index, documentation-only:

```text
https://home.treasury.gov/data/treasury-international-capital-tic-system/tic-press-releases-by-topic
```

The full research audit may issue exactly one GET for the research archive
index and may download only the 54 ZIP URLs selected from it. It must not
request the live metadata index or follow any value, table, press-release, or
unrelated link from the archive HTML page. The live URL above documents the
pre-boundary closure cross-check and is reserved for a later live boundary.

The complete production Treasury request set is exactly 109 GET requests:

1. one archive-index GET;
2. after durable `identity_finalization`, exactly one GET to each of the 54
   indexed starting archive URLs; and
3. exactly one manually redirected GET to each corresponding final URL.

No production `HEAD`, `OPTIONS`, conditional, range, guessed-URL,
live-metadata, browser-preflight, or extra metadata request is permitted.

### Starting archive URL

Every starting archive URL must:

- use `https`;
- have exact host `www.treasury.gov`;
- have no credentials, explicit port, query, or fragment;
- use exact path
  `/resource-center/data-chart-center/tic/Documents/`; and
- end in the exact indexed `ticrel_YYYYMMDD.zip` filename.

### Redirect and final URL

Exactly one redirect is required:

```text
www.treasury.gov
  -> ticdata.treasury.gov
```

The redirect must:

- use status 301, 302, 307, or 308;
- remain HTTPS;
- preserve the exact path byte-for-byte;
- add no credentials, port, query, or fragment; and
- terminate at exact host `ticdata.treasury.gov`.

The final response must be HTTP 200. Any zero-hop response, second redirect,
host change, path rewrite, query addition, CDN, mirror, or alternate Treasury
path rejects the source.

All URLs are checked after one strict percent-decoding pass and RFC 3986
dot-segment normalization. Invalid escapes, encoded or literal slash/backslash
ambiguity, control characters, user-info, host case variants, trailing-dot
hosts, second-decoding changes, or normalized path escape reject the source.

### HTTP behavior

The builder must:

1. use a contact-bearing `User-Agent`;
2. use direct TLS without a proxy, cookie, browser session, API key, or
   authorization;
3. send `Accept-Encoding: identity`;
4. issue at most one Treasury request per second, including redirects;
5. cap each index body at 4 MiB and each ZIP body at 32 MiB;
6. reject partial responses, non-identity content encoding, malformed or
   duplicate length headers, and bodies over the frozen cap;
7. accept only `text/html` for an index and `application/zip` or
   `application/octet-stream` for an archive, ignoring only an optional media
   parameter;
8. hash exact response bytes with SHA-256 before parsing; and
9. persist request time, starting URL, redirect status and location, final
   URL, final status, selected headers, byte count, and exact hash in ignored
   local source storage.

No retry is permitted for an index, redirect, or archive request during the
full source audit.

## Conservative availability clock

HTTP `Last-Modified` is not the historical release clock.

Metadata probes showed that older archives currently carry February 2026
`Last-Modified` timestamps and that successive recent archives can carry the
next release's timestamp. Those headers reflect migration or subsequent file
handling and would collapse historical availability into a later date.

TICRV-v1 instead uses the official release-vintage identity:

```text
release_available_at_utc
  = start of the calendar day after release_identity_date
    in America/New_York, converted to UTC
```

This is at least eight hours after the normal 4:00 p.m. Eastern release and
also handles the frozen delayed February 2022 filename and November 2025
co-release conservatively.

A later candidate may use a release only after a separately committed target
clock is strictly later than `release_available_at_utc`. It may never backdate
an archive to:

- the month described by the data;
- the label date when the filename date is later;
- the label or filename date when an explicit public-release date is later;
- 4:00 p.m. exactly;
- a ZIP member timestamp; or
- an HTTP `Last-Modified` timestamp.

No candidate may infer or use an intra-day order between the two archives
whose `release_identity_date` is `2025-11-18`.

Live collection must additionally record the first complete successful fetch:

```text
live_available_at_utc
  = max(release_available_at_utc, first_successful_fetch_utc)
```

## ZIP and member safety boundary

Each release must be a valid, unencrypted ZIP with:

- at least one and at most 512 members;
- no directory, symlink, device, absolute path, parent traversal, duplicate
  member name, nested archive, or unsupported compression;
- only STORE method 0 or DEFLATE method 8;
- one disk only, no ZIP64, no data-descriptor flag, and no archive comment or
  bytes before the first local header or after the end-of-central-directory
  record;
- general-purpose flags equal either zero or only bit 11 (`0x0800`, UTF-8);
- exact agreement between each central-directory entry and local header for
  raw filename, general-purpose flags, compression method, DOS timestamp,
  CRC-32, compressed size, and uncompressed size;
- at most 128 MiB per member;
- at most 512 MiB total declared and realized uncompressed bytes;
- at most a 100:1 compression ratio per member and in aggregate; and
- successful CRC verification of every member.

Any other compression method, encryption or unsupported general-purpose flag,
multi-disk structure, ZIP64 marker or extra field, overlapping region,
duplicate central/local record, malformed extra field, or central/local
disagreement is `TERMINAL_REJECT`.

Allowed regular-member suffixes are:

```text
.txt .csv .htm .html .pdf
```

Member names are validated from the raw filename bytes in both the central
directory and local header. The byte strings must be identical, non-empty, and
consist only of ASCII bytes `0x20` through `0x7E`. They must:

- contain no `/`, `\`, NUL, colon, drive, UNC, absolute, or parent form;
- not equal `.` or `..`;
- equal a single basename under both POSIX and Windows path semantics; and
- have no exact or ASCII-lowercase duplicate within the archive.

The UTF-8 flag, if present, must agree between headers; because all permitted
bytes are ASCII, decoding must reproduce the exact same basename. A Unicode
path extra field, if present, must validate its name CRC and decode to that
same ASCII basename. Any encoding or separator/path interpretation ambiguity
is `TERMINAL_REJECT`.

Case is normalized only for suffix validation and the frozen exact-family key;
original member names and bytes are hashed unchanged. Unknown suffixes reject
the source before value-level inspection.

The source audit may:

- hash every member;
- validate magic bytes without interpreting table cells;
- count members and bytes by suffix;
- record member basenames, suffixes, exact-name hashes, and support counts; and
- identify whether an exact stable machine-readable member basename exists
  across all 54 vintages.

The source audit may not emit:

- a table heading, line label, field name, country, sector, or instrument;
- a numeric or categorical cell;
- a revision amount, level, change, ratio, rank, or direction;
- a candidate-specific incidence count; or
- any market, funding, outcome, portfolio, or model value.

Every member name must be a printable ASCII basename. Source-family keys used
for the stable-family support gate are computed only for non-empty regular
members whose normalized suffix is `.txt`, `.csv`, `.htm`, or `.html`. An
eligible key is the exact ASCII-lowercase basename; no date stripping, token
deletion, fuzzy matching, edit distance, or manual alias is allowed.

PDF member basenames may be counted for ZIP-safety metadata but never enter
the stable machine-readable intersection and can never satisfy source support.

Every ZIP member's timestamp representations must be deterministic. The
central and local DOS date/time fields must be identical. Recognized
timestamp-bearing extra fields are:

```text
0x5455  Extended Timestamp
0x000A  NTFS
0x5855  Info-ZIP Unix
```

Every available modification-time representation from those fields in either
header must parse exactly once and resolve to the same calendar date as the
DOS timestamp; Unix and NTFS epochs are interpreted in UTC. Duplicate,
truncated, unsupported-version, out-of-range, or mutually inconsistent
timestamp fields reject the archive.

That common date must be no later than seven calendar days after
`release_identity_date`, where the identity already includes any later
explicit public-release date. A later timestamp rejects the archive because
point-in-time preservation cannot be supported. Invalid, zero, or pre-1980
timestamps also reject it.

## Source-support gates

TICRV-v1 passes only if the one-shot audit proves all of the following:

1. exactly 54 frozen index identities;
2. exactly the one February 2022 label/filename mismatch;
3. exactly the one October 2025 delayed-public-release annotation;
4. the exact one-hop official redirect for every archive;
5. 54 valid archive hashes and no hash or URL duplication;
6. safe ZIP structure and valid member hashes for every release;
7. at least one non-empty `.txt`, `.csv`, `.htm`, or `.html` member in every
   release;
8. a non-empty intersection of exact machine-readable source-family keys
   across all 54 releases;
9. no internal timestamp later than the release-plus-seven-day boundary;
10. at least 52 strictly later adjacent availability transitions plus exactly
    one documented same-day co-release transition between the October-label
    and November-label 2025 archive identities, with no inferred intra-day
    order;
11. more than four years between earliest and latest causal availability; and
12. no forbidden source or outcome access.

A PDF-only family cannot satisfy machine-readable support.

The source audit may report only:

- index and archive hashes;
- aggregate byte, member, suffix, exact-family, and release counts;
- member basenames and their per-release support counts;
- earliest and latest source availability;
- strictly-later and same-day adjacent availability-transition counts;
- support-floor booleans;
- aggregate timestamp-lag distributions;
- a generic failure stage and exception class; and
- whether mechanism preregistration is authorized.

It may not reveal which archive, member, or schema failed on terminal
rejection.

## One-shot, commit, disk, and no-repair guard

Before the first production Treasury request of any kind, the archive-index
GET, the runner must:

1. verify this decision's exact SHA-256 against an embedded constant;
2. verify the decision, builder, and tests are committed and match `HEAD`;
3. require a clean worktree;
4. require filesystem use below 300 GiB and at least 2 GiB free;
5. create an exclusive immutable start sentinel containing the commit,
   decision hash, runner Git blob, run ID, and UTC start; and
6. create an empty append-only, hash-chained, ignored-local retrieval manifest.

The attempt is consumed when the sentinel is created, before the single
permitted archive-index GET. Every production Treasury request, successful or
failed, is part of that consumed attempt. Unit and fixture tests must use
synthetic URLs and bytes and may not create the production sentinel.

After the archive-index response succeeds and the parser proves the frozen
54-row identity contract, the runner must append exactly one
`identity_finalization` record containing all 54 ordered source identities.
No redirect or archive request is authorized before that record is durable.
The sentinel and the complete manifest prefix must be revalidated before every
later request. A failed, interrupted, malformed, or nonconforming index
request, or any failure before identity finalization, is `TERMINAL_REJECT` and
may not be retried.

The sentinel is never deleted or overwritten. Once it exists:

- no retry or resume;
- no second archive-index request and no live-metadata request;
- no request outside the frozen 109-GET sequence;
- no changed host, URL, redirect, or user agent;
- no range reduction;
- no parser, ZIP, timestamp, suffix, exact-family, or support repair;
- no alternate official table, current-vintage page, mirror, or cache; and
- no mechanism or outcome stage after rejection.

Transport or support failure is `TERMINAL_REJECT`.

## Raw storage and publication boundary

Exact index, redirect, archive, and member bytes remain only in ignored local
source storage. The committed result contains hashes and aggregate support
metadata, not TIC values.

The source runner must suppress progress, URLs, filenames, member names,
labels, and values on stdout/stderr. A rejection report exposes only a generic
stage and exception class.

Before live shadow or real orders, a separately committed `TICRV-LIVE-v1`
boundary must prove:

- prospective current-release discovery;
- exact release time and first-seen logging;
- immutable raw archive versioning;
- research/live archive hash overlap;
- delayed government-release handling;
- stale, revised, missing, or changed archive fail-flat behavior;
- the later target-market clock boundary; and
- forced `ABSTAIN` on any ambiguity.

Treasury's public data and site access do not by themselves constitute legal,
commercial, investment, or model-use approval. Real-order promotion requires
separate documented legal/compliance review of the then-current policies and
the intended use:

- <https://home.treasury.gov/subfooter/site-policies-and-notices>;
- <https://home.treasury.gov/footer/privacy-act/privacy-policy>.

Until both source support and the later live boundary pass, TICRV-v1 is
research-only and contributes no live signal.

## Candidate ideas deliberately not selected

Plausible later families include:

- revision geometry across consecutive release vintages;
- foreign official versus private sponsorship disagreement;
- securities-flow versus financial-firm balance-sheet disagreement; or
- cross-country sponsorship breadth.

These names are explanatory only. They do not select a member, field,
direction, threshold, clock, hold, or action. Exactly one mechanism must be
committed after a clean source audit and before any candidate incidence is
opened.

## Stop conditions

Retire TICRV-v1 immediately if:

- the index or any frozen archive is unavailable;
- the exact redirect contract changes;
- the February 2022 exception does not reproduce exactly;
- the October 2025 delayed-public-release annotation does not reproduce
  exactly;
- any ZIP/member/hash/timestamp gate fails;
- no stable 54-release machine-readable family exists;
- effective causal support is not longer than four years;
- any TIC value is emitted before a mechanism decision;
- any market, funding, outcome, portfolio, or model source is accessed; or
- the one-shot run is interrupted.

The immediate next work unit is an independent critique of this decision.
Only after PASS may the tested source builder be implemented and committed.
