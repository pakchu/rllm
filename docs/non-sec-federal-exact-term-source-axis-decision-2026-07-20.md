# NFET source-axis decision — 2026-07-20

## Decision

Select **NFET (Non-SEC Federal Exact-Term publication salience)** as the next
outcome-blind source axis. NFET is a clock of U.S. Federal Register documents
that are discovered through a frozen search envelope, contain an exact frozen
crypto term in the official GovInfo document rendition, and are not assigned
to the Securities and Exchange Commission (SEC).

NFET is a source, not an alpha. This decision does **not** authorize a trading
direction, price response, holding period, language-model label, reward,
checkpoint, or portfolio weight. Those decisions remain sealed until the
source and novelty gates pass.

## Evidence boundary

No BTC bar, funding row, return, PnL, prior-alpha outcome, CAGR, MDD, or trade
clock was opened while selecting this axis.

A source-only FederalRegister.gov API envelope probe **was opened** over
2020–2023 and returned
486 unique document numbers across broad searches for bitcoin,
cryptocurrency, virtual currency, distributed ledger, and blockchain. It
covered 348 publication days and contained 218 SEC documents. These counts
show transport feasibility and material SEC overlap only. They are not final
NFET incidence because the API's term search can return documents that do not
contain the frozen exact phrase. Final exact GovInfo membership incidence was
not opened.

The GovInfo bulk XML is not membership authority. GovInfo's XML guide states
that the bulk XML is a converted/simplified rendition and is not part of the
official online format; GovInfo identifies the PDF and text renditions as the
official online content. A source-only probe also found that the 44,165-byte
`FR-2021-01-20.xml` object, SHA-256
`b4b99acbd6832b98f5d0b8c01c0e60d5520a6428518336c175986dfa5cfa7523`,
fails a standard XML parse at line 70. NFET therefore does not depend on bulk
XML for corpus completeness, membership, agency routing, or correction
handling.

Official references:

- [GovInfo Federal Register help](https://www.govinfo.gov/help/fr)
- [GovInfo sitemaps](https://www.govinfo.gov/sitemaps)
- [GovInfo API](https://github.com/usgpo/api)
- [GovInfo Federal Register XML guide](https://www.govinfo.gov/bulkdata/FR/resources/FDsys_OFR-XML_User-Guide-v1.pdf)
- [FederalRegister.gov API documentation](https://www.federalregister.gov/developers/documentation/api/v1)

## Frozen source hierarchy

NFET uses three layers with distinct authority:

1. **Expected issue inventory — GovInfo collection sitemap.**
   `https://www.govinfo.gov/sitemap/FR_{YYYY}_sitemap.xml` is fetched for each
   year 2020, 2021, 2022, and 2023. Only URLs exactly matching
   `https://www.govinfo.gov/app/details/FR-YYYY-MM-DD` are eligible package
   identities. Every response byte, retrieval time, HTTP validator, package
   ID, and parser SHA is sealed. Duplicate package IDs, out-of-year dates,
   malformed XML, or an absent year fail closed. Weekends and holidays are not
   inferred: a date absent from the sealed GovInfo sitemap is an expected
   non-issue date.
2. **Candidate discovery — FederalRegister.gov search API.** The API may only
   propose document numbers. All pages of every frozen query are fetched and
   hashed; `total_pages`, `count`, document numbers, publication dates, and
   response order are recorded. Search rank, snippets, and API term matching
   never establish membership.
3. **Membership authority — official GovInfo granule renditions.** For every
   search candidate, NFET fetches the predictable GovInfo MODS metadata and
   document HTML:
   `https://www.govinfo.gov/metadata/granule/FR-{publication_date}/{document_number}/mods.xml`
   and
   `https://www.govinfo.gov/content/pkg/FR-{publication_date}/html/{document_number}.htm`.
   Only exact terms found in the canonical visible text extracted from the
   HTML can establish membership. The corresponding `/pdf/{document_number}.pdf`
   rendition is fetched and hashed for every positive member as the
   archival/legal authority. PDF is never mined to add a candidate or used to
   repair an HTML mismatch. These routes require no API key.

FederalRegister.gov detail JSON is mandatory supplementary reconciliation for
each positive member. It supplies correction relationships and an independent
agency-slug check, but cannot create membership or override GovInfo metadata.
The detail endpoint is
`https://www.federalregister.gov/api/v1/documents/{document_number}.json`.

## Frozen candidate envelope

The interval is `[2020-01-01, 2024-01-01)`. Candidate discovery executes each
of these ten literal `conditions[term]` values independently:

1. `bitcoin`
2. `cryptocurrency`
3. `cryptocurrencies`
4. `virtual currency`
5. `virtual currencies`
6. `virtual-currency`
7. `virtual-currencies`
8. `blockchain`
9. `distributed ledger`
10. `distributed-ledger`

Every query also fixes publication-date `gte=2020-01-01`, publication-date
`lte=2023-12-31`, `order=oldest`, and `per_page=1000`. Pages are reconstructed
as integer pages `1..total_pages`; arbitrary next-page URLs are not followed.
The union key is `document_number`. A duplicate with conflicting publication
date, type, title, or GovInfo URL is fatal.

Every candidate publication date must appear in its sealed GovInfo issue
inventory. A candidate on a missing package date is fatal. The fixed search
envelope is intentionally named in the source. An exact-term GovInfo document
that the query union does not propose is outside NFET-v1. NFET does not claim
perfect full-corpus recall or a full semantic census.

## Frozen official-text parser

The parser contract is executable and version-bound:

- decode GovInfo HTML bytes as UTF-8 with strict errors;
- parse with Python standard-library `html.parser.HTMLParser`,
  `convert_charrefs=True`;
- ignore data inside `script` and `style` elements;
- append every other `handle_data` value in document order, including table
  cells and element tails;
- join data fragments with one ASCII space;
- normalize with Unicode NFKC using the preregistered Unicode database
  version;
- lowercase with `str.lower`;
- replace every Unicode code point whose general category starts with `C`
  with one ASCII space;
- collapse Python-Unicode `\s+` to one ASCII space and strip the ends; and
- run Python `re` with `re.ASCII`, no case-insensitive or multiline flags.

Every candidate's raw search page, GovInfo MODS, raw GovInfo HTML, canonical
visible text, and FederalRegister detail JSON are retained in a
content-addressed deterministic compressed archive. Positive members also
retain the GovInfo PDF and exact match records. This permits independent replay
of both membership and non-membership decisions. Gzip `mtime` is zero. The
disk guard aborts before each download if the output filesystem has reached
300 GiB used.

## Frozen exact membership

The normalized official GovInfo HTML text must contain at least one of these exact
ASCII-boundary regular expressions:

```text
(?<![a-z0-9_])bitcoin(?![a-z0-9_])
(?<![a-z0-9_])cryptocurrenc(?:y|ies)(?![a-z0-9_])
(?<![a-z0-9_])virtual[ -]+currenc(?:y|ies)(?![a-z0-9_])
(?<![a-z0-9_])blockchain(?![a-z0-9_])
(?<![a-z0-9_])distributed[ -]+ledger(?![a-z0-9_])
```

No stemming, fuzzy match, semantic expansion, synonym addition, API snippet,
search score, or LLM relevance decision may change membership. Every positive
stores the regex ID, exact supporting substring, character span, and official
HTML SHA-256.

The stable identity is `document_number`. GovInfo MODS `accessId`, `frDocNumber`,
and `FR Doc No.` identifier must equal it; the host `dateIssued` and package
URI must equal the candidate publication date; `collectionCode` must be `FR`;
and MODS `otherFormat` HTML/PDF links must exactly equal the predictable
GovInfo URLs. The FederalRegister detail document number and publication date
must also equal the candidate identity, and its `pdf_url` must equal the same
GovInfo PDF URL. The PDF must be nonempty and start with `%PDF-`; its SHA-256
and size are stored. Any absence or mismatch rejects the whole source without
quarantine or imputation.

## Frozen agency routing

The canonical agency set comes from GovInfo MODS extension `agency` values. Each
name is Unicode NFKC-normalized, uppercased with `str.upper`, has category-`C`
code points replaced by spaces, and has whitespace collapsed. The resulting
sorted, unique set must be nonempty; missing, empty, or duplicate names fail
closed. FederalRegister detail `agencies[].slug` remains a mandatory
independent cross-check.

- A document enters the mandatory SEC comparator if **any** canonical GovInfo
  agency name is exactly `SECURITIES AND EXCHANGE COMMISSION`.
- A mixed-agency document containing that name is comparator-only and cannot
  enter the primary stratum.
- Every other positive document enters the primary non-SEC stratum.
- Agency concentration uses fractional attribution: a primary document with
  `k` agencies contributes `1/k` to each agency. Each agency's share is its
  fractional sum divided by the number of primary documents.

The original GovInfo agency objects, canonical name set, and FederalRegister
agency objects are retained. GovInfo SEC membership must agree with whether
the detail slug set contains `securities-and-exchange-commission`; disagreement
is fatal. No alias map, manual agency relabel, or parent-agency rollup is
allowed.

## Causal availability and corrections

- Historical availability is publication date plus one calendar day at
  **12:00 UTC**.
- Advance-display and public-inspection times are ignored because equivalent
  historical point-in-time retrieval is not frozen.
- Live availability is the later of the historical floor and durable local
  receipt, parse, hash, metadata reconciliation, and manifest commit.
- A correction is a separate document at its own publication date and causal
  availability. `correction_of` and `corrections` are recorded from detail
  JSON, but the original record is never edited, deleted, relabeled, or
  retimed.
- A changed hash for a previously sealed sitemap, search page, GovInfo MODS,
  canonical visible text, or PDF halts NFET. History is not rebuilt in place.
  The raw HTML response SHA is recorded for transport audit but is not source
  identity: GovInfo's edge can vary email-obfuscation attributes while leaving
  visible text unchanged. The canonical visible-text SHA is the deterministic
  membership identity.

The source clock contains one event per unique primary publication date at the
frozen historical availability. Multiple primary documents on one date do not
create multiple trade clocks. Document counts and agency composition remain
source features for a later mechanism decision, not trading labels here.

## Frozen source-quality gates

Before any market input, semantic model, or prior-alpha outcome is opened, the
complete 2020–2023 source **under the frozen FederalRegister API candidate
envelope** must satisfy every gate.

### Primary non-SEC stratum

- at least 140 exact-member documents;
- at least 25 documents in every calendar year;
- at least 120 unique publication days and 20 in every year;
- at least five documents in every calendar quarter;
- no calendar month above 15% of primary documents; and
- no fractionally attributed agency share above 45%.

### Mandatory SEC comparator

- at least 100 exact-member documents;
- at least 15 documents in every calendar year; and
- at least 80 unique publication days.

### Integrity

- 100% of candidate discovery pages reconcile to the declared page counts;
- 100% of candidate publication dates exist in the sealed GovInfo issue
  inventory;
- 100% of positive members reconcile to nonempty GovInfo MODS, exact official
  GovInfo HTML, nonempty GovInfo PDF, and detail JSON;
- every positive has one document number, publication date, document type,
  nonempty canonical agency set, exact supporting substring, and correction
  record;
- every fetched byte stream, parser, decision, source protocol, output, and
  manifest is SHA-256 bound; and
- a deterministic rebuild from the sealed responses reproduces the exact
  selected-document artifact.

There is no 99.5% allowance. One unexplained absence, mismatch, duplicate, or
parse failure returns `REJECT_NO_REPAIR`. Rejected records do not count toward
incidence gates because the whole source is retired.

## Frozen novelty gate before economics

Passing source support does not establish alpha. A separate outcome-blind
novelty stage must create the deduplicated primary availability clock and
compare it to the committed GDELT GNRC, SEC EDGAR, Wikimedia, BitMEX Trollbox,
and executable live-portfolio clocks over common coverage.

For each comparator, sort unique UTC entries and perform greedy one-to-one
matching in primary chronological order. Match the unused comparator entry
with minimum absolute time distance; ties choose the earlier comparator. The
stage records:

- exact-entry Jaccard, which must be at most **0.20**;
- one-to-one Jaccard within `±24h`, which must be at most **0.35**; and
- primary containment within `±24h`, which must be at most **0.50**.

All three thresholds pass independently for every comparator. Missing,
malformed, unhashable, or coverage-empty comparator artifacts fail closed.
The NFET SEC stratum is an additional mandatory same-source comparator under
the same thresholds; it is never a fallback primary candidate. Comparator
paths, schemas, common coverage, and SHA-256 values must be committed in the
novelty access seal before the novelty evaluator opens any clock.

## Later LLM/RLLM boundary

Only a later commit may define a mechanism. If Gemma is used, it begins as a
quote-grounded bounded extractor/classifier over source text with synthetic,
prompt-injection, entity-swap, timestamp-swap, and memorization controls. A
single-model RLLM gate may be trained later on train-only data, but it may not:

- create, delete, or retime source events;
- infer source membership or agency routing;
- see eval/holdout rewards during prompt, adapter, or checkpoint selection;
- receive raw future prices or post-entry information; or
- reintroduce an analyzer/trader two-model split.

Any BTC reaction feature opens a market prefix and therefore needs its own
frozen pre-entry window, cost model, and access seal. None is authorized by
this source decision.

## Next authorized work

1. Commit an outcome-blind executable source protocol and immutable manifest.
2. In a later commit, fetch and seal only the 2020–2023 issue inventories and
   frozen candidate-query responses.
3. Stream candidate official sources; retain content-addressed raw and
   canonical replay artifacts for every candidate, plus PDFs and match records
   for positives.
4. Reconcile every positive to detail JSON and official GovInfo PDF.
5. Run the frozen source-quality evaluator.
6. Reject without repair on failure; otherwise preregister and run novelty
   before any market or model stage.
