# Federal Reserve Deliberation Communication Source-Axis Decision

## Decision

Open one source-only research axis:

**FRDCL-D1 — Federal Reserve Deliberation Communication Ledger**

FRDCL-D1 contains two official Federal Reserve document classes:

1. FOMC post-meeting or emergency policy statements; and
2. FOMC meeting minutes.

The source-release window is **2012-01-01 through 2020-12-31**. The window is
defined by public release date, not meeting date. It therefore includes the
minutes for the 2011-12-13 meeting, released on 2012-01-03, and excludes the
minutes for the 2020-12-16 meeting, released in 2021.

This decision authorizes only:

- retrieval of the ten fixed historical-year indexes for meeting years
  2011 through 2020;
- exact reconciliation against the committed metadata ledger;
- retrieval of the 147 frozen date-coded HTML pages;
- metadata checks on all 147 pages;
- canonical text extraction for the 145 rows frozen as eligible;
- structural source-support checks; and
- publication of one aggregate, text-free source verdict.

It does **not** authorize:

- Beige Book retrieval;
- the mutable current FOMC calendar;
- any 2021-or-later document body;
- human, agent, or model reading of raw or canonical corpus text;
- semantic labels, embeddings, sentiment, hawkishness, or topic extraction;
- a prompt, tokenizer, base model, adapter, or inference call;
- BTC, funding, price, reward, portfolio, checkpoint, CAGR, or MDD access; or
- selection among trading directions, horizons, gates, or position sizes.

A source pass permits one separately committed RLLM mechanism decision. That
mechanism must be committed **before any human, agent, or model reads any raw
or canonical corpus text**. A source rejection retires FRDCL-D1 without repair.

## Why this axis

Earlier source-first alpha attempts failed before economic evaluation because
the source was mutable, lacked a release-vintage archive, prohibited automated
ML use, or could not reproduce a historical live clock. FRDCL-D1 narrows the
problem to official, date-coded textual communications that can plausibly use
an LLM's language reasoning rather than asking an LLM to approximate arithmetic
over dense numeric rows.

This is not a profitability claim. It is a bounded test of whether the source
is causal, stable enough, numerous enough, and structurally reproducible enough
to justify one preregistered language mechanism.

## Why Beige Book and 2021+ are excluded

Legacy Beige Book pages exposed later page-update dates, including pages updated
years after release. They cannot satisfy this axis's first-party no-later-update
gate and are excluded rather than repaired.

The current FOMC calendar is mutable and contains post-window metadata. It is
not needed for a 2012-2020 release window and may not be requested. Restricting
production discovery to fixed historical-year pages leaves every 2021-or-later
body sealed for a later untouched evaluation.

## Official source contract

Only `https://www.federalreserve.gov` is authorized.

Official references:

- historical FOMC material:
  <https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm>
- historical-year index pattern:
  <https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm>
- Federal Reserve disclaimer:
  <https://www.federalreserve.gov/disclaimer.htm>

Production may request only:

```text
https://www.federalreserve.gov/monetarypolicy/fomchistorical2011.htm
...
https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm

the 147 exact official_url values committed in:
data/federal_reserve_deliberation_communication_identity_2012_2020.csv
```

The following are forbidden:

- `fomccalendars.htm`;
- any 2021-or-later historical index or item body;
- search engines, mirrors, browser caches, RSS, Wayback, FRED, or ALFRED;
- PDFs, transcripts, speeches, implementation notes, and press conferences;
- redirects that change the frozen identity; and
- a URL not already present in the committed ledger.

## Frozen metadata ledger

The authoritative identity and release-date ledger is:

```text
path:
data/federal_reserve_deliberation_communication_identity_2012_2020.csv

bytes:
29677

rows excluding header:
147

SHA-256:
a8586f749e2e1d3f3a83fb14de579f3c286fd1a8077af8fb9a5b67d247012bea
```

Its exact columns are:

```text
document_class
meeting_date
release_date
last_update_date
encoding
source_eligible
official_url
index_url
```

Rows are strictly sorted by:

```text
(release_date, document_class, official_url)
```

Frozen candidate counts:

| Release year | Statements | Minutes | Total |
|---:|---:|---:|---:|
| 2012 | 8 | 8 | 16 |
| 2013 | 8 | 8 | 16 |
| 2014 | 8 | 8 | 16 |
| 2015 | 8 | 8 | 16 |
| 2016 | 8 | 8 | 16 |
| 2017 | 8 | 8 | 16 |
| 2018 | 8 | 8 | 16 |
| 2019 | 9 | 8 | 17 |
| 2020 | 10 | 8 | 18 |
| **Total** | **75** | **72** | **147** |

The two pages below are identity members but are permanently quarantined
because the official page reports an update after its release:

| Class | Release | Last update | URL |
|---|---|---|---|
| statement | 2018-01-31 | 2018-02-01 | <https://www.federalreserve.gov/newsevents/pressreleases/monetary20180131a.htm> |
| statement | 2019-10-11 | 2019-10-15 | <https://www.federalreserve.gov/newsevents/pressreleases/monetary20191011a.htm> |

Quarantined pages are fetched only to reconcile identity, bytes, encoding, and
embedded `Last Update`. Their article text is never canonicalized, stored,
logged, emitted, or counted as source support.

Frozen eligible counts are therefore:

```text
statements: 73
minutes:    72
total:     145
```

The exact encoding ledger contains 146 `utf-8` rows and one
`windows-1252` row:

```text
https://www.federalreserve.gov/monetarypolicy/fomcminutes20111213.htm
```

No encoding sniffing, replacement decoding, fallback codec, or repair is
allowed.

## Frozen URL and index grammar

Every URL must be HTTPS, have no user information or explicit port, use host
`www.federalreserve.gov`, and contain no query, fragment, encoded separator,
path traversal, case variant, or redirect identity change.

### Statements

The path must full-match:

```text
/newsevents/pressreleases/monetaryYYYYMMDDa.htm
```

The historical-index anchor must normalize exactly to `Statement`. The path
date, meeting date, and release date must be equal. An `a1` page, PDF,
implementation note, press conference, projection, framework statement, or
generic press release is excluded.

### Minutes

The path must full-match:

```text
/monetarypolicy/fomcminutesYYYYMMDD.htm
```

The path date must equal the meeting date. The associated historical-index
record must contain exactly one explicit `Released` date, and that date must be
strictly later than the meeting date. Selection is by the released date.

### Indexes

`index_url` must full-match:

```text
https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm
```

where `YYYY` is 2011 through 2020 and equals the meeting year. Production must
reconstruct the candidate identities and release metadata from those indexes
and exactly match the committed ledger. Missing, added, duplicated, or changed
metadata is terminal rejection.

## First-party page-update gate

Current HTTP bytes cannot cryptographically prove the exact release-time body.
HTTP `Last-Modified` is also not useful because site-wide replatforming can
change it. FRDCL-D1 therefore uses the page publisher's embedded
`div#lastUpdate` as a predeclared first-party no-later-update assertion.

For every row:

1. exactly one embedded `Last Update` date must be parsed;
2. it must equal the frozen `last_update_date`;
3. an eligible row must also have `last_update_date == release_date`; and
4. a quarantined row must remain ineligible and must not be canonicalized.

This gate is an explicit source assumption, not proof of immutable historical
bytes. A changed, missing, duplicate, or malformed embedded date is terminal
rejection. The assumption may not be weakened after source access.

## Frozen causal clock

Every eligible document becomes available to a later policy at:

```text
00:05 America/New_York on the calendar day after release_date
```

The delayed clock avoids guessing historical intraday publication times.
Meeting date, retrieval time, HTTP headers, and page-navigation timestamps are
never availability evidence.

Canonical source records are ordered by:

```text
(available_at_utc, document_class, official_url)
```

## Deterministic article parser

The verifier uses only:

```python
html.parser.HTMLParser(convert_charrefs=True)
```

It must find exactly one `<div id="article">` in the full document. Attribute
names and tag names use `HTMLParser`'s lowercase representation; duplicate
`id` attributes and a second `div#article` are rejected.

Inside `div#article`, the following container tags are transparent:

```text
div section article main ul ol dl table thead tbody tfoot tr
```

The following tags create retained text blocks:

```text
h1 h2 h3 h4 h5 h6 p li blockquote dt dd caption th td
```

The following inline tags retain their visible text without adding a block
boundary:

```text
a em strong span sup sub b i u abbr cite code q small time
```

`br` ends the current non-empty block. `hr`, `img`, `picture`, and `source`
retain no text. The following subtrees are skipped completely:

```text
script style nav footer header form noscript svg template
```

Any subtree whose exact `id` is `lastUpdate` is also skipped. Any other tag
inside `div#article` is rejected. Non-whitespace visible text outside a
retained block is rejected.

Start/end nesting is checked explicitly inside `div#article`. An unmatched
close, wrong close order, nested table, `th`/`td` outside `tr`, unclosed
non-void tag, or unclosed article is rejected. Table cells are emitted in DOM
row-major order, one retained block per non-empty `th` or `td`. `rowspan` and
`colspan` do not alter that order.

For each retained block:

1. HTML character references are decoded by `HTMLParser`;
2. Unicode is normalized to NFC;
3. `U+200B`, `U+200C`, `U+200D`, `U+2060`, and `U+FEFF` are removed;
4. `U+00A0`, CR, LF, TAB, FF, VT, and every Unicode `Zs` character become one
   ASCII space;
5. consecutive ASCII spaces collapse to one;
6. leading and trailing spaces are stripped; and
7. empty blocks are discarded.

Canonical text is the retained blocks joined by one LF plus exactly one
terminal LF. Case, punctuation, numbers, section order, negation, modal verbs,
votes, and dissent names are preserved.

Before any network access, committed synthetic fixtures must lock:

- exact canonical text and SHA-256 for paragraphs, inline tags, and `br`;
- table row-major order;
- excluded subtree behavior;
- NFC, zero-width, non-breaking-space, and whitespace behavior;
- duplicate article rejection;
- unknown-tag rejection;
- malformed nesting rejection; and
- quarantine behavior that never invokes canonicalization.

## Source-support gates

`SOURCE_SUPPORT_PASS` requires every gate below.

### Identity and coverage

- exact ledger bytes and SHA-256 frozen above;
- exactly 147 candidate rows and 145 eligible rows;
- exactly 75 candidate statements and 72 candidate minutes;
- exactly 73 eligible statements and 72 eligible minutes;
- exact year counts frozen above;
- exact two quarantined URLs and reasons;
- exactly one frozen `windows-1252` row;
- zero duplicate URL or class/URL pair;
- exact reconstruction from the ten authorized indexes;
- zero current-calendar request; and
- zero 2021-or-later body request.

### Transport and metadata

- HTTP 200 and `text/html`;
- strict decode with the row's frozen encoding;
- no raw body larger than 2 MiB;
- exact URL identity after redirect validation;
- exact embedded `Last Update` reconciliation; and
- no canonicalization call for either quarantined row.

### Structural support

- exactly one accepted `div#article` per eligible row;
- one non-empty canonical text per eligible row;
- statement canonical length from 500 through 50,000 characters;
- minutes canonical length from 10,000 through 1,000,000 characters;
- no duplicate canonical text within a document class;
- raw and canonical hashes reproducible by offline replay; and
- no forbidden or skipped text in canonical output.

### Transition support

Without calculating semantic differences, the ordered eligible ledger must
support:

```text
72 statement predecessor pairs
71 minutes predecessor pairs
143 total same-class transitions
```

### No-leak integrity

- market, price, return, funding, portfolio, reward, or performance rows: 0;
- database connections: 0;
- model, tokenizer, adapter, prompt, or checkpoint reads: 0;
- embedding, semantic-label, or inference calls: 0;
- subprocess model execution: 0;
- article text emitted to stdout, stderr, logs, report, or manifest: 0; and
- 2021-or-later item bodies read: 0.

Any failed gate is `TERMINAL_REJECT`.

## Transport, storage, and one-shot execution

The production audit runs in a fresh, isolated, standard-library-only Python
process with:

- no cookie, credential, API key, browser session, proxy, or prompt;
- HTTPS and the fixed host only;
- fixed `User-Agent`, `Accept: text/html`, and `Accept-Encoding: identity`;
- a 30-second timeout;
- same-host redirect validation;
- a 2 MiB per-body and 256 MiB aggregate raw-body cap;
- at most two identical retries only after timeout, HTTP 429, or HTTP
  500/502/503/504, delayed by one then two seconds; and
- no retry after any identity, parse, support, or other HTTP failure.

Before the sentinel:

- this boundary, ledger, verifier, and synthetic tests are committed and equal
  to `HEAD`;
- the worktree is clean;
- the ledger hash is exact;
- filesystem `shutil.disk_usage(...).used` is below 300 GiB;
- at least 8 GiB is free; and
- no prior sentinel, manifest, raw directory, or aggregate report exists.

The attempt is consumed only when the sentinel is atomically created with:

```python
os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
```

The sentinel is created after all local preflight checks and before the first
network read. A pre-sentinel local failure is not an attempt. Once created, the
attempt is never resumed, repaired, or rerun, even after process interruption.

Every network intent and response is appended to a hash-chained NDJSON
manifest using append mode, flush, and `fsync`. Manifest records contain
metadata and hashes but no article text. Ignored local storage may retain exact
raw HTML and eligible canonical files for offline hash replay, but neither may
be opened by a human, agent, or model before the mechanism decision is
committed. Quarantined canonical files must not exist.

Only a deterministic aggregate JSON report may be committed. It contains
counts, gate booleans, hashes, code/commit bindings, and the terminal verdict;
it contains no document text, per-document semantic value, or market outcome.

## Mandatory mechanism freeze after source pass

If and only if the aggregate verdict is `SOURCE_SUPPORT_PASS`, the next work
unit is a single-small-LLM mechanism decision. Before anyone reads corpus text,
that decision must freeze:

- exact base model, revision, tokenizer, license, context length, and
  quantization;
- exact prior/current same-class input construction;
- chunking, truncation, and aggregation;
- prompt and output grammar;
- a compact symbolic ontology and explicit abstention;
- adapter or fine-tuning method;
- RL state, action, reward, and abstention treatment;
- train, test, and untouched evaluation windows;
- costs, funding, full-calendar CAGR, and strict MDD;
- non-LLM, shuffled-text, and direction-inversion controls;
- novelty comparators; and
- a fixed stopping rule.

The mechanism may use the language model for deductive comparison of
prior-versus-current communication and contradiction detection. It may not use
the model to search source classes, labels, directions, prompts, horizons, or
holds after BTC outcomes are opened.

## Stop condition

Stop and permanently retire FRDCL-D1 on:

- ledger or index drift;
- a missing, extra, or redirected identity;
- page-update mismatch;
- encoding or parser failure;
- structural-support failure;
- transport/storage failure after sentinel creation;
- no-leak violation; or
- aggregate report publication failure.

On pass, commit the mechanism decision before corpus inspection. On rejection,
publish the aggregate rejection and return to a different source axis.
