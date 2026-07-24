# DOL weekly-claims release-vintage source-axis decision — 2026-07-24

## Decision

Select:

```text
DOL-WCRV-D1 — DOL weekly-claims release-vintage and state-breadth ledger
```

as the next independent, outcome-blind BTC RLLM source axis.

DOL-WCRV-D1 asks only whether the exact unemployment-insurance information
published by the U.S. Department of Labor at each historical weekly release
can produce a causal, replayable, sufficiently dense national-and-state
release-vintage ledger. It does not assume that claims are bullish or bearish
for BTC.

This decision authorizes one source-only builder and one source-support audit.
It does **not** authorize:

- a trade direction, action, holding period, threshold, position, or weight;
- a prompt, tokenizer, adapter, checkpoint, reward, RL policy, or portfolio
  join;
- a BTC bar, price, return, funding, premium, open interest, liquidation, PnL,
  CAGR, MDD, existing-alpha incidence, or market clock; or
- a claim that unemployment-insurance releases predict BTC.

A separate committed mechanism decision must precede the first market join,
model input, semantic state label, or candidate comparison.

## Why this axis is selected

The immediately preceding GitHub Advisory first-add source was permanently
rejected at its committed source boundary:

- `docs/github-advisory-first-add-source-axis-decision-2026-07-24.md`
- `docs/github-advisory-first-add-source-rejection-2026-07-24.md`

That boundary explicitly retained DOL weekly claims as a separate reserve
source. DOL-WCRV-D1 does not repair, retry, relabel, or reuse the failed Git
transport.

Repository-wide searches found no previous implementation, source artifact,
or market evaluation based on DOL weekly claims, ETA 538, ETA 539, initial
claims, or continuing claims. The observable is therefore source-new to this
repository.

The source is worth one bounded audit because it has:

- a fixed official weekly release clock;
- an official DOL newsroom archive that preserves each release's first
  paragraph, including the advance national level and contemporaneously
  disclosed revision;
- date-coded official HTML state tables that preserve cross-state breadth,
  dispersion, concentration, and disagreement without reconstructing them
  from a later database snapshot;
- roughly one national event per week over a long history; and
- explicit revisions, allowing a later mechanism to distinguish newly
  reported information from already-known values.

The source is lower-text-entropy than the rejected advisory corpus. Its value
must come from causal vintage structure and a later bounded deductive state
mechanism, not from free-form sentiment or an LLM performing arithmetic.

## Metadata-only design probe

Before this boundary was frozen, an outcome-blind metadata probe opened only
the official archive calendars and counted links and suffixes. It opened no
state table, release body beyond isolated format examples, market data, model,
candidate, or performance artifact.

That probe found:

- national press artifacts change from `.asp` to `.pdf` over 2012–2014 and
  contain calendar-link errors, so they are not the canonical machine-readable
  ledger;
- the state-table archive is complete in most years but has only 23 dated
  tables in 2019 and 44 in 2025; and
- the state calendars contain two malformed extra links alongside the
  canonical dated tables:
  - `/unemploy/page8/2012/022312.html`, whose encoded date is a Thursday; and
  - `/unemploy/page8/2023/0223.html`, whose filename is not an `MMDDYY` date.

Those source-format facts are frozen here rather than discovered after market
access. The production audit may not repair them or reinterpret them using
outcomes.

A separate byte-prefix transport preflight used Python's standard-library HTTPS
client with no cookie or browser session. On 2026-07-24, the exact audit
identity:

```text
User-Agent: rllm-dol-wcrv-source-audit/1.0 (https://github.com/pakchu/rllm)
Accept: text/html
Accept-Encoding: identity
```

received HTTP 200 and `text/html; charset=UTF-8` from the authorized newsroom
index. A browser-emulation header profile was rejected with HTTP 403 and is
therefore forbidden. The production verifier must use the exact non-browser
identity above; it may not add browser headers or switch transport after a
failure.

## Official source contract

### Release schedule

Official Office of Unemployment Insurance archive landing page:

<https://oui.doleta.gov/unemploy/claims_arch.asp>

The page states that the release is normally published Thursday at 8:30 a.m.
Eastern time and that federal holidays may shift the release day. The audit
retrieves and hashes this landing page once to bind the schedule statement.

The page's link to the mutable current PDF is not a historical source.

### National release-vintage ledger

Official Employment and Training Administration newsroom archive:

<https://www.dol.gov/newsroom/releases/eta>

The audit may retrieve only:

```text
https://www.dol.gov/newsroom/releases/eta
https://www.dol.gov/newsroom/releases/eta?page=<N>
```

with canonical decimal page numbers starting at zero and following the exact
`rel="next"` chain exposed by each page. Parameters such as `published_at`,
`year`, `date`, search terms, language variants, Drupal APIs, JSON endpoints,
and search-engine caches are not authorized.

The authorized national source record is one ETA newsroom teaser whose title,
after HTML entity decoding and ASCII-whitespace collapse, equals:

```text
Unemployment Insurance Weekly Claims Report
ETA Press Release: Unemployment Insurance Weekly Claims Report
```

The record must expose:

- one numeric `data-history-node-id`;
- one canonical relative `about` path;
- one displayed release date;
- one authorized exact title;
- one `field-press-body` teaser; and
- one source-page identity and raw page hash.

The `about` path must match:

```text
/newsroom/releases/eta/eta<YYYYMMDD>
/newsroom/releases/eta/eta<YYYYMMDD>-<SUFFIX>
```

where the path date equals the displayed release date and `<SUFFIX>` is a
canonical non-negative decimal integer.

Newsroom detail paths are not fetched. The index teaser is the canonical
machine-readable national vintage. No detail-page format, redirect, or current
PDF may substitute for a missing teaser.

### Archived state-table inventory

The audit may submit the archive form only to:

```text
POST https://oui.doleta.gov/unemploy/archive.asp
Content-Type: application/x-www-form-urlencoded

report=page8&year=<YYYY>&submit=Submit
```

where `<YYYY>` is one of the integers 2012 through 2025.

Only links matching this exact ASCII form are selected:

```text
/unemploy/page8/<YYYY>/<MMDDYY>.html
```

The path date must:

- parse as a valid Gregorian date;
- have the requested calendar year;
- be a Saturday; and
- equal the embedded initial-claims week-ending date in the fetched table.

The known noncanonical calendar links:

```text
/unemploy/page8/2012/022312.html
/unemploy/page8/2023/0223.html
```

must each appear exactly once in their stated year, must not match the selected
Saturday-path contract, and must not be fetched. Any other malformed `page8`
link rejects the source.

### Archived state-table schema

The date-coded `page8` HTML is the only authorized state-level source. It must
contain two contemporaneous observation dates and one row for each of:

- the 50 states;
- District of Columbia;
- Puerto Rico; and
- Virgin Islands.

For each jurisdiction, the authorized numeric columns are:

- regular-state initial claims;
- initial-claims change from last week;
- initial-claims change from one year ago;
- UCFE initial claims;
- UCX initial claims;
- regular-state insured unemployment;
- not-seasonally-adjusted insured-unemployment rate;
- insured-unemployment change from last week;
- insured-unemployment change from one year ago;
- UCFE insured unemployment;
- UCX insured unemployment; and
- all-program insured unemployment excluding railroad retirement.

The exact totals row, table headings, footnote markers, URL, HTTP metadata, raw
byte length, raw SHA-256, and canonical structural hash are retained for
replay. No later state table, current query, or raw CSV may repair an archived
table.

### Excluded sources

The following are excluded:

- the mutable current PDF at `https://www.dol.gov/ui/data.pdf`;
- the current national/state query at
  `https://oui.doleta.gov/unemploy/claims.asp`;
- the current report generator, spreadsheets, XML, CSV, and raw database
  extracts;
- the date-coded press `.asp` and `.pdf` artifacts, which are not needed by
  this machine-readable source contract;
- FRED, ALFRED, BLS, BEA, Census, private calendars, vendor forecasts,
  consensus estimates, news articles, search results, mirrors, or caches;
- revised historical series reconstructed after the release;
- state data edited after the contemporaneous archived table;
- HTTP/PDF metadata used as a release-time substitute; and
- any market, model, portfolio, or existing-alpha data.

The ETA raw-data page explicitly says that files are refreshed from the
current National Office database and that states may edit prior data at any
time. Those files are operational snapshots, not historical vintages:

<https://oui.doleta.gov/unemploy/DataDownloads.asp>

## Frozen source window and causal clock

The national source-support window contains displayed release dates in:

```text
[2012-01-01, 2026-01-01)
```

The state-table source-support window contains initial-claims week-ending dates
in the same half-open interval.

The newsroom crawler must retrieve raw pages containing newer entries in order
to reach the frozen window. It may hash the raw response as a byte string. For
any teaser dated on or after 2026-01-01, a byte-level record scanner may decode
only:

- node ID;
- `about` path;
- displayed date; and
- exact title.

It must identify and skip the teaser-body byte span without HTML entity
decoding, text extraction, grammar parsing, numeric parsing, printing, or
publication. Thus the transport sees the enclosing page bytes, while the
source parser and output remain blind to the 2026 body.

For every selected national release date `d`, historical source availability
is:

```text
09:00:00 America/New_York on d
```

This is a fixed conservative 30-minute lag after DOL's stated 8:30 a.m.
Eastern publication time. It applies to normal Thursdays and holiday-shifted
release dates alike. HTTP `Date`, `Last-Modified`, node ID, week-ending date,
and observation date may not move availability earlier.

A state table becomes available at the 09:00 Eastern clock of the unique
national teaser whose parsed initial-claims week-ending date equals the
table's embedded initial-claims week-ending date. The table's encoded Saturday
is an observation date, not a publication clock.

Live availability is the later of:

- the same historical 09:00 Eastern floor;
- durable receipt of the exact schedule page, newsroom page, state inventory,
  and matched state table;
- successful structural parsing and hashing; and
- append-only manifest commit.

No live event may be backdated after receipt.

## Frozen national source schema

For every selected national teaser, the source builder may retain:

- numeric node ID;
- canonical newsroom path;
- displayed release date;
- normalized exact title;
- exact teaser UTF-8 bytes and SHA-256;
- source page number and raw page SHA-256;
- parsed initial-claims week-ending date;
- current seasonally adjusted advance initial-claims level;
- signed change from the previous reported level;
- whether the prior level was described as revised or unrevised;
- prior level before revision, when disclosed;
- prior revised level, when disclosed;
- current four-week moving average;
- signed change in the four-week moving average;
- prior four-week average before revision, when disclosed; and
- prior revised four-week average, when disclosed.

The parser may retain exact integer lexemes and grammar/field-presence flags.
It may not infer missing values, load a later release to rewrite a prior row,
or replace a contemporaneous value with the current historical series.

At source-support stage, pre-2026 teaser text may be opened only for strict
grammar, encoding, and numeric validation. It may not be prompted, embedded,
classified, summarized, scored, printed, or searched for a mechanism.

## Frozen state source schema

For every selected state-table artifact, the source builder may retain:

- encoded and embedded initial-claims week-ending dates;
- embedded insured-unemployment week-ending date;
- exact jurisdiction names and canonical ordering;
- the twelve authorized numeric columns for each jurisdiction;
- the exact totals row;
- table grammar and footnote variant;
- source URL and HTTP metadata;
- raw byte length, raw SHA-256, and canonical structural hash; and
- the uniquely matched national release identity and availability clock.

The state table becomes part of a national event only after exact week-ending
matching. An unmatched table remains a source-integrity failure, not a
standalone event.

## Aggregate publication boundary

The aggregate report may publish:

- ordered schedule, inventory, newsroom-page, teaser, state-table, manifest,
  parser, and output hashes;
- counts by year, quarter, release weekday, grammar variant, revision state,
  field-presence pattern, table variant, and jurisdiction coverage;
- aggregate gap, byte-size, and matching-coverage distributions;
- aggregate arithmetic/reconciliation counts without values;
- null, duplicate, malformed, mismatch, and conflict counts; and
- pass/reject booleans.

It may not publish:

- row-level release dates, week-ending dates, claims values, or state values;
- teaser prose;
- a source-derived state, direction, surprise, threshold, or rank;
- any market, model, candidate, or economic result; or
- any 2026 teaser body or value.

## Exact identity and structure rules

Every selected newsroom page must:

- be strict UTF-8 HTML without NUL or replacement code points;
- be no larger than 2 MiB;
- expose at most 40 teaser records;
- contain unique node IDs and `about` paths;
- expose one unambiguous canonical next-page link or terminate the chain;
- preserve reverse-chronological displayed-date ordering; and
- reproduce its exact raw and canonical structural hashes.

Every selected national release must:

- match exactly one authorized teaser;
- have a unique node ID, path, teaser hash, and release date;
- have a week-ending date strictly before its release date;
- contain finite base-10 integers without exponent or sign ambiguity;
- have non-negative levels and moving averages;
- arithmetically reconcile every disclosed prior-value revision;
- arithmetically reconcile every signed current change against the exact prior
  comparison level stated in the same teaser; and
- reproduce the same source row on replay from sealed local bytes.

Every state inventory must:

- be strict UTF-8 HTML without NUL or replacement code points;
- contain exactly one valid calendar for the requested year;
- contain no duplicate selected path or selected week-ending date;
- reproduce the frozen count and malformed-link exception; and
- reproduce its exact raw and canonical structural hashes.

Every selected state table must:

- be strict UTF-8 HTML without NUL or replacement code points;
- be no larger than 2 MiB;
- have a path date equal to its embedded initial-claims week-ending date;
- have an insured-unemployment week-ending date exactly seven calendar days
  earlier;
- contain exactly the frozen 53-jurisdiction set, once each;
- contain exactly twelve finite base-10 numeric cells per jurisdiction;
- contain one exact totals row;
- reconcile every totals field defined by the source as a jurisdiction sum;
- match exactly one national teaser by initial-claims week-ending date; and
- reproduce its exact raw and canonical structural hashes on replay.

Conflicting dates, duplicate nodes, duplicate selected paths, off-domain
redirects, malformed tables, missing jurisdictions, missing required teaser
fields, impossible arithmetic, or hash disagreement reject the source.

## Source-only support gates

All gates are frozen before the production source is systematically opened.

### National release coverage

- at least 700 selected national releases exist across 2012–2025;
- each year from 2012 through 2024 contains 50–54 releases;
- 2025 contains exactly 46 releases;
- every 2012–2024 quarter contains at least 12 releases;
- 2025 Q1–Q3 each contain at least 12 releases;
- 2025 Q4 contains at least seven releases;
- consecutive release gaps are 5–10 calendar days, except the exact
  publication interruption from 2025-09-25 through 2025-11-20;
- no selected release date is after 2025-12-31; and
- every release has exact 09:00 Eastern historical availability.

The 2025 count and interruption are source-schedule facts, not missing rows to
be backfilled from a revised current series.

### State inventory coverage

The exact selected table counts must be:

```text
2012  52    2013  52    2014  52    2015  52
2016  53    2017  52    2018  52    2019  23
2020  52    2021  52    2022  52    2023  52
2024  52    2025  44
```

The 2012 and 2023 archives must each additionally contain exactly their one
frozen malformed link. The links are counted as inventory metadata but are not
selected tables.

### Cross-surface identity

- 100% of selected state tables match exactly one national teaser by
  initial-claims week-ending date;
- every national teaser whose week-ending date is represented by a selected
  table maps to exactly one table;
- at least 90% of all national teasers in the full window have a matched state
  table;
- each year except 2019 and 2025 has at least 98% national-to-state coverage;
- 2019 has at least 40% coverage;
- 2025 has at least 90% coverage;
- zero node IDs, newsroom paths, selected table paths, week-ending dates,
  jurisdiction sets, teaser hashes, or table hashes conflict;
- the newsroom crawler reaches every selected year and terminates only after
  crossing below 2012; and
- exact replay reproduces every ordered hash and aggregate.

Missing 2019 and 2025 state tables remain explicit missing source fields. They
may not be imputed, forward-filled, reconstructed from current data, or used
to delete the corresponding national event.

### State-table structure

- every selected table contains exactly 53 jurisdiction rows and one totals
  row;
- every jurisdiction appears in every selected table;
- all twelve authorized numeric columns parse in every jurisdiction row;
- every rate is finite and in `[0.0, 100.0]`;
- every count/level field that is not a change is non-negative;
- every source-defined totals arithmetic relation reconciles exactly; and
- no state value is imputed or loaded from a current series.

### National numeric coverage

Across the full window and within every year:

- 100% parse a week-ending date, current advance initial-claims level, signed
  change, and current four-week moving average;
- at least 90% disclose whether the prior level was revised or unrevised;
- at least 75% disclose both prior-value revision endpoints;
- at least 90% parse the four-week-average signed change;
- every disclosed arithmetic relation reconciles exactly; and
- no value is imputed from another release or a mutable current series.

The audit does not relax a failed threshold, exclude an inconvenient year,
quarantine a selected malformed row, or change grammar after production
access.

## Transport, disk, and execution boundary

The source audit must run in a fresh isolated process with:

- no cookie, API token, database credential, browser session, proxy, or
  interactive prompt;
- a sealed environment and fixed standard-library-only Python runtime;
- the exact preflighted non-browser User-Agent, `Accept: text/html`,
  `Accept-Encoding: identity`, timeout, response-size cap, and redirect
  allowlist;
- HTTPS only and DNS names restricted to `oui.doleta.gov` and `www.dol.gov`;
- no persistent HTTP cache, search engine, mirror, or alternate endpoint; and
- no import from market, model, training, portfolio, or live-trading modules.

One production attempt is authorized. Within that attempt, an identical
request may be repeated at most twice after a socket timeout, HTTP 429, or HTTP
500/502/503/504 response. A retry must use the identical method, URL, body,
headers, and timeout, discard partial bytes, and wait fixed intervals of one
then two seconds. No other response or parse failure is retryable.

Before network access and before state-table materialization:

- this boundary, builder, and tests must be committed and equal to `HEAD`;
- the worktree must be clean;
- filesystem use must be below 300 GiB;
- at least 8 GiB must remain free;
- total raw source bytes must remain below 256 MiB; and
- an exclusive one-shot sentinel and append-only hash-chained manifest must be
  durable.

The source attempt is consumed when its sentinel is created. It is not
restarted, repaired, widened, narrowed, or redirected after a source or
structure failure.

Ignored local storage may retain exact official HTML bytes and the private
manifest. Only the aggregate report and immutable bindings may be committed.

## No-leak and no-outcome boundary

Before a passing source report is committed, the process may not open:

- BTC, ETH, equity, FX, commodity, rates, volatility, or other market data;
- funding, premium, open interest, liquidation, order flow, depth, returns,
  targets, rewards, PnL, CAGR, MDD, checkpoints, or portfolio data;
- existing alpha incidence, direction, weight, or performance;
- private forecasts, consensus estimates, surprise series, or revised current
  claims history;
- any LLM, embedding model, tokenizer, adapter, prompt, or RL policy; or
- any 2026 teaser body, state table, or claims value.

The audit must fail if market/model modules, paths, environment variables, or
database connections are imported or opened.

Synthetic fixtures may exercise all parser, clock, pagination, retry, hash,
replay, arithmetic, missing-table, and no-leak rules before production. They
may not contain real market outcomes or production release rows.

## Later RLLM boundary

A source pass authorizes exactly one later, separately committed mechanism
decision.

That decision may define:

- deterministic release-vintage arithmetic and trailing-state features;
- deterministic cross-state breadth, dispersion, concentration, persistence,
  and disagreement features from available archived state tables;
- explicit missing-state-table masks that never delete the national event;
- an outcome-blind causal sequence representation;
- one bounded Gemma 4 state reasoner over only information available by the
  event clock;
- synthetic arithmetic, temporal-order, revision, counterfactual,
  prompt-injection, and label-balance controls;
- an RL trader that receives the reasoner's bounded state plus causal
  price-action, regime, and current-position information; and
- train/test/eval boundaries with all of 2026 untouched until final
  evaluation.

Deterministic code must compute clocks, integer arithmetic, rolling windows,
positions, costs, risk, rewards, and performance. The LLM may reason about
relations among causal national and cross-state states; it may not calculate
or silently repair numeric facts.

No model may:

- create, delete, retime, impute, or rewrite source events or state tables;
- see a later revision while representing an earlier release;
- receive eval or 2026 rewards during prompt, adapter, checkpoint, or
  hyperparameter selection;
- receive future bars or post-entry facts; or
- restore the discarded analyzer/trader two-model split.

## Pass, reject, and next action

`SOURCE_SUPPORT_PASS` requires every integrity, inventory, cross-surface,
numeric-coverage, transport, disk, replay, and no-leak gate.

Any production source failure is `TERMINAL_REJECT` and permanently retires
DOL-WCRV-D1. There is no mutable-series fallback, parser repair, threshold
relaxation, PDF fallback, or market-assisted rescue.

On pass:

1. commit the aggregate source report;
2. commit one mechanism decision before market or model access;
3. freeze the causal representation, evaluator, and splits;
4. open train outcomes only;
5. select once on the committed test boundary; and
6. preserve 2026 as untouched final evaluation.

On reject, return to source selection without opening a DOL mechanism.
