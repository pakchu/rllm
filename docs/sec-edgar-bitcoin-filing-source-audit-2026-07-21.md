# SEC EDGAR Bitcoin filing source audit — passed

## Decision

Authorize preregistration of one text-native candidate from the frozen SEC
EDGAR source clock. Do **not** run a semantic model or open BTC outcomes yet.
The source is immutable, causally timestamped, sufficiently broad, and ends
before 2024.

- source artifact:
  `data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz`;
- source SHA-256:
  `c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce`;
- source canonical-row hash:
  `98793185f1e411d8c59736fb54c5ed529d539e81ccddf2c823f24127ecfcef0b`;
- audit artifact:
  `results/sec_edgar_bitcoin_8k_6k_source_audit_2026-07-21.json`;
- audit SHA-256:
  `c1e11d1f5089378ac787fdb2a80474f0feec33d5fb2296fb0c3014d6f1fafec1`;
- manifest hash:
  `b4234f71b559a6b98e4056491f3b726191e9a89c2c0bec1e549249d93840f575`;
- auditor SHA-256:
  `79e741cf3711f9ab1e041806de8cbf8daa3e909a321e5bff1e439d39fd6fe7b5`.

## Frozen source

The exact query is `bitcoin`, forms `8-K,6-K`, filing dates 2018-01-01 through
2023-12-31, with EFTS `sort=asc`. The response itself confirmed the deterministic
sort `(file_date ASC, _id ASC)` on every page. The committed artifact contains
only stable source metadata: accession, document name, CIK set, filing date,
acceptance timestamp, form, sequence, and description. It omits current ticker
and company-name fields.

| Source item | Count |
|---|---:|
| exact EFTS document hits | 3,496 |
| unique accessions | 2,543 |
| non-amendment 8-K/6-K accessions eligible for later semantics | **2,493** |
| distinct CIKs among eligible filings | 308 |
| distinct eligible acceptance days | 992 |
| amendments retained for audit but forbidden to emit | 50 |

The official [EDGAR full-text search](https://www.sec.gov/edgar/search/index.html)
covers filings since 2001. The [EDGAR data API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
documents submissions history by filer and real-time dissemination metadata.
SEC also states that an accepted dissemination cannot be rescinded in its
[filing submission guidance](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/attach-submit-filing-through-edgar-filing-website),
and that acceptance time is not adjustable in its
[filing-date adjustment guidance](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/request-filing-date-adjustment).

## Causal clock verification

All 2,543 accessions resolved to `acceptanceDateTime` through official
`data.sec.gov/submissions` records; 24 supplemental history files were required.
Six archive documents, one per year, were fetched twice and reproduced the same
SHA-256. Every sample contained `bitcoin`, and every archive header acceptance
field matched the submissions timestamp.

The raw archive header representation is not uniform across the sample: four
rows matched after interpreting the compact value as U.S. Eastern time, while
two matched it directly as UTC. Therefore the candidate must use the official
submissions API's UTC `acceptanceDateTime`, not infer a timezone from the raw
14-digit header. Likewise, 123 accessions had a filing date different from the
UTC acceptance calendar date; `file_date` is never the execution clock.

The collector respected the SEC's documented
[10 requests/second fair-access ceiling](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
The committed audit made 394 source-only calls.

## Outcome-blind support gates

Amendments are excluded from support and may not emit later signals.

| Period | Eligible accessions | Event days | CIKs | Top-1 share | Top-5 share | HHI |
|---|---:|---:|---:|---:|---:|---:|
| 2018–2020 source train | 407 | 301 | 116 | 8.35% | 35.38% | 0.0347 |
| 2021–2022 source test | 1,399 | 465 | 200 | 5.65% | 19.87% | 0.0187 |
| 2023 selection | 687 | 226 | 100 | 8.15% | 28.82% | 0.0312 |

Every preregistered support and concentration gate passed. These counts only
show that semantic extraction has enough raw material; they do not show an
alpha or profitability.

## Disclosed repair before freeze

An initial source-only run used EFTS's default relevance order. Page boundaries
were not deterministic, so the run failed closed before candidate or outcomes.
The frozen query adds `sort=asc`, whose returned query contract explicitly sorts
by filing date and document ID. A second source-only issue was the mixed raw
header time representation described above; the final audit validates both
representations against the official UTC submissions timestamp instead of
guessing one timezone. Neither repair inspected market data.

```text
BTC market rows    = 0
funding rows       = 0
future-return rows = 0
return/PnL fields  = 0
candidate signals = 0
```

The next authorized step is a singleton, outcome-blind semantic
preregistration. Model revision, quantization, prompt, redaction, quote
grounding, state transition, support gates, controls, and 2024+ embargo must all
be frozen before document bodies are classified.
