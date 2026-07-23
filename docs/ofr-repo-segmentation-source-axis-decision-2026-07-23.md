# OFR repo-segmentation source-axis decision — 2026-07-23

## Decision

Open one new **source-only** research axis: the Office of Financial Research
(OFR) U.S. Repo Markets preliminary release. The source exposes daily rates and
volumes across centrally cleared DVP, centrally cleared GCF, and tri-party repo,
with tenor and collateral subdivisions.

This decision authorizes only a hash-reproducible preliminary-vintage source
build, metadata audit, and 2019–2023 normalized panel. It does **not** authorize
a trading predicate, event count, side, hold, BTC market data, funding, forward
return, PnL, CAGR, MDD, parameter search, or model training. A separate committed
mechanism decision must precede the first candidate-derived value.

## Why this source axis

The repository has already evaluated EIA petroleum stock breadth (`EPSB-1`),
including 2020–2022 outcomes, and later rejected the related `RPDS-576` source
clock on novelty. Another petroleum transform would not be a new source axis.

New York Fed Primary Dealer Statistics are genuinely untried in this
repository, but the official page states that historical data may reflect
revisions since prior publication. The current all-history export therefore
cannot by itself reproduce what a live system knew each week.

The OFR repo release is selected because it provides:

- a previously unused daily source family;
- explicit `Preliminary` and `Final` vintages under separate mnemonics;
- segment, tenor, and collateral detail beyond the already tested SOFR level;
- an official unauthenticated JSON API suitable for deterministic live replay;
- data collected daily since 2019 and preliminary publication since
  2020-09-09; and
- a declared publication schedule and disclosure-edit semantics.

This is not globally pristine macro-liquidity research. The branch has already
opened outcomes for SOFR, overnight RRP, H.4.1/H.8, and related Federal Reserve
liquidity candidates. Any later OFR candidate must therefore pass strict
clock/exposure novelty against those frozen families and the live portfolio.

## Official economic object

OFR describes repo as secured short-term funding and publishes activity in
three market segments. The release contains rates and volumes subdivided by
tenor or underlying collateral. OFR states that the series complement, but are
not directly comparable with, New York Fed reference rates because transaction
sets differ and OFR rates are volume-weighted means rather than medians.

Official references:

- U.S. Repo Markets data release:
  <https://www.financialresearch.gov/short-term-funding-monitor/datasets/repo/>;
- preliminary/final release and methodology notes:
  <https://www.financialresearch.gov/short-term-funding-monitor/documentation/>;
- API overview:
  <https://www.financialresearch.gov/short-term-funding-monitor/api/>;
- full-dataset API contract:
  <https://www.financialresearch.gov/short-term-funding-monitor/api-specs/api-full-dataset/>;
- single-series data contract:
  <https://www.financialresearch.gov/short-term-funding-monitor/api-specs/api-data-single/>;
- official data-collection description:
  <https://www.financialresearch.gov/data/collections/>.

OFR states:

- preliminary data are updated most weekdays at 3 p.m. Eastern;
- centrally cleared observations are generally released with a one-business-
  day lag;
- tri-party observations are generally released with a two-business-day lag;
- government holidays can delay publication;
- missing values can represent no trading or confidentiality disclosure edits;
- preliminary values may differ from a later final release; and
- final series are published quarterly and are not revised afterward.

The source says nothing about BTC direction. Rate, volume, tenor, collateral,
fragmentation, scarcity, and balance-sheet language are research hypotheses,
not OFR claims.

## Frozen retrieval contract

The source builder will request only the official production API:

```text
GET https://data.financialresearch.gov/v1/series/dataset
    ?dataset=repo
    &vintage=p
    &start_date=2019-01-01
    &end_date=2023-12-31
```

No `periodicity`, `how`, `remove_nulls`, spread calculation, final-vintage
substitution, or post-2023 date is allowed. The builder must also retrieve the
official mnemonic/metadata context needed to prove that every retained series:

1. belongs to the `repo` data set;
2. is explicitly marked `Preliminary`;
3. has daily observation frequency;
4. is a rate or volume series for DVP, GCF, or tri-party repo; and
5. preserves its disclosure-edit subseries where the API exposes one.

The mnemonic response is also the frozen final-definition audit surface. Every
preliminary mnemonic must have exactly one corresponding final mnemonic, and
their series names must be identical after removing only the terminal
`(Preliminary)` / `(Final)` vintage label. No final time-series values may be
requested or read for this check.

The exact request URL, retrieval UTC time, HTTP status, content type, byte
count, response SHA-256, redirects, and API metadata must be written to a fetch
ledger. Raw API bytes are retained as deterministic gzip. Offline replay must
revalidate URL, bytes, hashes, schema, and normalized outputs.

The builder must reject redirects outside `data.financialresearch.gov`, HTML,
non-JSON responses, unknown top-level fields, duplicate mnemonics, duplicate
observation dates within a mnemonic, nonfinite values, a final/as-of series in
the preliminary payload, or an aggregation/normalized date after 2023-12-31.

The OFR full-dataset response can return a mnemonic's complete
`disclosure_edits` subseries even when `start_date` and `end_date` restrict its
aggregation rows. Those raw markers remain hash-bound in the response but are
not normalized outside 2019–2023. The audit must report total, retained,
pre-2019, and post-2023 marker counts; every retained marker must correspond to
an aggregation row. An out-of-window marker may never create, remove, fill, or
otherwise alter an in-window observation.

## Conservative causal availability

The API exposes observation dates and series update metadata, not a historical
publication timestamp for every point. A candidate may not backdate a row to
its observation day. Preliminary publication began on 2020-09-09, so older
historical rows also may not be treated as if the feed existed in 2019. The
normalized source uses one deliberately conservative clock for every segment:

```text
preliminary_feed_floor_utc = 2020-09-10 00:00 UTC
available_at_utc = max(
    observation_date 00:00 UTC + 8 elapsed calendar days,
    preliminary_feed_floor_utc,
)
```

This is later than the documented one- or two-business-day lag, remains later
than the normal 3 p.m. Eastern publication through DST, and avoids reconstructing
a holiday calendar from hindsight. The feed floor makes pre-publication history
audit-only until the first preliminary release could have been known. A missing
row is missing; it is never forward-filled, interpolated, converted to zero, or
borrowed from the final vintage.

The preliminary payload is retained as its own source vintage. The source
audit must compare metadata—not candidate features—between preliminary and
final definitions and explicitly reject the axis if the API does not preserve
a separately identifiable preliminary historical series.

## Allowed source audit

Before any mechanism is selected, the source audit may report only:

- series names, mnemonics, descriptions, units, segments, tenor/collateral
  subsets, start/end dates, and source metadata;
- row counts, date coverage, duplicates, nonfinite values, missing values, and
  disclosure-edit counts, including excluded out-of-window source markers;
- preliminary/final series-definition correspondence without value spreads;
- deterministic artifact and manifest hashes; and
- whether the conservative availability clock is reproducible.

It may not compute cross-segment spreads, ratios, ranks, z-scores, changes,
states, candidate incidence, sides, holds, correlations, or market outcomes.

## Candidate ideas deliberately not selected

Possible later mechanisms include segment-rate fragmentation, collateral
scarcity breadth, overnight-to-term volume migration, or disagreement between
cleared and tri-party funding. These are inventory only. This source-axis
decision selects none of them and authorizes no incidence calculation.

Exactly one later identity must freeze its economic mechanism, source fields,
strict-prior transform, direction, latency, hold, non-overlap, source-support
floors, controls, comparator cohort, and no-repair rule before values are
combined.

## Stop conditions

Retire the source axis before candidate work if any of the following occurs:

- preliminary-vintage history is not separately identifiable from final data;
- the API cannot replay complete 2019–2023 responses;
- source metadata cannot distinguish market segment and rate/volume semantics;
- disclosure edits or missingness prevent a stable multi-segment daily panel;
- the conservative availability clock cannot be reproduced;
- a required official endpoint becomes credentialed or non-replayable; or
- the builder accesses market prices, funding, returns, portfolio outcomes, or
  model values.

The next work unit is a tested source builder. It must be committed before the
first full OFR repo payload is retrieved.
