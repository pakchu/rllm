# Bybit public-trade source probe v1 invalidation and v2 correction — 2026-07-23

## Decision

The v1 `SOURCE_FEASIBILITY_PASS` is **invalidated and may not authorize a
mechanism, source build, support clock, or outcome evaluation**.  An independent
code review found that the committed probe could conceal full-header drift and
could decompress more source records than the frozen header-plus-one-record
boundary.

This is a probe-implementation failure, not an economic or source rejection.
No candidate incidence, Binance comparator, post-entry market value, funding,
return, PnL, CAGR, or strict MDD was opened.  The source axis may receive one
conservative v2 correction because the original source decision already
required explicit schema mapping and already classified block/RPI fields as
optional controls that are forbidden from the primary unless present through
the full historical window.

## Invalid v1 evidence retained

The invalid artifact is retained only for audit:

- path:
  `results/bybit_public_trade_sequence_source_feasibility_v1_invalid_2026-07-23.json`
- file SHA-256:
  `3e2872467acebfd07f91ff8b9ff0079eb9dc518f6f37ad79f83a6a47cf413536`
- stored v1 manifest hash:
  `fdc57f5ecfb34f516726fbd1d323faf1d397cd492c81ae23ef006354b3554227`
- v1 script SHA-256:
  `be9b214a24d3ace6d1a5eb7fc837bda970f44d2e7e21afbe9252332349f6aceb`

The exact archive prefixes opened by v1 were:

| day | compressed prefix bytes | prefix SHA-256 |
|---|---:|---|
| 2020-03-25 | 16,384 | `dfde89d406b7d179c03fef8116d2668ab52999199e39e07cdd55175a6e749821` |
| 2023-01-01 | 16,384 | `e91c088cdf75a06fcaecb42be75d59178706798029cdec098ef67e49cc2e7455` |
| 2026-07-22 | 16,384 | `ad70af6e771252b3c5331234882d7a60defece0813e9ca189c91b64bc93e6039` |

V1 observed the same ten-field base header in 2020 and 2023 and an additional
`RPI` field in 2026.  That fact is source schema, not alpha incidence.  V1
silently compared only canonical required-field mappings and therefore failed
the precommitted requirement to map full-header drift explicitly.

## Independent review findings accepted

V2 must correct all findings before source access:

1. compare and report full headers, not only canonical mappings;
2. classify every added, removed, or reordered field;
3. decompress no byte beyond the second logical CSV record and apply output
   limits before allocation;
4. hash the exact raw header and first-record bytes rather than a reconstructed
   CSV row;
5. reject redirects before contacting a redirected target;
6. count exact directory anchor `href` entries rather than arbitrary regex
   text;
7. declare the manifest-hash scope and test independent verification; and
8. bind the disk guard to the repository/data filesystem, not a caller-selected
   output path.

## Frozen v2 schema rule

V2 may accept only the following full-header relation:

- the six required canonical fields remain uniquely mapped at every boundary;
- no base field is removed or reordered;
- an exact `RPI` suffix is the only permitted addition; and
- `RPI` is explicitly recorded as recent-only and excluded from every primary
  historical feature.

This does not relax the source-axis decision.  That decision already said
block/RPI fields were optional controls and could not enter the primary without
full-window historical semantics.  Any other added field, any removed field,
any reordered base field, or any canonical mapping drift is a v2 rejection.

## Frozen v2 replay rule

V2 must access only the same directory and the same three archive days.  It
must require the three compressed-prefix SHA-256 values listed above.  A source
revision rejects the replay rather than selecting a new prefix.  The v2 code
and synthetic regression tests must be committed before it is run.

The v2 artifact must state that v1 is invalid, disclose its hash, and retain
zero source values beyond exact raw-record hashes and schema/type validation.
Only an independently reviewed v2 pass may authorize the next stage.
