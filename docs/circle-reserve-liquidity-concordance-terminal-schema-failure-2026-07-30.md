# CRLC-336 terminal schema failure — 2026-07-30

## Decision

`CRLC-336` is permanently retired as:

```text
REJECT_SCHEMA_INCOMPATIBLE_PREPRODUCTION
```

No production SEC daily-index path, Feed archive, Circle filing, liquidity
percentage, WAM, WAL, candidate clock, comparator clock, Gross9 clock, BTC
bar, funding row, return, PnL, CAGR, or strict MDD was opened.

The failure was established from the already disclosed official SEC XML
technical-specification packages before the source builder, source-support
evaluator, source-access claim, or any production request existed.

## Frozen contracts that conflict

The frozen source decision requires:

- original filings before local acceptance date `2024-06-11` to be
  `N-MFP2`;
- every retained original from report month `2022-11` through `2026-04` to
  contain all six frozen identities in its XML;
- every original to contain `govMoneyMrktFundFlag`;
- every original to contain 15–31 `liquidAssetsDetails` rows; and
- every such row to contain a source-native date plus daily and weekly
  liquidity percentages.

The frozen mechanism and preregistration consume that exact homogeneous path.
They do not authorize a form-family adapter, Friday-slot path, external
identity repair, missing-field exception, or reduced support envelope.

## Official schema evidence

The bounded metadata-only packages were:

```text
SEC N-MFP2 technical-specification ZIP SHA-256
  0f055d8f3660ad0d328f6abf973995734a9ac7a96fda17f95ada17ea405e0b4e
SEC N-MFP3 technical-specification ZIP SHA-256
  4a8daf4801d79e8a2f0484bc41a22e098771c1fec822045a3e96c6fb82b82ec7

extracted N-MFP2 eis_NMFP2_Filer.xsd SHA-256
  372808e0b8a348047d97d7631df505f627abb62d1bd8d3069172f9c12339d84a
extracted N-MFP2 eis_NMFP2_common.xsd SHA-256
  b6fa1b1a545845f1c17daf078ac610ea6a27164108c2d328c3dbbe94b92aeda7
extracted N-MFP3 eis_NMFP3_Filer.xsd SHA-256
  bf6fb5e217caac77b85ef90567da94f32fa41198c1b254ef132b7a2ee4b89e08
```

Exact element-declaration counts in the two filer schemas are:

| Element | N-MFP2 | N-MFP3 |
|---|---:|---:|
| `liquidAssetsDetails` | 0 | 1 |
| `govMoneyMrktFundFlag` | 0 | 1 |
| `registrantFullName` | 0 | 1 |
| `nameOfSeries` | 0 | 1 |
| `leiOfSeries` | 0 | 1 |
| `percentageDailyLiquidAssets` | 1 | 1 |
| `averagePortfolioMaturity` | 1 | 1 |

The N-MFP2 `percentageDailyLiquidAssets` element is not an N-MFP3-style
dated-row collection. Its exact common-schema type is
`PERCENTAGE_OF_TOTAL_ASSET_INVESTED_DAILY`, whose children are optional
`fridayDay1` through `fridayDay5`. The corresponding weekly type uses
`fridayWeek1` through `fridayWeek5`. N-MFP2 therefore cannot satisfy the
frozen 15–31 dated `liquidAssetsDetails` contract.

This is a form-schema incompatibility, not an observed Circle value or an
economic failure.

Official authorities:

- <https://www.sec.gov/info/edgar/specifications/form-n-mfp2-xml-tech-specs.htm>
- <https://www.sec.gov/edgar/filer-information/specifications/form-n-mfp3-xml-tech-specs>
- <https://www.sec.gov/submit-filings/technical-specifications>

## Sealed preregistration retained

The following evidence remains immutable and is not reused:

```text
producer/test commit
  9822f0e7b169c3e3e13db666c37a154ff1f151d6
artifact commit
  2ea164c5afc7dbe233f9c0eefc716ded8cebe547
verification commit
  334e144cf03ed2caa3a4003d420c82997d28d1a7

artifact
  results/circle_reserve_liquidity_concordance_preregistration_2026-07-30.json
artifact SHA-256
  a3da6ca20d42aa8253d0b126eb362774051e20a3e14540e81622efcb24483e70
manifest_hash
  d9bd957107bce86b446c640e7bc6b03e655489d4a30799616386b136d1eaffca
```

Its evidence boundaries remain false, including production daily indexes,
Feed archives, source incidence, liquidity values, WAM/WAL values,
comparators, Gross9, market, funding, and outcomes.

No CRLC source builder, support evaluator, novelty evaluator, economics
evaluator, source-access claim, source artifact, support clock, novelty
artifact, or economic artifact was created.

## No-repair consequence

`CRLC-336` may not be repaired by:

- treating N-MFP2 Friday slots as if they were N-MFP3 dated rows;
- weakening the six-identity XML requirement;
- dropping pre-transition months;
- changing support periods or floors;
- using current pages, bulk data, amendments, or inferred dates; or
- retaining the candidate name while changing its source schema.

A later research attempt may use the same official SEC family only under a
new candidate identity, a schema-transition-aware source decision, a fresh
mechanism review, and a fresh source-unseen preregistration. It may not import
any CRLC-336 incidence or outcome because none was opened.
