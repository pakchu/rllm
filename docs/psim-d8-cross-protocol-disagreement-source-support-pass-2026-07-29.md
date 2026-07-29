# PSIM-D8 Cross-Protocol Disagreement Source-Support Pass

Date: 2026-07-29  
Candidate family: `PSIM-D8-CDP1`  
Decision: `pass`

## Boundary

This run followed preregistration commit `42259cfe`. It opened only the frozen
`ARCHIVE_D90` PSIM-D8 cards and did not open:

- the PSIM-D8 event payload;
- BTC market rows;
- funding rows;
- any 2022 or 2023 economic outcome.

The implementation recomputed the frozen nine-component structural
disagreement score, its 3-card and 30-card half-life EWMAs, and source-only
signal incidence.

## Source capacity

| Year | Nonmissing daily scores | Eligible relation units |
|---|---:|---:|
| 2022 | 213 | 1,955 |
| 2023 | 272 | 1,267 |

The individual daily score values were not written to the result.

## Eligible frozen candidates

Two of the nine preregistered candidates passed every source-support condition
in both years:

| Candidate | 2022 accepted long / short | 2022 total | 2023 accepted long / short | 2023 total |
|---|---:|---:|---:|---:|
| `CDP_S35_G05` | 14 / 62 | 76 | 44 / 16 | 60 |
| `CDP_S50_G05` | 14 / 42 | 56 | 43 / 16 | 59 |

Both candidates had incidence in at least three quarters per year and stayed
below the preregistered 50% top-month concentration ceiling.

The remaining seven frozen thresholds are excluded from economic selection.
They cannot be restored through rank-2 substitution, threshold repair, or
family extension.

## Decision

Authorize 2022 economic selection for exactly:

- `CDP_S35_G05`
- `CDP_S50_G05`

The 2023 market and funding outcomes remain untouched and unauthorized until a
single 2022 top1 is selected and committed.

Result hash:
`3ac80ced7ea528dc53826e398fc4f7621664f79dd744f79de34f20ea2888c0d7`.
