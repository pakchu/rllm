# PSIM-D8-RLLM2-S7 2021 report-only transfer preregistration

Date: 2026-07-27 KST

Status: **PREREGISTERED — 2021 outcome payload still unopened by S7**

## Fixed candidate

The single promotable primary is fixed before outcome access:

```text
semantic_ridge_action_mean_residual_fqi
```

The complete family contains the sealed 25 S4, 8 S5, and 8 S6R1 schedules:
41 policies in fixed order. No 2021 metric may be used to alter, repair,
reselect, invert, threshold, or ensemble the primary.

This is a protocol-isolated policy-specific report-only transfer. Repository
2021 is not globally pristine because unrelated historical results exist.
Passing does not authorize live promotion without forward/live confirmation.

## Economic evaluation

- calendar: `2021-01-01T00:00:00Z` to
  `2022-01-01T00:00:00Z`;
- CAGR denominator: the complete calendar, including flat periods;
- target gross: `0.5`;
- base cost: 6 bp per changed notional side;
- stress cost: 10 bp per changed notional side;
- funding: exact frozen 8h funding/mark rows;
- strict MDD: global pre-entry high-water plus favorable-then-adverse held 5m
  OHLC path and terminal flatten;
- delayed evaluation: exact sealed +5m schedule;
- halves: Jan 1–Jul 1 and Jul 1–Jan 1;
- trade reporting: directional entries including flips, plus all target
  changes including terminal flatten.

## Fixed pass gate

Every check is required:

1. base absolute return > 0;
2. 10 bp stress absolute return > 0;
3. +5m delayed absolute return > 0;
4. first-half absolute return > 0;
5. second-half absolute return > 0;
6. base CAGR / strict MDD >= 1.0;
7. at least 80 non-flat intervals;
8. long and short shares each >= 20%;
9. primary beats the strongest fixed nonsemantic control on both absolute
   return and CAGR / strict MDD;
10. action-code permutation schedule identity remains exact; and
11. weekly shared-sign familywise max-stat p-value < 0.25.

The max-stat family is all 41 policies, clustered by Monday-00:00 UTC weeks,
with 100,000 Rademacher draws, seed `20260725`, and one-sided positive
studentized means.

## Outcome source

The frozen stage-local source is:

- market: 105,120 BTCUSDT 5m rows;
- funding: 1,095 8h rows;
- source-manifest SHA-256:
  `1d12d8dad47eda810933ddce7ac2d911a7a4f85262ecc02e91e22df2488c6e2d`;
- source-manifest self-hash:
  `d71ade5504ebe4d729bd9892afbceec0e03675f2ad197ac937d0a46cb0bd64c9`;
- expected market gzip SHA-256:
  `a6b66b41aec8484f8ac30d22d3513ceb2cacda57260fd25f6c9456b59f119f6d`;
- expected funding gzip SHA-256:
  `65d6bceacdb3655062e8e5f5ca95b2dd4a129607966d814bb6b787b5ad15901d`.

At preregistration, S7 opened, read, hashed, or parsed zero market/funding
payload paths, bytes, or numeric rows. The committed runner must write its
attempt before opening, reading, hashing, or parsing either payload.

## Immutable preregistration

- file SHA-256:
  `daec6511fa318383d0f4988d32d8a91a78d552e48e2581268cb122718337a84e`;
- manifest hash:
  `2d7683f6f3e0d3c158fb5609924cfb2010756be81ad90f54560dde90fb4655fa`.

Next authorized action:

```text
IMPLEMENT_REVIEW_COMMIT_AND_PUSH_S7_RUNNER_THEN_EXECUTE_THE_SINGLE_REPORT_ONLY_2021_TRANSFER
```
