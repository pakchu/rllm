# SABLE-8 source-support implementation contract

Date: 2026-07-25

## Authority

This unit implements only Stage 0 of
[`sable8-symbolic-alpha-braid-boundary-2026-07-25.md`](sable8-symbolic-alpha-braid-boundary-2026-07-25.md).
It may:

- verify the three frozen full-history containers and physical headers;
- stream-project the authorized columns into deterministic pre-2024 cuts;
- calculate the fourteen causal primitives;
- create strictly-prior ordinal bands and six-line sequence signatures; and
- decide the frozen source/language support gates.

It may not calculate a forward interval return, action label, reward, PnL,
CAGR, MDD, policy action, model prediction, or any 2024+ numeric source value.

## Files

```text
training/preregister_sable8_symbolic_alpha_braid.py
training/build_sable8_symbolic_alpha_braid_support.py
tests/test_preregister_sable8_symbolic_alpha_braid.py
tests/test_build_sable8_symbolic_alpha_braid_support.py
results/sable8_symbolic_alpha_braid_preregistration_2026-07-25.json
```

The real runner refuses to execute unless these files, this contract, and the
boundary are tracked and byte-clean against `HEAD`.

## Projection transaction

For each source, the runner performs this order:

1. verify full-file SHA-256;
2. read one physical header line and verify its exact bytes and order;
3. locate the timestamp and authorized projection indices;
4. for each row, tokenize CSV framing and convert the timestamp field first;
5. skip before converting any other cell when timestamp is before
   `2020-01-01T00:00:00Z`;
6. stop before validating row width or converting any other cell when timestamp is
   `>=2024-01-01T00:00:00Z`;
7. only for in-range rows, validate exact row width and emit authorized fields;
8. write deterministic gzip with `mtime=0`, UTF-8, LF, and the exact projected
   header; and
9. write a manifest containing input/output hashes, included bounds, and
   timestamp-only skipped-row bounds.

Unprojected cells are never numerically converted, retained, aggregated,
value-hashed, returned, or written. The output cuts are ignored data artifacts;
their tracked manifest is the reproducibility record.

Zero-volume market rows with
`quote_asset_volume=taker_buy_quote=0` are valid source rows but invalidate
quote-dependent primitives through normal missingness. A negative quote,
negative taker quote, taker quote above total quote, or nonzero taker quote
with zero total quote is source failure.

## Support table

The derived table has one row per canonical `00/08/16 UTC` boundary. It may
contain clocks, readiness flags, source-freshness flags, categorical bands,
canonical lines, and sequence hashes. It contains no raw primitive value,
position, action, reward, return, or economic metric.

A line is ready only after every core primitive has a current finite value and
180 strictly prior valid values. Context values emit `STALE` when their current
source or strictly-prior rank history is unavailable. A sequence is ready only
when the current and previous five boundary lines are all ready and exactly
eight hours apart. Invalid boundaries are represented and later force `FLAT`;
they are never skipped.

## Prefix replay

The runner recomputes the source language after physically slicing all three
cuts at `2023-01-01T00:00:00Z`. Every pre-2023 clock, readiness flag, band,
canonical line, and sequence hash must equal the same prefix produced from the
full pre-2024 cuts. Any mismatch is a Stage 0 failure.

Core-missing and source-freshness metrics are calculated separately for
development `2020-2022` and report-only `2023`; both periods must pass their
frozen threshold. One period cannot rescue another.

## Pre-commit review incident

An independent code reviewer ran a temporary-output projection before this
implementation unit was committed. It decoded funding/premium support rows and
market rows through the first zero-volume row at
`2021-03-02T01:05:00Z`, discovering 26 pre-2024 rows with
`quote_asset_volume=taker_buy_quote=0`. No forward return, reward, action,
token incidence, PnL, CAGR, MDD, or post-2023 numeric value was opened, and no
repository artifact was written.

This is an execution-order breach and is recorded rather than hidden. The
resulting fix does not weaken a support threshold: the boundary already states
that zero quote invalidates the corresponding primitive. The implementation
now expresses that frozen rule as missingness instead of aborting the support
transaction. The official write-once run is no longer claimed to be the first
source-value read; it remains the first complete support-gate decision.

## Stop condition

The one-shot result is `PASS` only when every frozen support gate passes.
Otherwise the exact SABLE-8 candidate is retired before reward construction.
No threshold, window, availability rule, primitive, band, or sequence repair
is authorized after the result is known.
