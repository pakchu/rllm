# SCAF-48 normalized-source schema amendment — pre-incidence

## Status

This amendment is committed before any SCAF feature, causal batch, candidate
incidence, comparator row, BTC row, funding value, or outcome is decoded.

It supersedes only:

1. item 4 of the mechanism-proof checklist; and
2. the provisional RLLM token `operation-type agreement/disagreement`

in:

```text
docs/soma-collateral-allocation-fracture-boundary-2026-07-24.md
```

All other source, no-repair, novelty, sequencing, and RLLM boundaries remain
unchanged.

## Schema conflict

The SCAF boundary required simultaneous-availability batching and
strict-prior operation-type history. The immutable normalized operation and
detail schemas do not expose `operation_type`.

The exact normalized columns are already frozen in:

```text
data/new_york_fed_securities_lending_2019_2023/build_manifest.json
```

The operation schema contains operation identity, dates, publication timing,
notes, and operation totals. The detail schema contains operation identity,
CUSIP, security description, submitted/accepted amounts, weighted fee,
holdings, available inventory, and outstanding loans. Neither contains an
official normalized operation-type token.

SCAF may not repair this by reopening raw JSON fields, parsing free-form
`security_description`, modifying the audited source builder, or silently
inferring a type from CUSIP or operation ID.

## Effective causal-batch rule

The exact causal unit is now a **distinct `available_at_utc` batch**:

1. every normalized operation with the same exact `available_at_utc` belongs
   to one batch;
2. every normalized detail atom remains identified by the exact pair
   `(operation_id, cusip)`;
3. simultaneous operations never enter one another's prior state;
4. batch-level topology is computed over the complete set of valid detail
   atoms in that batch, without merging equal CUSIPs across operations;
5. a batch may compare only with the immediately preceding complete batch at a
   strictly earlier `available_at_utc`; and
6. an invalid or incomplete batch breaks transition continuity rather than
   being skipped or imputed.

The later mechanism must freeze exact batch completeness, zero/null handling,
distributional formulas, transition direction, and controls before a value row
is decoded.

The replacement provisional RLLM token is:

```text
batch-component agreement/disagreement
```

It may be derived only from the frozen symbolic directions of SCAF's
distributional components inside one complete causal batch. It may not encode
or infer operation type, `security_description`, CUSIP, operation ID, date, or
timestamp.

## Why this is not candidate repair

No SCAF incidence or outcome exists. This change resolves a schema-level
impossibility visible from the committed manifest alone. It does not change an
observed event count, side balance, calendar concentration, novelty statistic,
return, or risk metric.

Batch aggregation is more conservative than inferring an unavailable type:
all simultaneously public information is treated as one causal release, and
no within-timestamp ordering can create hidden lookahead.

## Evidence boundary

This amendment inspected only committed boundary text, builder/schema code, and
the build-manifest column lists. It did not parse either normalized CSV and did
not open any candidate, comparator, market, funding, return, PnL, CAGR, or MDD
value.
