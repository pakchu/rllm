# SMAF-72 source-support implementation contract — 2026-07-30

## Decision

Implement one deterministic, outcome-blind evaluator for the committed
`SMAF-72` preregistration. This document does not alter any parser, feature,
rank, tail, onset, side, scheduling, support, control, novelty, or economic
threshold. The preregistration remains the sole policy authority.

The evaluator and synthetic tests must be committed before any real SMAF
source row is decoded. The real run is separately authorized only from that
clean committed state.

## Frozen preregistration bindings

| Artifact | SHA-256 / canonical hash |
|---|---|
| `docs/soma-maturity-allocation-fracture-preregistration-2026-07-30.md` | `0ca0b00c77bd55e3360abe1f36409938a8e95dc450a599b89370d1265b4491f9` |
| `training/preregister_soma_maturity_allocation_fracture.py` | `1712afb46872a3b94cc9838b9225c596d2c19b6d4ec5da672a4e9d562e73e554` |
| `tests/test_preregister_soma_maturity_allocation_fracture.py` | `d68e80cd6bfa4c49d8d98698c2930b72fd2949e21ba3d27d59d5098ef559fb88` |
| `results/soma_maturity_allocation_fracture_preregistration_2026-07-30.json` | `4809586182ab2777d1aa32cc249e223700419983fb85a3d149532c62c0e0d01d` |
| preregistration internal `manifest_hash` | `a5949fd0aa723aebf271a966371222c219093fbff1f3d34ba383f5f66620682b` |
| preregistration commit | `01bdc8b923f1ddd4e218df28239e2b814fc47f62` |

Any drift or uncommitted evaluator protocol file fails before source access.

## Implementation and outputs

- evaluator:
  `training/build_soma_maturity_allocation_fracture_support.py`;
- synthetic tests:
  `tests/test_build_soma_maturity_allocation_fracture_support.py`;
- symbolic clock:
  `data/soma_maturity_allocation_fracture_clocks_2020_2023.csv.gz`;
- source-support report:
  `results/soma_maturity_allocation_fracture_support_2026-07-30.json`.

The clock and report are write-once. Existing canonical bytes verify; any
existing drift, symlink, non-regular file, unsafe parent, unsupported secure
open primitive, write failure, or publication race fails closed.

## Source access and memory

The evaluator validates the preregistration, source, and comparator hashes and
exact headers before decoding rows. It uses the Python CSV reader over gzip and
the exact preregistered allowlists. It does not load the 182,616-row detail
panel into a dataframe.

Operations are retained in a small identity map. Detail rows are streamed once
and update per-operation exact-rational accumulators:

- row count and unique `(operation_id,cusip)` identity;
- submitted, accepted, and available-to-borrow sums;
- each weight multiplied by integer maturity distance;
- invalid reason counts.

Only operation-level exact features and symbolic clocks remain after the
stream. The evaluator never retains every detail amount, maturity, or CUSIP
after its operation accumulator is complete. The unique identity set is the
only detail-cardinality structure.

No network, subprocess other than the two protocol Git checks, market, funding,
return, model, or GPU access is allowed.

## Integrity and failure reporting

The evaluator implements the preregistered seven-stage source order:

1. frozen identity and exact header;
2. schema, join, uniqueness, and exact reconciliation;
3. parser coverage and complete operations;
4. singleton causal batches;
5. rank coverage and tail selectivity;
6. primary event support;
7. internal component distinctness.

Malformed rows are never repaired or dropped into a partial feature. When an
identity can be attributed, the whole operation becomes invalid. A global
schema, timestamp, unjoinable identity, or duplicate-key failure prevents all
incidence. Attributable numeric, parser, joined-value, reconciliation, and
weight failures are accumulated through the end of the source scan. Their
operation receives no feature value, its whole causal batch is invalid, and
the report records occurrence counts, invalid-operation and invalid-batch
counts, and the earliest failed stage. A global failure records the exact
one-based number of rows decoded through its failing row.

An invalid causal batch resets all five scalar histories and tail states. A
valid batch contains exactly one complete operation. Split boundaries never
reset history.

## Exact operation features

All arithmetic is `fractions.Fraction`. Centroids and the five scalar controls
are the exact formulas frozen in the preregistration. Current values are
compared against the latest 126 strictly prior complete operations in the same
uninterrupted segment by exact fraction comparison.

For each scalar:

- classify LOW, HIGH, or NEUTRAL with the exact midrank integer inequalities;
- count every rank-ready classification for selectivity;
- emit only false-to-true LOW/HIGH onset;
- update source state regardless of later reservation suppression.

The source controls reserve independently. Primary side/delay controls are
derived only from accepted primary parents under the committed rules.

## Symbolic clock

The exact columns are:

```text
control,signal_id,parent_signal_id,decision_time,entry_time,exit_time,split,side,tail
```

No amount, centroid, fracture, raw rank numerator, price, funding, return, PnL,
CAGR, MDD, reward, label, CUSIP, or operation ID is written.

Rows use canonical UTC second timestamps, exact `LONG`/`SHORT` sides, and
`LOW`/`HIGH` tails. They are sorted by the preregistered control order then
`(entry_time,signal_id)`.

Primary and scalar-control IDs are the preregistered hashes. Every primary-side
or delay-control ID is lowercase hex
`SHA256(UTF8("SMAF-72|<control>|<parent_primary_signal_id>"))`. The random-side
control uses that digest only as its signal ID. Its side is independently
drawn from the first byte of
`SHA256(UTF8("SMAF-72|<parent_primary_signal_id>|RANDOM_SIDE"))`, exactly as
preregistered. `parent_signal_id` is empty for independently derived scalar
clocks and is the exact primary ID for every primary-side/delay control.

CSV uses UTF-8, LF, and fixed columns. Gzip uses an empty filename, `mtime=0`,
and compression level 9, so two builds are byte-identical.

## Source-support statistics

Coverage uses `available_at_utc` and reports full, warmup, train, and selection
numerators and denominators exactly. Rank-ready and raw-tail selectivity use
complete primary operation features. Accepted-event statistics include all
fixed year, half, and quarter cells, including zeros.

Count/share gates use integer comparison whenever possible. Gaps use elapsed
UTC seconds. Month and quarter concentration, active months, and same-side runs
use accepted chronological entries. Ratio evidence is constrained to `[0,1]`
and Pearson evidence to `[-1,1]` before any threshold check.

Internal component distinctness compares each non-primary source control with
primary separately in train and selection:

- exact entry-set Jaccard;
- exact-entry same-side reproduction;
- signed five-minute occupied-exposure Pearson correlation.

The complete split grid is allocated one control pair at a time and discarded
after correlation, bounding memory.

## Comparator gating and novelty

No comparator row is decoded unless all seven source stages pass.

When authorized, each entire immutable comparator file is validated before
selected-group filtering:

- exact full group vocabulary;
- exact side vocabulary;
- canonical timestamps and interval ordering;
- distinct entries and non-overlap within every group;
- five-minute boundaries;
- exact common-window containment accounting.

Each selected SLCS and SCAF group is then evaluated separately with the four
preregistered novelty metrics. The 24-hour matcher is the exact sorted
two-pointer algorithm. Signed occupancy uses one candidate/comparator pair at a
time.

A raw/contained/before/after/crossing count is reported for every allowed
group, including groups not selected for novelty metrics. Metric values and
threshold checks are added only for the preregistered selected groups.

A comparator contract failure is serialized with rows decoded, reason, and
terminal novelty failure. It never falls through to economic authorization.

## Report and first-failure decision

The canonical report contains:

- all frozen bindings and implementation identities;
- source rows read and structural counts;
- coverage, selectivity, event-support, and internal-control metrics/checks;
- comparator access authorization, rows read, metrics, and checks;
- earliest failed stage and check;
- `advance_to_economic_evaluator_freeze` only if every source and novelty check
  passes;
- otherwise `retire_SMAF_72_unchanged_before_outcomes`;
- exact output clock file SHA-256;
- a canonical self `manifest_hash`;
- the outcome boundary below.

JSON uses sorted keys, two-space indent, UTF-8, one LF, `ensure_ascii=true`, and
`allow_nan=false`. The internal hash uses compact sorted-key JSON without the
hash field. Validation independently recomputes source pass, novelty pass, and
the advance-or-retire decision from the exact preregistered check-key schemas.
Every source and novelty check is also recomputed from its reported counts and
metrics with the same frozen formulas, including integer cross-multiplication
for count/share gates. Rank-ready counts are bounded by complete operations;
source and internal event totals reconcile to exact per-control clock counts;
subperiod counts obey feasible containment inequalities; booleans are never
accepted as numbers; and gap/run domains are enforced. Validation additionally
checks complete comparator row accounting, the distinct terminal
source/comparator failure schemas, and the reported first failed stage/check;
recomputing the self-hash cannot make a contradictory or evidence-free
decision valid.

## Outcome boundary

Both synthetic development and the real source run must report:

- BTC market rows loaded: `0`;
- funding rows loaded: `0`;
- forward-return rows loaded: `0`;
- PnL/CAGR/MDD opened: `false`;
- economic evaluator authorized: `false` until all source and novelty gates
  pass and a later evaluator is separately committed;
- network calls: `0`;
- protocol Git subprocess calls: exactly `2` only in the real run;
- model/GPU calls: `0`.

Synthetic tests may build source frames in memory, but may not read the real
source or comparator files. The first real run occurs only after this document,
the evaluator, and tests are committed and clean.
