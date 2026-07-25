# CLOR-D1 source-support implementation contract

Date: 2026-07-25

Status: **frozen before any CLOR-D1 source value row is decoded.**

## Authority

```text
policy
  CLOR-D1
boundary
  docs/collateral-liquidity-ordering-relation-target-policy-boundary-2026-07-25.md
  commit c82dccf1aea20e4f71e9b676e3d3f22b00b92e77
  SHA256 bc537d568701215e72199e632fcc196724927d68348f847e42a47776d248f9df
preregistration producer
  training/preregister_collateral_liquidity_ordering_relation.py
  commit 52efeed17445157ac585b246fde16830e23236b3
  SHA256 322b8d4e19649bcda4d99a5e2a7888f3e38908f443037e6c4569c7ab31075942
preregistration artifact
  results/collateral_liquidity_ordering_relation_preregistration_2026-07-25.json
  commit 41d77b9184c1f4bf839a7f44d963689afc44f7a5
  SHA256 7aee03d42daade588a0e785133632ff6f9f9e2a8d23117b49ffd405a41341e89
  manifest_hash 881f4c631f924e26e827e71359ce8df1f9add309d0b828478a545c7262f00b2b
  scientific_contract_hash 59b52c826eef8315dc81a15a0467a6a494c9fcad8c54f7bf2c73c8ecf344a22a
```

The evaluator must validate this complete authority before source access. It
may not reinterpret a boundary ambiguity, change a threshold, or substitute a
control after incidence is known.

## Implementation and output paths

```text
runner
  training/build_collateral_liquidity_ordering_relation_source_support.py
tests
  tests/test_build_collateral_liquidity_ordering_relation_source_support.py
execution seal
  results/clor_d1_source_support_execution_seal_2026-07-25.json
pass source language
  data/collateral_liquidity_ordering_relation_source_2020_2023.csv.gz
pass controls
  data/collateral_liquidity_ordering_relation_controls_2020_2023.csv.gz
pass report
  results/collateral_liquidity_ordering_relation_source_support_2026-07-25.json
rejection report
  results/collateral_liquidity_ordering_relation_source_rejection_2026-07-25.json
```

Runner and tests share one commit. A separately committed execution seal binds
their exact bytes and commit before the official run decodes any source value.
Every final path is create-once, repository-relative, and a regular
non-symlink file.

The seal is canonical sorted ASCII JSON with exactly:

```text
protocol_version = clor_d1_source_support_execution_seal_v1
policy_id
runtime
contract
boundary
preregistration
preregistration_producer
runner
tests
shared_commit
synthetic_verification
forbidden_access
seal_hash
```

`runtime` binds `sys.executable` realpath/SHA-256, exact Python version, pandas
version and `pandas.__file__` SHA-256, and the preregistered Git authority.
Every authority/implementation object contains exact repository path, 40-hex
commit, and 64-hex byte SHA-256; preregistration additionally contains its two
frozen manifest hashes. Runner and tests must have the same last-modifying
commit, equal to `shared_commit`.

`synthetic_verification.self_check` binds the exact argv
`[sys.executable, runner, "self-check"]`, exit code zero, exact stdout SHA-256,
the parsed self-check manifest hash, and zero source/predecessor value rows and
zero forbidden counters. The self-check output is canonical JSON without
timing text. `synthetic_verification.pytest` binds repository-relative argv
`[".venv/bin/pytest","-q",tests]`, environment override
`{"PYTHONPATH":"."}`, exit code zero, and parsed integer passed-test count;
failure/skip/error counts must all be zero. `forbidden_access` is the exact
all-zero map below. `seal_hash` is SHA-256 of canonical JSON over every
preceding field.

Seal creation requires a clean worktree and validates the exact test command:

```text
PYTHONPATH=. .venv/bin/pytest -q tests/test_build_collateral_liquidity_ordering_relation_source_support.py
```

An absent, malformed, hash-drifted, environment-mismatched, nonzero-counter,
or implementation-mismatched seal hard-aborts before source access and writes
no terminal report.

## Source-access boundary

Before the execution seal is committed, tests use synthetic in-memory records
only. Importing the runner must not open a source, predecessor, market, or
outcome file.

The official run may decode only the four preregistered source allowlists. Each
physical CSV loader receives its exact allowlist through `usecols`; loading a
wider frame and dropping fields later is forbidden. Strings remain strings
until strict parsing. Source arithmetic uses `Decimal` converted to exact
`Fraction`; binary floating point is forbidden.

Predecessor files are whole-file and gzip-header hash checked only. No
predecessor value/action row is decoded in source support.

## Runtime precondition

Before source access:

1. validate hash-bound `/usr/bin/git`, sanitized Git environment, repository
   top level, and disabled user/system Git config through the preregistration
   authority;
2. validate the contract, boundary, preregistration producer, preregistration
   artifact, source files, source manifests, source headers, and predecessor
   headers;
3. validate the execution seal, runner, and test bytes and shared commit;
4. require a clean worktree;
5. require no partial or conflicting terminal state; and
6. require every forbidden-access counter to be zero.

Executable-name lookup for Git, network access, wall-clock data, environment
model selection, and mutable registry scans are forbidden.

## Exact physical validation

All dates are exact ASCII `YYYY-MM-DD`. Source timestamps are exact
whole-second UTC strings emitted by the bound builders and are serialized in
derived artifacts as `YYYY-MM-DDTHH:MM:SSZ`. Empty, fractional-second,
timezone-naive, non-UTC, or post-2023 source timestamps fail.

### Treasury

Validate exactly 445 physical rows, with exactly 440 `source_complete=true`
and five `source_complete=false`, matching the bound manifest. Physical
identity `(auction_date,result_available_at_utc,original_security_term)` is
unique. Physical rows are nondecreasing by `auction_date`; term order is
applied only inside token construction.

An incomplete row must have all four authorized amount fields empty and emits
no token. A complete row has canonical finite nonnegative amounts,
`competitive_accepted_usd > 0`, and exact bidder-sum reconciliation. A complete
row's term is in the seven-term frozen order and its result time is strictly
before 2024.

Group all rows by exact result availability. A batch containing any incomplete
row explicitly invalidates Treasury state and emits no token. Otherwise sort
by frozen term order and serialize each term's exact `P,D,I` weak order.
Duplicate terms inside a batch fail. A batch token joins `<term>:<order>` with
`|`.

### SOMA

Validate exactly 1,259 operation rows and 182,616 detail rows. Operation IDs
are nonempty and unique. Detail identity `(operation_id,operation_date,
available_at_utc)` is not required to be unique because one operation has many
securities; no forbidden security identifier is loaded.

Every amount is canonical, finite, and nonnegative; accepted never exceeds
submitted. Every operation has at least one detail, operation/detail
date/availability agree exactly, and detail submitted/accepted sums reconcile
to operation totals.

Group operations by exact availability. Batch submitted must be positive.
Compute exact submitted, accepted, and accepted/submitted. The first complete
batch establishes a baseline. Every later complete batch emits three exact
`UP|DOWN|EQUAL` comparisons to the immediately previous complete batch. A
structurally invalid batch fails Gate 1 rather than being repaired.

### OFR

Validate exactly 77,369 physical rows and the bound source window. Immediately
retain only the six frozen mnemonics after loading the exact five-column
allowlist. Physical identity `(mnemonic,observation_date)` is unique.

For each observation date, a complete authorized date has exactly six rows,
all `disclosure_edit == "0"`, and six nonempty finite exact values. Volume
values are nonnegative and their three-venue sum is positive. An authorized
date with a missing row, empty value, or disclosure edit is incomplete and
does not contribute a token.

Group authorized dates by exact availability. Every complete date in a batch
is validated; the greatest complete date becomes the new OFR state. A batch
with no complete authorized date explicitly invalidates the OFR state. The
state token is the exact `DVP,GCF,TRIV1` weak ordering for rates and volumes.

A Treasury incomplete batch or OFR no-complete-date batch is itself a source
invalidation event. It receives the same latency formula, enters its joint
execution group, appears in CSV `updated`, marks that source invalid before
joint validity is evaluated, emits an invalid joint row when in a research
split, and clears history. Relation controls preserve every invalidation event
and do not transform it.

## Joint schedule

For each source token or explicit invalidation batch:

```text
execution_time = ceil_to_5m(available_at_utc) + 5 elapsed minutes
```

An exact five-minute availability still receives five minutes. Group equal
execution times and apply `(available_at_utc,source_order)` with
`TREASURY < SOMA < OFR`. Emit one joint row after the group.

Warmup source state carries into TRAIN, but source history and position reset
at every TRAIN/TEST/EVAL boundary. A boundary reset occurs before processing a
source group exactly at the boundary. Source state itself does not reset.

At a joint execution time, current states are valid only at ages:

```text
TREASURY <= 14 elapsed days
SOMA     <=  4 elapsed days
OFR      <=  4 elapsed days
```

Missing, explicitly invalidated, or stale state emits an invalid row, forces
the source-only safety target `TARGET_FLAT`, and clears history. No model row
or selected action is built.

A valid line is exact preregistered canonical text. Number consecutive valid
lines from one after split/invalid reset. Line `N >= 12` is a model-decision
schedule row using the trailing twelve lines including `N`. Sequence bytes are
the twelve canonical lines joined by LF with no terminal LF. `sequence_sha256`
is SHA-256 of those bytes; `line_sha256` is SHA-256 of line bytes.

Every model-decision schedule row derives
`decision_expiry_time = execution_time + 72 elapsed hours`. This is a
source-only timer candidate, not an action, position, trade, or model output.

The source-language CSV contains exactly:

```text
split
execution_time
valid
invalid_reason
model_decision
updated
treasury
soma_submitted_step
soma_accepted_step
soma_coverage_step
ofr_rate_order
ofr_volume_order
line_text
line_sha256
sequence_sha256
decision_expiry_time
```

Rows cover TRAIN, TEST, and EVAL execution groups only. Boolean fields are
`0|1`. Exact row-state semantics are:

| Row kind | `valid` | `invalid_reason` | `model_decision` | `updated` | primitive fields + `line_text,line_sha256` | `sequence_sha256,decision_expiry_time` |
|---|---:|---|---:|---|---|---|
| invalid | `0` | nonempty | `0` | nonempty | all empty | both empty |
| valid non-decision line | `1` | empty | `0` | nonempty | all populated | both empty |
| valid model-decision line | `1` | empty | `1` | nonempty | all populated | both populated |

CSV `updated` joins members in frozen source order with `|`; canonical
`line_text` serializes the same members after `UPDATED=` joined with commas.
Gate 4 reads only the CSV `updated` column and splits it on `|`. Invalid reasons
use only
`MISSING_<SOURCE>`, `INVALID_<SOURCE>`, and `STALE_<SOURCE>` and are joined with
`|` in source order `TREASURY,SOMA,OFR`; within a source, exactly one reason is
possible.

## Frozen controls

Controls use the exact preregistered definitions.

- Label rotation, SOMA stale, within-year reversal, and random relations
  transform source-token batches first, then rebuild carried state and the
  joint schedule.
- `one_merged_update_stale` is line-level. The first valid line after each
  split or invalid reset remains primary; every later consecutive valid line
  receives the prior consecutive valid primary line's four field values at the
  current execution time.
- All six relation falsifications retain primary execution rows, validity,
  split, and model-decision schedule exactly.
- `future_append` appends only the three frozen synthetic post-2023 token
  batches. It must leave every ordered pre-2024 source row, line/hash, sequence
  hash, model-decision time, and derived expiry time byte-identical.

The controls CSV contains the source schema above with leading `control`.
Only the six relation-falsification controls are published; future append is
reported as an invariance check.

For every relation control, compare sequence hashes at identical primary model
decision timestamps. It must be globally hash-distinct and change at least
10% of eligible hashes. Primary and control model-decision timestamps must be
unique and form an exact bijection. `eligible_count` is the size of that common
timestamp set and must be positive:

```text
changed_fraction =
  count(primary.sequence_sha256 != control.sequence_sha256 at same timestamp)
  / eligible_count
```

Empty, duplicate, missing, extra, or misaligned schedules fail.

## Frozen gate sequence

Run exactly and stop after the first failing gate:

1. `source_schema_chronology_reconciliation`;
2. `causal_schedule_split_append_invariance`;
3. `model_decision_count`;
4. `source_update_support`;
5. `maximum_decision_gap`;
6. `calendar_support`;
7. `primitive_diversity`;
8. `state_signature_concentration`;
9. `sequence_uniqueness`;
10. `relation_falsification_controls`;
11. `forbidden_access`.

Gate 3 requires 450/180/180 TRAIN/TEST/EVAL model decisions.

Gate 4 counts a source only when it appears in `UPDATED` on a model-decision
line and requires:

```text
TREASURY 40/20/20
SOMA     200/90/90
OFR      200/90/90
```

Gate 5 converts timestamps to integer UTC seconds and measures
`first_decision - split_start`, every
`next_decision - previous_decision`, and
`split_end - last_decision`. Splits are start-inclusive/end-exclusive and every
decision must satisfy `split_start <= decision < split_end`. The maximum is at
most exactly `864000` seconds.

Gate 6 requires at least 30 TRAIN decisions in
`[2020-09-10T00:00:00Z,2021-01-01T00:00:00Z)`, at least 50 in each 2021 UTC
quarter, and at least 40 in every 2022 TEST and 2023 EVAL UTC quarter. A quarter
is exactly `[YYYY-{01|04|07|10}-01T00:00:00Z,next_quarter_start)`.

Gate 7 uses these six primitive fields on every valid primary line:

```text
treasury
soma_submitted_step
soma_accepted_step
soma_coverage_step
ofr_rate_order
ofr_volume_order
```

In every split, each field has at least two observed levels and no level share
above 0.95. Missing fields fail.

Gate 8 groups exact `line_text` on every valid primary line; no signature may
exceed 0.25 of a split.

Gate 9 requires at least 150/70/70 unique sequence hashes in
TRAIN/TEST/EVAL.

Gate 10 requires exact row/schedule alignment and at least 0.10 changed
sequence-hash fraction for each of the six relation controls.

Gate 11 requires every forbidden counter to remain zero.

## Determinism and terminal publication

Build source and reached control records twice and require exact equality.
Pass CSVs use canonical LF CSV and deterministic gzip with `mtime=0` and empty
filename. Reports use sorted ASCII JSON with no wall-clock timestamp.

The report records exact authority, execution seal, decoded row counts,
source audit, schedule funnel, reached gate records, first failure, artifact
hashes when built, canonical row hashes, forbidden counters, and a
self-consistent result hash.

On first failure:

- publish no pass source/control CSV;
- publish no pass report;
- atomically write only the rejection report; and
- return `retire_clor_d1_unchanged_before_outcomes`.

On complete pass:

- atomically publish source gzip, controls gzip, and pass report as one
  rollback-capable group; and
- return `authorize_clor_d1_economic_rllm_evaluator_freeze_only`.

An existing valid terminal state returns idempotently. Partial, conflicting,
unreadable, hash-drifted, or semantically invalid terminal state hard-aborts
without writes. A rejection cannot coexist with pass outputs.

## Forbidden evidence

All remain zero:

```text
post_2023_source_value_rows_opened
comparator_action_rows_opened
market_rows_opened
funding_rows_opened
future_return_rows_built
reward_rows_built
model_rows_built
selected_action_rows_built
trade_rows_built
pnl_cagr_mdd_values_computed
network_calls
```

Source support cannot validate alpha, profitability, an LLM, a reward, or live
trading.

## Execution order

1. commit this contract;
2. implement runner/tests with synthetic data only;
3. pass tests and independent review;
4. commit runner/tests together;
5. create the execution seal only;
6. add seal regression evidence and commit;
7. validate all tests on a clean worktree;
8. invoke the official source run once; and
9. stop permanently on first failure or commit the complete pass.
