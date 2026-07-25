# CEFS-D1 source-support implementation contract — 2026-07-25

## Authority

This contract implements but cannot alter:

```text
docs/cboe-edge-flip-sequence-policy-boundary-2026-07-25.md
SHA256 d0b522a7ac87e3526d6cd740bb81304bd73042bc327978660eb551b159c16ec3
latest boundary commit b7297ccf9235da98edcbe9b35d84009245972321

training/preregister_cboe_edge_flip_sequence_policy.py
SHA256 1c4668e72846eadf66011c582c62e8574af3679204500c1ff9631101ecbb7ac1
producer commit ec8eae23226d39f0c62b6c5711d6080f2bf990a4

results/cboe_edge_flip_sequence_policy_preregistration_2026-07-25.json
SHA256 5e515663e99ef4aa322cae25cfb2c07f69b3e24f289bc2f0f79463aca64a8878
manifest_hash 9aa7c891ec241d4733db215068bed3507f41c03cbae7198c906a079ddb6467bf
artifact commit d60d9744e82f48350be506f1da63bcb2e706cf03
```

The implementation may expose only Cboe source validity, exact relation-edge
tokens, action-independent schedules, prompt templates, and source-only
controls. It may not open BTC market rows, funding, future returns, rewards,
models, selected actions, trades, PnL, CAGR, MDD, comparators, or post-2023
source values.

## Exact implementation and artifact paths

```text
runner
  training/build_cboe_edge_flip_sequence_policy_support.py
tests
  tests/test_build_cboe_edge_flip_sequence_policy_support.py
execution seal
  results/cefs_d1_source_support_execution_seal_2026-07-25.json
pass source output
  data/cboe_edge_flip_sequence_policy_source_2020_2023.csv.gz
pass control output
  data/cboe_edge_flip_sequence_policy_controls_2020_2023.csv.gz
pass report
  results/cboe_edge_flip_sequence_policy_source_support_2026-07-25.json
failure report
  results/cboe_edge_flip_sequence_policy_source_rejection_2026-07-25.json
```

The runner and tests must be committed together. A separate committed
execution-seal artifact must bind their commit and exact bytes before the
first real source row is decoded. Synthetic fixtures are allowed before the
seal.

## Physical parsing

Before source access the runner verifies the boundary, preregistration
producer, preregistration artifact, contract, all three source files,
manifests, audits, physical headers, hashes, and execution seal.

For each physical file:

1. read the exact UTF-8 header and reject drift;
2. parse the first `observation_date` field before any non-date field;
3. require exact ten-byte `YYYY-MM-DD`, strict Gregorian round-trip, unique
   strictly increasing dates, and exact physical column count;
4. scan pre-2020 term/tail rows as date-only;
5. parse allowed non-date fields only for dates from 2020-01-01 through
   2023-12-31;
6. reject any at-or-after-2024 row before parsing a non-date field;
7. parse allowed decimal text with `decimal.Decimal`;
8. reject exponent notation, sign prefixes, whitespace, commas, underscores,
   non-finite values, zero, and negative values;
9. parse volumes as positive base-ten integers; and
10. reject malformed rows rather than skipping, filling, carrying, or
    repairing them.

For option flow:

- decode only the frozen relation columns;
- validate `response_sha256` only as 64 lowercase hexadecimal characters;
- retain no response hash in a relation, prompt, or output; and
- leave every other field as unparsed text after exact whole-file/header hash
  validation.

The exact sorted intersection of the three date sets is the only CEFS date
set. The term and tail `VIX_close` values must be decimal-exact on every common
date.

## Exact edge and sequence construction

Import the frozen state labels, edge order, comparison levels, formulas,
position contexts, controls, and fixed clock from the preregistration module.
The runner may not duplicate them with different literals.

For each common date after the first:

- construct the twelve exact relation edges against the immediately preceding
  common date;
- compare positive ratios by exact cross multiplication only;
- persist no decoded numeric source value; and
- form one current twelve-edge signature in frozen order.

For each common date with five complete edge states:

- construct the sequence `EARLIEST, EARLY, MIDDLE, LATE, CURRENT`;
- serialize exactly three primary position templates in the frozen position
  order;
- derive entry at calendar `D+1 09:35 America/New_York`;
- derive exit at entry plus exactly 288 five-minute bars; and
- create the schedule before any policy action exists.

## Reservation and role containment

Sort candidate schedules by `(entry_utc, observation_date)`. Accept a schedule
when its entry is greater than or equal to the prior accepted exit. Suppress,
never queue, a strict overlap. Equality is a direct rebalance.

Reservation occurs globally before role containment. Every sequence-ready
source date remains in the source artifact with:

```text
ACCEPTED
SUPPRESSED_OVERLAP
```

An accepted interval is `model_eligible` only when entry and exit both lie
inside exactly one role:

```text
TRAIN  [2020-01-01T00:00:00Z,2022-01-01T00:00:00Z)
TEST   [2022-01-01T00:00:00Z,2023-01-01T00:00:00Z)
EVAL   [2023-01-01T00:00:00Z,2024-01-01T00:00:00Z)
```

An accepted cross-boundary candidate remains an action-independent reservation
record marked `ROLE_CROSSING`, but it is not a policy/economic interval:

- it deterministically holds `TARGET_FLAT`;
- it invokes no model;
- it opens no market or funding row;
- it resets the next policy interval's current target to `TARGET_FLAT`; and
- it still occupies its reservation clock so split filtering cannot release a
  neighboring opportunity.

A suppressed candidate is `SUPPRESSED`, invokes no model, opens no market or
funding row, and is not model-eligible.

For this contract, a **complete primary schedule row** means exactly:

```text
reservation_state == ACCEPTED
and role in {TRAIN, TEST, EVAL}
and model_eligible == true
```

`ROLE_CROSSING` and `SUPPRESSED` rows are action-independent audit records,
not complete primary schedule rows and never economic intervals. Thus the
boundary's Gate 3 `no role-crossing interval` condition is evaluated over the
complete primary schedule set and must have a zero crossing count.

## Primary source artifact

The pass source CSV contains one audit row per sequence-ready source date,
including the complete primary schedules and the non-economic suppressed or
role-crossing records, with exact columns:

```text
observation_date
available_utc
entry_utc
exit_utc
reservation_state
role
model_eligible
current_signature
sequence_signature
prompt_target_flat
prompt_target_long
prompt_target_short
```

Encoding:

- timestamps are `YYYY-MM-DDTHH:MM:SSZ`;
- booleans are lowercase `true|false`;
- signatures join frozen categorical levels with one ASCII `|`;
- sequence signature joins five current signatures with one ASCII `/`;
- prompt fields contain the exact 61-line UTF-8 prompt with final newline;
- CSV uses frozen column order, Python minimal quoting, `\n`, and UTF-8; and
- gzip omits filename and uses `mtime=0`.

No decoded source number, hash metadata, market value, return, reward, action,
or economic metric may be persisted.

## Source-only controls

Controls are built for every complete primary schedule row and all three
position contexts. Apply the eight frozen transformations exactly. This is
exactly the boundary's control scope; suppressed and role-crossing audit rows
are not complete primary schedules.

The long-form deterministic control CSV has columns:

```text
observation_date
position_context
control_id
control_prompt
control_prompt_sha256
semantic_difference
```

`semantic_difference` is lowercase `true|false`. It compares ordered
state/edge values, not serialized line order. Thus `group_order_rotation` is
expected to be byte-different while semantically equal.

The control order is schedule order, frozen position order, then frozen
control order.

## Gate metrics and first-stop sequence

Compute only the next gate after the previous gate passes.

### Gate 1 — authority and forbidden access

- every authority/seal/hash/commit/header binding passes;
- runner/test bytes match the seal's committed runner revision and the current
  worktree is clean; HEAD may be the later commit that seals the execution;
- the pre-run artifact-state check found no existing terminal or partial
  result; and
- every forbidden-access counter is zero.

### Gate 2 — schema and chronology

- parser/type/date/order/positivity rules pass;
- exact common coverage is 2020-01-02 through 2023-12-29;
- exact common-date count is 1,006;
- maximum common-date gap is at most ten calendar days;
- term/tail VIX equality holds on every common date; and
- no post-2023 non-date field is parsed.

### Gate 3 — schedule support

Using only complete primary schedule rows:

- total at least 920;
- each entry year 2020, 2021, 2022, and 2023 at least 230;
- every entry quarter at least 50;
- no accepted intervals overlap;
- every accepted interval is exactly 288 five-minute bars; and
- complete-primary role-crossing count is exactly zero;
- append/delete of later source rows cannot alter a prior formed interval.

### Gate 4 — primitive edge support

Roles use model-eligible rows:

```text
train/development = TRAIN
test              = TEST
eval              = EVAL
```

For `TERM_FRONT_LEVEL` and `TERM_BACK_LEVEL`, each role requires at least two
observed levels and no level above 0.98 share.

For each of the ten change edges, each role requires:

- `LOWER` share at least 0.10;
- `HIGHER` share at least 0.10; and
- no level above 0.88 share.

### Gate 5 — diversity and stability

Using only complete primary schedule rows, for each entry year:

- at least 40 distinct current signatures;
- largest current-signature share at most 0.15;
- at least 0.80 of complete sequence signatures unique; and
- largest sequence-signature share at most 0.02.

For each edge/level pair, absolute share drift from TRAIN to TEST and TRAIN to
EVAL is at most 0.25.

### Gate 6 — control construction

- each mask differs bytewise on every row;
- group-order rotation differs bytewise on every row;
- reverse-sequence byte difference share is at least 0.95;
- stale-current semantic difference share is at least 0.35; and
- within-group rotation semantic difference share is at least 0.50.

Every eligible schedule has exactly 24 controls: three positions times eight
control identities.

### Gate 7 — determinism and append replay

Build twice and require canonical source/control rows byte-identical. Rebuild
physical prefixes ending before:

```text
2021-01-01
2022-01-01
2023-01-01
2024-01-01
```

For each prefix, compare every schedule whose observation date exists in that
prefix and whose entry/exit were already fully formed from its own observation
date. Later rows may append schedules but cannot change any existing edge,
sequence, reservation, role, prompt, or control.

During the one full physical parse, seal immutable prefix record snapshots
immediately before the first row at each cutoff enters the parser. A prefix
rebuild receives only that sealed truncated record container. It may not
access, scan, count, or inspect any later row or date field. Later physical
rows therefore cannot participate in prefix reconstruction.

Also append one in-memory synthetic valid common row strictly after the frozen
end and prove all pre-append records byte-identical. This row is constructed
only after the physical parser has closed; it does not open a post-2023 source
row or change a forbidden counter. Its fields are fixed positive exact
decimals/integers and are never used in support statistics.

## Failure and pass semantics

Before Gate 1 and before any source access:

- an existing valid terminal pass or rejection report causes an idempotent
  terminal return with no writes and no new gate evaluation;
- an unreadable, hash-drifted, conflicting, or partial terminal state causes a
  hard abort with no writes;
- pass CSVs without the terminal pass report are a partial terminal state and
  cause the same no-write hard abort; and
- this pre-run refusal is not converted into a new candidate rejection.

At the first failure, stop later metrics and atomically write only:

```text
results/cboe_edge_flip_sequence_policy_source_rejection_2026-07-25.json
```

Exact failure action:

```text
retire_cefs_d1_unchanged_before_outcomes
```

Do not write either pass CSV or the pass report.

Only after every gate passes, atomically write deterministic source/control
gzip files and then:

```text
results/cboe_edge_flip_sequence_policy_source_support_2026-07-25.json
```

Exact pass action:

```text
authorize_economic_rllm_evaluator_freeze_only
```

Neither result is profitability evidence.

## Report contract

The report or rejection binds:

- boundary, contract, preregistration, producer, runner, tests, seal, sources,
  manifests, audits, and exact hashes;
- gates in executed order only;
- parser counters;
- common-date and schedule metrics if reached;
- edge counts/shares if reached;
- diversity/stability metrics if reached;
- control metrics if reached;
- replay metrics if reached;
- canonical source/control row hashes if built;
- all forbidden counters;
- pass output paths/hashes/row counts only on pass; and
- bounded exception type/message on implementation failure.

The report must not contain decoded source values, dates selected by an
outcome, returns, rewards, model values, trades, PnL, CAGR, or MDD.

## One-shot execution sequence

1. commit this contract;
2. implement runner and synthetic tests;
3. commit runner and tests together;
4. run only `create-seal`, which opens no source row;
5. commit the execution seal and its exact artifact test;
6. run all source-support tests, including seal validation;
7. execute `run` once;
8. commit the immutable pass or rejection result; and
9. only a pass may authorize economic/RLLM evaluator work.
