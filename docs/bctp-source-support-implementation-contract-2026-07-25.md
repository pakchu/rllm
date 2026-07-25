# BCTP-12H source-support implementation contract

Date: 2026-07-25
Status: frozen before BCTP sequence incidence, market data, funding, rewards, or
model outcomes

## Purpose

This contract freezes the one permitted source-only implementation for
`BCTP-12H`. It may establish that the already frozen BCRT block-topology
representation can form enough deterministic three-release source sequences
for a later target-position MDP. It may not establish that any sequence
predicts BTC returns.

The real source-support command is legal only after this document, the builder,
and its synthetic tests are committed and byte-identical to `HEAD`.

`BCRT-72` remains terminally retired. This implementation does not change,
relax, retry, or reinterpret BCRT's failed three-calendar-day gap gate.

## Bound inputs

The builder may open only:

- the canonical BCTP preregistration JSON, boundary, and this contract;
- the frozen BCRT preregistration source and artifact;
- the frozen BCRT source-support source and artifact;
- the terminal BCRT retirement document;
- the frozen UTXO/fee block source CSV and its source manifest;
- the frozen independent basic block-summary reference CSV; and
- its own committed source and tests for protocol binding.

It may not open BTC market bars, spot or futures prices, funding, premium,
open interest, liquidation, Kimchi, DXY, comparator, portfolio, label, reward,
return, PnL, CAGR, MDD, checkpoint, or post-2023 data.

The source and reference are loaded only through the exact hash-bound BCRT
loader and exact preregistered allowlists. Load-and-drop is forbidden.

## Exact BCRT replay

The builder reuses the hash-bound BCRT functions for:

- source/reference validation;
- causal twelve-hour bucket construction;
- the eight primitive values;
- strictly prior ranks;
- the twelve relational tokens;
- BCRT signal identifiers; and
- the legacy BCRT clock replay used only as an integrity comparator.

Before any BCTP sequence is constructed, the replay must reproduce:

```text
formed buckets       = 2918
rank-complete states = 2792
token-ready states   = 2791
```

These full-period values and the full common projection are frozen structural
integrity assertions, not support Booleans. A mismatch aborts the noncanonical
run without retiring, selecting, repairing, or authorizing BCTP. The exact
common projection of every enriched BCTP token-ready row must be byte-identical
to the row returned by the frozen BCRT token builder. The BCTP extension adds
only the positive integer `confirmation_height` required for same-release
batching. It does not alter a BCRT time, token, or identifier.

The legacy BCRT reservation and split-containment path is replayed only as an
integrity comparator. It reports whether:

- its deterministic clock SHA equals the frozen BCRT support artifact;
- its train and 2022 marginal-token reports equal the frozen artifact;
- its complete train/2022 token-support Boolean map equals the frozen
  artifact; and
- every one of those token-support Booleans is true.

Only the train/2022 report and Boolean-map comparisons enter BCTP support.
Full-clock SHA, frame, calendar, and 2023 comparator fields are report-only
because they include 2023 incidence. The failed BCRT gap Boolean and the BCRT
retirement decision are not copied into the BCTP support decision.

## Same-release batching

The BCTP action clock begins from all 2,791 token-ready BCRT source states,
before BCRT reservation or split filtering.

Rows are grouped by exact `entry_time`. One actionable source release is kept
per group using, in order:

1. latest `bucket_start`;
2. greatest `confirmation_height`; and
3. lexically greatest `signal_id`.

A duplicate full identity
`(entry_time,bucket_start,confirmation_height,signal_id)` is invalid.
Suppressed rows remain causal predecessors inside the already frozen BCRT
token chronology, but they do not consume an action or sequence warm-up slot.
After batching, actionable entry times must be strictly increasing.

## Three-release sequence

For actionable releases `A < B < C`, the sequence at `C` is:

```text
S_MINUS_2 = A's twelve source tokens
S_MINUS_1 = B's twelve source tokens
S_0       = C's twelve source tokens
```

The first two actionable releases are warm-up only. Every later actionable
release emits exactly one sequence. Source signatures hash only the 36
oldest-first categorical token lines; timestamps, identifiers, positions, and
outcomes are excluded.

Development releases are validated strictly against the frozen vocabulary.
Releases in `[2023-01-01,2024-01-01)` use the same structural clock, identity,
batch, order, schema, and signature rules but tolerate an unknown categorical
value so it can be reported under `report_only_2023` and mapped to
`TARGET_FLAT`. A missing token column, invalid clock, invalid confirmation
height, or duplicate identity remains structural corruption and fails closed.
Release times at or after 2024 are omitted from the 2020-2023 sequence
artifact.

The stored sequence has the exact preregistered schema. It contains only:

- deterministic sequence and source identifiers;
- the actionable UTC entry time;
- the exact 36 categorical source tokens; and
- the source-only signature.

It contains no confirmation height, primitive, numeric rank, position, action,
side, price, market, funding, return, label, reward, PnL, CAGR, or MDD field.

## Future-append invariance

The exact BCRT prefix replay must pass. In addition, at every completed
same-release group boundary in 2020-2022, rebuilding sequences from that
prefix must equal the corresponding prefix of the full development sequence
frame byte-for-byte. Appending a later release may add sequences but may not
change an already completed sequence. The same 2023 replay is emitted as a
report-only diagnostic.

Synthetic tests independently cover later source append, shuffled input,
same-release suppression, duplicate identity rejection, and the two-release
warm-up.

## Development-only support decision

Only sequence rows whose current `S_0` `entry_time` is in
`[2020-01-01,2023-01-01)` may enter a support Boolean.

The exact Boolean checks are:

- all BCTP/BCRT protocol and source bindings match;
- development replay counts are positive, ordered
  `formed >= rank-complete >= token-ready`, and contain at least 1,502
  token-ready states;
- the development BCTP common token-ready projection is identical to BCRT;
- the development BCRT prefix replay passes;
- the tolerant sequence builder is byte-identical to the frozen strict builder
  throughout 2020-2022;
- the frozen BCRT train/2022 token reports and train/2022 token checks
  reproduce exactly;
- every frozen BCRT train/2022 token check is true;
- 2020-2022 actionable release times are strictly increasing;
- 2020-2022 sequence count equals actionable release count minus two;
- all development sequence tokens use the frozen BCRT vocabularies;
- 2020-2022 future-append sequence replay passes;
- at least 500 sequences occur in each of 2020, 2021, and 2022;
- all twelve months are active in both 2021 and 2022;
- maximum exact 36-token source-signature share is at most 5% separately in
  2020-2022 development, 2020-2021 train, and 2022 selection; and
- the emitted sequence schema is exact and outcome-free.

Calendar-boundary maximum entry gaps are reported for development, each year,
and 2023, but are not support Booleans.

Every 2023 incidence, month, gap, signature, and vocabulary statistic is
emitted only under `report_only_2023`. It cannot change a Boolean, decision,
failure stage, threshold, sequence rule, parameter, retirement, or repair.
Unknown 2023 vocabulary is reported for the already frozen operational
`TARGET_FLAT` behavior; it does not become a source-support selection rule.

Failure retires `BCTP-12H` unchanged before market or funding access. Success
authorizes only a separately frozen economic evaluator and cheap-policy
family.

Artifact eligibility is not a caller-supplied Boolean or module-level
capability. The shared payload constructor is permanently non-authorizing.
Only the final inline block inside the guarded real-source path may replace
that state, and only after exact committed-protocol checks, source bindings,
loaded raw/reference frame schemas and row counts, and replay construction
have completed. Synthetic or injected evidence cannot call a reusable
authorization helper or construct an authorization-bearing report.

## Deterministic artifacts

Outputs:

- `data/block_clearing_target_position_mdp_sequences_2020_2023.csv.gz`
- `results/block_clearing_target_position_mdp_support_2026-07-25.json`

CSV gzip uses an empty filename and `mtime=0`. Rows are sorted by actionable
entry time and sequence id. JSON uses sorted compact keys, ASCII, finite values
only, and a terminal newline. Both outputs are write-once; an existing byte
mismatch fails closed.

The report binds:

- preregistration, boundary, implementation, BCRT, source, manifest, reference,
  and retirement hashes;
- source, bucket, rank, token, legacy-clock, batching, warm-up, append-replay,
  and sequence funnels;
- development-only Boolean checks and their first failure;
- a separate non-Boolean 2023 report;
- sequence path, schema, row count, SHA, and deterministic frame hash; and
- explicit zero counters for every forbidden outcome family.

## Mandatory synthetic tests

Before the real source may be decoded, tests must cover:

- BCTP preregistration and protocol binding without decoding source rows;
- exact BCRT common-row preservation while adding confirmation height;
- every same-release tie-break level and duplicate full-identity rejection;
- two-release warm-up and oldest-first sequence order;
- future-append and shuffled-input determinism;
- strictly increasing actionable releases;
- development-only support versus adversarial 2023 report-only incidence;
- report-only maximum gaps;
- exact 36-token signature concentration checks;
- exact schema and forbidden-column rejection;
- deterministic gzip and write-once drift rejection;
- unknown 2023 vocabulary surviving builder execution as report-only;
- synthetic/injected builds being unable to authorize market access; and
- refusal to run or open the source loader while protocol files are untracked
  or differ from `HEAD`.

The inherited BCRT regression suite remains mandatory for the reused source,
bucket, primitive, rank, token, clock, reservation, and split-replay code.
