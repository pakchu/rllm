# GIPR-D1 source-support implementation contract

Date: 2026-07-25

## Binding

This contract freezes the GIPR-D1 source evaluator before any governance event
incidence is opened.

```text
preregistration commit
50269717b64bbef02b8da96bb2bdc010851b47d7

results/governance_intent_payload_relation_preregistration_2026-07-25.json
SHA256 319ac26108e936331f95d047a69f739ffecfd2bdc777573100a1ce83c771c197
manifest_hash daccd86ac2b218552f6312189fe4cd0ce6c775fe3b91f37ace31e8d12e1b1645
```

No Ethereum governance log, description, payload, lifecycle incidence, BTC
market row, funding row, future return, reward, model output, action, trade,
PnL, CAGR, or strict MDD was opened while writing this contract.

## Authorized implementation

The next code unit may implement one source-only evaluator:

```text
training/build_governance_intent_payload_relation_source_support.py
tests/test_build_governance_intent_payload_relation_source_support.py
```

The implementation may use the repository's existing bounded Ethereum
JSON-RPC transport and canonical header helpers only if their exact bytes are
hash-bound by the evaluator and later execution seal.

The evaluator must expose:

```text
self-check   synthetic-only; no network
run          sealed source execution; exactly two Ethereum transports
```

`run` must require two nonempty, distinct RPC URLs. Provider host names are
not artifact identity and must never be persisted. A resumable cache may store
only protocol-hash-bound canonical JSON-RPC chunks.

## Gate-1 boundary audit before event access

Before calling `eth_getLogs`, both transports must independently reproduce:

1. chain ID `1`;
2. the five year-boundary block numbers, hashes, and timestamps;
3. every governor's empty runtime code at `first_code_block - 1`;
4. every governor's frozen runtime-code byte length and SHA256 at
   `first_code_block`;
5. every governor's frozen runtime-code byte length and SHA256 at
   `18,908,894`; and
6. a head at or beyond block `18,908,958`, the last source block plus 64
   confirmations.

Any mismatch fails Gate 1 before event incidence. No fallback address,
provider, block, code hash, or confirmation rule is allowed.

## Exact log replay

After Gate 1 passes, each transport independently replays block `9,601,459`
through `18,908,894` in ascending chunks of at most 5,000 blocks.

Each request contains:

- the exact five frozen governor addresses;
- topic zero equal to one of the exact four frozen event topics; and
- no other address, topic, receipt-time, market, or post-2023 source filter.

The canonical raw-log identity is:

```text
address
topics
data
removed
block_number
block_hash
transaction_hash
transaction_index
log_index
```

Hexadecimal addresses, topics, hashes, and data are normalized to lowercase.
Hex quantities are normalized to nonnegative integers. `removed` must be
exactly `false`. Rows are sorted by:

```text
block_number, transaction_index, log_index, address
```

Duplicate `(block_hash, transaction_hash, log_index)` identities are
forbidden. Each row must lie inside the frozen source envelope and match one
frozen address/topic. The two canonical raw-log lists and their hashes must be
identical.

## Strict dynamic ABI grammar

`ProposalCreated` contains exactly nine non-indexed values:

```text
uint256 proposal_id
address proposer
address[] targets
uint256[] values
string[] signatures
bytes[] calldatas
uint256 start_block
uint256 end_block
string description
```

The top-level head is exactly nine 32-byte words. Dynamic offsets must be
32-byte aligned, point beyond the head, remain in bounds, and define
non-overlapping tails. Dynamic-array element offsets are relative to the
first byte after the array-length word. Every byte/string length and zero
padding byte must be canonical. All parsed intervals must cover the event
data exactly: no gap, alias, overlap, or trailing byte is allowed.

Additional invariants:

- proposer and target words must be canonical left-padded addresses;
- the four action arrays must have equal length in `[1, 10]`;
- `end_block` must be strictly greater than `start_block`;
- description and signatures must decode as strict UTF-8;
- description and signatures may not contain NUL;
- description length is at most 131,072 bytes;
- each signature is at most 512 bytes;
- each calldata is at most 65,536 bytes;
- aggregate calldata is at most 262,144 bytes;
- empty signatures and empty calldata are preserved;
- values and other uint256 words remain exact nonnegative integers; and
- ordered actions may never be sorted or treated as a set.

The three lifecycle events have exact fixed heads:

```text
ProposalCanceled(uint256)             1 word
ProposalQueued(uint256,uint256)       2 words
ProposalExecuted(uint256)             1 word
```

No padding, extra word, or trailing byte is allowed.

## Canonical headers and availability

For every event block `N`, both transports must reproduce identical canonical
headers for `N` and `N+64`. The event's block hash must equal the canonical
header at `N`.

```text
event_at    = canonical timestamp at N
available_at = canonical timestamp at N+64
```

An event enters source state at the first UTC `00:00:00` boundary greater than
or equal to `available_at`. Event block time, transaction receipt time,
provider time, local time, and later indexer time are forbidden substitutes.

## Proposal and lifecycle identity

Proposal identity is exactly:

```text
(governor_address, proposal_id)
```

Lifecycle order is canonical chain order, not lifecycle-rank order. Exactly
these transitions are allowed:

```text
proposal_created -> proposal_queued
proposal_created -> proposal_canceled
proposal_queued  -> proposal_executed
proposal_queued  -> proposal_canceled
```

Every queue, execute, or cancel must reference one previously created proposal
under the same governor. Duplicate creation, queue, execute, or cancel events
are forbidden. Execution without queue, post-execution events, and
post-cancellation events are forbidden.

## Exact structural normalization

For each proposal, deterministic code derives:

### Action-count bucket

```text
1       -> ONE
2       -> TWO
3..5    -> THREE_TO_FIVE
6..10   -> SIX_TO_TEN
```

### Target role

The exact seven-address preregistered registry is used. A matched target emits
`<protocol>:<ROLE>`; every other target emits `UNKNOWN_TARGET`. Raw unknown
addresses are retained in the sealed source proposal file for reproducibility
but are forbidden model input.

Target-role mix is:

```text
ALL_UNKNOWN
KNOWN_ONLY
MIXED
```

### Selector token

- nonempty signature: exact UTF-8 signature;
- empty signature with calldata length at least four: lowercase first four
  calldata bytes as `0x????????`;
- empty signature with shorter calldata: `EMPTY_CALLDATA`.

### Selector-count bucket

```text
1 distinct       -> ONE
2 distinct       -> TWO
3 or more        -> THREE_PLUS
```

### Native-value bucket

```text
all zero         -> ZERO
all nonzero      -> ALL_NONZERO
otherwise        -> MIXED
```

### Description-structure token

Description byte length is:

```text
0                -> EMPTY
1..256           -> SHORT
257..2048        -> MEDIUM
2049..131072     -> LONG
```

It is combined with:

```text
HAS_URL | NO_URL
HAS_MARKDOWN_HEADING | NO_MARKDOWN_HEADING
```

URL detection is the ASCII case-insensitive presence of `http://` or
`https://`. Markdown-heading detection is a line beginning with one to six
`#` characters followed by one ASCII space.

## Daily source state

Decision timestamps are every UTC midnight from `2020-01-29T00:00:00Z`
through `2023-12-31T00:00:00Z`, inclusive. The 28-day history is fixed and is
never fit from source incidence.

For each protocol and decision timestamp:

- activity counts use creation availability in `(decision - W, decision]`
  for `W = 1, 7, 28, 90` elapsed days;
- each activity count is bucketed `ZERO`, `ONE`, or `TWO_PLUS`;
- the latest proposal is the most recent creation by
  `(available_at, block_number, transaction_index, log_index,
  governor_address, proposal_id)`;
- lifecycle state includes only lifecycle events available by the decision;
- proposal age is elapsed time since creation availability;
- lifecycle age is elapsed time since the latest available lifecycle event;
- age buckets are `<2d D0_1`, `<8d D2_7`, `<29d D8_28`, `<=90d D29_90`,
  otherwise `STALE_OR_NONE`;
- after 90 elapsed days the protocol state is explicitly
  `STALE_OR_NO_PROPOSAL`; and
- no field is silently forward-filled beyond that boundary.

The complete protocol card contains only:

```text
protocol
activity_1d
activity_7d
activity_28d
activity_90d
proposal_age_bucket
lifecycle_state
lifecycle_age_bucket
action_count_bucket
target_role_mix
selector_count_bucket
native_value_bucket
description_structure
same_day_event_order_fingerprint
```

The cross-protocol card contains:

```text
activity_28d_relation        COMPOUND_GT|UNISWAP_GT|EQUAL
lifecycle_relation           SAME|DIFFERENT|ONE_STALE|BOTH_STALE
action_complexity_relation   COMPOUND_GT|UNISWAP_GT|EQUAL|UNAVAILABLE
relation_transition          PERSIST|FLIP|INITIAL
```

Raw numbers, block data, timestamps, proposal IDs, unknown addresses, calldata
values, prices, returns, ranks, PnL, and model reasoning are absent from daily
cards.

## Exact split support

The preregistered per-split minimums are applied without change.

In addition, each split must satisfy:

- nonempty-description fraction at least `0.95`;
- unique normalized-description SHA fraction at least `0.90`;
- at least two action-count buckets;
- at least two target-role-mix tokens;
- at least two selector-count buckets;
- `proposal_created` plus at least two other observed lifecycle event types;
- at least 30 unique complete daily-card hashes in TRAIN and 15 in each of
  TEST and EVAL;
- no complete daily-card hash fraction above `0.35`;
- each protocol's `STALE_OR_NO_PROPOSAL` fraction at most `0.75`; and
- `BOTH_STALE` cross-protocol fraction at most `0.60`.

Fractions use exact integer cross-multiplication; binary floating-point is not
used for pass/fail.

## Future append

The evaluator adds one synthetic, schema-valid proposal after
`2024-01-01T00:00:00Z` to an in-memory copy. Filtering at the frozen source
cutoff must leave every normalized pre-2024 event, proposal, lifecycle state,
daily card, relation card, and canonical hash byte-identical.

No real post-2023 governance event may be fetched for this control.

## Six controls

Every control is deterministic and source-only. A changed fraction of at
least `0.10` is required over its defined eligible denominator in every split.

1. `protocol_label_swap`: swap Compound and Uniswap labels; denominator is all
   daily cards.
2. `within_day_event_order_reverse`: reverse canonical available-event order
   within each decision day; denominator is decision days containing at least
   two events. At least two eligible days per split are required.
3. `text_payload_pair_permutation`: rotate proposal payload bundles forward by
   one proposal within each split while descriptions remain fixed;
   denominator is all proposals.
4. `ordered_action_permutation`: reverse the action order for proposals with
   at least two actions; denominator is those proposals. At least one eligible
   proposal per split is required.
5. `lifecycle_event_rotation`: rotate non-creation event labels
   `QUEUED -> EXECUTED -> CANCELED -> QUEUED` only for a control fingerprint;
   denominator is non-creation lifecycle rows. At least two eligible rows per
   split are required.
6. `availability_plus_seven_days`: shift every event's in-memory availability
   by exactly seven elapsed days; denominator is all daily cards.

The text/payload and ordered-action controls compare exact relation
fingerprints, not market outcomes. Controls may be structurally invalid by
design; they are never promoted to source data or model input.

## Twelve gates and first-failure semantics

The evaluator executes exactly:

1. `source_schema_chronology_reconciliation`
2. `dual_transport_canonical_replay`
3. `dynamic_abi_and_payload_integrity`
4. `canonical_log_and_header_identity`
5. `lifecycle_ordering`
6. `split_proposal_action_support`
7. `cross_protocol_support`
8. `structural_vocabulary_diversity`
9. `daily_schedule_coverage_and_staleness`
10. `future_append_invariance`
11. `relation_control_sensitivity`
12. `forbidden_access_zero`

The evaluator stops logically at the first failed gate and publishes only the
terminal rejection artifact:

```text
results/governance_intent_payload_relation_source_rejection_2026-07-25.json
```

The terminal action is:

```text
REJECT_GIPR_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No pass source file, pass report, control report, market evaluator, or model
artifact may exist after rejection.

Only all twelve passing gates authorize:

```text
data/ethereum_governance_intent_payload_2020_2023/
  governance_events_2020_2023.csv.gz
  governance_proposals_2020_2023.jsonl.gz
  gipr_daily_source_cards_2020_2023.jsonl.gz

results/governance_intent_payload_relation_source_controls_2026-07-25.json
results/governance_intent_payload_relation_source_support_2026-07-25.json
```

All gzip files use deterministic metadata with `mtime=0`. JSON uses sorted
keys, UTF-8, no NaN, and one final newline. Repeated publication with the same
semantic payload is idempotent; conflicting existing artifacts fail closed.

## Synthetic self-check and execution seal

The synthetic self-check must cover:

- dual-transport canonical equivalence under order/case/hex-width changes;
- removed, out-of-boundary, wrong-topic, duplicate, and header-mismatch rows;
- valid dynamic arrays and ordered actions;
- length mismatch, misalignment, alias, overlap, invalid padding, invalid
  UTF-8, NUL, parser bounds, and trailing bytes;
- all valid and invalid lifecycle transitions;
- N+64 availability across UTC midnight;
- exact age/stale boundaries;
- raw-forbidden-field absence in daily cards;
- exact-threshold and threshold-minus-one split gates;
- collapsed vocabularies;
- future append invariance;
- all six controls;
- ordered first failure;
- forbidden counters on success and rejection paths; and
- canonical idempotent publication.

`self-check` must make zero network calls and leave no source result artifact.
The implementation, tests, preregistration, helper bytes, self-check command,
stdout hash, forbidden counters, and synthetic test evidence must then be
bound in a separately committed execution seal before `run` is authorized.

## Stop condition

Do not execute `run` until the implementation, tests, and execution seal have
all been independently reviewed, committed, and verified. After sealing,
execute the source gate exactly once. Do not repair GIPR-D1 after observing
source incidence.
