# PSIM-D6 Source-Support Implementation Contract

Date: 2026-07-26 KST  
Scope: source representation only; no market, model, reward, trade, PnL, or
outcome access.

## Frozen predecessor

- PSIM-D6 preregistration commit:
  `a2ff036d03f01750da3527666e3be3d44737cbe2`.
- PSIM-D5 terminal rejection commit:
  `0f69f7472d89474052186bbb2b13fa8d6bf5d77f`.
- The terminal PSIM-D5 evaluator and rejection artifact are immutable. They
  must not be rerun, repaired, or reused as a candidate.
- D5 Git acquisition, first-parent traversal, path grouping, targeted batch
  hydration, object-store invariants, availability schedules, relation
  controls, and thirteen-gate order remain frozen.
- The official D6 source namespace is the fresh root
  `/tmp/psim-d6-source` with sealed ref `refs/psim-d6/sealed-tip`.
  `/tmp/psim-d5-source`, any D1-D5 object store, alternates, worktree, or
  cache must not be reused. A new attempt requires this root to be absent;
  even an empty pre-existing directory is rejected before clone or hydration.

## Exact migration restoration

There is no generic administrative-to-valid exception. A restoration is
accepted only when it reproduces one of the exact 365 three-step ERC migration
episodes bound by the frozen D5 census:

- exact proposal roster;
- exact commit, day, path, redirect class, target, blob OID, and SHA-256
  continuity;
- exact per-proposal episode receipt; and
- exact 365-receipt manifest
  `abf21a4691e4407158efc61a267cc6eaec8522751c25fa531aed6f782accdc07`.

The public authorizer accepts only an episode. A caller cannot inject a receipt
map or manifest. An accepted restoration remains a model-hidden
administrative quarantine with an empty model chunk roster. Proposal, path,
commit, blob identity, authority receipts, and quarantine reason are
audit-only. The audit receipt carries the exact three-step causal episode;
Gate 4 revalidates its frozen receipt hash and requires the final step's event,
commit, day, paths, blob OIDs, and blob SHA-256 values to equal the emitted
source event.

## Lossless model-text transport

D5 causal rows are serialized in their existing order as
`section|direction|line`. The first two pipes delimit section and direction;
everything after the second pipe is opaque line text and is never reparsed.

The serialized text is transported without truncation or summarization:

- strict UTF-8;
- greedy contiguous chunks of at most 8,192 bytes;
- at most eight chunks and 65,536 bytes per event;
- exact byte reconstruction;
- canonical partition, index, and chunk-count validation; and
- typed event error when a ninth chunk or invalid transport would be required.

Model-visible chunk fields are exactly
`normalized_text_delta_chunk`, `chunk_index`, and `chunk_count`. Full-text
hashes, chunk hashes, byte counts, and reconstruction receipts are audit-only.
Chunk-level model-output aggregation remains
`UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION`; this source implementation
does not choose or run a model.

## Gate 4 totality and publication

After successful hydration, every retained 2020-2023 proposal-group event in
all four fresh replicas receives exactly one typed audit outcome. Event-level
decode, grammar, migration, UTF-8, or chunk errors are accumulated rather than
escaping the roster loop.

Gate 4 verifies:

1. the complete four-replica roster before making a decision;
2. replica event and typed-outcome identity;
3. the exact Ethereum blob-class census and all-valid Bitcoin roster;
4. hydration/OID/decode-ledger identity;
5. event-to-outcome, event-to-chunk-receipt, and
   event-to-migration-receipt bindings;
6. exactly 365 authorized restorations and the exact frozen roster of 190
   multi-chunk Ethereum events;
7. no more than eight chunks per event; and
8. zero semantic error outcomes for a pass.

Any error or receipt mismatch rejects unchanged before market, model, or
outcome access. Rejection artifacts contain only typed error identifiers,
counts, and hashes. Raw proposal text, normalized text, chunk text, and
exception messages are forbidden from rejection publication. Full causal
migration episodes and their commit/path/blob identities are also redacted
from rejection summaries; only their count and roster hash may be published.

## Execution discipline

`self-check` and the implementation test battery are synthetic/local-only.
They do not authorize the official source run. Official execution remains
forbidden until:

1. the D6 runner, tests, and this contract share one reviewed clean
   implementation commit;
2. the complete D1-D6 PSIM regression roster passes with no skipped or expected
   failures; and
3. a direct-child commit adds only the canonical D6 execution seal and its
   seal test.

Only that direct-child seal may authorize the one permitted fresh-root D6
source attempt. No official PSIM-D6 evaluator is run as part of this
implementation unit.
