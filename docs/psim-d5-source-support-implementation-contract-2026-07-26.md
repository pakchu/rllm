# PSIM-D5 Source-Support Implementation Contract

Date: 2026-07-26 KST  
Scope: source support only; no market, model, reward, trade, or outcome access.

## Frozen predecessor and authority

- PSIM-D5 preregistration commit: `4e2b403c1f369bf2e76b5edeb1e4166b9d2f8779`.
- PSIM-D4 terminal rejection remains immutable forensic authority.
- PSIM-D4 Git acquisition, first-parent traversal, path grouping, targeted batch
  hydration, object-store invariants, and thirteen-gate order are preserved.
- The fresh source namespace is `/tmp/psim-d5-source`; no D1-D4 source-object
  store, alternates, worktree, or cache may be reused.

## Authorized semantic delta

PSIM-D5 replaces the D4 parser boundary with the frozen D5 event-semantics
probe. Each blob is exactly one of:

1. D4-valid;
2. an exact administrative ERC migration redirect;
3. a preregistered known-invalid metadata state; or
4. unknown grammar, which fails closed before any model or outcome access.

Known-invalid metadata is never repaired. Its dependency state is
`UNKNOWN_INVALID_METADATA` and its count is null. Exact administrative events
remain in the source event artifact with path/hash/audit evidence, but are
excluded from cards, counterpart history, staleness state, relation controls,
and memorization challenges.

## Source/model boundary

- Source artifact model text field: `normalized_text_delta`.
- The legacy `intent_text` field is forbidden.
- Model text is restricted to the preregistered approved body sections.
- Exact paths, blob/commit identity, path-identity hash, and full normalized
  diff hash/count are audit-only and never enter card payloads.
- Explicit known-invalid metadata states may enter cards as categorical tokens.

## Gate 4

Each replica must decode every requested blob exactly once. Receipt hashes,
OID-manifest identity, hydration/decode ledger totals, class-count totals,
replica replay, and total fraction `1.0` are verified. Ethereum class counts
must exactly equal the frozen 5,206-blob census; Bitcoin must remain entirely
D4-valid, with its exact count bound to the Gate-3-derived hydration request
count and OID-manifest hash for each replica. No post-source Bitcoin count is
selected. Any missing, extra, or unknown class or any hydration/decode mismatch
rejects the candidate before downstream gates.

## Execution discipline

`self-check` is synthetic-only. The official fresh-root run is forbidden until
runner, tests, and this contract share one clean implementation commit, an
independent review passes, and a direct-child execution-seal commit binds the
exact verification outputs. The official source evaluator is one-shot and may
not be repaired, retried, or provider-swapped after terminal publication.
