# PSIM-D7 Source-Support Implementation Contract

Date: 2026-07-27 KST
Scope: source representation only. Market, model, reward, trade, PnL,
profitability, and outcome access remain forbidden.

## Frozen predecessor and rebased authority

- PSIM-D7 preregistration is the canonical artifact at
  `results/protocol_specification_intent_maturity_d7_preregistration_2026-07-26.json`
  (SHA-256
  `e9402b984232a9c30a5bc427ee8b828b4e61b7f355746e36ee5fe986be3ae79d`,
  manifest
  `7b6ac7c514bd3c0c8fad54a69707bb682a8a97bae020a603940c3410ddea378d`).
- The current rebased preregistration commit is
  `b26c92acf053553c5f4b02eee6b6229229d7e737`.
- The pre-rebase hash-bound PSIM authority chain remains reachable from
  `archive/psim-authority-pre-rebase-20260727-107c6fe`. The D7 runner binds
  the byte-identical rebased paths used by the current branch and separately
  revalidates the canonical preregistration's frozen predecessor hashes.
- Seal verification is dual-epoch rather than suppressing rebase-sensitive
  failures. The complete inherited D1-D7-preregistration battery runs in a
  fresh detached worktree at pre-rebase commit
  `107c6fe172c2dcba604b06ca67f23f136507b6e9`; the current D7 mechanism,
  preregistration, and evaluator battery then runs at the implementation
  commit. Both epochs require zero failures, skips, errors, expected failures,
  or unexpected passes, exact frozen pass counts, a pytest environment with
  ambient selection/plugin controls removed, and removal of the temporary
  worktree.
- The terminal PSIM-D6 result remains a rejection. PSIM-D6 must not be rerun,
  repaired, or reused as a D7 source candidate.
- D6 acquisition, first-parent traversal, path grouping, availability
  schedules, relation controls, thirteen-gate order, exact ERC restoration,
  and lossless UTF-8 chunk transport remain unchanged.

## Exact D7 Bitcoin grammar overlay

The overlay applies only to Bitcoin proposal blobs. Ethereum decoding remains
the frozen D5/D6 semantics.

For every Bitcoin blob:

1. verify the Git blob object identity before parsing;
2. invoke the frozen initial BIP header parser first;
3. permit later-header fallback only for the exact initial failure
   `ValueError: PSIM malformed header line`;
4. require balanced, non-nested exact `<pre>` fences;
5. require exactly one parseable later BIP header in the entire normalized
   document;
6. require its BIP number to equal the path proposal number;
7. reject a prior fenced block, ambiguous candidate, mismatched identity, or
   any other malformed grammar;
8. preserve the prefix, selected header, body, full normalized lines, and raw
   SHA-256 without rewriting source text; and
9. parse dependency fields only as bare decimal or exact uppercase
   `BIP-<decimal>` tokens, with D6 SP/HTAB trimming, positive/non-self/unique
   constraints, and the frozen dependency-count bound.

Removing the `BIP-` prefix is an integer-edge interpretation only. The
model-visible causal text retains the original source token.

Each accepted Bitcoin blob carries a deterministic mechanism receipt hash.
The official Gate-4 class roster is frozen to:

- `D4_VALID`: 426;
- `D7_BIP_LATER_HEADER`: 7; and
- `D7_BIP_PREFIXED_DEPENDENCY`: 1.

Unknown or ambiguous grammar produces the typed event failure
`ERROR_UNKNOWN_GRAMMAR`; strict normalization failure remains
`ERROR_STRICT_UTF8`. Materialization continues through the complete event
roster and then rejects before any model or outcome access.

## Frozen D6 representation mechanisms

- Exact receipt-bound ERC restoration remains the D6 mechanism and protocol.
- Causal model text remains strict UTF-8, greedily partitioned into at most
  eight contiguous 8,192-byte chunks.
- A ninth chunk fails closed; truncation and summarization are forbidden.
- Chunk aggregation policy remains
  `UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION`.
- Audit identities, raw source, full normalized source, mechanism receipts,
  exception messages, and migration episodes never enter model payloads or
  rejection publication.

## Fresh source and execution discipline

- The only official source root is `/tmp/psim-d7-source`.
- It must be absent before an official attempt. Existing, stale, symlinked,
  D1-D6, alternate-object, worktree, or cache roots are forbidden.
- `self-check` and tests are synthetic/local-only and do not authorize an
  official source run.
- The runner, evaluator tests, and this document must share one reviewed,
  clean implementation commit.
- A direct-child commit may add only the canonical D7 execution-seal JSON and
  its seal test.
- The official D7 source run remains forbidden until that direct-child seal
  validates at the exact current HEAD.

No official PSIM-D7 source execution occurs in this implementation unit.
