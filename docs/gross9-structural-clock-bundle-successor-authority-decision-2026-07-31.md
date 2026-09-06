# Gross9 structural clock bundle successor authority decision — 2026-07-31

## Decision and status

Freeze `G9CB-2` as a new candidate-independent structural-clock
infrastructure identity.

`G9CB-2` is not `G9CB-1D`, a v3 preregistration, a repair of the failed
`G9CB-1` chain, or a retry of a consumed production attempt. It is the new
infrastructure identity, decision, preregistration, claim, sentinel, and
publication required by the single-use rule in:

```text
docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
```

This decision adopts without alteration the candidate-independent economic
object and runtime mechanics frozen by:

```text
docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
```

The five sleeves, weights, domain, interval geometry, Rank7 learner and
source-routed exits, source paths and hashes, environment, static closure,
isolated runtime, exactly-two-pass rule, one-use anonymous-pipe
capabilities, counters, deterministic serialization, publication allowlist,
manifest-last rule, one-shot terminal rule, candidate independence, and
prohibition on portfolio economics and overlap computation are unchanged.

Only these successor changes are authorized:

1. the new identity and new v1-under-`G9CB-2` protocol versions;
2. a new authority decision and exact `A2 -> Q2 -> P2 -> C2 -> D2` history;
3. distinct successor artifact paths;
4. immutable bindings for both failed `G9CB-1` preregistrations;
5. a closed optional Git-metadata pair contract that distinguishes tracked,
   untracked, and canonical external regular files; and
6. names whose old `G9CB-1` spelling would otherwise misstate the new
   identity, including the terminal exception/action and bytecode-cache
   prefix.

This decision is not operative merely because this file exists. It must be
committed and pushed alone as `A2`. A later clean pushed protocol commit
`Q2` must implement and test every requirement below before a successor
preregistration may be written.

## Recorded `G9CB-1` preclaim failure

The immutable operative-v2 preregistration attempt for `G9CB-1` is:

```json
{
  "path": "results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json",
  "path_type": "regular_file",
  "sha256": "5e6fe5e23f78103e5e4c6a288bb12df5f6aaa4e00028a211a175221a58b48e84",
  "git_blob": "6bf7c4fd62818c639b11da943f25353946d141b6",
  "git_mode": "100644",
  "filesystem_mode_octal": "0444",
  "seal_commit": "c5c5120cb5af931294524d4833f44440f8949327",
  "protocol_implementation_commit": "d4ebec8f151fc5db6d318734ca0b6a79afaad1e1",
  "protocol_version": "gross9_structural_clock_bundle_preregistration_v2",
  "manifest_hash": "e83d2bec1300c34401931c2b45c6c0b8715f4237eba0ae01811c665718b11a54",
  "status": "historical_nonoperative_preclaim_git_metadata_contract_failure"
}
```

Its metadata-only claim preflight authenticated the preregistration and the
`Q -> P` seal, entered `_validate_regular_hashed_inputs()`, opened the
following regular file only as opaque bytes, read its exact 66,696,659 bytes
for SHA-256 authentication, and then failed:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz

TerminalG9CB1Failure:
incomplete bound input Git metadata:
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz
```

The source binding correctly contained both optional Git fields as JSON null
because the repository-relative source is untracked:

```json
{"git_blob": null, "git_mode": null}
```

The producer intentionally emitted that pair, but the builder interpreted
every present pair as two required strings. This was a closed-contract
contradiction, not source drift.

The same preregistration contains seven paired-null source bindings:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz
data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz
data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz
/tmp/btcusdt_open_interest_5m_2020_2026.csv
data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl
data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl
data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl
```

At failed-preflight commit `c5c5120c`, each of the six repository-relative
paths produced no `git ls-files --error-unmatch` match and no `git ls-tree
HEAD` entry. The one absolute path is canonical and outside the repository.

No gzip stream was decompressed. No CSV or JSONL header, row, field, source
value, model array, model-history value, Gross9 clock row, comparator-clock
row, candidate row, return, PnL, funding cash, CAGR, MDD, economic rank,
candidate metric, or overlap metric was opened or computed. No generic
runtime module was imported.

The following `G9CB-1` paths remained absent:

```text
results/gross9_structural_clock_bundle_access_claim_2026-07-31.json
results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_manifest_2026-07-31.json
```

There was therefore no `G9CB-1` claim, sentinel, worker, publication, or
consumed production attempt. Nevertheless, the `G9CB-1C` single-use rule
makes the v2 protocol failure non-restorable under identity `G9CB-1`.
`G9CB-1` is closed as failed, immutable, and non-operative.

## Immutable failed-predecessor bindings

The `G9CB-2` preregistration must authenticate, in this order, both complete
failed predecessor objects under:

```text
bindings.failed_predecessor_preregistrations
```

The first object is:

```json
{
  "path": "results/gross9_structural_clock_bundle_preregistration_2026-07-31.json",
  "path_type": "regular_file",
  "sha256": "3580a3663b54509d004dc2edac0f18ff9c79cb80b199e8de5e9b1a9feb98d472",
  "git_blob": "61992d68beff0da255b002776d0efdb4ef96ab93",
  "git_mode": "100644",
  "filesystem_mode_octal": "0444",
  "seal_commit": "3810a3b7e24b83591866f2ccf9b63167795718c5",
  "protocol_parent_commit": "05437c3d8f2a9c556fde4e950a815b9901f7fc98",
  "protocol_version": "gross9_structural_clock_bundle_preregistration_v1",
  "manifest_hash": "5ddf4c5c0aef42e1fb24defa78fccbd4142c8274bc22fd0a7d7e97fa9e8bb9bb",
  "status": "historical_nonoperative_preclaim_validation_failure"
}
```

The second object is the exact v2 object recorded in the preceding section.
For each predecessor the producer and builder must verify:

- regular-file and no-symlink type;
- exact SHA-256, Git blob, Git mode, and filesystem mode `0444`;
- canonical compact JSON with one trailing LF;
- exact protocol version and internal manifest hash;
- exact implementation/seal ancestry and artifact-only seal diff;
- exactly one addition commit for its path;
- continued tracked presence at `HEAD`; and
- exact failure status.

The v1 artifact keeps its sealed two-row `G9CB-1A`/`G9CB-1B` amendment list.
The v2 artifact keeps its sealed three-row
`G9CB-1A`/`G9CB-1B`/`G9CB-1C` list. Neither predecessor may be accepted as
the active preregistration or rewritten to describe `G9CB-2`.

## Closed optional Git-metadata pair contract

The keys `git_blob` and `git_mode` are optional only as one closed pair.
For every qualifying path/SHA binding, pair shape, path classification, and
stage-zero index/HEAD classification must finish before any bound file bytes
are opened:

1. neither key is present: the binding makes no Git metadata declaration;
2. both keys are strings: the path must be normalized repository-relative,
   `git_blob` must be 40 lowercase hexadecimal characters, `git_mode` must be
   `100644`, `git ls-files --stage -- <path>` must return exactly one
   stage-zero row whose path, mode, and blob equal the declaration,
   `git ls-files --error-unmatch -- <path>` must succeed with the exact path,
   and `git ls-tree HEAD -- <path>` must equal the same path, mode, and blob;
3. both keys are exact JSON null:
   - for a normalized repository-relative path,
     `git ls-files --error-unmatch -- <path>` must fail with no matched path
     while `git ls-files --stage -- <path>` and
     `git ls-tree HEAD -- <path>` must both be empty; or
   - for an already-canonical absolute no-symlink regular-file path whose
     resolved path is strictly outside the canonical repository root, no
     Git-tree claim is made and no repository lookup is required.

An absolute path equal to or contained by the canonical repository root is
terminal and must instead be represented as normalized repository-relative
POSIX text.

Every other form is terminal before opaque-byte authentication:

- exactly one key present;
- null paired with a string;
- a boolean, integer, float, list, mapping, or other type;
- string metadata on an absolute path;
- paired null for a tracked repository-relative path;
- string metadata for an untracked repository-relative path;
- a Git mode other than `100644`;
- a malformed blob; or
- a stage-zero index/HEAD disagreement.

Repeated bindings retain the inherited conflict rule. A repeated path may
omit both fields where its schema never declares Git metadata, but any two
present declarations must be identical. A null pair and a string pair for
the same path conflict terminally.

During the subsequent inherited single no-follow opaque-byte authentication,
the builder must derive the Git blob ID from the same raw bytes already read
for size and SHA-256:

```text
SHA1(b"blob " + ASCII(decimal byte length) + b"\0" + raw bytes)
```

For a declared string pair, that derived blob ID must equal the declared,
stage-zero, and `HEAD` blob before the binding is accepted. The builder must
not reopen the path to compute the Git blob. A mismatch is terminal before
decode, import, claim creation, or any production output.

The producer's `_optional_git_metadata()` must deterministically emit:

- the exact string pair for a tracked normalized repository-relative regular
  file after proving stage-zero index and HEAD-tree identity and authenticating
  the worktree blob from opaque bytes without decoding them;
- the exact null pair for an untracked normalized repository-relative
  regular file after proving index and HEAD absence; and
- the exact null pair for an already-canonical absolute no-symlink regular
  file strictly outside the canonical repository root.

The builder must first materialize and validate the complete pair inventory,
then perform inherited no-follow regular-file, size, and SHA-256
authentication. It must never parse, decompress, import, or decode a bound
value during this preflight.

## Successor identity, versions, and paths

The exact successor values are:

```text
identity = G9CB-2
preregistration protocol =
  gross9_structural_clock_bundle_g9cb2_preregistration_v1
builder/publication protocol =
  gross9_structural_clock_bundle_g9cb2_v1
terminal action =
  TERMINAL_G9CB2_ATTEMPT_CONSUMED_NO_RETRY
```

The terminal exception class is `TerminalG9CB2Failure`. The fixed absent
bytecode-cache prefix is:

```text
results/.g9cb2-bytecode-cache-disabled
```

The active successor paths are exactly:

| Role | Path |
|---|---|
| successor authority decision | `docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md` |
| preregistration | `results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json` |
| access claim | `results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json` |
| attempt sentinel | `results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json` |
| worker ledger pass 1 | `results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json` |
| worker ledger pass 2 | `results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json` |
| canonical CSV gzip | `results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz` |
| final manifest | `results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json` |

The staged CSV/core/receipt names and every embedded identity or protocol
literal must use `G9CB-2` naming where identity is represented. The canonical
CSV `identity` column is always `G9CB-2`.

No `G9CB-1` output path may be created, reused, renamed, deleted, or treated
as a `G9CB-2` alias. All failed predecessor files and authority documents
remain immutable.

The top-level `authority_decision` binding in the active successor
preregistration names this new decision and its standalone commit `A2`.
The inherited ordered `authority_amendments` rows remain exactly
`G9CB-1A`, `G9CB-1B`, and `G9CB-1C`; they authenticate inherited mechanics
and do not make `G9CB-2` a continuation of the failed identity. This decision
and all four predecessor authority documents must appear in the exact
protocol inventory. The claim binds the complete successor preregistration
path, SHA-256, and manifest hash, and its exact protocol-blob inventory
therefore carries this decision through the sentinel, receipts, cores, and
final manifest without adding a second authority field to those inherited
schemas.

## Exact commit topology

Let:

- `F2` be the immutable failed-v2 preregistration seal
  `c5c5120cb5af931294524d4833f44440f8949327`;
- `A2` be the standalone pushed commit adding only this decision;
- `Q2` be the clean pushed successor protocol implementation commit;
- `P2` be the clean pushed direct child adding only the successor
  preregistration;
- `C2` be the clean pushed direct child adding only the successor claim; and
- `D2` be the clean pushed direct child adding only the five successor
  publication files.

Every seal uses the existing exact branch:

```text
codex/gross9-structural-clock-bundle-20260731
```

`A2` must satisfy:

```text
A2 has exactly one parent
first_parent(A2) == c5c5120cb5af931294524d4833f44440f8949327
diff(F2, A2) ==
  A docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md
HEAD == A2 == @{upstream}
the worktree and index are clean
```

`Q2` must satisfy:

```text
Q2 has exactly one parent
first_parent(Q2) == A2
diff(A2, Q2) ==
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py
HEAD == Q2 == @{upstream}
the worktree and index are clean
```

The rows are a sorted exact set. `Q2` may change only identity/version/path
literals, successor/predecessor authority and topology bindings, the closed
Git-metadata pair validator/producer, names that falsely retain `G9CB-1`, and
tests for those changes. It may not alter any source path/hash, domain,
sleeve, weight, side, hold, barrier, Rank7 formula/learner/policy, feature,
counter event, environment lock, dependency, serializer geometry, runtime
guard, capability, two-pass rule, economic prohibition, overlap prohibition,
or publication algorithm.

No successor preregistration, claim, sentinel, worker ledger, canonical CSV,
or final manifest may exist at `Q2`.

After all tests pass, the successor preregistration may be built only while
`HEAD == Q2 == @{upstream}` and the worktree/index are clean. `P2` must then
satisfy:

```text
P2 has exactly one parent
first_parent(P2) == Q2
diff(Q2, P2) ==
  A results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json
HEAD == P2 == @{upstream}
the worktree and index are clean
```

Claim preflight derives `P2` from `HEAD`, authenticates both failed
predecessors and the complete pair inventory before opaque-byte hashing,
proves the exact `Q2 -> P2` diff, and creates no output. Only after that
preflight passes may the write-once claim be created. `C2` must satisfy:

```text
C2 has exactly one parent
first_parent(C2) == P2
diff(P2, C2) ==
  A results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json
HEAD == C2 == @{upstream}
the worktree and index are clean
```

The sentinel is then the first operation that consumes the one production
attempt. It must precede every generic import and value-row access. The exact
publication diff is:

```text
D2 has exactly one parent
first_parent(D2) == C2
diff(C2, D2) ==
  A results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz
  A results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json
HEAD == D2 == @{upstream}
the worktree and index are clean
```

No authority, protocol, test, failed predecessor, or preregistration byte may
change between `P2`, `C2`, and `D2`.

## Required regression evidence

Before `Q2` is sealed, tests must prove at least:

1. this decision is the sole `A2` diff and direct child of `F2`;
2. both failed predecessor artifacts remain byte-identical, tracked, mode
   `0444`, canonical, topologically exact, and non-operative;
3. `G9CB-2` uses only the new versions and paths;
4. the exact `A2 -> Q2 -> P2 -> C2 -> D2` ancestry/diffs are required;
5. a tracked repository-relative binding accepts only an exact string pair
   whose stage-zero index row and HEAD tree row match before byte access and
   whose Git blob derived from the single opaque read also matches before
   binding acceptance;
6. an untracked repository-relative binding accepts only an exact null pair
   plus index/HEAD absence;
7. a canonical external absolute binding accepts only an exact null pair;
8. an absolute spelling of a path inside the repository fails before bound
   bytes are read;
9. staged blob/mode drift, non-stage-zero entries, and index/HEAD drift each
   fail before bound bytes are read;
10. worktree/blob drift fails during the single opaque read, before binding
    acceptance, decode, import, or output creation;
11. every mixed, partial, malformed, or path-inconsistent pair fails before
    bound bytes are read;
12. duplicate null/string declarations conflict terminally;
13. producer and builder classify the same synthetic tracked, untracked, and
    external absolute files identically;
14. malformed preregistration or Git metadata creates no claim or production
    artifact;
15. the active successor preregistration reproduces byte-for-byte at `P2`
    and `D2`;
16. the exact inherited frozen clock/runtime tests still pass; and
17. committed publication validation reconstructs and authenticates every
    successor artifact at `D2`.

Synthetic tests may use synthetic bytes and Git repositories. They may not
decode official source/model/history values or compute portfolio economics.

## Completion and terminal rule

`G9CB-2` is complete only when `A2`, `Q2`, `P2`, `C2`, and `D2` each satisfy
their exact clean-pushed topology; exactly two fresh workers produce
byte-identical authenticated outputs; the final manifest is published last;
the committed-publication verifier passes at `D2`; and the worktree remains
clean with `HEAD == @{upstream}`.

Before the sentinel, any non-restorable protocol failure closes `G9CB-2`
without value access and without mutation under the same identity. After the
sentinel, every failure is terminal:

```text
TERMINAL_G9CB2_ATTEMPT_CONSUMED_NO_RETRY
```

There is no retry, resume, repair, v2-under-`G9CB-2`, or amendment that
changes source, clock, runtime, metric, economics, overlap, or publication
semantics after failure. Any future successor requires another new
infrastructure identity and decision.

This decision authorizes no alpha candidate, source-value exploration,
candidate ranking, portfolio return, PnL, funding cash, CAGR, MDD, economic
metric, or overlap metric. Those remain forbidden until `G9CB-2` has a clean,
pushed, verified `D2` publication and a later candidate has its own
preregistration.
