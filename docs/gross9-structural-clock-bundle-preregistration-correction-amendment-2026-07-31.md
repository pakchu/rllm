# Gross9 structural clock bundle preregistration correction amendment — 2026-07-31

## Status

This candidate-independent amendment is `G9CB-1C`. It supplements
`G9CB-1A` and `G9CB-1B` and supersedes only:

- the pre-sentinel no-repair wording that would otherwise prevent this exact
  metadata-schema correction;
- the operative preregistration version and path;
- the exact builder/publication `protocol_version` literal without changing
  any surrounding schema shape;
- the two-amendment binding cardinality;
- the protocol/preregistration commit topology; and
- the completion wording needed to preserve the superseded preregistration as
  immutable historical evidence.

Every other requirement of these authorities remains in force:

- `docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md`;
- `docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md`;
  and
- `docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md`.

In particular, this amendment does not change the `G9CB-1` identity, domain,
five sleeves, weights, clock geometry, Rank7 learner, Rank7 policy, label
formula, bundle files, sources, source hashes, environment, exactly-two-pass
rule, anonymous-pipe capabilities, one-shot sentinel, publication allowlist,
manifest-last rule, candidate independence, or prohibition on portfolio
economics and overlap computation.

This amendment is not operative merely because it exists or is committed. It
must first be committed and pushed as one standalone metadata-only authority
commit. A later protocol commit must implement and test every requirement
below before a replacement preregistration may be created.

## Preclaim failure and zero-access finding

The committed preregistration at:

```text
results/gross9_structural_clock_bundle_preregistration_2026-07-31.json
```

is canonical and reproducible as metadata, but it cannot be used as the
operative preregistration for a claim. Its bound builder rejected the
preregistration's own exact permanent prohibited counters during the first
step of metadata-only claim preflight:

```text
TerminalG9CB1Failure:
preregistration records prohibited computation: cagr_values_computed
```

The contradiction is:

```text
creation_evidence_boundary.*_computed = false
permanent_prohibited_counters.*_values_computed = 0
```

while the builder applied:

```text
if key.endswith("_computed") and item is not False:
    fail
```

The integer-zero counter name also ends in `_computed`, so a correct,
prohibited-computation counter was rejected solely because it was not the
boolean singleton `false`.

The failure occurred inside `validate_preregistration()` before
`_validate_regular_hashed_inputs()`, environment validation, static-closure
validation, claim creation, sentinel publication, generic runtime import, or
value-row access. The claim path, sentinel path, both worker-ledger paths,
canonical CSV path, and final-manifest path were all absent after the failure.

No source row, model array, history row, Gross9 clock row, comparator-clock
row, candidate row, return, PnL, funding cash, CAGR, MDD, economic rank,
candidate metric, or overlap metric was opened or computed to identify or
document this defect. Only committed authority text, protocol source,
repository/Git metadata, and the committed preregistration metadata were
inspected.

Because no attempt-consumed sentinel exists, the post-sentinel no-retry rule
was not triggered. `G9CB-1C` narrowly authorizes correction of this
pre-sentinel metadata contradiction without authorizing a production retry:
there has been no production attempt.

## Superseded preregistration evidence

The old preregistration remains immutable at its existing path. It must not be
deleted, renamed, replaced, modified, untracked, or used to create a claim.
Its exact historical binding is:

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

The replacement preregistration must authenticate this complete object under:

```text
bindings.superseded_preregistration
```

It must verify the old path's regular-file type, SHA-256, Git blob, Git mode,
filesystem mode, seal ancestry, protocol version, manifest hash, canonical
JSON bytes, and continued tracked presence without opening any source, model,
history, clock, candidate, return, PnL, or metric value. The exact historical
topology is:

```text
first_parent(3810a3b7e24b83591866f2ccf9b63167795718c5) ==
  05437c3d8f2a9c556fde4e950a815b9901f7fc98
diff(
  05437c3d8f2a9c556fde4e950a815b9901f7fc98,
  3810a3b7e24b83591866f2ccf9b63167795718c5
) ==
  A results/gross9_structural_clock_bundle_preregistration_2026-07-31.json
```

All later claim, sentinel, worker, per-pass core, and final-manifest authority
flows through the replacement preregistration binding and the direct
`G9CB-1C` amendment binding. They must not bind the old preregistration as an
operative preregistration.

## Corrected closed zero-access schema

Suffix matching does not authorize any field. The replacement builder must
validate exact location-aware objects and reject every unknown, additional, or
misplaced computation/counter key.

`creation_evidence_boundary` retains exactly these inherited keys:

```text
source_bytes_hashed
source_value_rows_opened
pre2025_anchor_value_rows_opened
runtime_modules_imported
esdi_runtime_or_private_invocations
model_files_loaded
model_or_history_rows_opened
market_rows_opened
open_interest_rows_opened
funding_rows_opened
premium_rows_opened
outcome_dependent_ohlc_rows_opened
gross9_clock_rows_opened
candidate_rows_opened
comparator_clock_rows_opened
portfolio_return_or_pnl_computed
funding_cash_computed
economic_metric_computed
candidate_or_overlap_metric_computed
```

`source_bytes_hashed` must be the boolean singleton `true`. The fourteen keys
from `source_value_rows_opened` through `comparator_clock_rows_opened` must
each satisfy:

```text
type(value) is int
value == 0
```

The final four named computation declarations must each be the boolean
singleton `false`.

`permanent_prohibited_counters` retains exactly these eleven inherited keys:

```text
pre2025_anchor_value_rows_opened
candidate_rows_opened
comparator_clock_rows_opened
portfolio_return_values_computed
portfolio_pnl_values_computed
funding_cash_values_computed
cagr_values_computed
mdd_values_computed
economic_rank_values_computed
candidate_metric_values_computed
overlap_metric_values_computed
```

Every value in that exact object must satisfy:

```text
type(value) is int
value == 0
```

`pre2025_anchor_boundary` retains exactly its four inherited fields.
`pre2025_anchor_bytes_hashed` and
`pre2025_anchor_git_blob_authenticated` must be `true`;
`pre2025_anchor_json_parsed` must be `false`; and
`pre2025_anchor_value_rows_opened` must be an exact integer zero.

`candidate_independence` retains exactly its four inherited fields.
`candidate_identity_present` and `candidate_artifacts_opened` must be `false`;
`comparator_clock_rows_opened` must be an exact integer zero; and
`comparator_clocks_preseen_by_research_program` must be `true`.

Within the operative v2 preregistration object only, the eight keys ending in
`_values_computed` may appear only in `permanent_prohibited_counters`. The
three row-opened permanent keys may appear only at their exact inherited
preregistration locations:

```text
pre2025_anchor_value_rows_opened:
  creation_evidence_boundary
  permanent_prohibited_counters
  pre2025_anchor_boundary
candidate_rows_opened:
  creation_evidence_boundary
  permanent_prohibited_counters
comparator_clock_rows_opened:
  creation_evidence_boundary
  permanent_prohibited_counters
  candidate_independence
```

Every creation computation key may appear only in
`creation_evidence_boundary`. Any occurrence outside those exact locations is
terminal. An unknown or misplaced mapping key ending in `_computed` or
`_values_computed` is terminal; the suffix itself never makes a key valid.
This location rule does not alter any inherited downstream claim/sentinel
zero-access field or `G9CB-1B`'s exact
`evidence_boundary.prohibited_output_counters` placement in each per-pass core
and the final manifest; those downstream schemas and placements remain
unchanged.

Python booleans must not satisfy integer-counter fields. Integer zero must not
satisfy boolean fields. A negative integer, positive integer, float, decimal
string, null, container, or alternate false-like value is terminal before
hashed-input authentication.

Regression tests must prove at least:

1. the exact replacement preregistration passes the corrected validator;
2. every authorized integer counter accepts only built-in integer zero;
3. each such counter rejects `false`, `0.0`, `"0"`, null, and containers;
4. integer `0` in any boolean declaration fails;
5. every unknown or misplaced computed/counter key fails;
6. a malformed preregistration fails before hashed-input, environment,
   closure, Git-seal, or output-creation operations;
7. preflight-only validation creates no claim or production artifact;
8. the old v1 artifact is rejected as operative because of its exact
   path/version/status, not because its valid integer-zero counters are
   reclassified as computations;
9. the historical v1 artifact remains byte-identical, tracked, mode `0444`,
   and non-operative;
10. `A`, `Q`, `P`, `C`, and `D` satisfy their exact ancestry and diffs; and
11. the active v2 artifact reproduces at `P` and at `D`.

No economic or overlap computation is authorized by accepting an exact
integer zero.

## Replacement versions and planned-path extension

The active replacement versions are exactly:

```text
preregistration protocol =
  gross9_structural_clock_bundle_preregistration_v2
builder/publication protocol =
  gross9_structural_clock_bundle_v2
```

The builder/publication value replaces the inherited
`gross9_structural_clock_bundle_v1` literal at every builder-owned
`protocol_version` field in the active chain: claim, sentinel, each worker
consumption ledger, each pass receipt, each per-pass core, and final manifest.
Only this exact scalar value changes; each surrounding object, key order, hash
rule, receipt rule, and publication schema shape remains inherited and exact.

The sole active preregistration path becomes:

```text
results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json
```

The old path remains historical evidence only. A claim naming the old path,
old preregistration protocol version, old manifest hash, or old artifact SHA
is terminal before source/runtime/value access.

`G9CB-1C` supersedes the prior planned-path exclusivity only to append:

| Stage | Path | Contract |
|---|---|---|
| preregistration-correction authority | `docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md` | Standalone metadata-only `G9CB-1C` authority |
| historical preregistration evidence | `results/gross9_structural_clock_bundle_preregistration_2026-07-31.json` | Immutable, tracked, non-operative v1 evidence |
| active preregistration artifact | `results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json` | Write-once operative v2 preregistration |

No other implementation, test, claim, sentinel, ledger, CSV, manifest,
sidecar, cache, or publication path is added or renamed.

## Ordered amendment binding

The canonical ordered authority binding becomes:

```text
authority_amendments[0].identity = "G9CB-1A"
authority_amendments[1].identity = "G9CB-1B"
authority_amendments[2].identity = "G9CB-1C"
```

The existing exact row schema remains:

```text
identity
path
path_type = "regular_file"
sha256
git_blob
git_mode = "100644"
authority_commit
```

The complete ordered three-row list must be identical throughout the active
v2 chain at every location where `G9CB-1B` requires the amendment list. An
active two-row, reordered, additional, missing, renamed, or changed list is
terminal before capability consumption, generic import, or value-row access.
The immutable historical v1 artifact is the sole exception: it is validated
under its sealed two-row `G9CB-1A`, `G9CB-1B` cardinality only as historical
evidence and can never become operative.

The replacement preregistration's ordered protocol inventory must include this
amendment path in addition to every previously required protocol path.

## Commit topology

Let:

- `A` be the standalone pushed metadata-only `G9CB-1C` authority commit;
- `Q` be the later clean pushed protocol implementation commit;
- `P` be the clean pushed direct child that adds the active v2
  preregistration artifact;
- `C` be the claim-only direct child of `P`; and
- `D` be the unchanged five-file publication-only direct child of `C`.

`A` must satisfy exactly:

```text
A has exactly one parent
first_parent(A) == 3810a3b7e24b83591866f2ccf9b63167795718c5
diff(3810a3b7e24b83591866f2ccf9b63167795718c5, A) ==
  A docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
HEAD == A
HEAD == @{upstream}
the worktree and index are clean
```

No intervening commit, merge parent, rename, deletion, modification, amend,
rebase, replacement, or force-push is permitted.

`Q` must contain the corrected producer, builder, isolated runtime files,
tests, all earlier authority files, this exact amendment, and the immutable
historical v1 artifact. It must contain no active v2 preregistration, claim,
sentinel, worker ledger, canonical CSV, or final manifest.

`Q` must satisfy exactly:

```text
Q has exactly one parent
first_parent(Q) == A
diff(A, Q) ==
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py
HEAD == Q
HEAD == @{upstream}
the worktree and index are clean
```

The rows are a sorted exact set. `Q` may not modify the isolated runtime,
primitives, original authority documents, historical v1 artifact, source,
model, config, lock, clock, counter event, serializer, economic rule, overlap
rule, or publication schema shape. The exact protocol-version scalar
substitution enumerated above, active path, historical binding, three-row
amendment binding, exact Git checks, closed zero-access validation, and their
tests are the only authorized changes.

After all tests pass, the active v2 preregistration may be built only while:

```text
HEAD == Q
HEAD == @{upstream}
the worktree and index are clean
```

`P` must then satisfy exactly:

```text
P has exactly one parent
first_parent(P) == Q
diff(Q, P) ==
  A results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json
HEAD == P
HEAD == @{upstream}
the worktree and index are clean
```

The active v2 artifact must bind the exact `Q` protocol blobs, the base
decision commit `e6ff406444a95068100cfacf617a3a23bcf918e3`
separately, and the exact ordered three amendment commits for `G9CB-1A`,
`G9CB-1B`, and `G9CB-1C`. The historical v1 artifact remains unchanged in
`P`. The active v2 artifact must also contain the exact top-level field:

```text
protocol_implementation_commit = Q
```

Claim preflight derives `P` from `HEAD` and must prove, before hashed-input
authentication:

```text
first_parent(P) == preregistration.protocol_implementation_commit
diff(preregistration.protocol_implementation_commit, P) ==
  A results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json
```

`Q` is exclusively the protocol-implementation commit. `P` is exclusively the
preregistration-seal commit and direct parent of the claim. The inherited claim
field name `protocol_parent_commit` is retained for schema compatibility but
its exact value is `P` and its role is the preregistration-seal/direct-parent
commit; it must never be substituted for `Q`.

The pre-access claim contract then remains structurally unchanged except that
its `protocol_parent_commit` is this new `P`, its preregistration binding names
the active v2 path/version/hash, and its amendment list has three rows:

```text
C has exactly one parent
first_parent(C) == P
diff(P, C) ==
  A results/gross9_structural_clock_bundle_access_claim_2026-07-31.json
```

Claim preflight authenticates both roles before hashed-input access: it reads
`Q` from the active preregistration's exact
`protocol_implementation_commit`, derives `P` from `HEAD`, proves the
`Q -> P` artifact-only topology, and records `P` in
`claim.protocol_parent_commit`. The claim hash binds the active
preregistration path, file SHA-256, and manifest hash and therefore binds `Q`.
The sentinel repeats the exact claim and preregistration bindings and records
`P`; each ledger, receipt, per-pass core, and final manifest authenticates the
same claim/preregistration/sentinel hash chain. Every later validator must
therefore authenticate both `Q` and `P` in their distinct roles without
substitution.

No protocol, authority, test, historical artifact, or active preregistration
byte may change between `P`, `C`, and `D`. The `G9CB-1B` exact five-file
publication diff and all one-shot runtime rules remain unchanged.

## Completion and failure rule

The earlier v1 preregistration commit is not `P` and never authorized a claim.
The replacement sequence is complete only when:

1. `A` is committed and pushed alone as metadata-only authority;
2. `Q` implements the corrected validator, historical binding, v2 versions,
   active path, three-amendment list, and regression tests and is clean and
   pushed;
3. `P` adds only the active write-once v2 preregistration and is clean and
   pushed;
4. the active artifact reproduces from the exact `Q` blobs and authenticates
   the unchanged historical v1 artifact;
5. `C` adds only the active v2-bound claim and is clean and pushed;
6. the sentinel precedes every generic import and value-row access;
7. exactly two fresh workers and all `G9CB-1B` capability, counter,
   byte-identity, and receipt checks pass without retry;
8. `D` adds exactly the unchanged five publication files, manifest last; and
9. committed artifact validation passes with `HEAD == D == @{upstream}` and a
   clean worktree and index.

Before the sentinel, any failure outside this exact correction stops without
value access. It may not be repaired by changing alpha, clock, source,
environment, metric, economics, or publication semantics. After the sentinel,
the existing terminal rule remains exact:

```text
TERMINAL_G9CB1_ATTEMPT_CONSUMED_NO_RETRY
```

This correction authority is single-use. If the operative v2 path fails for
any non-restorable reason before sentinel publication, no `G9CB-1D`, v3
preregistration, second correction, or further protocol mutation may retain
`G9CB-1`. A successor requires a new infrastructure identity, decision,
preregistration, and claim.

## Decision

`G9CB-1C` preserves the pushed record, distinguishes boolean declarations from
integer physical counters, and restores a usable metadata-only claim gate. It
authorizes no candidate, source/value access, alpha choice, economic
calculation, overlap calculation, or production retry.
