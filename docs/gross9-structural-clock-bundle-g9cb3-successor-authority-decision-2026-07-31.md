# Gross9 structural clock bundle G9CB-3 successor authority decision — 2026-07-31

## Decision

Freeze `G9CB-3` as a new candidate-independent structural-clock
infrastructure identity.

`G9CB-3` is not a retry, resume, repair, amendment, v2, or completion of
`G9CB-2`. The single official `G9CB-2` production attempt published its
attempt-consumed sentinel and then failed terminally before either worker
consumed its capability, imported a generic runtime module, or opened an
official source-value row. The `G9CB-2` authority therefore requires another
new infrastructure identity and decision:

```text
docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md
```

This decision inherits without alteration the candidate-independent economic
object and all compatible mechanics frozen by:

```text
docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md
```

The domain, five sleeves, weights, sides, interval geometry, holds, barriers,
Rank7 learner and source-routed exits, source paths and hashes, environment,
static closure, isolated runtime, exactly-two-pass rule, one-use
anonymous-pipe capabilities, counters, deterministic serialization,
publication allowlist, manifest-last rule, one-shot terminal rule, candidate
independence, and prohibitions on portfolio economics and overlap computation
remain unchanged.

Only these successor changes are authorized:

1. the new `G9CB-3` identity and v1-under-`G9CB-3` protocol versions;
2. the exact `A3 -> T2 -> Q3 -> P3 -> C3 -> D3` history below;
3. a terminal-evidence seal for the already-existing immutable `G9CB-2`
   sentinel;
4. immutable bindings for the complete failed `G9CB-2` attempt;
5. a split between complete parent Git authentication and stdlib-only guarded
   worker metadata authentication;
6. a distinct `G9CB-3` worker-stage prefix; and
7. identity-specific artifact, exception, action, and bytecode-prefix names.

This decision is not operative merely because this file exists. It must be
committed and pushed alone as `A3` while `HEAD` is the exact clean pushed
`G9CB-2` claim commit. Only then may the existing sentinel be force-added
alone as historical terminal evidence `T2`.

## Immutable G9CB-2 terminal finding

The exact predecessor chain is:

```text
A2 = 0a2847c8589908def4243890727c3640f806e109
Q2 = f48634af22dcad84ffde885fa970635d133cc126
P2 = 04550a47686ee039f82dfdb412d3c3eec4b5d6a1
C2 = 731f093eb963b9e7213778ed4f259ee5466cd893
```

The operative `G9CB-2` preregistration is:

```json
{
  "path": "results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json",
  "sha256": "84cea282bda82270d5c1f10c2606f78ac8fddd40527598c3e2aaafa6089b38ec",
  "git_blob": "31bd51bdca5cec5da428b9ae3db067635d6d04b2",
  "git_mode": "100644",
  "filesystem_mode_octal": "0444",
  "seal_commit": "04550a47686ee039f82dfdb412d3c3eec4b5d6a1",
  "protocol_implementation_commit": "f48634af22dcad84ffde885fa970635d133cc126",
  "protocol_version": "gross9_structural_clock_bundle_g9cb2_preregistration_v1",
  "manifest_hash": "070b0dded30c1ffcf8744c232ddf37ef22985f80546ad8d6ce4d0d3c72b84b0b"
}
```

The operative `G9CB-2` claim is:

```json
{
  "path": "results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json",
  "sha256": "0d0cea614cc8ddc51106989c0d68362ed27684d1ce46b1d4daea41c6bfb0be23",
  "git_blob": "05adab10031399c3599168c9673923812cecdc09",
  "git_mode": "100644",
  "filesystem_mode_octal": "0444",
  "seal_commit": "731f093eb963b9e7213778ed4f259ee5466cd893",
  "protocol_parent_commit": "04550a47686ee039f82dfdb412d3c3eec4b5d6a1",
  "claim_hash": "28faeb0a7f9662c3264374785b7e53376c4d6500817f26e7d94e0afeab25979d"
}
```

The single official command was invoked exactly once:

```text
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/time -v uv run python -B \
  -m training.build_gross9_structural_clock_bundle --produce
```

It returned status `1`. The diagnostic stderr/time file had SHA-256:

```text
b20ea5ecf066b3c286025af4167364b3fea6bda0607eb4375340da674d30fc65
```

Captured stdout was empty and had SHA-256:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The `/tmp` logs are diagnostic disclosure, not durable artifacts or future
inputs. The durable failure evidence is the canonical sentinel:

```json
{
  "path": "results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json",
  "path_type": "regular_file",
  "size_bytes": 3288,
  "sha256": "3bfb5c3259d398b4b16e029d180c72390fd3f835acffa1010fac7b9c40eeac83",
  "git_blob": "8263cd1be0f061f349d7e93a8fb49ce2c72c08dd",
  "git_mode": "100644",
  "filesystem_mode_octal": "0444",
  "manifest_hash": "100ae22658c5dda3351761f3ea09db406ac47377c619fb39803f70af8646a3b5",
  "claim_commit": "731f093eb963b9e7213778ed4f259ee5466cd893",
  "status": "attempt_consumed_before_runtime_or_value_access",
  "resume_allowed": false,
  "retry_allowed": false
}
```

The sentinel's parent-authentication SHA-256 is:

```text
ee916c2ce7249c75744f9b7717adbcbbe8b341cab84c5b030922b417e022e7ca
```

The first fresh worker installed the required isolation guard, then attempted
to re-run failed-predecessor Git authentication. That path called
`subprocess.run`, which the already-installed guard correctly rejected. The
failure was therefore a control-flow contradiction between parent-only Git
authentication and guarded-worker recomputation. No actual worker Git child
was launched.

The failure occurred before the first capability read, worker ledger,
repository runtime import, source decode, source-value read, model/history
read, structural-clock computation, or output row. Exact findings are:

```text
worker capabilities consumed = 0
worker ledgers published = 0
runtime modules imported = 0
source value rows opened = 0
candidate rows opened = 0
comparator clock rows opened = 0
pre-2025 anchor value rows opened = 0
portfolio return or PnL values computed = 0
CAGR or MDD values computed = 0
overlap metric values computed = 0
```

The following `G9CB-2` paths were absent after failure and are permanently
reserved as absent:

```text
results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json
```

Their later creation is terminal. The sentinel alone is not `D2`, is not a
clock authority, and does not complete `G9CB-2`.

The sentinel binds two exact historical transient stage paths. Their states
after failure are:

```text
slot 1:
  path =
    results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef
  state = empty_directory
  filesystem_mode_octal = 0700
slot 2:
  path =
    results/.gross9-structural-clock-worker-2c9f266762f8864bf5e24691
  state = absent
```

Neither path is a canonical artifact. Neither may be committed, read as a
value source, reused, renamed, or treated as authority. This decision does
not authorize creating the slot-2 path, deleting or populating the slot-1
path, or changing the slot-1 mode. The exact states above must remain through
`T2` and `D3`. `G9CB-3` must use a distinct stage prefix and must ignore the
preserved slot-1 directory when checking only its own stages.

## T2 terminal-evidence seal

Let `T2` be the direct child of `A3` that force-adds only the already-existing
sentinel. `T2` is historical terminal-attempt evidence, not `D2`.

Before `T2`, the sentinel must reproduce the exact SHA-256, prospective Git
blob, internal manifest hash, canonical JSON bytes, status, claim binding,
zero-access counters, and filesystem mode above. The four permanently absent
`G9CB-2` paths must remain absent. The exact slot-1 and slot-2 transient states
must also remain unchanged.

`T2` must satisfy:

```text
T2 has exactly one parent
first_parent(T2) == A3
diff(A3, T2) ==
  A results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json
HEAD == T2 == @{upstream}
the worktree and index are clean
```

Git records regular files as mode `100644`; the worktree file must remain
mode `0444`. The committed Git blob must be:

```text
8263cd1be0f061f349d7e93a8fb49ce2c72c08dd
```

No other `G9CB-2` output may be added at or after `T2`.

## G9CB-3 identity and paths

The exact successor values are:

```text
identity = G9CB-3
preregistration protocol =
  gross9_structural_clock_bundle_g9cb3_preregistration_v1
builder/publication protocol =
  gross9_structural_clock_bundle_g9cb3_v1
terminal exception = TerminalG9CB3Failure
terminal action = TERMINAL_G9CB3_ATTEMPT_CONSUMED_NO_RETRY
bytecode prefix = results/.g9cb3-bytecode-cache-disabled
worker stage prefix = results/.gross9-structural-clock-g9cb3-worker-
```

The active paths are exactly:

| Role | Path |
|---|---|
| authority | `docs/gross9-structural-clock-bundle-g9cb3-successor-authority-decision-2026-07-31.md` |
| preregistration | `results/gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json` |
| access claim | `results/gross9_structural_clock_bundle_g9cb3_access_claim_2026-07-31.json` |
| attempt sentinel | `results/gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json` |
| worker ledger pass 1 | `results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_2026-07-31.json` |
| worker ledger pass 2 | `results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass2_2026-07-31.json` |
| canonical CSV gzip | `results/gross9_structural_clock_bundle_g9cb3_2026-07-31.csv.gz` |
| final manifest | `results/gross9_structural_clock_bundle_g9cb3_manifest_2026-07-31.json` |

No `G9CB-1` or `G9CB-2` output path may be reused, replaced, renamed, deleted,
or treated as an alias.

## Split parent and guarded-worker authentication

The parent remains responsible before sentinel publication for the complete
existing metadata preflight:

- canonical preregistration and manifest validation;
- authority and failed-predecessor authentication;
- exact Git branch, ancestry, index, `HEAD`, mode, and blob validation;
- complete optional Git-pair classification before bound-byte reads;
- no-follow single-read size, SHA-256, and derived Git-blob authentication;
- environment and static-closure authentication;
- claim and commit-topology validation;
- bytecode and output/stage absence checks; and
- exact permanent absence of the four failed `G9CB-2` publication paths.

The top-level `parent_authentication` object has exactly these keys:

```text
environment
hashed_inputs
preregistration_authentication
runtime_import_closure
```

No missing or additional key is allowed. The first, second, and fourth values
retain their inherited exact schemas. The new value at
`parent_authentication.preregistration_authentication` is exactly:

```json
{
  "manifest_hash": "<active preregistration manifest_hash>",
  "path": "results/gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json",
  "protocol_implementation_commit": "<exact Q3 commit>",
  "sha256": "<active preregistration complete-file SHA-256>"
}
```

The angle-bracket values are not literal artifact strings. At `P3` they are
replaced by the exact lowercase hashes and commit authenticated by the parent.
The record has exactly those four keys. `path`, `manifest_hash`, `sha256`, and
`protocol_implementation_commit` are strings; both digests are exactly 64
lowercase hexadecimal characters and the commit is exactly 40.

The complete `parent_authentication` object is serialized by the inherited
canonical JSON rule and its SHA-256 is bound in the sentinel before either
worker starts. Before capability access, each worker computes the complete
preregistration file SHA-256 and internal manifest hash from the bytes it
reads. Those two values, the exact path, and the recorded implementation
commit must equal the four corresponding fields in
`preregistration_authentication`; the path, file SHA-256, and manifest hash
must also equal the corresponding claim/sentinel preregistration binding. A
missing, additional, mistyped, malformed, or unequal field is terminal before
capability consumption, runtime import, or value access.

The guarded worker must not re-run Git. Its preregistration path must be
stdlib-only and must:

- read the exact active preregistration as canonical no-symlink regular-file
  bytes;
- verify mode `0444`, file SHA-256, internal manifest hash, identity, protocol,
  zero-access schema, and embedded predecessor bindings;
- verify predecessor regular-file bytes, modes, SHA-256 values, canonical
  hashes, and derived Git blobs without invoking Git;
- compare the active preregistration binding and implementation commit to the
  exact parent-authentication record whose hash is bound by the sentinel; and
- continue the inherited claim, sentinel, capability, runtime, and counter
  checks.

No guarded-worker call path may reach `_run_git`, `_git_result`,
`subprocess.*`, process creation, descriptor duplication, or another external
metadata helper. The isolation guard remains installed at the earliest
bootstrap point and remains unchanged in strength.

Allowing Git or another child process inside the worker is forbidden. Moving
the guard later is forbidden. Treating official production as synthetic is
forbidden.

## Authority flow and exact protocol inventory

The active `G9CB-3` preregistration has one top-level
`authority_decision` binding with exactly:

```text
path
path_type
sha256
git_blob
git_mode
authority_commit
```

It binds this decision, its complete-file SHA-256, Git blob, mode `100644`,
regular-file type, and the exact standalone `A3` commit. No other decision is
the active authority.

The inherited ordered `bindings.authority_amendments` list remains exactly:

```json
[
  {
    "authority_commit": "f1ae4e68bfb0d0b861cd9979762f87e51a55f69d",
    "git_blob": "0c7781ebe25178c592bb526ac51ee00c5ba840e2",
    "git_mode": "100644",
    "identity": "G9CB-1A",
    "path": "docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md",
    "path_type": "regular_file",
    "sha256": "a99b1a2b3d738ecc1cea8595eed2d88759c9b5fa7faf751a53b643fcc1a808cb"
  },
  {
    "authority_commit": "2550e0b8ee348b4217744a73d9781dba1e1e91a3",
    "git_blob": "c2da15ff249e46a8fac2040d67f531a683b7fd7e",
    "git_mode": "100644",
    "identity": "G9CB-1B",
    "path": "docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md",
    "path_type": "regular_file",
    "sha256": "354ae3870dd6dedf738b38bdd266d85b24389fe5de10d1fa0b3dbdde18d1c2de"
  },
  {
    "authority_commit": "eee3383c9b2f88f4ea28f5bfe3a5ff6a650cec0f",
    "git_blob": "94c0f3e13680f9e0ebbdb07ae7646b9505891e46",
    "git_mode": "100644",
    "identity": "G9CB-1C",
    "path": "docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md",
    "path_type": "regular_file",
    "sha256": "b79151c3378960017ddb30b7c1040f3027be538acad00776315380c267c6acaf"
  }
]
```

`bindings.protocol` is a sorted exact list of bindings for these 15 paths:

```text
docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb3-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md
execution/gross9_rank7_clock_runtime.py
tests/test_build_gross9_structural_clock_bundle.py
tests/test_gross9_rank7_clock_runtime.py
tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
tests/test_gross9_structural_clock_primitives.py
tests/test_preregister_gross9_structural_clock_bundle.py
training/build_gross9_structural_clock_bundle.py
training/gross9_structural_clock_primitives.py
training/preregister_gross9_structural_clock_bundle.py
```

Each row has exactly `path`, `path_type`, `sha256`, `git_blob`, and
`git_mode`; every path is a tracked regular file at `Q3`, every mode is
`100644`, and every SHA/blob is derived from the exact current `Q3` bytes.
The two successor decisions are both in the protocol inventory: this decision
is active, while the `G9CB-2` decision is immutable predecessor authority.

The claim binds the complete active preregistration path, SHA-256, and
manifest hash. That binding carries the active decision, the unchanged
amendments, the exact protocol inventory, and all failed-predecessor evidence
through the sentinel, worker receipts, per-pass cores, and final manifest.
Those downstream artifacts do not add an alternate authority field.

## Failed-predecessor binding

The `G9CB-3` preregistration retains the exact two failed `G9CB-1`
preregistration bindings inherited by `G9CB-2`.

It must additionally contain exactly one
`bindings.failed_predecessor_attempts` row for `G9CB-2`.

In the closed schema below, `"<A3>"` and `"<T2>"` are metavariables, not
literal artifact values. At `P3` they are replaced by the exact lowercase
40-hex commits derived from the required topology. Every other scalar is the
exact literal shown. No key may be added or omitted:

```json
{
  "access_claim": {
    "claim_hash": "28faeb0a7f9662c3264374785b7e53376c4d6500817f26e7d94e0afeab25979d",
    "filesystem_mode_octal": "0444",
    "git_blob": "05adab10031399c3599168c9673923812cecdc09",
    "git_mode": "100644",
    "path": "results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json",
    "path_type": "regular_file",
    "protocol_parent_commit": "04550a47686ee039f82dfdb412d3c3eec4b5d6a1",
    "seal_commit": "731f093eb963b9e7213778ed4f259ee5466cd893",
    "sha256": "0d0cea614cc8ddc51106989c0d68362ed27684d1ce46b1d4daea41c6bfb0be23"
  },
  "attempt_sentinel": {
    "claim_commit": "731f093eb963b9e7213778ed4f259ee5466cd893",
    "filesystem_mode_octal": "0444",
    "git_blob": "8263cd1be0f061f349d7e93a8fb49ce2c72c08dd",
    "git_mode": "100644",
    "manifest_hash": "100ae22658c5dda3351761f3ea09db406ac47377c619fb39803f70af8646a3b5",
    "path": "results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json",
    "path_type": "regular_file",
    "protocol_version": "gross9_structural_clock_bundle_g9cb2_v1",
    "resume_allowed": false,
    "seal_commit": "<T2>",
    "sha256": "3bfb5c3259d398b4b16e029d180c72390fd3f835acffa1010fac7b9c40eeac83",
    "size_bytes": 3288,
    "status": "attempt_consumed_before_runtime_or_value_access",
    "retry_allowed": false
  },
  "authority_decision": {
    "authority_commit": "0a2847c8589908def4243890727c3640f806e109",
    "git_blob": "4904a47fa75cb455cd3c5007373e149267b7f198",
    "git_mode": "100644",
    "path": "docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md",
    "path_type": "regular_file",
    "sha256": "b80ad199c803623c4de289d32beea913b3e1c38541e1c4f035ce3d33fa049410"
  },
  "classification": "terminal_guarded_worker_git_subprocess_rejected_before_capability_or_value_access",
  "failure_counters": {
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "generic_runtime_modules_imported": 0,
    "pre2025_anchor_value_rows_opened": 0,
    "source_value_rows_opened": 0,
    "worker_capabilities_consumed": 0,
    "worker_git_children_launched": 0,
    "worker_ledgers_published": 0
  },
  "identity": "G9CB-2",
  "permanently_absent_outputs": [
    "results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz",
    "results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json"
  ],
  "preregistration": {
    "filesystem_mode_octal": "0444",
    "git_blob": "31bd51bdca5cec5da428b9ae3db067635d6d04b2",
    "git_mode": "100644",
    "manifest_hash": "070b0dded30c1ffcf8744c232ddf37ef22985f80546ad8d6ce4d0d3c72b84b0b",
    "path": "results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json",
    "path_type": "regular_file",
    "protocol_implementation_commit": "f48634af22dcad84ffde885fa970635d133cc126",
    "protocol_version": "gross9_structural_clock_bundle_g9cb2_preregistration_v1",
    "seal_commit": "04550a47686ee039f82dfdb412d3c3eec4b5d6a1",
    "sha256": "84cea282bda82270d5c1f10c2606f78ac8fddd40527598c3e2aaafa6089b38ec"
  },
  "protocol_implementation": {
    "commit": "f48634af22dcad84ffde885fa970635d133cc126",
    "files": [
      {
        "git_blob": "ef5b6e0f480fad3fd221e290c4b4f88b75dc4395",
        "git_mode": "100644",
        "path": "tests/test_build_gross9_structural_clock_bundle.py",
        "sha256": "d50cbcd24ac0af74a5c531161766ca99662af0d24e2f3c5ad4ae364d84165c33"
      },
      {
        "git_blob": "03fc8aa8dd22bcd29fd6e8d51ad5d040bba1ce95",
        "git_mode": "100644",
        "path": "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
        "sha256": "d50ad1eed00e0b1f8e55c09998dcfb576548885aaf5ec8b4650797fa20684700"
      },
      {
        "git_blob": "d199ca4395cf3653860ea65b21a3ed573ee1b092",
        "git_mode": "100644",
        "path": "tests/test_preregister_gross9_structural_clock_bundle.py",
        "sha256": "21ef33be2aa1654b6a34d5771d57c829cc712d8ad0dc6178e81dd8c034042bb6"
      },
      {
        "git_blob": "488a405fd39092c288a34b84a3a27811968f7050",
        "git_mode": "100644",
        "path": "training/build_gross9_structural_clock_bundle.py",
        "sha256": "48a94fe63ae1aeb040bef7fa632d87522d97550d754c112e069776a8b6692132"
      },
      {
        "git_blob": "641db72d2c5f395147a844b38c27427c691a2d8d",
        "git_mode": "100644",
        "path": "training/preregister_gross9_structural_clock_bundle.py",
        "sha256": "07496f055b2e8ce2cdadd237f21eee0794679595ec82f4330bca1e82188c55bf"
      }
    ],
    "parent_commit": "0a2847c8589908def4243890727c3640f806e109"
  },
  "protocol_version": "gross9_structural_clock_bundle_g9cb2_v1",
  "residue": {
    "slot1_stage": {
      "committed": false,
      "filesystem_mode_octal": "0700",
      "path": "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef",
      "state": "empty_directory"
    },
    "slot2_stage": {
      "committed": false,
      "path": "results/.gross9-structural-clock-worker-2c9f266762f8864bf5e24691",
      "state": "absent"
    }
  },
  "status": "historical_terminal_attempt_consumed_no_clock_authority",
  "topology": {
    "g9cb2_authority_commit": "0a2847c8589908def4243890727c3640f806e109",
    "g9cb2_claim_commit": "731f093eb963b9e7213778ed4f259ee5466cd893",
    "g9cb2_preregistration_commit": "04550a47686ee039f82dfdb412d3c3eec4b5d6a1",
    "g9cb2_protocol_commit": "f48634af22dcad84ffde885fa970635d133cc126",
    "g9cb3_authority_commit": "<A3>",
    "terminal_evidence_commit": "<T2>"
  }
}
```

The producer and parent builder must authenticate the complete row with Git.
The guarded worker must authenticate the same embedded row from bytes and the
sentinel-bound parent authentication without calling Git.

## Exact commit topology

Every seal uses:

```text
codex/gross9-structural-clock-bundle-20260731
```

Let:

- `C2` be `731f093eb963b9e7213778ed4f259ee5466cd893`;
- `A3` add only this decision;
- `T2` add only the immutable `G9CB-2` sentinel;
- `Q3` be the clean pushed `G9CB-3` protocol implementation;
- `P3` add only the `G9CB-3` preregistration;
- `C3` add only the `G9CB-3` claim; and
- `D3` add only the five successful `G9CB-3` publication files.

`A3` must satisfy:

```text
A3 has exactly one parent
first_parent(A3) == 731f093eb963b9e7213778ed4f259ee5466cd893
diff(C2, A3) ==
  A docs/gross9-structural-clock-bundle-g9cb3-successor-authority-decision-2026-07-31.md
HEAD == A3 == @{upstream}
the worktree and index are clean
```

`T2` must satisfy the terminal-evidence rules above.

`Q3` must satisfy:

```text
Q3 has exactly one parent
first_parent(Q3) == T2
diff(T2, Q3) ==
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py
HEAD == Q3 == @{upstream}
the worktree and index are clean
```

`Q3` may change only identity/version/path names, successor/predecessor
bindings, exact topology checks, the parent/worker metadata-authentication
split, the preregistration-authentication record in parent authentication,
the distinct worker-stage prefix, exception/action/bytecode names, and tests
for those changes. It may not alter a source path/hash, domain, sleeve,
weight, side, hold, barrier, feature, Rank7 learner/policy, counter event,
environment lock, dependency, runtime computation, serializer geometry,
guard strength, capability, two-pass rule, economic prohibition, overlap
prohibition, or successful publication algorithm.

No `G9CB-3` artifact may exist at `Q3`.

`P3` must satisfy:

```text
P3 has exactly one parent
first_parent(P3) == Q3
diff(Q3, P3) ==
  A results/gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json
HEAD == P3 == @{upstream}
the worktree and index are clean
```

`C3` must satisfy:

```text
C3 has exactly one parent
first_parent(C3) == P3
diff(P3, C3) ==
  A results/gross9_structural_clock_bundle_g9cb3_access_claim_2026-07-31.json
HEAD == C3 == @{upstream}
the worktree and index are clean
```

`D3` must satisfy:

```text
D3 has exactly one parent
first_parent(D3) == C3
diff(C3, D3) ==
  A results/gross9_structural_clock_bundle_g9cb3_2026-07-31.csv.gz
  A results/gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb3_manifest_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass2_2026-07-31.json
HEAD == D3 == @{upstream}
the worktree and index are clean
```

No authority, protocol, test, predecessor, preregistration, or claim byte may
change between `P3`, `C3`, and `D3`.

## Required regression evidence

Before `Q3` is sealed, tests must prove at least:

1. exact `C2 -> A3 -> T2 -> Q3 -> P3 -> C3 -> D3` topology and diffs;
2. the `G9CB-2` sentinel is exact terminal evidence, not `D2`;
3. both `G9CB-2` ledgers, its CSV, and its manifest remain absent;
4. the failed worker launched no child and reached neither capability
   consumption nor runtime/value access;
5. parent mode still performs complete Git/topology/input authentication
   before the sentinel;
6. guarded-worker mode validates exact canonical predecessor and active
   preregistration bytes without any Git or subprocess call;
7. a positive non-synthetic guarded-worker metadata path reaches capability
   consumption with zero child-process events;
8. the old bug fails a regression that would reject any worker-side
   `subprocess.run`;
9. the isolation guard remains installed before worker argument parsing;
10. the `G9CB-3` stage prefix is distinct and the old random stage cannot
    block, satisfy, or alias a `G9CB-3` stage, while the old slot-1 stage
    remains empty mode `0700` and old slot-2 remains absent through `D3`;
11. every field of `preregistration_authentication` rejects deletion,
    addition, type drift, malformed hashes, and parent/worker byte drift
    before capability access;
12. explicit traps prove guarded-worker metadata validation cannot reach
    `_run_git`, `_git_result`, `subprocess.*`, descriptor duplication, or an
    external helper;
13. all `G9CB-3` stages, receipts, and bytecode paths are absent after `D3`;
14. the active preregistration reproduces byte-for-byte at `P3` and `D3`;
15. the exact inherited frozen clock/runtime suites still pass; and
16. committed publication validation authenticates every `G9CB-3` artifact
    at `D3`.

Synthetic tests may use synthetic bytes and Git repositories. They may not
decode official source/model/history values or compute portfolio economics.

## Completion and terminal rule

`G9CB-3` is complete only when `A3`, `T2`, `Q3`, `P3`, `C3`, and `D3`
satisfy their exact clean-pushed topology; exactly two fresh workers produce
byte-identical authenticated outputs; the final manifest is published last;
the committed-publication verifier passes at `D3`; and the worktree remains
clean with `HEAD == @{upstream}`.

Before the `G9CB-3` sentinel, any non-restorable protocol failure closes
`G9CB-3` without mutation under the same identity. After the sentinel, every
failure is terminal:

```text
TERMINAL_G9CB3_ATTEMPT_CONSUMED_NO_RETRY
```

After any `G9CB-3` sentinel is published, every retry, resume, repair,
amendment, protocol mutation, metadata correction, second production
invocation, v2-under-`G9CB-3`, or completion attempt under the same identity is
forbidden, regardless of whether it would change source, clock, runtime,
metric, economics, overlap, or publication semantics. Any future successor
requires another new infrastructure identity and decision.

This decision authorizes no alpha candidate, source-value exploration,
candidate ranking, portfolio return, PnL, funding cash, CAGR, MDD, economic
metric, or overlap metric. Those remain forbidden until `G9CB-3` has a clean,
pushed, verified `D3` publication and a later candidate has its own
preregistration.
