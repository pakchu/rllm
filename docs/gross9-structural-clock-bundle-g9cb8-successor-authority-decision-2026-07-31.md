# Gross9 structural clock bundle G9CB-8 successor authority decision — 2026-07-31

## Decision

Freeze `G9CB-8` as a new candidate-independent structural-clock
infrastructure identity and the only authorized successor after the
pre-sentinel production closure of `G9CB-7`.

The exact last clean pushed `G9CB-7` claim commit is:

```text
C7 = ff1a8907d19c97beeef0bd7d2797e3bacce17617
```

The sole observed non-synthetic `G9CB-7 --produce` invocation faulted before
the publication context, capability probe, attempt-consumed sentinel, worker
stage, worker process, or source-value decode. Under the one-shot closure rule
frozen by `A7`, `G9CB-7` closed immediately. It is not retryable, resumable,
repairable, or completable under the same identity. `P7` and `C7` remain
immutable historical metadata. `D7` and a `G9CB-7` terminal-evidence seal do
not exist and must never be created.

The only canonical continuation authorized by this decision is:

```text
C7 -> A8 -> Q8 -> P8 -> C8 -> D8
```

Every commit is single-parent, clean, pushed, and a fast-forward on:

```text
codex/gross9-structural-clock-bundle-20260731
```

History rewriting and force-pushing remain forbidden.

This decision inherits every compatible candidate-independent economic and
structural mechanic frozen by the authority chain through `A7`. It changes
only:

1. the active infrastructure identity from `G9CB-7` to `G9CB-8`;
2. active output, residue, exception, action, phase, and protocol-version
   literals from `g9cb7` to `g9cb8`;
3. the exact `C7 -> A8 -> Q8 -> P8 -> C8 -> D8` topology;
4. one exact pre-sentinel-production closure record for `G9CB-7`;
5. authenticated historical bindings for `P7` and `C7`;
6. inclusion of that complete closure in both manifest and initial bootstrap
   declaration construction;
7. one authority-ordered removal of the exact ignored repository bytecode
   caches recorded below;
8. expansion of the strict repository-bytecode preflight to reject
   `__pycache__`, `.pyc`, and `.pyo`, plus shared placement of that check as
   the first post-argument filesystem gate for `P8`, `C8`, and `D8`, before
   topology, parent snapshot, or input authentication; and
9. synthetic regressions for the closure, declaration parity, cache-removal
   boundary, and preflight ordering.

No candidate, candidate rank, comparator clock, return, PnL, funding cash,
CAGR, MDD, economic metric, overlap metric, sleeve weight, signal rule, hold,
barrier, model, feature, source content, source digest, source logical path,
environment requirement, serialization rule, or publication rule may change.

This file is not operative merely because it exists. It becomes `A8` only
when committed and pushed as the sole change in the direct child of exact
`C7`.

## Evidence provenance and recoverability

This decision separates four evidence classes:

1. **repository-authenticated facts** reproducible from committed Git objects,
   current immutable metadata artifacts, and filesystem metadata;
2. **observed execution evidence** supplied by the current session's sole
   `P7` invocation, sole `C7 --create-claim` invocation, and sole
   non-synthetic `G9CB-7 --produce` invocation;
3. **control-flow deductions** from exact committed `Q7` source, the traceback,
   the failure location, and authenticated declaration sizes; and
4. **unrecoverable facts** that remain JSON null and are not inferred from
   artifact absence.

No standalone byte-exact shell-input file, raw stdout/stderr capture, syscall
trace, duration record, or resource-usage record was preserved. The exact
tool-submitted command text, traceback, and exit status were observed in this
session but were not saved as standalone files. The bound path and byte totals
below are control-flow deductions, not syscall counts.

Exactly one `P7` invocation, one `C7 --create-claim` invocation, and one
non-synthetic `G9CB-7 --produce` invocation are observed in this continuation.
The production invocation did not conform byte-for-byte to the `A7` wrapper,
as disclosed below. No same-identity retry is authorized.

## Authenticated chain and bindings

At drafting time:

- `HEAD == @{upstream} == C7`;
- `C7` is the direct child of `P7` and adds only the active access claim;
- `P7` is the direct child of `Q7` and adds only the active preregistration;
- `Q7` is the direct child of `A7` and changes exactly the five authorized
  protocol and test files;
- the index and tracked worktree are clean;
- the tracked top-level `results/` inventory contains 1,341 entries;
- the actual top-level inventory contains only those entries plus the two
  exact retained predecessor residue directories;
- only the tracked immutable `P7` preregistration and `C7` claim exist for
  active `G9CB-7`; and
- every `G9CB-7` sentinel, worker ledger, canonical CSV, final manifest,
  publication stage, worker stage, capability probe, and fixed worker
  bytecode path is absent.

The exact `A7` authority binding is:

```json
{
  "authority_commit": "ad5a7e5f6d3edeac0928c1ef93fd0fd2209a9279",
  "git_blob": "53860caaefbeb964a46d5668660793f98a929ed4",
  "git_mode": "100644",
  "path": "docs/gross9-structural-clock-bundle-g9cb7-successor-authority-decision-2026-07-31.md",
  "path_type": "regular_file",
  "sha256": "faf5b5f427882c97e2437fe32bb1a0b280f87fe780393e4016911a02ce6c2624",
  "size_bytes": 27054
}
```

The exact `Q7` implementation binding is:

```json
{
  "builder_git_blob": "f4fb347cad4bb7eb71ad3f6a55cc06ea60b06326",
  "builder_path": "training/build_gross9_structural_clock_bundle.py",
  "builder_sha256": "5931e78dcc28246564c222c33161bf5e065766978a8116a951022e0f82e9de76",
  "builder_size_bytes": 417345,
  "commit": "39cd0c4233cc879a0a5461be2ab76f3bd30ae36c",
  "preregistration_git_blob": "1d9a7458d0850575e9414371bbd392aac990386a",
  "preregistration_path": "training/preregister_gross9_structural_clock_bundle.py",
  "preregistration_sha256": "8e5ff615d2b62e218dcfe568df0f7a2b8ef5ca51c2237922eefe57c54c9572a7",
  "preregistration_size_bytes": 201993
}
```

The exact `P7` binding is:

```json
{
  "filesystem_mode_octal": "0444",
  "git_blob": "1e626c079319a8390e794ea3883c62ccde1d785b",
  "git_mode": "100644",
  "manifest_hash": "6103318b5ebeb13b1a63450d8ed6e02d7f05785bc212128016f76846a0c21aca",
  "path": "results/gross9_structural_clock_bundle_g9cb7_preregistration_2026-07-31.json",
  "path_type": "regular_file",
  "protocol_implementation_commit": "39cd0c4233cc879a0a5461be2ab76f3bd30ae36c",
  "protocol_version": "gross9_structural_clock_bundle_g9cb7_preregistration_v1",
  "seal_commit": "ededa5df4c5b5b91588765995ed7b1c502332925",
  "sha256": "981caec711869f3c13b295d655b9b90ac6a863d3ce818f66e9df16213d939a94",
  "size_bytes": 59280
}
```

The exact `C7` binding is:

```json
{
  "claim_hash": "6b431ad2a1a389917e66d4911182d494c2141e6edef68d40bcab447bb0cc5159",
  "filesystem_mode_octal": "0444",
  "git_blob": "a3bb363b08a9b05deb255df216fcc35403761505",
  "git_mode": "100644",
  "path": "results/gross9_structural_clock_bundle_g9cb7_access_claim_2026-07-31.json",
  "path_type": "regular_file",
  "protocol_parent_commit": "ededa5df4c5b5b91588765995ed7b1c502332925",
  "protocol_version": "gross9_structural_clock_bundle_g9cb7_v1",
  "seal_commit": "ff1a8907d19c97beeef0bd7d2797e3bacce17617",
  "sha256": "f1bfdd168194c618d945e183eac8035c73d4cbdc095f50039e044d7278a74fdc",
  "size_bytes": 16901
}
```

`C7` records exact zero access at claim publication:

```json
{
  "candidate_rows_opened": 0,
  "comparator_clock_rows_opened": 0,
  "pre2025_anchor_value_rows_opened": 0,
  "runtime_modules_imported": 0,
  "source_value_rows_opened": 0
}
```

These claim-time counters do not by themselves prove the later production
attempt's exposure. That exposure is fixed separately below from control flow.

## G9CB-7 pre-sentinel production closure

### Controlling rule

`A7`, `P7`, and `C7` freeze a one-shot policy with `retry_allowed = false`,
`resume_allowed = false`, and terminal closure on any non-synthetic production
fault. Cleanup of an environmental residue cannot authorize a second
`G9CB-7 --produce` invocation.

### Sole observed production invocation and command-shape deviation

The exact tool-submitted shell command was:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --produce
```

`A7` required the following exact wrapper:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
```

The submitted command omitted only the `/usr/bin/time -v` diagnostic wrapper;
the child `uv run python -B -m ... --produce` command and environment
assignments were otherwise identical. The omission is nevertheless an exact
command-shape violation. It independently closes `G9CB-7` under `A7` and
cannot be repaired by a second invocation. This document does not relabel the
submitted command as the byte-exact canonical wrapper.

It exited with status `1`. Stdout was empty. The terminal exception was:

```text
TerminalG9CB7Failure: repository bytecode-cache directory exists: /tmp/rllm-alpha-orthogonal-20260718/tests
```

The relevant exact control-flow order was:

```text
main
  -> produce_one_shot
  -> _validate_claim_commit
  -> _validate_environment
  -> _validate_regular_hashed_inputs
  -> _validate_static_closures
  -> parent_authentication_sha256
  -> _validate_bytecode_preflight
  -> os.walk(repository_root)
  -> reject tests/__pycache__
```

The failure occurred before construction of `_PublicationContext` and therefore
before all of the following:

```text
C7_PRODUCTION_PREFLIGHT path-state validation
O_TMPFILE publication capability probe
worker-stage reservation
attempt-consumed sentinel publication
worker capability creation or consumption
worker process creation
runtime-root import inside a worker
source-value decode
feature construction or model/history deserialization or value access
candidate interval construction
canonical CSV or final-manifest publication
```

No probe leaf, stage, sentinel, worker ledger, CSV, or final manifest was
created.

### Authenticated opaque exposure

Before the bytecode preflight, the parent authenticated the claim-only Git
topology, environment, static closure, protocol bindings, and all declared
regular hashed inputs. The retained parent snapshot contained exactly:

```text
60 unique preregistration-bound paths
+ P7 active preregistration
+ C7 active claim
= 62 unique paths
```

The exact authenticated byte total was:

```text
178,534,197 preregistration-bound bytes
+    59,280 P7 bytes
+    16,901 C7 bytes
= 178,610,378 bytes
```

The process opened those bytes for authentication, including model and history
artifacts as opaque bytes, and decoded permitted metadata JSON and static
Python ASTs. It did not deserialize an official model/history artifact or
decompress or decode an official source gzip, CSV, JSONL, or NPZ value; did not
launch a worker; and did not compute a candidate, comparator, return, PnL,
funding cash, CAGR, MDD, economic metric, or overlap metric. This is a
control-flow deduction from the exact failure point, not a syscall trace.

### Root cause

The repository contained two ignored, untracked mode-`0755` bytecode-cache
directories before the observed non-synthetic production invocation:

```text
training/__pycache__
tests/__pycache__
```

They contained exactly five ignored, untracked mode-`0644` `.pyc` files with
a total size of `710,879` bytes:

```json
[
  {
    "observed_sha256": "fb67bf1d93b48da2567234512d2638936d38d61b55490e79e2d5f0ba2bb66a64",
    "observed_size_bytes": 262663,
    "relative_path": "tests/__pycache__/test_build_gross9_structural_clock_bundle.cpython-310.pyc"
  },
  {
    "observed_sha256": "07b987af9ea0c36ff22393f570b7ef3bea5d54b41bd1719bd24720ca677f7193",
    "observed_size_bytes": 72711,
    "relative_path": "tests/__pycache__/test_preregister_gross9_structural_clock_bundle.cpython-310.pyc"
  },
  {
    "observed_sha256": "aaca7fe793f25e49fdb062101800e4bb919ff3c92fe13427f370393809f20005",
    "observed_size_bytes": 190,
    "relative_path": "training/__pycache__/__init__.cpython-310.pyc"
  },
  {
    "observed_sha256": "5328dfa8045ccdd65aae22fcd67dd079fdd111e602f204719f8392c8c7e1a1bf",
    "observed_size_bytes": 261500,
    "relative_path": "training/__pycache__/build_gross9_structural_clock_bundle.cpython-310.pyc"
  },
  {
    "observed_sha256": "83dd1b636e39a20294563478ae6d6acb5244af087e889ed6fc672e837011b939",
    "observed_size_bytes": 113815,
    "relative_path": "training/__pycache__/preregister_gross9_structural_clock_bundle.cpython-310.pyc"
  }
]
```

The exact process that created each ignored cache is not recoverable and must
not be inferred. Their presence, exact contents, and Git-ignored status are
repository-authenticated facts. The preflight rejection is observed execution
evidence consistent with those facts. The active fixed worker cache prefix
`results/.g9cb7-bytecode-cache-disabled` remained absent.

The protocol correctly rejected repository bytecode. The operational defect
was failure to preserve the required no-bytecode repository state between Q7
validation and the observed production attempt. Expanding and sharing the
preflight at the first P8/C8/D8 filesystem gate narrows any future failure
exposure but does not change candidate or output semantics.

### Closure conclusion

The exact conclusions are:

```text
G9CB-7 complete = false
G9CB-7 retryable = false
G9CB-7 resumable = false
G9CB-7 repairable = false
P7 exists = true
C7 exists = true
D7 exists = false
T7 exists = false
observed G9CB-7 preregistration invocations = 1
observed G9CB-7 claim invocations = 1
observed non-synthetic G9CB-7 production invocations = 1
observed byte-exact A7 production-wrapper invocations = 0
deduced sentinel publications in the sole production invocation = 0
deduced worker launches in the sole production invocation = 0
```

The exact permanently absent production outputs are:

```text
results/gross9_structural_clock_bundle_g9cb7_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb7_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb7_manifest_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb7_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb7_worker_capability_consumed_pass2_2026-07-31.json
```

The following active residues are also absent:

```text
results/.g9cb7-bytecode-cache-disabled
results/.g9cb7-otmpfile-probe-*
results/.gross9-structural-clock-g9cb7-worker-*
results/.gross9_structural_clock_bundle_g9cb7_*.stage-*
```

`P7` and `C7` are historical metadata, not a candidate clock authority.

## Authority-ordered bytecode cleanup

Only after this document is committed and pushed alone as `A8` may the two
ignored cache directories be removed. Cleanup is authorized only if all five
files still match the exact relative paths, modes, sizes, and SHA-256 values
above and neither directory contains another entry.

The cleanup must:

1. require the exact branch, `HEAD == @{upstream} ==` exact committed `A8`, a
   clean index and tracked worktree, `A8` as the single-parent direct child of
   `C7`, and an `A8` diff that adds only this authority document;
2. confirm both cache paths are ignored and untracked non-symlink mode-`0755`
   directories and retain their parent and directory descriptors with
   `O_DIRECTORY | O_NOFOLLOW`;
3. open each of the exact five leaves descriptor-relative with `O_NOFOLLOW`,
   require a non-symlink mode-`0644` regular file, hash opaque bytes from that
   same descriptor, and require the exact size and SHA-256 above;
4. retain and compare device/inode/type/mode/size/time tokens before and after
   hashing, require exact directory-entry inventories, and perform an
   immediate final descriptor-based identity and inventory recheck before any
   deletion;
5. abort without deleting anything if any Git, path, type, mode, identity,
   inventory, size, hash, or final-recheck condition differs;
6. unlink only the five authenticated leaf names relative to the retained
   cache-directory descriptors, verify both retained directories are empty,
   and remove exactly the two cache-directory names relative to retained
   non-symlink `training` and `tests` parent descriptors;
7. verify that both paths are absent and that no `__pycache__`, `.pyc`, or
   `.pyo` remains anywhere in the worktree outside `.git`;
8. leave tracked bytes, index, `P7`, `C7`, and all historical results
   unchanged; and
9. record no source, model, history, candidate, or economic value.

The cleanup is successor preparation. It does not reopen `G9CB-7` and does not
authorize a second `G9CB-7 --produce` invocation.

## Required Q8 protocol

`Q8` must be the direct child of clean pushed `A8` and must change exactly:

```text
tests/test_build_gross9_structural_clock_bundle.py
tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
tests/test_preregister_gross9_structural_clock_bundle.py
training/build_gross9_structural_clock_bundle.py
training/preregister_gross9_structural_clock_bundle.py
```

It must:

1. bind `A8`, `A7`, `Q7`, `P7`, and `C7` at their exact Git objects and
   hashes;
2. add one ordered `failed_predecessor_pre_sentinel_closures` container whose
   sole row is the exact `G9CB-7` closure stated here;
3. include that container through the same shared predecessor-state helper in
   both manifest and initial bootstrap declaration construction;
4. require the exact historical `P7` and `C7` bytes, modes, Git pairs, hashes,
   manifest/claim hashes, and direct-child topology;
5. require permanent absence of all five `G9CB-7` production outputs and all
   `g9cb7` probe, stage, worker-stage, and fixed-cache residues;
6. expand one shared strict repository-bytecode helper to reject
   `__pycache__`, `.pyc`, and `.pyo`;
7. invoke that helper for `P8` before topology/bootstrap snapshot work, for
   `C8` before claim preflight/snapshot work, and for `D8` before
   `_validate_claim_commit` or any parent-input snapshot construction; each
   call must be the first filesystem gate after CLI/worker discrimination and
   repository-root canonicalization;
8. preserve strict paired-null Git absence proof for ignored frozen OI;
9. preserve exact bootstrap/manifest bound-path parity with sorted mismatch
   diagnostics;
10. preserve every inherited candidate-independent mechanic unchanged; and
11. add synthetic regressions for all of the above, including `__pycache__`,
    `.pyc`, and `.pyo` cases and proof at each P8/C8/D8 entry point that a
    bytecode failure occurs before Git pair classification, bound-file open,
    publication context construction, probe, or link.

All validation and review commands after authority-ordered cleanup must set
`PYTHONDONTWRITEBYTECODE=1`; Python entry points must also use `-B`. Reviewers
must not regenerate repository bytecode.

Before `Q8` may be committed, the exact five-file suite must pass, AST parsing
and diff checks must pass, Ruff and Pyright must introduce no diagnostics on
added lines, bootstrap/manifest declaration parity must be proven, all active
`G9CB-8` paths must be absent, and an independent reviewer must approve.

## P8, C8, and D8 gates

After clean pushed `Q8`:

```text
P8: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
C8: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
D8: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
V8: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
```

1. confirm the shared bytecode preflight is clean and run the sole canonical
   `P8` preregistration command once;
2. authenticate, force-add, commit, and push only `P8` as the direct child of
   `Q8`;
3. confirm the shared bytecode preflight is still clean and run the sole
   canonical `C8 --create-claim` command once;
4. authenticate, force-add, commit, and push only `C8` as the direct child of
   `P8`;
5. confirm the shared bytecode preflight is still clean;
6. run the sole canonical `D8 --produce` command once with the exact
   `/usr/bin/time -v` wrapper required by the successor protocol;
7. on success, force-add exactly the sentinel, two worker ledgers, canonical
   CSV gzip, and final manifest as the direct child of `C8`;
8. push and run committed-publication verification; and
9. only after successful committed verification may any candidate or economic
   evaluator open the clock output.

Any non-synthetic failure closes `G9CB-8` immediately. No same-identity retry,
resume, repair, partial completion, or output salvage is authorized.

## Final authority

`G9CB-7` is permanently closed with immutable `P7` and `C7`, one observed
pre-sentinel production failure, zero sentinel publications, zero worker
launches, and no candidate clock authority. `G9CB-8` is the sole authorized
continuation, preserving the exact candidate-independent object while closing
the bytecode-hygiene gap and binding the complete predecessor history.
