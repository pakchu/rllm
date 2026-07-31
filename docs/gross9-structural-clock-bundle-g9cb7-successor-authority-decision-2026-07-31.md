# Gross9 structural clock bundle G9CB-7 successor authority decision — 2026-07-31

## Decision

Freeze `G9CB-7` as a new candidate-independent structural-clock
infrastructure identity and the only authorized successor after the
prepublication closure of `G9CB-6`.

The exact last clean pushed `G9CB-6` implementation commit is:

```text
Q6 = 86c7076e415ed667560bfe41c942ab4a00c75a4d
```

The sole canonical non-synthetic `G9CB-6` preregistration invocation faulted
before publication. Under the controlling pre-sentinel closure rule in `A6`,
`G9CB-6` closed immediately. It is not retryable, resumable, repairable, or
completable under the same identity. `P6`, `C6`, `D6`, and a `G9CB-6`
terminal-evidence commit do not exist and must never be created.

The only canonical continuation authorized by this decision is:

```text
Q6 -> A7 -> Q7 -> P7 -> C7 -> D7
```

Every commit is single-parent, clean, pushed, and a fast-forward on:

```text
codex/gross9-structural-clock-bundle-20260731
```

History rewriting and force-pushing remain forbidden.

This decision inherits every compatible candidate-independent economic and
structural mechanic frozen by the authority chain through `A6`. It changes
only:

1. the active infrastructure identity from `G9CB-6` to `G9CB-7`;
2. the active output, residue, exception, action, phase, and protocol-version
   literals from `g9cb6` to `g9cb7`;
3. the exact `Q6 -> A7 -> Q7 -> P7 -> C7 -> D7` topology;
4. an exact prepublication-closure record for `G9CB-6`;
5. inclusion of the complete inherited prepublication-closure binding in the
   initial retained snapshot declaration set;
6. one authority-ordered, opaque, byte-identical materialization of the
   ignored frozen-OI logical path as a regular file;
7. correction of active frozen-OI provenance after that materialization;
8. paired-null Git preauthentication for declared ignored inputs while
   preserving required-tracked classification for protocol and active
   authority paths; and
9. synthetic regressions for those changes.

No candidate, candidate rank, comparator clock, return, PnL, funding cash,
CAGR, MDD, economic metric, overlap metric, sleeve weight, signal rule, hold,
barrier, model, feature, source content, source digest, or source logical path
may change.

This file is not operative merely because it exists. It becomes `A7` only
when committed and pushed as the sole change in the direct child of exact
`Q6`.

## Evidence provenance and recoverability

This decision separates four evidence classes:

1. **repository-authenticated facts** are reproducible from committed Git
   objects, the current worktree, and filesystem metadata;
2. **observed execution evidence** is supplied by the current session's sole
   canonical `G9CB-6` command result;
3. **control-flow deductions** follow from exact committed `Q6` source, the
   exception location, and the declaration sets; and
4. **unrecoverable facts** remain JSON null and are not inferred from artifact
   absence.

No standalone raw stdout/stderr capture file, kernel-level open trace,
wall-clock timestamp record, duration record, or resource-usage record was
preserved. The normalized command, Python traceback text, and process exit
status were observed in the current session and are frozen here, but are not
represented as byte-exact shell-input or raw-stream artifacts.

The 58-path and `105,571,805`-byte exposure counts are control-flow
deductions from the sorted bootstrap declaration set, the completed retained
snapshot, the exact failure location, and authenticated file sizes. They are
not syscall-trace counts. The in-memory manifest's 59-path declaration set is
reproducible without opening or decoding an official source value.

Exactly one preregistration invocation is observed in the current `Q6`
continuation. This decision does not infer a global production-invocation
count from artifact absence. The `G9CB-6` official production invocation
count therefore remains JSON null.

## Authenticated repository facts

At drafting time, the following facts are authenticated without decoding a
new candidate, comparator clock, or economic result:

- `HEAD == @{upstream} == Q6`;
- `Q6` is `86c7076e415ed667560bfe41c942ab4a00c75a4d`;
- `Q6` is the direct child of `A6`,
  `2695ee61fbb9b5e053dbb9da597ebe2729aad361`;
- the `A6 -> Q6` diff changes exactly the five authorized protocol and test
  files;
- the index and tracked worktree are clean;
- every active `G9CB-6` result path is absent;
- every active `G9CB-6` publication stage, worker stage, capability-probe
  leaf, and fixed-bytecode path is absent;
- the corrected tracked top-level `results/` inventory contains 1,339
  entries and the actual inventory contains only those entries plus the two
  exact retained predecessor residue directories; and
- this authority decision is the sole new tracked candidate for `A7`.

The exact `A6` authority binding is:

```json
{
  "authority_commit": "2695ee61fbb9b5e053dbb9da597ebe2729aad361",
  "git_blob": "eb743d9f8ecd878b83f8f8873697c58cccef9f1b",
  "git_mode": "100644",
  "path": "docs/gross9-structural-clock-bundle-g9cb6-successor-authority-decision-2026-07-31.md",
  "path_type": "regular_file",
  "sha256": "b64f9480741eeb4f69ac86736589fbcf8fb75565c436d76316b73f5e076acfca"
}
```

The exact `Q6` implementation binding is:

```json
{
  "builder_git_blob": "09cb9757a230c349cd7b7df9f7ce4a20cfa9b30c",
  "builder_path": "training/build_gross9_structural_clock_bundle.py",
  "builder_sha256": "4fe465368fa074536e85e2e0b54e4ff4800b4cd8a034510015bef78a66d9db93",
  "commit": "86c7076e415ed667560bfe41c942ab4a00c75a4d",
  "preregistration_git_blob": "af809793347a647632f07ab1d74f5fbeabaac122",
  "preregistration_path": "training/preregister_gross9_structural_clock_bundle.py",
  "preregistration_sha256": "5a04a8616a7c8416e67f349f8fd4a846fda87786c0f54fb1415dcf924bb17374"
}
```

## G9CB-6 closure

### Controlling rule

`A6` states that, before the `G9CB-6` sentinel, any non-synthetic protocol
fault closes `G9CB-6` without same-identity repair and that continuation
requires a new identity and a standalone successor authority decision.

The sole canonical preregistration invocation therefore closed `G9CB-6`
even though no active artifact was published.

### Operative closure event

The normalized invocation was:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
```

It exited with status `1`. The exact terminal exception was:

```text
ValueError: manifest bound path set differs from bootstrap
```

The failure occurred in this control-flow chain:

```text
main
  -> _bootstrap_q6_snapshot
  -> build_manifest
  -> write_once
  -> _snapshot_declarations
  -> bound-path set comparison
```

`_bootstrap_q6_snapshot` constructed its declaration payload from exactly:

```text
predecessor
successor_preregistrations
failed_attempts
failed_closures
```

It omitted the already-authorized and manifest-required container:

```text
failed_prepublication_closures
```

After inherited-protocol replacement, the retained bootstrap snapshot
contained exactly 58 unique paths. It opened all 58 once as opaque bytes,
authenticated their declared SHA-256 values where present, authenticated
their paired Git state where required, and retained their descriptors. The
exact total was `105,571,805` bytes.

The process then constructed and validated an in-memory manifest. It decoded
permitted metadata JSON, parsed static Python ASTs, authenticated environment
and Git metadata, and used only retained opaque bytes for declared current
files. It did not decode or load an official source gzip, CSV, JSONL, NPZ,
model, or history value and did not compute a candidate, comparator, return,
PnL, funding cash, CAGR, MDD, economic metric, or overlap metric.

The complete in-memory manifest declaration set contained exactly 59 unique
paths. The exact set difference was:

```text
manifest minus bootstrap:
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz

bootstrap minus manifest:
<empty>
```

That path entered the manifest through the exact historical `G9CB-5`
prepublication-closure recovery-exposure binding. At failure time its ignored
worktree leaf was a symbolic link whose link text was exactly:

```text
/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz
```

The declared referent was a regular file of `72,898,508` bytes with SHA-256:

```text
dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192
```

The mismatch check occurs before `_validate_closed_path_state`, before the
publication capability probe, before `O_TMPFILE` preparation, before the
snapshot final recheck, and before any hard-link publication. The process
therefore created no file, directory, probe leaf, stage, sentinel, worker,
claim, bundle, manifest, bytecode cache, or active authority artifact.

### Closure conclusion

The exact closure conclusions are:

```text
G9CB-6 complete = false
G9CB-6 retryable = false
G9CB-6 resumable = false
G9CB-6 repairable = false
P6 exists = false
C6 exists = false
D6 exists = false
T6 exists = false
observed G9CB-6 preregistration invocations = 1
observed G9CB-6 canonical publication links = 0
G9CB-6 official production invocations = null
```

The JSON null is unknown and must not be interpreted as zero.

The exact permanently absent `G9CB-6` outputs are the following sorted list:

```text
results/gross9_structural_clock_bundle_g9cb6_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb6_access_claim_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb6_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb6_manifest_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass2_2026-07-31.json
```

Every path above remains absent through `D7`. No future identity may reuse,
replace, rename, or hard-link any of them as an active artifact.

The exact `G9CB-6` residue state is:

```json
{
  "bytecode_cache": {
    "path": "results/.g9cb6-bytecode-cache-disabled",
    "state": "absent"
  },
  "capability_probes": {
    "glob": "results/.g9cb6-otmpfile-probe-*",
    "state": "absent"
  },
  "publication_stages": {
    "glob": "results/.gross9_structural_clock_bundle_g9cb6_*.stage-*",
    "state": "absent"
  },
  "worker_stages": {
    "glob": "results/.gross9-structural-clock-g9cb6-worker-*",
    "state": "absent"
  }
}
```

## Required G9CB-6 prepublication-closure binding

`P7` must preserve the exact `G9CB-5` row and append one exact `G9CB-6` row
to:

```text
bindings.failed_predecessor_prepublication_closures
```

The list order is exactly `G9CB-5`, then `G9CB-6`. No other predecessor list
changes.

The `G9CB-6` row has exactly these top-level keys:

```text
authority_decision
classification
failure
identity
input_materialization
permanently_absent_outputs
protocol_implementation
protocol_version
residue
root_cause
status
topology
```

Its exact scalar classification is:

```json
{
  "classification": "pre_preregistration_publication_bootstrap_manifest_bound_path_set_mismatch",
  "identity": "G9CB-6",
  "protocol_version": "gross9_structural_clock_bundle_g9cb6_v1",
  "status": "historical_prepublication_closure_no_preregistration_no_attempt_no_clock_authority"
}
```

`authority_decision` is the exact `A6` binding above.
`protocol_implementation` is the exact `Q6` binding above.

`failure` is exactly:

```json
{
  "bytes_opened": 105571805,
  "exception": "ValueError: manifest bound path set differs from bootstrap",
  "exit_status": 1,
  "manifest_constructed": true,
  "metadata_json_decoded": true,
  "normalized_invocation": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle",
  "observed_preregistration_invocations": 1,
  "official_production_invocations": null,
  "paths_opened": 58,
  "preregistration_published": false,
  "publication_capability_probe_started": false,
  "runtime_python_ast_parsed": true,
  "snapshot_final_recheck_completed": false,
  "source_model_or_history_values_decoded_or_loaded": false,
  "status": "authorized_first_invocation_closed_identity"
}
```

`root_cause` is exactly:

```json
{
  "bootstrap_bound_path_count": 58,
  "bootstrap_missing_container": "failed_prepublication_closures",
  "bootstrap_minus_manifest": [],
  "manifest_bound_path_count": 59,
  "manifest_minus_bootstrap": [
    {
      "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
      "sha256": "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
      "size_bytes": 72898508
    }
  ],
  "publication_state_validation_started": false,
  "set_comparison_location": "write_once_retained_snapshot_before_results_parent_lookup"
}
```

`input_materialization` records the exact post-`A7`, pre-`Q7` recovery and is
exactly:

```json
{
  "authority_order": "after_clean_pushed_A7_before_Q7",
  "destination": {
    "git_blob": null,
    "git_mode": null,
    "mode_octal": "0444",
    "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
    "path_type": "regular_file",
    "sha256": "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
    "size_bytes": 72898508
  },
  "source": {
    "absolute_path": "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
    "expected_sha256": "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
    "path_type": "regular_file",
    "size_bytes": 72898508
  },
  "source_values_decoded": false,
  "status": "opaque_byte_identical_symlink_replaced_by_regular_file"
}
```

`topology` is exactly:

```json
{
  "g9cb6_authority_commit": "2695ee61fbb9b5e053dbb9da597ebe2729aad361",
  "g9cb6_protocol_commit": "86c7076e415ed667560bfe41c942ab4a00c75a4d",
  "g9cb7_authority_commit": "<A7>",
  "preregistration_commit": null,
  "terminal_evidence_commit": null
}
```

`"<A7>"` is a metavariable replaced at `P7` by the exact lowercase 40-hex
commit that adds only this decision.

## Authority-ordered opaque input materialization

The `G9CB-5` recovery disclosure made the frozen OI logical path a required
SHA-bound declaration. At the `G9CB-6` failure it remained an ignored
symbolic link. The retained snapshot intentionally uses component-wise
`O_NOFOLLOW` and accepts only regular-file leaves. That protection must not be
weakened or special-cased for `G9CB-7`.

Only after clean pushed `A7`, and before any `Q7` test that constructs the
official snapshot, this decision authorizes one local recovery operation:

1. open the exact absolute source with `O_NOFOLLOW` and require a regular
   file;
2. require stable descriptor metadata, size `72,898,508`, and SHA-256
   `dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192`;
3. require the destination leaf still to be the exact symbolic link and link
   text recorded above;
4. copy bytes opaquely to a create-exclusive temporary regular file in the
   destination directory without decompression, parsing, or value access;
5. require the temporary destination's device/inode identity to differ from
   the source, set its mode to `0444`, and fsync it;
6. atomically replace only that symbolic-link leaf, fsync the destination
   directory, and remove any temporary residue; and
7. require the destination to be a non-symlink, non-hard-link regular file
   with mode `0444` and the exact size and SHA-256 above.

The operation changes no tracked byte, source content, source digest, or
logical path. It changes only the ignored worktree leaf type from symbolic
link to byte-identical regular file so the existing no-follow snapshot can
authenticate and retain it without alias or TOCTOU weakening.

If any precondition differs, the operation is forbidden and `Q7` must not be
created. A symlink, hard-link alias, wrong digest, wrong size, wrong source
path, wrong destination path, or leftover temporary file is a stop condition.

## G9CB-7 identity and paths

The exact active literals are:

```text
identity = G9CB-7
preregistration protocol = gross9_structural_clock_bundle_g9cb7_preregistration_v1
builder/publication protocol = gross9_structural_clock_bundle_g9cb7_v1
terminal exception = TerminalG9CB7Failure
terminal action = TERMINAL_G9CB7_ATTEMPT_CONSUMED_NO_RETRY
preregistration phase = Q7_PREREGISTRATION_PUBLICATION
fixed bytecode prefix = results/.g9cb7-bytecode-cache-disabled
worker stage prefix = results/.gross9-structural-clock-g9cb7-worker-
```

The exact active paths are:

| Role | Path |
|---|---|
| authority | `docs/gross9-structural-clock-bundle-g9cb7-successor-authority-decision-2026-07-31.md` |
| preregistration | `results/gross9_structural_clock_bundle_g9cb7_preregistration_2026-07-31.json` |
| access claim | `results/gross9_structural_clock_bundle_g9cb7_access_claim_2026-07-31.json` |
| attempt sentinel | `results/gross9_structural_clock_bundle_g9cb7_attempt_consumed_2026-07-31.json` |
| worker ledger pass 1 | `results/gross9_structural_clock_bundle_g9cb7_worker_capability_consumed_pass1_2026-07-31.json` |
| worker ledger pass 2 | `results/gross9_structural_clock_bundle_g9cb7_worker_capability_consumed_pass2_2026-07-31.json` |
| canonical CSV gzip | `results/gross9_structural_clock_bundle_g9cb7_2026-07-31.csv.gz` |
| final manifest | `results/gross9_structural_clock_bundle_g9cb7_manifest_2026-07-31.json` |

No `G9CB-6` active path may be reused.

## Complete bootstrap declaration rule

The `Q7` bootstrap payload must contain exactly the same inherited binding
containers that the in-memory `P7` manifest contains before active `Q7`
protocol paths are overlaid:

```text
predecessor
successor_preregistrations
failed_attempts
failed_closures
failed_prepublication_closures
```

The last container must include both exact rows in order. It may not be
omitted, projected, lazily appended after file opening, or treated as a
literal-only disclosure. Every SHA binding recursively reachable through it,
except the existing explicit `protocol_implementation` exclusion, belongs to
the initial declaration set.

`Q7` must construct these inherited containers through one shared helper used
by both manifest composition and bootstrap composition, rather than two
independently maintained literal dictionaries.

After the active `PROTOCOL_PATHS` overlay and replacement of superseded
historical protocol paths, the bootstrap and complete manifest bound-path
sets must be identical before publication state validation. Every declared
leaf must be opened once into the retained initial snapshot as a no-follow
regular file, and every declared digest and Git pair must be authenticated.
Any mismatch exception must include both sorted path-only differences,
`manifest_minus_bootstrap` and `bootstrap_minus_manifest`; it must not expose
file contents or continue to publication-state validation.

No recovery-exposure key is globally exempted from SHA binding discovery. No
absolute-path allowlist is broadened. No symlink-following behavior is added.

## Active frozen-OI provenance and paired-null Git classification

After the authority-ordered materialization, the live frozen-OI leaf resolves
to itself inside the `Q7` worktree. The external `/home/pakchu/rllm/...` path
remains only historical `G9CB-5` recovery provenance and the exact source of
the authorized opaque copy. `P7` must not describe that external path as the
live resolved input.

The active `P7` disclosure must report the frozen-OI logical and resolved
repository paths as the same normalized repository-relative path:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz
```

It must report live path type `regular_file`, exact mode `0444`, size
`72,898,508`, and the frozen SHA-256. With an active retained snapshot, those
facts must be checked against the cached no-follow descriptor and bytes. With
no active snapshot, they must be checked by a no-follow regular-file read.
A live symlink is forbidden.

The frozen-OI file and other declared runtime inputs under ignored `data/`
paths are intentionally untracked. Their manifest bindings carry exact
paired-null Git metadata:

```json
{"git_blob": null, "git_mode": null}
```

`Q7` must correct builder preauthentication so repository-relative bindings
with paired-null declarations are required to remain absent from both index
and `HEAD`, while declarations with non-null Git pairs or explicit
required-tracked authority remain exact tracked stage-zero blobs. Absolute
allowlisted inputs remain paired-null and outside Git. This distinction may
not weaken tracking requirements for protocol files, authority decisions,
preregistration/claim/sentinel artifacts, or any other binding that declares
a Git blob and mode.

## Frozen value-access and economic boundary

The historical `G9CB-5` OI recovery exposure and the `G9CB-6` opaque reads
remain explicit and immutable. `P7` creation-evidence counters are
process-local counters for the single successful `P7` process; they do not
erase or contradict historical exposure.

The authority-ordered materialization copies opaque bytes only. It does not
decompress the gzip member, inspect a header, decode a row, load a field, or
compute any candidate or economic quantity.

No candidate identity or candidate artifact has been introduced. The exact
domain, source bytes and hashes, feature logic, schedules, five sleeves,
order, configured weights, sides, holds, barriers, Rank7 model and history
bytes, two-pass worker-capability protocol, counters, serialization, and
economic prohibitions remain unchanged from `A6`.

## Authorized Q7 diff

`A7` adds only this file. `Q7` is the direct child of clean pushed `A7` and
may modify exactly these five paths:

```text
tests/test_build_gross9_structural_clock_bundle.py
tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
tests/test_preregister_gross9_structural_clock_bundle.py
training/build_gross9_structural_clock_bundle.py
training/preregister_gross9_structural_clock_bundle.py
```

`Q7` may make only the identity, topology, closure-binding, complete-bootstrap
declaration, permanent-absence, materialization-verification, active-
provenance, paired-null preauthentication, and synthetic-regression changes
authorized here. It may not alter any frozen source byte, economic object,
structural clock rule, model, schedule, signal, side, hold, barrier, weight,
serializer, worker-capability ordering, or publication ordering.

No active `G9CB-7` artifact may exist at `Q7`.

## Required regression evidence before Q7

Before `Q7` is committed and pushed, regressions must prove at minimum:

1. the exact `Q6 -> A7 -> Q7` single-parent topology and exact diffs;
2. the `G9CB-6` closure binding contains the exact `A6`, `Q6`, failure,
   root-cause, materialization, permanent-absence, residue, and topology facts
   above;
3. every `G9CB-6` active path remains absent;
4. the existing predecessor lists remain unchanged and the prepublication-
   closure list is exactly ordered `G9CB-5`, `G9CB-6`;
5. the active protocol inventory includes exact `A7`, `A6`, and every
   inherited protocol path once each;
6. the bootstrap and complete manifest declaration sets are equal and include
   every binding recursively reachable through both prepublication closures;
7. omission of either prepublication-closure row or its unique OI binding is
   rejected before publication with both sorted path-only set differences;
8. the frozen OI logical leaf is an untracked, paired-null, non-symlink
   mode-`0444` regular file with exact size and SHA-256, a distinct inode from
   the materialization source, and live repository-relative provenance;
9. no symlink, hard-link alias, descriptor swap, Git pair mutation, directory
   substitution, inventory drift, or path-set drift is accepted;
10. builder preauthentication accepts exact paired-null ignored and absolute
    inputs, rejects tracked-state drift for them, and still requires exact Git
    pairs for tracked protocol and authority bindings;
11. all `g9cb6` active path, phase, exception, action, stage, and bytecode
    literals are historical-only and cannot become active authority;
12. all active outputs use only `g9cb7` paths;
13. synthetic `P7`, `C7`, `D7`, and committed-verifier topology succeeds;
14. every inherited authentication, boundary, isolation, serialization,
    two-pass, exact-results-inventory, and publication regression remains
    passing; and
15. no test decodes an official source value or computes economics.

## Canonical commands and one-shot rule

After clean pushed `Q7`, the sole canonical preregistration command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
```

After `P7` is committed and pushed, the sole canonical claim command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
```

After `C7` is committed and pushed, the sole canonical production command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
```

After `D7` is committed and pushed, the metadata-only committed verifier is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
```

No rehearsal, official-source probe, partial production, retry, or second
invocation is authorized. Before the `G9CB-7` sentinel, any non-synthetic
protocol fault closes `G9CB-7` without same-identity repair. After the
sentinel, every failure is terminal under:

```text
TERMINAL_G9CB7_ATTEMPT_CONSUMED_NO_RETRY
```

Any continuation after either closure requires another new identity and a
new standalone successor authority decision.

## Stop conditions

Stop before `P7` unless all of the following are true:

- `A7` is the sole-change direct child of exact `Q6` and is pushed;
- the authority-ordered OI materialization completed exactly and left a
  non-symlink regular file with the frozen size and SHA-256;
- `Q7` is the exact-five-change direct child of `A7` and is pushed;
- the full owned regression suite and independent review pass;
- the complete declared runtime-input set is present as exact regular files
  before launching the canonical process;
- the bootstrap and manifest bound-path sets are independently shown equal;
- active frozen-OI provenance is repository-relative and builder paired-null
  preauthentication has passed;
- the actual top-level `results/` inventory equals the corrected authenticated
  inventory;
- every `G9CB-6` and `G9CB-7` active artifact and residue is absent; and
- the worktree and index are clean with `HEAD == @{upstream}`.

If any condition differs, do not invoke the canonical command.
