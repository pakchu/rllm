# Gross9 structural clock bundle G9CB-5 successor authority decision — 2026-07-31

## Decision

Freeze `G9CB-5` as a new candidate-independent structural-clock
infrastructure identity and the direct successor of the clean pushed `G9CB-4`
preregistration seal:

```text
P4 = 01de73258902d754905319b906345c865a016558
```

`G9CB-5` is not a retry, resume, repair, amendment, v2, or completion of
`G9CB-4`. `G9CB-4` closed before its access claim and before its sentinel under
the controlling rule at lines 1068–1071 of its authority decision. It is
incomplete, not retryable, and has no `C4`, `D4`, or `T4`.

The only canonical history authorized by this decision is:

```text
P4 -> A5 -> Q5 -> P5 -> C5 -> D5
```

Every commit is single-parent, clean, pushed, and a fast-forward on:

```text
codex/gross9-structural-clock-bundle-20260731
```

History rewriting and force-pushing are forbidden.

This decision inherits without alteration the candidate-independent economic
object and every compatible structural mechanic frozen by:

```text
docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb3-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb4-successor-authority-decision-2026-07-31.md
```

The exact source paths and hashes, source bytes, domain, boundary, signals,
models, model history, features, schedules, sleeves, weights, sides, holds,
barriers, Rank7 policy, counters, economics prohibitions, overlap
prohibitions, dependencies, environment, serializer, two-pass protocol, and
worker-capability token semantics remain frozen, subject only to the exact
ledger-FD transport/publication supersession below.

Only these changes are authorized:

1. the new `G9CB-5` identity and v1-under-`G9CB-5` protocol literals;
2. the exact `P4 -> A5 -> Q5 -> P5 -> C5 -> D5` history;
3. a closed predecessor schema that distinguishes preregistrations,
   attempt-consumed failures, and pre-access/pre-sentinel closure;
4. the exact `G9CB-4` closure and exposure record in this decision;
5. correction of the keyword-only call that closed `G9CB-4`;
6. snapshot-wide Git classification before the first worktree bound-file
   read;
7. one component-wise no-follow open per unique bound path, with a retained
   descriptor, one initial cache read, one final verification read,
   hard-link alias rejection, and descriptor-derived metadata;
8. preregistration-only and preregistration-plus-claim parent snapshot modes;
9. immutable HEAD/index rechecks without worktree `git hash-object` reads;
10. descriptor-anchored unnamed-`O_TMPFILE` write-once publication resistant
    to results-directory or inode substitution;
11. the exact two-descriptor worker handoff and sole guarded ledger-link
    exception defined below;
12. a committed-publication verifier using the same preclassified cached
    snapshot discipline; and
13. synthetic regressions proving each correction without opening official
    values or computing economics.

This file is not operative merely because it exists. It becomes `A5` only
when committed and pushed as the sole change in the direct child of exact
`P4`.

## Evidence boundary

This decision separates authenticated repository facts, supplied execution
evidence, control-flow conclusions, and unrecoverable observations.

### Authenticated repository facts

The following facts are reproducible from Git metadata, the current
preregistration, implementation source, tests, and filesystem metadata
without decoding an official source, model, or history value:

- `HEAD == @{upstream} == P4`;
- `P4` is
  `01de73258902d754905319b906345c865a016558`;
- the sole parent of `P4` is `Q4`,
  `750c837a10c4d4ac39fbc8f6097465c82b6dc3ec`;
- `P4` adds only the `G9CB-4` preregistration;
- the current index and every tracked worktree path match `P4`;
- this decision is the sole current untracked path;
- only the `G9CB-4` preregistration exists among `G9CB-4` result artifacts;
- that preregistration is a tracked Git mode-`100644` regular file and a
  filesystem mode-`0444` regular file;
- every `G9CB-4` claim, sentinel, ledger, CSV, manifest, stage, and fixed
  bytecode-cache path is absent; and
- the implementation reaches the invalid call before claim payload
  construction or write-once publication.

The sealed `G9CB-4` preregistration facts are:

```json
{
  "filesystem_mode_octal": "0444",
  "git_blob": "76f9011d5752282c058feb531442b203a0bbdb0d",
  "git_mode": "100644",
  "manifest_hash": "fa3dab6f7e6ab86428c03fc5c3d7b005e0a165cd76662bba9a7c3cd5941beeed",
  "path": "results/gross9_structural_clock_bundle_g9cb4_preregistration_2026-07-31.json",
  "path_type": "regular_file",
  "protocol_implementation_commit": "750c837a10c4d4ac39fbc8f6097465c82b6dc3ec",
  "protocol_version": "gross9_structural_clock_bundle_g9cb4_preregistration_v1",
  "seal_commit": "01de73258902d754905319b906345c865a016558",
  "sha256": "f65aaf5fd2219f90421912e6fc9065ddffb54f5adf881196986f25185fe7342e",
  "size_bytes": 41289
}
```

The operative `G9CB-4` authority binding is:

```json
{
  "authority_commit": "1156e2fd80957d5ef0a6027a09e08ff59349a80d",
  "git_blob": "2610246e4d9fb89d775fe7d8d1998282d23e5961",
  "git_mode": "100644",
  "path": "docs/gross9-structural-clock-bundle-g9cb4-successor-authority-decision-2026-07-31.md",
  "path_type": "regular_file",
  "sha256": "9199955f62abbb99c8665a5eeee6a32cf9605ba637e2b034d929b1ac91ace626"
}
```

The exact builder SHA-256 at `Q4` and `P4` is:

```text
c7c3bf1f9971e058e719139b50379c356f45a0fcc8f62c12aab100f70fa64c63
```

### Supplied failure evidence

Supplied history states that the tracked worktree and index were clean at
`P4` before this decision was drafted. That historical cleanliness is not a
current authenticated fact and has no independent durable transcript.

The supplied normalized failed invocation was:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
```

This is a normalized invocation, not byte-exact shell text: no raw command
capture survives. The same normalized invocation is also the sole canonical
`C5` metadata command after clean pushed `P5`; it is stated once here
intentionally.

The exact exception was:

```text
TypeError: _validate_git_pair_preflight() takes 2 positional arguments but 4 positional arguments (and 1 keyword-only argument) were given
```

The call site was `_head_blob_binding`, which passed two keyword-only
arguments positionally. The exception occurred during protocol binding inside
claim preflight, before claim payload construction and before any claim
publication operation.

No durable raw stdout or stderr capture remains. Exact raw stdout bytes,
stderr bytes, stdout and stderr hashes, invocation timestamps, duration,
resource usage, and exit status are unrecoverable and must not be invented by
any `G9CB-5` artifact.

The supplied failed process did not enter production. No production artifact
or production residue exists. Whether any separate historical production
invocation occurred is not independently durable and its count is
unrecoverable.

### Conservative pre-closure exposure

Control flow and the sealed `G9CB-4` preregistration establish this exact
conservative exposure before the `TypeError`:

- 63 qualifying bindings representing 55 unique paths were opened as opaque
  bytes and SHA-256 authenticated;
- the eight source files totalled `100,551,601` bytes;
- one Rank7 history gzip and five NPZ models totalled `2,121,609` bytes;
- the pre-2025 anchor and Rank7 manifest totalled `14,680` bytes;
- historical metadata JSON was decoded and checked for canonical form and
  internal hashes where defined; and
- runtime Python source in the declared closure was UTF-8 decoded and parsed
  as Python AST.

The eight authenticated source files were the frozen market, funding,
premium, open-interest, and four REX source paths already bound by `P4`.
Their paths, bytes, sizes, and SHA-256 values remain exactly those in the
`G9CB-4` preregistration. This decision does not restate them as new inputs
and does not authorize changing them.

No gzip member was decompressed. No CSV, JSONL, or NPZ value was decoded,
parsed, or loaded. No source, model, or history value was opened. No feature,
model, schedule, sleeve, worker, capability, economics, or overlap path was
reached by the supplied failed process. It performed no claim write, sentinel
publication, production stage creation, fixed bytecode-cache creation, or
pycache creation.

The 63-to-55 duplication was binding duplication, not duplicate file reads.
The following eight paths appeared in two bindings each and were read once
per unique path:

```text
docs/gross9-structural-clock-bundle-g9cb4-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb3-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md
docs/gross9-structural-clock-bundle-successor-authority-decision-2026-07-31.md
execution/gross9_rank7_clock_runtime.py
training/gross9_structural_clock_primitives.py
```

The statement that all 55 paths were opened is a control-flow conclusion from
the failure location after regular hashed-input, environment, and static
closure validation. No kernel-level open trace survives. This limitation does
not authorize a broader value-access inference.

### Closure conclusion

The failure was pre-access-claim, pre-sentinel, and non-restorable without a
tracked protocol mutation. Under the specific `G9CB-4` pre-sentinel closure
rule, it closes `G9CB-4` without an attempt-consumed record.

Therefore:

```text
G9CB-4 complete = false
G9CB-4 retryable = false
G9CB-4 resumable = false
G9CB-4 repairable = false
C4 exists = false
D4 exists = false
T4 exists = false
```

No same-identity mutation, rerun, metadata correction, protocol v2,
completion attempt, claim creation, or production action is permitted.

## Immutable G9CB-4 pre-sentinel closure

The exact predecessor chain through closure is:

```text
A4 = 1156e2fd80957d5ef0a6027a09e08ff59349a80d
T3 = 04b9e53272ab58537235ad290551607dd071ee17
Q4 = 750c837a10c4d4ac39fbc8f6097465c82b6dc3ec
P4 = 01de73258902d754905319b906345c865a016558
```

The `G9CB-5` preregistration must encode the `G9CB-4` closure as a singleton
closed-schema row separate from attempt-consumed failures. The row has
exactly these keys:

```text
authority_decision
classification
exposure
failure
identity
permanently_absent_outputs
preregistration
protocol_implementation
protocol_version
residue
status
topology
```

Its exact scalar classification fields are:

```json
{
  "classification": "pre_access_claim_pre_sentinel_keyword_only_call_contract_failure",
  "identity": "G9CB-4",
  "protocol_version": "gross9_structural_clock_bundle_g9cb4_v1",
  "status": "historical_pre_access_pre_sentinel_closure_no_attempt_no_clock_authority"
}
```

`authority_decision` is exactly the authority binding above.
`preregistration` is exactly the sealed `P4` preregistration binding above.
`protocol_implementation` is exactly:

```json
{
  "builder_path": "training/build_gross9_structural_clock_bundle.py",
  "builder_sha256": "c7c3bf1f9971e058e719139b50379c356f45a0fcc8f62c12aab100f70fa64c63",
  "commit": "750c837a10c4d4ac39fbc8f6097465c82b6dc3ec"
}
```

`topology` is exactly:

```json
{
  "g9cb4_authority_commit": "1156e2fd80957d5ef0a6027a09e08ff59349a80d",
  "g9cb4_preregistration_commit": "01de73258902d754905319b906345c865a016558",
  "g9cb4_protocol_commit": "750c837a10c4d4ac39fbc8f6097465c82b6dc3ec",
  "g9cb5_authority_commit": "<A5>",
  "terminal_evidence_commit": null
}
```

`"<A5>"` is a metavariable replaced at `P5` by the exact lowercase 40-hex
commit that adds only this decision. `terminal_evidence_commit` is literal
JSON null because there is no `T4`.

`failure` has exactly:

```json
{
  "claim_payload_constructed": false,
  "claim_write_attempted": false,
  "exception": "TypeError: _validate_git_pair_preflight() takes 2 positional arguments but 4 positional arguments (and 1 keyword-only argument) were given",
  "official_production_invocations": null,
  "raw_capture_recoverable": false,
  "sentinel_published": false
}
```

`failure` must not contain raw-capture hashes, sizes, timestamps, duration,
resource usage, or exit status. JSON null means unknown and unrecoverable; it
must not be interpreted as zero.

`exposure` has exactly:

```json
{
  "bindings_authenticated": 63,
  "candidate_rows_opened": 0,
  "claim_files_published": 0,
  "economics_or_overlap_computed": false,
  "features_constructed": false,
  "gzip_csv_jsonl_npz_values_decoded_or_loaded": false,
  "historical_metadata_json_decoded": true,
  "model_or_history_values_opened": 0,
  "pre2025_anchor_and_rank7_manifest_bytes": 14680,
  "production_invocations": null,
  "rank7_history_and_model_bytes": 2121609,
  "runtime_python_ast_parsed": true,
  "schedules_reached": false,
  "source_files": 8,
  "source_files_bytes": 100551601,
  "source_values_opened": 0,
  "unique_paths_authenticated": 55,
  "worker_capabilities_consumed": 0,
  "workers_started": 0
}
```

The `production_invocations` JSON null is unknown and unrecoverable. It must
not be interpreted as zero.

The exact permanently absent `G9CB-4` canonical outputs are:

```text
results/gross9_structural_clock_bundle_g9cb4_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb4_access_claim_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb4_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb4_manifest_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass2_2026-07-31.json
```

`permanently_absent_outputs` is the sorted exact list above. No item may be
added, omitted, renamed, or created through `D5`.

`residue` is exactly:

```json
{
  "bytecode_cache": {
    "path": "results/.g9cb4-bytecode-cache-disabled",
    "state": "absent"
  },
  "publication_stages": {
    "glob": "results/.gross9_structural_clock_bundle_g9cb4_*.stage-*",
    "state": "absent"
  },
  "worker_stages": {
    "glob": "results/.gross9-structural-clock-g9cb4-worker-*",
    "state": "absent"
  }
}
```

The preregistration is not residue. It remains permanently present with its
exact `P4` bytes, Git mode, filesystem mode, blob, SHA-256, manifest hash,
protocol commit, and seal commit.

## G9CB-5 identity and paths

The exact literals are:

```text
identity = G9CB-5
preregistration protocol =
  gross9_structural_clock_bundle_g9cb5_preregistration_v1
builder/publication protocol =
  gross9_structural_clock_bundle_g9cb5_v1
terminal exception = TerminalG9CB5Failure
terminal action = TERMINAL_G9CB5_ATTEMPT_CONSUMED_NO_RETRY
fixed bytecode prefix = results/.g9cb5-bytecode-cache-disabled
worker stage prefix = results/.gross9-structural-clock-g9cb5-worker-
```

The active paths are exactly:

| Role | Path |
|---|---|
| authority | `docs/gross9-structural-clock-bundle-g9cb5-successor-authority-decision-2026-07-31.md` |
| preregistration | `results/gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json` |
| access claim | `results/gross9_structural_clock_bundle_g9cb5_access_claim_2026-07-31.json` |
| attempt sentinel | `results/gross9_structural_clock_bundle_g9cb5_attempt_consumed_2026-07-31.json` |
| worker ledger pass 1 | `results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass1_2026-07-31.json` |
| worker ledger pass 2 | `results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass2_2026-07-31.json` |
| canonical CSV gzip | `results/gross9_structural_clock_bundle_g9cb5_2026-07-31.csv.gz` |
| final manifest | `results/gross9_structural_clock_bundle_g9cb5_manifest_2026-07-31.json` |

No older path may be reused, replaced, renamed, deleted, hard-linked as an
active artifact, or treated as an alias.

## Frozen structural and economic semantics

The exact five sleeves, order, configured weights, side rules, entry delays,
holds, barriers, source routing, and total configured weight `9.0` remain
unchanged.

Weight remains provenance only. It must not become quantity, equity
allocation, leverage, return, PnL, or a portfolio path. The canonical CSV
contains only integer side `1` or `-1`; side `0`, null, `AUTO`, `FLAT`, and
every other side are forbidden.

Intervals remain half-open:

```text
[entry_time_utc, exit_time_utc)
```

Every timestamp remains exact UTC-second text on the Unix-epoch 300-second
grid. Entry is the next five-minute open after a completed causal signal.
Intervals are strictly ordered by entry within each sleeve, duplicate entries
are forbidden within a sleeve, touching intervals are allowed, and
cross-sleeve overlap remains allowed. The per-sleeve non-overlap rule is not a
portfolio-wide deduplication rule.

The exact fixed holds remain:

| Sleeve | Hold |
|---|---:|
| `cand_rex_veto_7` | `144` five-minute bars |
| `markov_transition_long` | `576` five-minute bars |
| `rex_taker_low_range_position` | `144` five-minute bars |

`fresh_kimchi_fx` retains maximum hold `288`, take `400` bps, stop `250` bps,
and stop-before-take resolution when both barriers touch in one occupied bar.

`frozen_annual_rank7` retains:

- funding leg maximum hold `576`, take `400` bps, no enabled stop;
- premium leg maximum hold `144`, stop `300` bps, no enabled take; and
- long-only side `1`.

Barrier exits remain the first five-minute boundary after the first occupied
bar touching the frozen barrier. Fixed and no-hit exits remain structural
boundaries. The CSV never contains a price, OHLC value, barrier level, return,
PnL, funding cash, or economic metric.

This decision authorizes no candidate, candidate ranking, comparator access,
portfolio return, PnL, funding cash, CAGR, MDD, economic rank, economic
metric, or overlap metric.

## Exact half-open value and boundary model

The economic domain remains:

```text
DOMAIN_START = 2023-06-01T00:00:00Z
DOMAIN_END = 2026-06-01T00:00:00Z
step_seconds = 300
interval = [DOMAIN_START, DOMAIN_END)
```

`DOMAIN_END` is a boundary, not an authorized value-row timestamp.

The authenticated working market retains every inherited pre-domain warm-up
row needed by the frozen feature, model, and schedule logic. End filtering
remains exclusive at `DOMAIN_END`; the working market must not be truncated
at `DOMAIN_START` or lose warm-up history.

After the inherited ordering and raw duplicate rule
`sort_values("date").drop_duplicates("date", keep="last")`, let:

```text
value_opens = [v[0], v[1], ..., v[n-1]]
```

The exact value-open contract is:

```text
v[0] <= DOMAIN_START
DOMAIN_START is an element of value_opens
v[i + 1] == v[i] + 300 for every 0 <= i < n - 1
v[n - 1] + 300 == DOMAIN_END
v[i] < DOMAIN_END for every 0 <= i < n
```

Raw duplicates retain sorted `keep="last"` resolution. Uniqueness and
complete-grid checks apply after normalization and to already-normalized
synthetic or reference inputs.

The producer rejects:

- a missing value open;
- a duplicate remaining in a normalized or synthetic/reference grid;
- an off-grid value open;
- a missing aligned `DOMAIN_START` value open;
- an earlier final value open;
- any gap;
- a value row equal to `DOMAIN_END`; and
- any extra end-boundary value row.

The producer must not fabricate a market, funding, premium, open-interest,
feature, OHLC, prediction, model, label, or price value at `DOMAIN_END`.

The derived boundary vector remains:

```text
boundaries = [v[0], v[1], ..., v[n-1], DOMAIN_END]
len(boundaries) == len(value_opens) + 1
```

Its final element is geometry only. It is derived after validating the
authenticated 300-second grid. It is not a source row and increments no
decoded, handed-off, feature, model, OHLC, label, price, or source-value
counter.

The indexing contract remains:

- source-backed values, features, masks, predictions, OHLC, and labels use
  indices `0 .. n-1`;
- boundary-only exit geometry uses indices `0 .. n`;
- every entry indexes a physical value open;
- fixed exits use the boundary vector;
- barrier hits after physical row `i` use boundary `i + 1`;
- a fixed or no-hit structural exit may equal boundary `n`, exactly
  `DOMAIN_END`;
- no source, feature, OHLC, open-price, model, label, prediction, or mask
  access may use index `n`; and
- no value timestamp array may be treated as the `n + 1` boundary vector.

For each inherited split `[split_start, split_end)`, define:

```text
M[i] = split_start <= value_opens[i] < split_end, for 0 <= i < n
```

Schedule eligibility remains:

- every signal and physical entry retains its physical-row checks;
- for fixed or no-hit boundary index `j < n`, eligibility is exactly `M[j]`;
- for the sole derived terminal boundary `j == n`, fixed or no-hit
  eligibility is `split_start <= boundaries[n] < split_end`;
- a barrier exit tests its occupied physical hit-row index `h` with exactly
  `M[h]` and emits boundary `h + 1`; and
- no other boundary predicate may replace a physical-row mask.

An interior split's exclusive end remains rejected for a fixed exit when that
end has a physical row and `M[j]` is false. A barrier hit on the last occupied
row before an interior split end remains eligible under `M[h]` and emits the
split-end boundary. The derived `j == n` rule applies only to the global
terminal boundary and cannot expand an interior schedule.

A structural horizon equal to `len(value_opens)` is valid only when no OHLC,
open, price, model, label, or other value is consumed at that horizon. A
horizon greater than `len(value_opens)` is invalid. Barrier scanning may
consume existing physical rows before the horizon and use the derived
boundary for a no-hit exit. It may not consume a fabricated cap row.

Rank7 and every economic label engine requiring a cap or exit price still
require a physical value. A derived boundary cannot substitute for a price.
Label construction, training history, source routing, features, and economics
remain unchanged.

Direct, reference, and production-shared reconstruction paths implement this
same contract. Any out-of-range mask, seconds, boundary, value, feature,
prediction, label, or model index is terminal.

## Closed predecessor schema

The `G9CB-5` preregistration must keep the two inherited historical
nonoperative `G9CB-1` preregistration rows byte-for-byte semantically
unchanged.

It must then distinguish these successor records:

| Class | Exact count | Identities |
|---|---:|---|
| successor preregistration bindings | 3 | `G9CB-2`, `G9CB-3`, `G9CB-4` |
| attempt-consumed failures | 2 | `G9CB-2`, `G9CB-3` |
| pre-access/pre-sentinel closures | 1 | `G9CB-4` |

`bindings.failed_predecessor_attempts` remains an exact two-row list for
`G9CB-2` and `G9CB-3`; no `G9CB-4` row may be added there.

`bindings.failed_predecessor_closures` is a new exact one-row list containing
only the closed `G9CB-4` row defined above.

Across those lists, the preregistration binding for each of `G9CB-2`,
`G9CB-3`, and `G9CB-4` occurs in its own identity row. A preregistration is
not an attempt sentinel. A pre-sentinel closure is not an attempt-consumed
failure.

The existing `G9CB-2` and `G9CB-3` rows, terminal evidence, permanent
absences, residue paths, residue modes, and exposure disclosures remain
exactly those sealed by `P4`. `Q5` may add validation for the new closure
class but may not rewrite either old row.

The exact preserved `G9CB-2` residue remains:

```text
slot 1 =
  results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef
  empty directory, filesystem mode 0700
slot 2 =
  results/.gross9-structural-clock-worker-2c9f266762f8864bf5e24691
  absent
```

The exact permanently absent `G9CB-2` outputs remain:

```text
results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json
```

The exact preserved `G9CB-3` residue remains:

```text
fixed bytecode cache =
  results/.g9cb3-bytecode-cache-disabled
  absent
slot 1 =
  results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23
  empty directory, filesystem mode 0700
  staged CSV absent
  staged core absent
  staged receipt absent
slot 2 =
  results/.gross9-structural-clock-g9cb3-worker-26e64bf0a62646afad3d77e6
  absent
```

The exact permanently absent `G9CB-3` outputs remain:

```text
results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb3_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb3_manifest_2026-07-31.json
```

All `G9CB-2`, `G9CB-3`, and `G9CB-4` predecessor states remain exact through
`D5`. No old residue may block, satisfy, alias, or be reused as a `G9CB-5`
path.

## Authority flow and exact protocol inventory

The top-level `authority_decision` in the `G9CB-5` preregistration binds only
this decision, with exact path, regular-file type, SHA-256, Git blob, Git mode
`100644`, and exact standalone `A5` commit.

The ordered `bindings.authority_amendments` list remains unchanged and
contains exactly:

| Identity | Authority commit | Path |
|---|---|---|
| `G9CB-1A` | `f1ae4e68bfb0d0b861cd9979762f87e51a55f69d` | `docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md` |
| `G9CB-1B` | `2550e0b8ee348b4217744a73d9781dba1e1e91a3` | `docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md` |
| `G9CB-1C` | `eee3383c9b2f88f4ea28f5bfe3a5ff6a650cec0f` | `docs/gross9-structural-clock-bundle-preregistration-correction-amendment-2026-07-31.md` |

No amendment may be added, removed, reordered, or reclassified.

`bindings.protocol` is a sorted exact list of 17 unique tracked mode-`100644`
regular-file bindings:

```text
docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb3-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb4-successor-authority-decision-2026-07-31.md
docs/gross9-structural-clock-bundle-g9cb5-successor-authority-decision-2026-07-31.md
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

These are the existing `G9CB-4` 16 paths plus this `A5` decision. Every row
has exactly:

```text
git_blob
git_mode
path
path_type
sha256
```

Every path appears once. No additional protocol path is allowed.

The claim binds the complete active preregistration path, SHA-256, manifest
hash, and protocol parent. That binding carries this decision, the unchanged
amendments, exact protocol inventory, and every predecessor row through the
sentinel, ledgers, per-pass cores, receipts, CSV, and final manifest.
Downstream artifacts must not introduce another authority field.

## Q5 security-correction contract

`Q5` changes only the exact five implementation and test files listed in the
topology section. It must implement all requirements in this section as one
indivisible correction set.

### Keyword-only correction

The uncached `_head_blob_binding` path must call
`_validate_git_pair_preflight` with `repository_relative`, `declaration`, and
`verify_git` by keyword. No compatibility wrapper, positional fallback,
monkeypatch, or bypass is authorized.

The corrected uncached branch must be exercised by an actual synthetic Git
repository test. A mocked return value or cache-only path does not qualify.

### Threat model and authenticated snapshot policy

The security boundary is one unprivileged parent process operating on a local
filesystem with retained descriptors. The protocol must fail closed on every
observable path, byte, metadata, Git, directory-inventory, or inode drift
within that boundary.

This decision does not claim that a malicious privileged actor cannot alter
an opened inode after the final verification or conceal such a write from the
available kernel metadata. It claims only that the unprivileged process
detects every drift observable through the required descriptor reads,
`fstat` fields, Git classifications, and directory inventories before it
publishes or returns success.

Each unique bound regular-file path has exactly one component-wise secure
`open`. The descriptor remains open through the final snapshot recheck. Two
content reads from that same descriptor are required and permitted:

1. one initial read caches the complete bytes and computes SHA-256 and Git
   blob identity; and
2. one final `pread` from offset zero, or one rewind followed by a complete
   read, verifies the same bytes from the same descriptor.

The final read is not a pathname reopen. No third content read is allowed.
Before and after those reads, `fstat` the same descriptor and require exact
equality of:

```text
st_dev
st_ino
regular-file type
st_mode
st_size
st_mtime_ns
st_ctime_ns
```

The initial cached bytes define the authenticated snapshot. Every parser,
closure check, historical validation, source binding, pre-sentinel consumer,
and verifier consumer uses those cached bytes. No consumer may read the
descriptor directly or reopen the pathname.

Distinct canonical paths may not share `(st_dev, st_ino)`. Multiple identical
declarations of one path share the same retained descriptor and cached bytes.
Conflicting declarations fail before the first worktree bound-file open.

### Git classification and cleanliness policy

Git-object, HEAD-tree, and index metadata classification does not prove full
worktree cleanliness.

Before the first bound-file open, the parent must classify the exact
HEAD/index pair for every repository-relative bound path and every tracked
protocol path. Bound-path worktree cleanliness is then proven by computing
the Git blob identity from the initial cached bytes and comparing it with the
preclassified pair.

Global clean-tree commit gates may run `git status` before or after the
descriptor snapshot and again at the final gate. Those Git processes may
perform their own internal worktree opens; such implementation-internal Git
opens are outside the protocol's one-secure-open counter. They do not provide
bound-path authentication and cannot replace either same-descriptor content
read.

No worktree `git hash-object` operation is allowed anywhere in
preregistration publication, claim preflight, production, or committed
verification.

### Claim-preflight bootstrap

At clean pushed `P5`, claim preflight proceeds in this order:

1. classify branch, HEAD, upstream name, upstream commit, index state, and
   exact `A5 -> Q5 -> P5` topology using Git objects and index metadata;
2. run the global clean-tree commit gate;
3. read the active preregistration from the Git object
   `HEAD:results/gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json`;
4. decode that Git-object payload without reading the worktree
   preregistration;
5. enumerate and normalize every qualifying binding, active metadata path,
   tracked protocol path, and closed path-state entry;
6. classify every repository-relative HEAD/index pair globally;
7. reject every missing, additional, unmerged, intent-to-add, wrong-stage,
   wrong-mode, wrong-blob, partial-null, malformed, or contradictory pair;
8. open each unique bound worktree path exactly once through component-wise
   no-follow traversal and retain every descriptor;
9. perform the initial content read, descriptor metadata capture, SHA-256,
   and locally framed Git blob calculation;
10. reject hard-link aliases and compare cached bound bytes with the
    preclassified Git blob;
11. capture and validate the initial closed path-state inventory;
12. validate all canonical metadata, predecessor history, environment,
    static closures, permanent absences, and residue from cached bytes and
    retained directory descriptors;
13. run the publication capability probe, validate its restored entry
    inventory, and rebaseline directory timestamps;
14. construct the complete canonical claim bytes, create its unnamed
    `O_TMPFILE` inode, write, `fchmod(0444)`, file-`fsync`, `fstat`, and
    same-FD-verify it without creating the canonical leaf;
15. perform the sole full immutable bound-snapshot recheck, including the
    final same-descriptor content/metadata verification, Git metadata
    reclassification, closed inventory check, and global clean-tree gate;
16. link the already verified unnamed inode create-only to the canonical
    claim leaf; and
17. perform only canonical inode/bytes/mode verification, exact one-leaf
    entry-delta validation, results-directory `fsync`, and timestamp
    rebaselining, then close publication descriptors.

Step 17 performs no bound-file reopen or read and no second immutable
snapshot recheck.

The bootstrap `git show` reads a Git object, not a worktree bound file. It
does not count as the secure worktree open and cannot substitute for either
same-descriptor content read.

### Parent snapshot content modes and entry-point states

`_preauthenticate_parent_snapshot` supports exactly two content modes:

| Content mode | Active metadata paths | Required at |
|---|---|---|
| preregistration-only | active preregistration | `P5` claim preflight |
| preregistration-plus-claim | active preregistration and active claim | `C5` production and `D5` committed verification |

The content mode is explicit and is not inferred from path existence.
Preregistration-only mode must not open the absent claim.
Preregistration-plus-claim mode must authenticate both metadata files.

The closed path-state entry point is also explicit:

| Entry-point phase | Active preregistration | Active claim | Five `G9CB-5` publication leaves | Worker stages | Fixed pycache |
|---|---|---|---|---|---|
| `Q5_PREREGISTRATION_PUBLICATION` | absent | absent | all absent | none | absent |
| `P5_CLAIM_PREFLIGHT` | present, tracked, mode `100644` in Git and `0444` in the worktree | absent | all absent | none | absent |
| `C5_PRODUCTION_PREFLIGHT` | present, tracked, mode `100644` in Git and `0444` in the worktree | present, tracked, mode `100644` in Git and `0444` in the worktree | all absent | none | absent |
| `D5_COMMITTED_VERIFICATION` | present, tracked, mode `100644` in Git and `0444` in the worktree | present, tracked, mode `100644` in Git and `0444` in the worktree | all present, tracked, mode `100644` in Git and `0444` in the worktree | none | absent |

These are entry-point states. No other entry-point combination is valid.

At clean pushed `Q5`, the preregistration writer must:

1. prove the exact clean-pushed `A5 -> Q5` topology and exact `Q5` diff;
2. classify the global current Git pairs for the protocol inventory and every
   repository-relative binding before the first protocol-owned worktree
   bound-file open;
3. securely open and cache the exact `Q5` protocol and bound metadata under
   the one-open/two-read descriptor policy;
4. capture the exact `Q5_PREREGISTRATION_PUBLICATION` path-state inventory;
5. run the required publication capability probe, validate its exact
   temporary entry delta and restored entry inventory, and rebaseline
   authorized directory timestamps;
6. construct the complete canonical preregistration bytes, create its unnamed
   `O_TMPFILE` inode, write, `fchmod(0444)`, file-`fsync`, `fstat`, and
   same-FD-verify it without creating the canonical leaf;
7. perform the sole full immutable bound-snapshot recheck immediately before
   the canonical link;
8. link that already verified unnamed inode create-only as
   `results/gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json`;
9. perform only canonical inode/bytes/mode verification, exact one-leaf
   entry-delta validation, results-directory `fsync`, and timestamp
   rebaselining, with no bound-file reopen/read or second immutable recheck;
10. require the final state to contain that preregistration as a regular
   mode-`0444` leaf while the claim, all five publication leaves, every
   `G9CB-5` worker stage, and the fixed pycache path remain absent; and
11. close the unnamed and canonical publication descriptors.

That exact one-file state is then committed as `P5`. The capability probe
does not authorize a claim, sentinel, ledger, CSV, manifest, stage, pycache,
or any other persistent leaf at `Q5`.

### Component-wise no-follow traversal

Every builder and preregistration reader uses component-wise
directory-descriptor traversal.

For every repository-relative binding, traversal starts from a retained
repository-directory descriptor:

1. validate path syntax before traversal;
2. reject absolute paths where repository-relative paths are required;
3. reject empty components, `.` and `..`;
4. open every parent relative to the preceding descriptor with directory-only
   and no-follow flags;
5. verify every parent with `fstat` as a directory;
6. open each present leaf relative to its authenticated parent with
   no-follow semantics;
7. verify the leaf with the same descriptor's `fstat`; and
8. retain all required file and parent descriptors through final recheck.

No `Path.resolve`, path-only `lstat` followed by `open`, parent pathname
reopen, or leaf-only no-follow check qualifies. Parent or leaf substitution
must fail closed.

The only currently authorized absolute binding is exactly:

```text
/tmp/btcusdt_open_interest_5m_2020_2026.csv
```

An authorized absolute binding must be a normalized canonical absolute path,
must be outside the repository tree, and must match the closed allowlist
exactly. Its traversal starts from a retained filesystem-root `/` descriptor,
not from the repository descriptor. Open and retain `/` and every parent
component through directory-only, no-follow, dirfd-relative operations;
verify each with `fstat`; then open the leaf no-follow relative to its retained
parent. Include the root and every retained absolute-parent identity in the
initial and final tokens. Reject every unlisted absolute path, repository-
internal absolute spelling, repeated separator, `.`, `..`, empty component,
noncanonical spelling, and symlink component.

Because the closed absolute-binding allowlist is currently nonempty, every
guarded worker descriptor graph must also contain a child-local registered
filesystem-root `/` anchor. The guarded component walker may derive and
register retained `/tmp` only from that root anchor, then open and authenticate
only the exact allowlisted leaf
`/tmp/btcusdt_open_interest_5m_2020_2026.csv`. The worker's initial and final
descriptor tokens include the root anchor, retained `/tmp`, and absolute leaf
device, inode, type, mode, size, `mtime_ns`, and `ctime_ns` as applicable.
No repository or results anchor may be reinterpreted as an absolute root, and
no other absolute traversal, derived absolute parent, leaf, or dirfd use is
authorized.

Common parent descriptors are opened once and reused within the normalized
descriptor graph. Each unique bound regular-file leaf still has exactly one
protocol-owned secure open, one initial cache read, and one final
same-descriptor verification read.

The preregistration reader and writer use this same discipline. No weaker
metadata-only reader is permitted.

### Closed path-state inventory

Each snapshot phase has one closed inventory with exactly these classes:

```text
bound regular-file leaves
tracked protocol leaves
active preregistration and optional claim leaves
predecessor permanent-absence leaves
predecessor residue directories and exact modes
predecessor residue glob/prefix states
active G9CB-5 publication leaves
active G9CB-5 worker-stage prefix
active G9CB-5 fixed-pycache leaf
retained repository-directory descriptor identity
retained results-directory descriptor identity
retained active worker-stage descriptor identity and complete entry enumeration
retained filesystem-root and absolute-binding parent identities
complete results-directory entry enumeration
```

The predecessor permanent-absence leaves are the exact `G9CB-2`, `G9CB-3`,
and `G9CB-4` lists frozen above. The residue directories, exact modes, globs,
and prefixes are also the exact frozen predecessor states above.

The active publication leaves are exactly the sentinel, pass-1 ledger,
pass-2 ledger, canonical CSV gzip, and final manifest. The worker-stage
prefix and fixed-pycache leaf are the exact `G9CB-5` literals in this
decision.

Open and retain the repository and `results` directory descriptors through
the snapshot. Check present leaves and directories with dirfd-relative
no-follow operations. Check absent leaves with dirfd-relative no-follow
lookups that distinguish absence from a symlink or wrong type. Enumerate
`results` from the retained descriptor and compare the complete initial and
final entry inventories. During production, open each authorized stage
directory once relative to the retained `results` descriptor, retain its
descriptor until exact cleanup, and enumerate its complete contents from that
descriptor at every applicable internal transition.

Every directory token records `st_dev`, `st_ino`, type, mode, `st_mtime_ns`,
and `st_ctime_ns`, plus the sorted entry-name/type inventory required by the
active state. For every retained directory, `st_dev`, `st_ino`, type, and mode
are invariant for the entire operation.

`st_mtime_ns` and `st_ctime_ns` equality is required only across an interval
in which this protocol authorizes no namespace mutation in that directory.
For each exact capability-probe, stage, or publication transition, validate
the exact authorized entry delta first, then record the resulting timestamps
as the baseline in the next inventory token. The next interval compares
against that new token. A capability probe must restore the exact entry
inventory and preserve directory identity, type, and mode; it is not required
to restore directory timestamps. After the restoration check, rebaseline
those timestamps.

For a nonmutation interval, initial and final entry inventories and directory
timestamps are exactly equal. No timestamp change can excuse an entry drift,
and no expected timestamp change can substitute for exact entry-delta
validation.

Ignored result paths, predecessor residue, worker stages, publication leaves,
and the fixed-pycache leaf are proven by this descriptor-relative inventory.
They must not be inferred from Git status.

### Immutable final recheck

For `Q5` preregistration publication, `P5` claim publication, and the `C5`
sentinel, first finish the unnamed artifact's write, mode change, file
`fsync`, `fstat`, and same-FD byte verification. Then perform the operation's
sole full immutable bound-snapshot recheck immediately before the canonical
link. At `D5`, perform the sole full recheck immediately before successful
verifier return. Require agreement of:

```text
branch
HEAD commit
upstream name and commit
index/HEAD pair for every repository-relative bound and protocol path
locally computed Git blob for each initial cached bound-file payload
final same-descriptor bytes and SHA-256
final same-descriptor fstat fields
closed protocol path-state inventory
retained repository/results directory identities
retained filesystem-root/absolute-parent identities when applicable
results-directory enumeration and exact authorized entry deltas
directory timestamp baselines for the current no-mutation interval
global clean-tree commit gate
```

The final recheck performs no pathname reopen of a bound regular file. After
the preregistration, claim, or sentinel link, perform no bound-file reopen or
read and no second immutable recheck. Only canonical inode/bytes/mode
verification, the exact atomic entry delta, directory `fsync`, and timestamp
rebaselining remain.

After the `C5` sentinel link, the parent must never reread or reopen any bound
snapshot input. Worker ledgers use the guarded-worker metadata final recheck
defined below immediately before their links. The canonical CSV and manifest
use only exact internal-transition inventory plus unnamed-FD/canonical-FD
publication verification. No post-sentinel parent action may invoke
`_preauthenticate_parent_snapshot`, consume a retained bound descriptor, or
perform another bound snapshot content/metadata recheck.

### Unnamed O_TMPFILE publication

Named publication staging files are forbidden. All canonical write-once
publication uses an unnamed `O_TMPFILE` inode on the retained `results`
directory filesystem.

The authoritative runtime capability probe must establish this outcome:

```text
REQUIRED:
O_TMPFILE succeeds
linkat(
  AT_FDCWD,
  "/proc/self/fd/<unnamed_fd>",
  results_dir_fd,
  canonical_leaf,
  AT_SYMLINK_FOLLOW
) = success
published inode == unnamed inode
published mode == 0444

OPTIONAL:
linkat(unnamed_fd, "", results_dir_fd, canonical_leaf, AT_EMPTY_PATH)
may be selected only when this active probe returns success
```

No supplied preliminary observation is authenticated historical evidence for
this capability. The runtime probe at the active entry point is authoritative.

Before creating any artifact or publishing the sentinel, the unguarded parent
must capability-probe:

- `O_TMPFILE` creation on the retained results filesystem;
- procfs descriptor-reference availability;
- the exact procfd `linkat` operation above;
- create-only canonical-link behavior;
- no-follow canonical reopening;
- exact inode and mode preservation; and
- results-directory `fsync`.

The capability probe may use one unique reserved probe leaf, but it is not a
staging leaf or artifact. The parent must open and authenticate it, unlink it
relative to the retained results descriptor, `fsync` the directory, and prove
that the results entry inventory returned exactly to its initial state and
that the retained directory's device, inode, type, and mode remained exact.
The probe need not restore directory timestamps; after exact restoration it
records a new timestamp baseline. If any capability is unavailable,
preregistration publication or production, as applicable, fails before its
first artifact, worker stage, capability consumption, or sentinel
publication.

The procfd descriptor-reference fallback is explicitly authorized in the
unguarded parent publisher and in the one narrowly defined guarded-worker
ledger operation below. `/proc/self/fd/<fd>` is a kernel-managed reference to
the already opened unnamed inode, not a staging pathname. No other guarded
worker procfs access or link operation is authorized.

For each parent-owned canonical publication—the preregistration, claim,
sentinel, canonical CSV gzip, and manifest—the unguarded parent must:

1. use the retained authenticated results-directory descriptor;
2. open an unnamed regular inode with `O_TMPFILE` on that directory;
3. retain its descriptor and initial `fstat`;
4. write the complete canonical bytes through that descriptor;
5. `fchmod` the descriptor to `0444`;
6. file-`fsync` the descriptor;
7. `fstat` it again, require unchanged device, inode, and regular-file type,
   and capture the exact completed mode, size, `mtime_ns`, and `ctime_ns`;
8. verify complete bytes from the same unnamed descriptor;
9. for the preregistration, claim, or sentinel only, perform the operation's
   sole full immutable bound-snapshot recheck immediately before linking; for
   the post-sentinel CSV or manifest, explicitly perform no parent bound
   snapshot reopen, read, or recheck;
10. create the absent canonical leaf through the retained results descriptor
   using the capability-probed procfd `linkat` operation, or a separately
   successful `AT_EMPTY_PATH` probe;
11. fail without replacement or repair on canonical `EEXIST`;
12. open the canonical leaf relative to the retained results descriptor with
    `O_NOFOLLOW`;
13. require exact unnamed/canonical `(st_dev, st_ino)`, bytes, SHA-256, size,
    regular-file type, and mode `0444`;
14. validate the exact absent-to-present one-leaf entry delta and unchanged
    results-directory device, inode, type, and mode;
15. directory-`fsync` the retained results descriptor and rebaseline its
    timestamps; and
16. close the unnamed and canonical descriptors.

Steps 10 through 16 never reopen or reread a bound snapshot file and never
perform a second immutable snapshot recheck.

There is no named publication staging leaf and no staging-name unlink step.
Create-only behavior remains absolute.

#### Narrow runtime-isolation dirfd supersession

The inherited runtime-isolation amendment currently requires every guarded
call to reject any non-`None` `dir_fd`, `src_dir_fd`, or `dst_dir_fd`,
including integer `AT_FDCWD`, before filesystem access. It also permits the
worker to open results and own-stage directory descriptors only after
metadata authentication and permits those descriptors only for
`fsync`/`fdatasync`, not dirfd-relative resolution.

This decision supersedes only those clauses, and only in the following closed
sequence:

1. the worker first completes the inherited mandatory
   `prctl(PR_SET_PDEATHSIG, SIGKILL)` setup and two-parent-PID race check;
2. before guard installation or metadata authentication, it validates the raw
   hidden arguments and exact inherited descriptor table;
3. while its current working directory is the exact canonical repository
   root, it opens the child-local repository anchor as `.` with
   `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, then the child-local results
   anchor relative to that repository FD with the same flags;
4. because the closed absolute-binding allowlist is nonempty, it opens a third
   child-local anchor, exact filesystem root `/`, with
   `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`;
5. it `fstat`s all three anchors, requires exact directory type, device,
   inode, and mode bindings, and installs the guard with those three
   registered anchors;
6. during guarded metadata authentication, only the prebound
   component-walker helper may use a registered anchor or helper-derived
   directory FD read-only to traverse an exact bound allowlist path;
7. for the sole allowlisted absolute binding, the helper may derive and
   register `/tmp` only from the registered `/` anchor and may authenticate
   only `/tmp/btcusdt_open_interest_5m_2020_2026.csv`; every other absolute
   path or absolute dirfd derivation is forbidden;
8. the helper opens every component with no-follow semantics, registers every
   derived directory and regular-file FD with its exact path and
   device/inode/type/mode, and returns cached bytes under the one-open/two-read
   policy; no caller receives general dirfd resolution authority;
9. after metadata authentication, the inherited own-stage directory
   durability exception remains: its registered stage FD may be used only for
   the inherited stage `fsync`/`fdatasync` operations;
10. before runtime or value access, exactly one own-ledger procfd `linkat`
   through the registered results FD and exactly one own canonical-ledger
   `O_NOFOLLOW` openat through that FD are authorized; and
11. every other non-`None` `dir_fd`, `src_dir_fd`, or `dst_dir_fd`, procfs
   operation, link, openat, unregistered descriptor, path, or mutation remains
   rejected and counted.

The repository, results, and conditionally required filesystem-root anchors
are opened by the child; none is inherited. `pass_fds` remains exactly the
capability FD plus that child's ledger FD. Every other inherited process,
descriptor, path, mutation, IPC, network, bytecode, cross-stage,
cross-ledger, and capability isolation rule remains unchanged.

#### Minimal worker ledger-FD handoff

This subsection supersedes only the inherited clauses that require
`pass_fds=(that_read_fd,)`, describe the capability FD as the child's sole
descriptor, authorize named/path-based ledger staging, or categorically
forbid the exact registered-anchor/component-walker/worker-ledger operations
specified here. Together with the immediately preceding dirfd subsection,
this is the complete supersession. All other capability-token, isolation,
stage-output, counter, and pre-runtime/pre-value semantics remain frozen.

Before sentinel publication, the unguarded parent creates one unnamed
`O_TMPFILE` ledger FD for each slot on the retained `results` filesystem. It
uses read/write access, forces mode `0600`, and requires a regular file of
size zero. Each sentinel `worker_capabilities` row has exactly these fields,
in this order:

```text
slot
parent_pid
stage_directory
carrier_kind = "anonymous_pipe_v1"
carrier_device
carrier_inode
token_sha256
consumed_ledger_path
ledger_carrier_kind = "unnamed_otmpfile_v1"
ledger_device
ledger_inode
ledger_initial_type = "regular_file"
ledger_initial_mode = "0600"
ledger_initial_size = 0
```

`consumed_ledger_path` remains the slot's exact canonical ledger path. The
process-local integer ledger FD is not serialized. The parent retains its
reference to both ledger FDs until the corresponding workers have exited and
parent validation is complete. The two ledger identities, the two canonical
paths, the two capability-pipe identities, and the two stage paths must each
be pairwise unique as applicable.

Each child receives the hidden argument `--worker-ledger-fd <fd>`.
`subprocess.Popen` uses exactly:

```text
close_fds=True
pass_fds=(capability_read_fd, ledger_fd)
```

The two descriptors must be distinct and bound to the child's own slot. The
capability FD remains the sole token/read carrier. The ledger FD remains the
sole ledger publication/write carrier. The other slot's capability and ledger
descriptors are not inherited and are inaccessible.

After mandatory `prctl` setup and before metadata authentication, the
worker's early bootstrap validates both raw hidden arguments and the inherited
descriptor table. It requires two distinct nonnegative descriptor numbers, a
FIFO capability FD, and a regular ledger FD of mode `0600` and size zero;
rejects aliasing, extra inherited protocol descriptors, or wrong types. The
inherited open-descriptor set is exactly standard descriptors `0`, `1`, and
`2` plus those two passed FDs; validate it by descriptor operations without
procfs or `/dev/fd`. It then opens the child-local repository and results
anchors plus the conditionally required child-local filesystem-root anchor
and installs the guard exactly as specified above, registering only its own
unnamed ledger FD as the pending ledger carrier while both canonical ledger
paths remain forbidden.
After canonical metadata and sentinel authentication, but before capability
consumption, it requires both descriptor identities, its stage, and its
canonical ledger path to match exactly one sentinel row and requires that
neither descriptor identity matches the other slot. Only then may the guard
bind the exact own canonical destination and one-use procfd-link exception;
the other ledger remains inaccessible.

After consuming and closing the capability FD, but before any runtime import
or value access, the guarded tracked worker must:

1. create the exact canonical ledger bytes in memory;
2. write all bytes to its own unnamed ledger FD;
3. `fchmod` that FD to `0444`, file-`fsync` it, and verify exact bytes, SHA-256,
   size, regular-file type, device, inode, and mode from that same FD;
4. perform its sole guarded-worker metadata final recheck from the retained
   registered descriptors: final same-descriptor verification of its cached
   preregistration, claim, sentinel, authority, amendment, parent-
   authentication, and other exact metadata; exact repository, results,
   filesystem-root, retained `/tmp`, and allowlisted absolute-leaf descriptor
   identities and modes; exact slot/capability/ledger bindings; empty
   mode-`0700` own stage; absent own canonical ledger; and all inherited
   pre-runtime isolation counters at their required values;
5. perform exactly one prebound
   `linkat(AT_FDCWD, "/proc/self/fd/<ledger_fd>", results_dir_fd,
   canonical_leaf, AT_SYMLINK_FOLLOW)` to its own absent canonical ledger;
6. open that canonical leaf once relative to the securely retained
   results-directory FD with `O_NOFOLLOW`;
7. require exact unnamed/canonical inode, bytes, SHA-256, size, regular-file
   type, and mode `0444`; and
8. validate the exact absent-to-present ledger entry delta, `fsync` the
   retained results-directory FD, and rebaseline its timestamps.

Steps 5 through 8 perform no metadata pathname reopen/read and no second
worker metadata final recheck.

That exact prebound procfd `linkat` is the sole guarded-worker procfs/link
exception and is counted exactly once. It is not a staging pathname. Every
other procfs access, link call, ledger destination, or path mutation remains
forbidden and counted by the guard. The worker proceeds directly after its
own durable canonical-ledger verification; no parent acknowledgment, pipe,
handshake, or other IPC is added.

Immediately after each successful `Popen`, the parent closes its copy of that
child's capability read FD; the inherited token-write/close sequence remains
frozen. The child closes its capability FD after exact token consumption,
retains its ledger FD through worker completion, and closes that ledger FD in
its final cleanup. The parent never passes either slot the other slot's ledger
FD and does not close its own ledger reference before the child exits. After
each child exits, the parent uses its retained ledger FD and retained
results-directory FD to validate the exact canonical link, inode, bytes,
SHA-256, size, regular-file type, and mode before accepting that worker.
Parent cleanup then closes that retained unnamed ledger FD; neither process
unlinks a canonical ledger. Once the sentinel exists, any ledger write, link,
validation, worker, or cleanup failure is terminal.

The sentinel capability schema therefore binds both the capability token
carrier and the worker's precreated unnamed ledger identity and canonical
path. All canonical artifacts other than the two worker ledgers are published
by the unguarded parent through the general unnamed-`O_TMPFILE` algorithm
above.

### Production entry

At clean pushed `C5`, production uses preregistration-plus-claim mode with the
`C5_PRODUCTION_PREFLIGHT` path state. It validates topology, active metadata,
predecessor closure, bound inputs, environment, static closures, residue, and
all permanent absences before the publication capability probe, worker-stage
creation, capability consumption, or sentinel publication.

The initial byte cache supplies every pre-sentinel consumer. Retained
descriptors supply only the one final verification read.

After the sentinel, the inherited two fresh isolated workers, capability
consumption, source loading, reconstruction, two-pass equality, serializer,
canonical CSV, and manifest-last semantics remain unchanged.

The following table defines stable checkpoints, not every syscall-visible
state. Both random stage names are generated and fixed in the two sentinel
capability rows before the sentinel is linked. “Exact stage outputs” means
only these three regular mode-`0400` leaves in the stated mode-`0700`
directory:

```text
gross9_structural_clock_bundle.csv.gz
gross9_structural_clock_bundle_core.json
gross9_structural_clock_bundle_pass_receipt.json
```

Throughout every stable checkpoint and authorized helper-local transient, the
preregistration and claim remain exact, mode-`0444`, and unchanged; all
predecessor residue and permanent absences remain exact; and no fixed pycache
path exists.

Before checkpoint 3, the parent constructs, writes, `fchmod`s, file-`fsync`s,
`fstat`s, and same-FD-verifies the unnamed sentinel, then performs the sole
full parent immutable bound-snapshot recheck immediately before linking it.
That recheck requires the exact checkpoint-2 path inventory and current
rebaselined directory tokens; the already validated capability-probe and
stage-creation deltas are the only differences from the
`C5_PRODUCTION_PREFLIGHT` entry inventory.
After that link, the parent performs only canonical sentinel
inode/bytes/mode verification, the exact entry-delta check, results-directory
`fsync`, and timestamp rebaselining. It performs no bound-file reopen/read or
second immutable recheck.

| Stable checkpoint | Exact `G9CB-5` state |
|---|---|
| 0. `C5` preflight | all five publication leaves absent; no stage; fixed pycache absent |
| 1. capability probe complete | exact entry inventory restored; retained directory identity/type/mode unchanged; timestamps rebaselined |
| 2. slot 1 prepared | slot-1 random stage exists, empty, mode `0700`; slot-2 random path is reserved and absent; both names and both precreated ledger-FD identities/paths are fixed in capability rows; five publication leaves absent |
| 3. sentinel linked | sentinel present mode `0444`; both ledgers, CSV, and manifest absent; slot-1 stage empty; slot-2 path absent |
| 4. pass-1 ledger linked | sentinel and pass-1 ledger present mode `0444`; pass-2 ledger, CSV, and manifest absent; this occurs before pass-1 runtime import or value access |
| 5. pass-1 output ready | slot-1 stage contains exactly the three mode-`0400` stage outputs; no other change |
| 6. slot transition | slot-1 stage and its contents are absent; slot-2 stage exists, empty, mode `0700`; publication leaves unchanged |
| 7. pass-2 ledger linked | sentinel and both ledgers present mode `0444`; CSV and manifest absent; this occurs before pass-2 runtime import or value access |
| 8. pass-2 output ready | slot-2 stage contains exactly the three mode-`0400` stage outputs; no other change |
| 9. canonical CSV linked | after exact two-pass compressed/decompressed CSV and core equality, sentinel, both ledgers, and canonical CSV present mode `0444`; manifest absent; slot-2 exact outputs remain |
| 10. manifest linked last | all five publication leaves present mode `0444`; slot-2 exact outputs remain; no later canonical artifact link is permitted |
| 11. final cleanup | all five publication leaves present mode `0444`; no worker stage; fixed pycache absent |

No other stable checkpoint is valid. Helper-local transient namespace states
are authorized only by the exact closed protocols below and can never be
accepted as a checkpoint, recovery point, or resume state.

### Closed helper-local transient protocols

The only valid transitions between stable checkpoints are:

1. **Capability probe:** the reserved probe leaf moves through exactly
   `absent -> one linked probe inode -> inode/mode/bytes verified -> unlinked
   -> absent`. Validate the one-entry addition and rebaseline results
   timestamps after the link and results-directory `fsync`; validate the
   one-entry removal, restored exact inventory, and invariant results
   device/inode/type/mode after unlink and another directory `fsync`; then
   rebaseline timestamps again.
2. **Stage creation:** the exact reserved random stage leaf moves from absent
   directly to one empty mode-`0700` directory. Open it no-follow, bind its
   descriptor identity, validate the exact one-entry addition and empty
   inventory, `fsync` `results`, and rebaseline timestamps. No wrong-mode,
   symlink, file, nonempty, or alternate stage leaf is transiently valid.
3. **Worker stage-file creation:** in exact order CSV, core, then receipt,
   each exact leaf moves through
   `absent -> exclusive in-progress regular inode bound to the worker FD ->
   fully written -> mode 0400 -> file-fsynced -> same-inode/same-bytes
   verified stable leaf`. At exclusive creation, validate the exact one-entry
   stage delta and rebaseline the stage-directory timestamps. Until final
   same-inode verification, the in-progress leaf is helper-local only and is
   not a stable output. After final verification, `fsync` the stage directory
   and require its exact stable inventory. No additional leaf, ordering,
   content, mode, inode replacement, or partially written leaf may qualify.
4. **Stage cleanup:** unlink exactly CSV, then core, then receipt. After each
   unlink, validate the exact one-entry removal, invariant stage-directory
   identity/type/mode, and exact remaining inventory, `fsync` the stage
   directory, and rebaseline its timestamps. After receipt removal, require
   and `fsync` the empty directory; remove that exact stage directory;
   validate the exact results-directory one-entry removal, `fsync` `results`,
   and rebaseline its timestamps. For the slot-1-to-slot-2 transition only,
   then execute the exact stage-creation protocol for the already reserved
   slot-2 name.
5. **Canonical unnamed-inode publication:** preregistration, claim, sentinel,
   each worker ledger, CSV, and manifest each have exactly one namespace
   transition, `canonical leaf absent -> canonical leaf present`, performed
   by the authorized create-only link of an already complete and same-FD-
   verified unnamed inode. Postlink checks require the exact canonical
   inode/bytes/mode, exact one-entry delta, directory `fsync`, and timestamp
   rebaseline. No named staging leaf or intermediate canonical content is
   authorized.

Every helper validates each listed namespace delta through retained
directory descriptors and records the next timestamp token. Any fault after
the sentinel link is terminal. Any fault before the sentinel link during an
authorized non-synthetic `G9CB-5` operation closes the identity. A transient
state never authorizes retry, resume, repair, or acceptance as a stable
checkpoint.

### Committed-publication verifier

The committed verifier must work end-to-end at exact `D5` using the
preregistration-plus-claim content mode and
`D5_COMMITTED_VERIFICATION` path state.

It must:

1. establish exact clean pushed `D5` and topology with Git objects, index
   metadata, and the global clean-tree gate;
2. bootstrap preregistration and claim from HEAD blobs;
3. globally classify every repository-relative HEAD/index pair;
4. securely open each unique bound and committed publication file once and
   retain all descriptors;
5. cache each initial payload and use it for all verification;
6. capture the complete descriptor-relative path-state inventory;
7. reject path, symlink, hard-link, parent, mode, byte, blob, internal-hash,
   canonical-hash, topology, residue, permanent-absence, or inventory drift;
8. validate sentinel, ledgers, receipts, per-pass core, canonical CSV gzip,
   final manifest, manifest-last authority, and two-pass byte equality from
   cached bytes;
9. perform exactly one final same-descriptor content read and metadata check
   per opened regular file;
10. repeat Git metadata, global clean-tree, retained-directory, protocol
    path-state, and results-entry inventory checks; and
11. return success only after every check passes.

The verifier must not use path-based `_tracked_head_bytes`, worktree
`git hash-object`, pathname `read_bytes`, a second open, or a third
same-descriptor content read for any bound path.

## Exact commit topology and diffs

Let:

- `P4` be exact
  `01de73258902d754905319b906345c865a016558`;
- `A5` add only this decision;
- `Q5` modify exactly the five authorized implementation/test files;
- `P5` add only the `G9CB-5` preregistration;
- `C5` add only the `G9CB-5` access claim; and
- `D5` add only the five successful `G9CB-5` publication files.

There is no `T4`.

`A5` must satisfy:

```text
A5 has exactly one parent
first_parent(A5) == 01de73258902d754905319b906345c865a016558
diff(P4, A5) ==
  A docs/gross9-structural-clock-bundle-g9cb5-successor-authority-decision-2026-07-31.md
HEAD == A5 == @{upstream}
the worktree and index are clean
```

`Q5` must satisfy:

```text
Q5 has exactly one parent
first_parent(Q5) == A5
diff(A5, Q5) ==
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py
HEAD == Q5 == @{upstream}
the worktree and index are clean
```

Those five paths are exact. `Q5` may change only:

- `G9CB-5` identity-specific literals and paths;
- active authority, protocol inventory, topology, and history bindings;
- the new `G9CB-4` closure schema and its validation;
- the exact security corrections in this decision; and
- synthetic regressions for those changes.

`Q5` may not alter any frozen economic or structural semantic listed in this
decision. No `G9CB-5` artifact may exist at `Q5`.

`P5` must satisfy:

```text
P5 has exactly one parent
first_parent(P5) == Q5
diff(Q5, P5) ==
  A results/gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json
HEAD == P5 == @{upstream}
the worktree and index are clean
```

`C5` must satisfy:

```text
C5 has exactly one parent
first_parent(C5) == P5
diff(P5, C5) ==
  A results/gross9_structural_clock_bundle_g9cb5_access_claim_2026-07-31.json
HEAD == C5 == @{upstream}
the worktree and index are clean
```

`D5` must satisfy:

```text
D5 has exactly one parent
first_parent(D5) == C5
diff(C5, D5) ==
  A results/gross9_structural_clock_bundle_g9cb5_2026-07-31.csv.gz
  A results/gross9_structural_clock_bundle_g9cb5_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb5_manifest_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass1_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass2_2026-07-31.json
HEAD == D5 == @{upstream}
the worktree and index are clean
```

No authority, predecessor, protocol, test, preregistration, or claim byte may
change between `P5`, `C5`, and `D5`.

## Canonical commands and execution gates

After clean pushed `Q5`, the sole canonical metadata command that may create
or byte-verify the `P5` preregistration is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
```

The sole canonical metadata command for `C5` is the same normalized command
stated once in the supplied failure-evidence section. It may run only at clean
pushed `P5`, after every required synthetic regression passes. It may create
only the canonical `G9CB-5` claim.

The official production command may run exactly once and only after clean
pushed `C5`:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
```

No official-source rehearsal, probe, partial production run, synthetic
production against the canonical repository, retry, or second invocation is
authorized.

After `D5` is committed and pushed, the exact committed-verifier command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
```

It performs metadata, authentication, and publication validation only. It may
not recompute official source values or economics.

## Required regression evidence before Q5

All new `G9CB-5` regressions use synthetic bytes, synthetic timestamps,
synthetic artifacts, and temporary synthetic Git repositories. They must not
read, decompress, decode, parse, or load an official source, model, history,
anchor, clock, candidate, comparator, or economic value.

The inherited committed `G9CB-4` artifact regression is the sole
non-synthetic-artifact exception. It may authenticate current repository
metadata and opaque committed hashes only. It must never decompress, decode,
parse, or load source, model, history, anchor, clock, candidate, comparator,
or economic values.

Before `Q5` is sealed, tests must prove:

1. an actual uncached `_head_blob_binding` call succeeds against a real
   temporary Git repository and authenticates the exact HEAD blob;
2. the old positional call shape is absent and the keyword-only call is used;
3. a complete `Q5_PREREGISTRATION_PUBLICATION` path succeeds in a synthetic
   clean pushed temporary Git repository, fully prepares and same-FD-verifies
   the unnamed preregistration, performs its sole immutable bound-snapshot
   recheck immediately before link, performs no postlink bound reread or
   second recheck, publishes only the mode-`0444` preregistration, and ends in
   the exact pre-`P5` state;
4. the preregistration-publication path rejects wrong topology, a dirty tree,
   any pre-existing active leaf, stage, or fixed pycache path, or any final
   state other than preregistration-present/every-other-active-path-absent;
5. a complete actual claim-preflight and claim-creation path succeeds against
   synthetic temporary Git metadata and synthetic bound bytes with the same
   unnamed-prepare, sole-recheck-before-link, and postlink-no-reread order;
6. preregistration-only mode succeeds at synthetic `P5` without probing an
   absent claim;
7. preregistration-plus-claim mode succeeds at synthetic `C5` and synthetic
   `D5`;
8. at preregistration publication, claim, production, and committed-verifier
   entry points, instrumentation proves all HEAD/index pairs are classified
   before the first protocol-owned worktree bound-file open;
9. at each entry point, instrumentation proves exactly one component-wise
   secure leaf open per unique path, one initial content read, one final
   same-descriptor content read, zero pathname reopens, and zero third reads;
   for `Q5`, `P5`, and `C5`, that final read occurs after unnamed-artifact
   preparation and immediately before canonical link;
10. every pre-sentinel and verifier consumer uses initial cached bytes rather
    than reading a descriptor or pathname, and after the sentinel the parent
    performs no bound snapshot reopen, read, or recheck;
11. the final same-descriptor read detects byte drift and final `fstat`
    comparison detects any observable device, inode, type, mode, size,
    `mtime_ns`, or `ctime_ns` drift;
12. duplicate declarations for one path share one retained descriptor,
    initial cache, and final descriptor verification;
13. conflicting duplicate declarations fail before any worktree bound-file
    open;
14. two distinct paths hard-linked to one inode fail before claim,
    production, or verifier success;
15. a symlink in any parent component fails in both builder and
    preregistration readers;
16. replacing a parent component during traversal fails closed;
17. leaf symlinks, non-regular leaves, wrong parent types, `.` components,
    `..` components, and empty components fail closed;
18. mode checks use the same opened descriptor's `fstat`, and path-level mode
    substitution cannot qualify;
19. tracked worktree bytes are compared with the preclassified Git blob by
    locally computing blob identity from initial cached bytes;
20. no worktree `git hash-object` command is issued at preregistration
    publication, claim, production, or verifier entry;
21. tests distinguish Git metadata classification from global worktree
    cleanliness and prove the global clean-tree gate cannot replace cached
    bound-byte authentication;
22. a HEAD move, upstream move, branch change, index mutation, bound-byte
    mutation, Git-mode mutation, or Git-pair mutation between initial and
    final checks fails closed when observable;
23. each of the four closed entry-point phases accepts only its exact
    preregistration, claim, five-publication-leaf, stage, and fixed-pycache
    state;
24. dirfd-relative checks detect permanent-absence, predecessor-residue,
    worker-stage, fixed-pycache, publication-leaf, repository-directory, or
    results-directory inventory drift;
25. retained-directory device, inode, type, and mode drift always fails;
    timestamps must agree within no-mutation intervals, while every authorized
    probe, stage, or publication delta is validated before timestamp
    rebasing;
26. capability probing restores exact entry inventory and directory
    identity/type/mode without requiring impossible timestamp restoration;
27. complete results-directory enumerations agree within no-mutation
    intervals and differ only by the exact transition currently authorized;
28. ignored results paths are proven through retained directory descriptors,
    never inferred from Git status;
29. the exact authorized absolute binding traverses from a retained `/`
    descriptor through retained directory-only/no-follow parents and carries
    every root/parent identity through final recheck; when the allowlist is
    nonempty, each worker opens and registers child-local `/` as its third
    anchor and derives retained `/tmp` only through that anchor;
30. unlisted, repository-internal, noncanonical, or symlink-component
    absolute bindings fail before leaf-byte access, and no repository/results
    anchor or unregistered dirfd can satisfy an absolute binding;
31. the publication capability probe succeeds before preregistration or claim
    publication and before production artifact or sentinel creation, and
    fails closed when `O_TMPFILE`, procfs descriptor references, procfd
    `linkat`, no-follow reopen, inode preservation, or directory `fsync` is
    unavailable;
32. `AT_EMPTY_PATH` is used only after an actual successful runtime
    capability probe, and its `ENOENT` disposition does not block the required
    procfd fallback;
33. fault injection at each unnamed-FD write, `fchmod`, file-`fsync`,
    same-FD verification, procfd link, canonical open, inode comparison, and
    directory-`fsync` step fails closed and proves no postlink bound reread or
    second immutable recheck can occur;
34. no named publication staging leaf is created;
35. results-directory substitution before canonical link fails closed;
36. canonical `EEXIST` never overwrites, repairs, or accepts another inode;
37. every parent-published canonical leaf has the exact unnamed-FD inode,
    bytes, SHA-256, size, regular-file type, and mode `0444`, with
    preregistration/claim/sentinel ordering exactly unnamed preparation,
    sole full recheck, atomic link, canonical verification, entry-delta
    validation, directory `fsync`, and timestamp rebaseline;
38. each sentinel capability row binds its slot's unique capability carrier,
    random stage name, precreated unnamed ledger device/inode/type/mode/size,
    and exact canonical ledger path;
39. each child receives exactly two distinct bound descriptors through
    `pass_fds=(capability_read_fd, ledger_fd)`, validates the raw arguments and
    inherited descriptor table after mandatory `prctl`, opens repository and
    results anchors plus the conditionally required third filesystem-root
    anchor in the child before guard installation, inherits none of those
    anchors, and cannot access the other slot's descriptors;
40. before metadata authentication, blanket non-`None` dirfd rejection is
    relaxed only for the registered-anchor component walker; each derived FD
    is registered, the sole absolute binding derives registered `/tmp` only
    from registered `/`, general or other absolute dirfd resolution fails,
    the inherited post-auth stage-fsync exception remains, and every
    unregistered dirfd, `src_dir_fd`, `dst_dir_fd`, openat, procfs, or link
    use is rejected and counted;
41. the guarded worker writes and same-FD-verifies only its own ledger FD,
    performs its sole metadata final recheck immediately before exactly one
    prebound procfd link, canonical-opens no-follow exactly once, proves
    inode/bytes/mode, validates the delta, directory-`fsync`s, rebaselines
    timestamps, and performs no second metadata recheck;
42. every other guarded-worker procfs or link call remains forbidden and
    counted, and no parent acknowledgment, handshake pipe, or extra IPC
    exists;
43. each canonical worker ledger is durable before that worker's first
    runtime import or value access; after worker exit, the parent validates it
    against its retained unnamed FD, closes all retained ledger FDs, and
    performs no bound snapshot reread;
44. every stable production checkpoint succeeds in order, and instrumentation
    proves every capability-probe, stage-mkdir, ordered stage-file creation,
    per-file ordered cleanup, stage-rmdir/next-mkdir, and atomic canonical-link
    transient delta with exact inventory validation, directory `fsync`, and
    timestamp rebasing; no transient is accepted as a checkpoint or resume
    state, and fault classification is pre-sentinel closure or post-sentinel
    terminal;
45. committed-publication verification succeeds end-to-end on a synthetic
    clean pushed `D5`;
46. the verifier detects each artifact byte, descriptor metadata, internal
    hash, Git blob, topology, predecessor closure, residue, permanent
    absence, and path-state inventory mutation;
47. the exact 17-path protocol inventory is unique, sorted, and complete;
48. the authority-amendment list is byte-for-byte semantically unchanged;
49. the predecessor schema contains three successor preregistrations, two
    attempt-consumed failures, and one pre-access/pre-sentinel closure;
50. the `G9CB-4` closure binds exact `P4`, exact preregistration SHA-256,
    exact manifest hash, exact builder SHA-256, exact exposure, and exact
    permanent absences;
51. the closure schema rejects a `G9CB-4` sentinel, attempt, `C4`, `D4`, or
    `T4`;
52. all preserved `G9CB-2`, `G9CB-3`, and `G9CB-4` residue states and
    permanent absences remain exact;
53. the `G9CB-5` stage and fixed bytecode prefixes cannot alias or be
    satisfied by predecessor residue;
54. the aligned half-open grid ending at `DOMAIN_END - 300 seconds` succeeds;
55. a physical value row at `DOMAIN_END`, an earlier final row, a gap, an
    off-grid row, or a normalized duplicate fails;
56. raw duplicate handling retains sorted `keep="last"` behavior;
57. fixed and barrier structural exits may equal `DOMAIN_END` only through
    the derived boundary vector;
58. horizon `n` succeeds only for geometry without value consumption, while
    horizon greater than `n` fails;
59. the derived end boundary never enters a value array or increments a value
    counter;
60. Rank7 and economic label paths still require a physical cap price;
61. direct, reference, and production-shared paths retain exact interval and
    counter parity, including interior split-boundary cases;
62. two fresh synthetic workers still produce byte-identical compressed CSV,
    decompressed CSV, and core JSON;
63. the frozen committed `G9CB-4` preregistration regression passes
    explicitly under its metadata-and-opaque-hash-only allowance;
64. the active pre-`P5` `G9CB-5` absence regression passes and is not skipped;
65. the preregistration reproduces byte-for-byte at synthetic `P5` and `D5`;
66. no official source value or economic calculation occurs in any test; and
67. every inherited authentication, boundary, isolation, serialization,
    two-pass, and publication regression remains passing.

The frozen inherited baseline is:

```text
owned suites = 482 passed, 1 skipped
frozen suites = 194 passed
combined suites = 676 passed, 1 skipped
```

`Q5` must preserve at least 676 inherited passing tests and only the one
inherited D-publication skip. New `G9CB-5` regressions are additive; they
may not replace, weaken, skip, or delete an inherited test. No new skip is
allowed. The frozen committed `G9CB-4` preregistration test must pass. The
active pre-`P5` `G9CB-5` absence test must pass and must not be the inherited
publication skip.

The exact five Python files changed by `Q5` must parse as Python AST.
Repository diff checks must pass. No test may create a canonical `G9CB-5`
artifact in the real repository.

## Publication and completion

`G9CB-5` is complete only when:

- `A5`, `Q5`, `P5`, `C5`, and `D5` satisfy their exact single-parent,
  clean-pushed topology and diffs;
- the `G9CB-5` preregistration and claim are exact write-once mode-`0444`
  artifacts committed at `P5` and `C5`;
- every required publication capability is probed successfully before any
  artifact, worker stage, capability consumption, or sentinel publication;
- every bound regular-file path uses exactly one component-wise secure open,
  one initial cache read, and one final same-descriptor verification read,
  with no pathname reopen or third read;
- for preregistration, claim, and sentinel publication, the complete unnamed
  artifact is written, mode-set, file-`fsync`ed, `fstat`ed, and same-FD-
  verified before the sole full immutable bound-snapshot recheck; that recheck
  occurs immediately before canonical link;
- after each such link, only canonical inode/bytes/mode verification, exact
  one-leaf entry-delta validation, results-directory `fsync`, and timestamp
  rebasing occur—never a bound-file reopen/read or second immutable recheck;
- the initial and final Git classifications, descriptor metadata,
  repository/results/root/absolute-parent identities, protocol path states,
  and complete results-directory inventories agree with the active closed
  entry point or exact authorized internal transition;
- every retained directory preserves device, inode, type, and mode; directory
  timestamps agree in no-mutation intervals and are rebaselined only after an
  exact authorized entry delta is validated;
- the sentinel is published before either worker, any runtime import, or any
  official value access;
- after the sentinel, the parent never reopens, rereads, or rechecks the bound
  snapshot; worker ledgers use one guarded-worker metadata final recheck
  immediately before link, and CSV/manifest publication uses only internal-
  transition and unnamed-FD/canonical-FD checks;
- exactly two fresh isolated workers consume exactly one capability each;
- each worker completes mandatory `prctl` and raw FD validation, opens exactly
  child-local repository and results anchors plus, because the absolute
  allowlist is nonempty, a third child-local filesystem-root `/` anchor before
  guard installation, and inherits only its capability and ledger FDs;
- guarded dirfd use is limited to the registered read-only metadata walker,
  including `/tmp` derived only from registered `/` for the exact allowlisted
  absolute leaf, the inherited post-authentication stage durability
  exception, and the exact one-link/one-open own-ledger exception; every other
  absolute path, dirfd, procfs, openat, or link use remains rejected and
  counted;
- each worker receives exactly its distinct capability and unnamed ledger FDs,
  links and verifies its own canonical ledger through the sole guarded
  procfd/link exception, and makes that ledger durable before importing
  runtime code or accessing values, without parent acknowledgment or extra
  IPC;
- after each worker exits, the parent verifies that canonical ledger against
  its retained unnamed FD and closes every retained ledger descriptor;
- the workers produce byte-identical compressed CSV, decompressed CSV, and
  core JSON;
- every canonical artifact is published create-only from an unnamed
  `O_TMPFILE` inode through the retained results-directory descriptor, by the
  worker only for its own ledger and by the unguarded parent for every other
  artifact;
- no named publication staging leaf is created;
- every stable checkpoint in the production table occurs in order, and every
  helper-local probe, stage creation, ordered staged-file creation, per-file
  cleanup, stage removal/next creation, and canonical-link transient follows
  its exact closed delta protocol;
- no helper-local transient state is accepted as a stable checkpoint,
  completion state, recovery point, or resume authority;
- the final manifest is published last;
- `D5` contains exactly the five specified publication files;
- the committed-publication verifier passes end-to-end at clean pushed `D5`;
- every predecessor residue and permanent absence remains exact;
- no worker stage, capability-probe leaf, unnamed temporary FD, or fixed
  bytecode-cache path remains; and
- `HEAD == @{upstream}` with a clean index and worktree.

Before the `G9CB-5` sentinel, any protocol fault during an authorized
non-synthetic `G9CB-5` operation closes `G9CB-5` without mutation under the
same identity. Such a closure has no attempt sentinel and no terminal-evidence
commit. A helper-local transient state never makes that operation retryable.

After the sentinel, every failure is terminal:

```text
TERMINAL_G9CB5_ATTEMPT_CONSUMED_NO_RETRY
```

After a `G9CB-5` sentinel exists, every retry, resume, repair, amendment,
protocol mutation, metadata correction, second production invocation,
v2-under-`G9CB-5`, or completion attempt under the same identity is forbidden.

Any future continuation after either kind of closure requires another new
infrastructure identity and a standalone successor authority decision.
