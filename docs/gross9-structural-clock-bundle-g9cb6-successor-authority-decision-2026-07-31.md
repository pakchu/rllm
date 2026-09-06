# Gross9 structural clock bundle G9CB-6 successor authority decision — 2026-07-31

## Decision

Freeze `G9CB-6` as a new candidate-independent structural-clock
infrastructure identity and the only authorized successor after the
prepublication closure of `G9CB-5`.

The exact last clean pushed `G9CB-5` implementation commit is:

```text
Q5 = 02c3c83a5253684057f44f51ee96bcb089b40b2f
```

`G9CB-5` has no preregistration seal, access claim, attempt sentinel, worker
ledger, clock CSV, final manifest, or terminal-evidence commit. Its first
non-synthetic preregistration invocation faulted before publication. Under
the controlling pre-sentinel closure rule in its authority decision,
`G9CB-5` closed immediately and is not retryable, resumable, repairable, or
completable under the same identity.

A second same-identity preregistration invocation occurred after that closure.
It was unauthorized, also faulted before publication, and created no
authority. This decision records it conservatively; it does not ratify it.

The only canonical continuation authorized by this decision is:

```text
Q5 -> A6 -> Q6 -> P6 -> C6 -> D6
```

Every commit is single-parent, clean, pushed, and a fast-forward on:

```text
codex/gross9-structural-clock-bundle-20260731
```

History rewriting and force-pushing remain forbidden.

This decision inherits every compatible candidate-independent economic and
structural mechanic frozen by the authority chain through `A5`. It changes
only:

1. the active infrastructure identity from `G9CB-5` to `G9CB-6`;
2. the active output, residue, exception, action, phase, and protocol-version
   literals from `g9cb5` to `g9cb6`;
3. the exact `Q5 -> A6 -> Q6 -> P6 -> C6 -> D6` topology;
4. an exact prepublication-closure record for `G9CB-5`;
5. an exact post-closure incident and recovery-exposure disclosure;
6. preservation of tracked nested top-level `results/` directories in every
   exact inventory check; and
7. synthetic regressions for those changes.

No candidate, candidate rank, comparator clock, return, PnL, funding cash,
CAGR, MDD, economic metric, overlap metric, sleeve weight, signal rule, hold,
barrier, model, feature, source byte, or source path may change.

This file is not operative merely because it exists. It becomes `A6` only
when committed and pushed as the sole change in the direct child of exact
`Q5`.

## Evidence provenance and recoverability

This decision separates four evidence classes:

1. **repository-authenticated facts** are reproducible from committed Git
   objects, the current worktree, and filesystem metadata;
2. **observed execution evidence** is supplied by the current session's two
   command results and recovery operations;
3. **control-flow deductions** follow from the exact committed `Q5` source and
   the recorded exception locations; and
4. **unrecoverable facts** remain JSON null and are not inferred from artifact
   absence.

No standalone raw stdout/stderr capture file, kernel-level open trace,
wall-clock timestamp record, duration record, or resource-usage record was
preserved for either preregistration invocation. The normalized command,
Python traceback text, and process exit status were observed in the current
session and are frozen here, but they are not represented as byte-exact shell
input or raw stream artifacts.

The 16-path and 57-path exposure counts are control-flow deductions from the
sorted declaration set, exact failure locations, and authenticated file
sizes. They are not syscall-trace counts. The recovery ordering and transform
are observed execution evidence, and the resulting sizes and SHA-256 values
are independently reproducible from the surviving files.

Exactly two preregistration invocations are observed in the current `Q5`
continuation. This decision does not infer an unobserved production-invocation
count from artifact absence. The historical `G9CB-5` production invocation
count is therefore unrecoverable and remains JSON null.

## Authenticated repository facts

At drafting time, the following facts are authenticated without decoding a
new candidate, comparator clock, or economic result:

- `HEAD == @{upstream} == Q5`;
- `Q5` is `02c3c83a5253684057f44f51ee96bcb089b40b2f`;
- `Q5` is the direct child of `A5`,
  `1ca718d9dab1077b041e753f3b011fbf5b23f047`;
- the `A5 -> Q5` diff changes exactly the five authorized protocol and test
  files;
- the index and tracked worktree are clean;
- every active `G9CB-5` result path is absent;
- every active `G9CB-5` stage and fixed-bytecode path is absent; and
- this authority decision is the sole new tracked candidate for `A6`.

The exact `A5` authority binding is:

```json
{
  "authority_commit": "1ca718d9dab1077b041e753f3b011fbf5b23f047",
  "git_blob": "e0bb4b1d26a67c4baf681d8a48e988307c92f9f5",
  "git_mode": "100644",
  "path": "docs/gross9-structural-clock-bundle-g9cb5-successor-authority-decision-2026-07-31.md",
  "path_type": "regular_file",
  "sha256": "d0b2e14417b4cd46213708597220067c2195d22308da9eb95921bcb59da27385"
}
```

The exact `Q5` implementation binding is:

```json
{
  "builder_git_blob": "8af92fbdf7200b2e67275d9b41d3e40ebc1449a8",
  "builder_path": "training/build_gross9_structural_clock_bundle.py",
  "builder_sha256": "d7edaa3277b581c675f81b2364421d862c1897e89cc149335d912753bb182802",
  "commit": "02c3c83a5253684057f44f51ee96bcb089b40b2f",
  "preregistration_git_blob": "1f74ddbb8fa019884f674466a29cf0bfb5ec9af1",
  "preregistration_path": "training/preregister_gross9_structural_clock_bundle.py",
  "preregistration_sha256": "2c989f97f8046154d8a479d541c1d4b3cb8f70ab1394d2e610fc203207854e1f"
}
```

## G9CB-5 closure

### Controlling rule

The `A5` authority decision states that, before the `G9CB-5` sentinel, any
protocol fault during an authorized non-synthetic `G9CB-5` operation closes
the identity without mutation and that any continuation requires a new
infrastructure identity and standalone successor authority decision.

The first preregistration invocation therefore closed `G9CB-5` even though no
active artifact was published.

### First invocation: operative closure event

The normalized invocation was:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
```

The exact terminal exception was:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'binance_um_aux_btc_2020_2026'
```

The failure occurred in this control-flow chain:

```text
main
  -> _bootstrap_q5_snapshot
  -> _prepare_declared_snapshot
  -> _PreregistrationSnapshot.open_initial
  -> _PreregistrationSnapshot._parent
  -> _open_component_no_follow
```

The missing component belonged to the declared funding path:

```text
data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz
```

Before the first bound-file open, Git pair classification completed for all
57 declared paths. Sorted snapshot opening then read exactly 16 unique paths
and `21,801,778` bytes once as opaque bytes. All 16 received SHA-256
authentication. Fifteen were tracked paths and additionally received locally
framed Git-blob SHA-1 and Git-mode validation against the already-classified
index/HEAD pair; the absolute `/tmp` OI path had the required paired-null Git
classification. It did not construct the preregistration manifest and did
not run the snapshot final verification read.

The exact 16 paths and sizes were:

```text
19657777 /tmp/btcusdt_open_interest_5m_2020_2026.csv
11848 artifacts/rank7/frozen_annual_rank7_2026/manifest.json
21132 artifacts/rank7/frozen_annual_rank7_2026/models/seed_2026.npz
21643 artifacts/rank7/frozen_annual_rank7_2026/models/seed_7.npz
21251 artifacts/rank7/frozen_annual_rank7_2026/models/seed_71.npz
21353 artifacts/rank7/frozen_annual_rank7_2026/models/seed_715.npz
21580 artifacts/rank7/frozen_annual_rank7_2026/models/seed_71515.npz
2014650 artifacts/rank7/frozen_annual_rank7_2026/state/completed_hourly_history.csv.gz
1811 configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json
778 configs/live/rex_veto_7_candidate.json
1362 configs/shadow/fresh_kimchi_fx_2026-07-16.json
945 configs/shadow/frozen_annual_rank7_2026-07-16.json
1107 configs/shadow/markov_transition_long_2026-07-16.json
1799 configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json
1928 configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json
814 configs/shadow/rex_taker_low_range_position_2026-07-16.json
```

No gzip member, CSV row, JSONL row, NPZ array, model value, history value,
feature, schedule, candidate, comparator, return, PnL, funding cash, CAGR,
MDD, economic metric, or overlap metric was decoded, loaded, constructed, or
computed by this first invocation.

It performed no publication capability probe, unnamed-file preparation,
hard-link publication, result-file write, stage creation, worker start,
sentinel publication, or bytecode-cache creation.

### Second invocation: unauthorized post-closure incident

After local runtime-input restoration, the same normalized command was
invoked a second time. That invocation was not authorized because the first
fault had already closed `G9CB-5`.

The exact terminal exception was:

```text
FileExistsError: Q5 exact results inventory differs
```

The second invocation opened exactly 57 unique declared paths and
`105,499,876` bytes once into the retained initial snapshot. It then
constructed an in-memory manifest from cached bytes. This included metadata
JSON validation, static Python AST parsing, environment inventory, Git
classification, opaque source hashing, and predecessor validation.

It did not run the snapshot final verification read. It failed at the first
`_validate_closed_path_state` call, before the publication capability probe,
before `O_TMPFILE` preparation, and before any canonical link.

No official source gzip, CSV, JSONL, NPZ, model, or history values were
decoded or loaded by the second invocation. No runtime model path, feature,
schedule, sleeve, candidate, comparator, return, PnL, funding cash, CAGR,
MDD, economic metric, or overlap metric was reached.

The second invocation created no authority and cannot make `G9CB-5`
retryable. Its only durable consequence is the disclosure in this decision.

### Recovery exposure before and between invocations

The recovered worktree had lost ignored runtime inputs and historical
filesystem modes when its `/tmp` worktree was recreated. Recovery restored
only byte-identical inputs and declared historical filesystem state.

The frozen open-interest-enriched gzip was authenticated as:

```json
{
  "logical_path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
  "resolved_path": "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
  "sha256": "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
  "size_bytes": 72898508
}
```

Before the first canonical preregistration invocation, the exact `/tmp`
open-interest artifact was absent. Recovery used pandas `2.3.3` to execute
the already-existing deterministic transform:

```python
pd.read_csv(source, usecols=["date", "open_interest"]).to_csv(target, index=False)
```

This recovery decompressed the entire gzip stream, decoded its 30-column
header, traversed `674,785` data rows, and selected the `date` and
`open_interest` fields for all `674,785` rows. It produced exactly:

```json
{
  "columns": ["date", "open_interest"],
  "path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
  "rows": 674785,
  "sha256": "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
  "size_bytes": 19657777
}
```

This was real source-value access and must not be represented as zero or
hidden by `P6`. It was an environment-recovery transform, not a candidate or
economic evaluation. No return, PnL, funding cash, CAGR, MDD, candidate
metric, or overlap metric was computed.

After the first invocation and before the unauthorized second invocation, six
additional missing ignored inputs were authenticated against their frozen
SHA-256 values and copied byte-for-byte as regular files. Those six recovery
operations were opaque byte copies; they did not decompress, parse, or load
their values.

Before `Q5` verification and the first canonical invocation, historical result
files were restored to their declared mode `0444`, the two declared
predecessor residue directories were restored empty at mode `0700`, and the
frozen OI gzip logical path was restored to its exact authenticated target.
Those actions changed no tracked byte and created no active `G9CB-5`
artifact.

### Closure conclusion

The exact closure conclusions are:

```text
G9CB-5 complete = false
G9CB-5 retryable = false
G9CB-5 resumable = false
G9CB-5 repairable = false
P5 exists = false
C5 exists = false
D5 exists = false
T5 exists = false
observed G9CB-5 preregistration invocations = 2
observed G9CB-5 canonical publication links = 0
G9CB-5 production invocations = null
```

The production-invocation JSON null is unknown and unrecoverable. It must not
be interpreted as zero.

The exact permanently absent `G9CB-5` outputs are the following sorted list:

```text
results/gross9_structural_clock_bundle_g9cb5_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb5_access_claim_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb5_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb5_manifest_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass2_2026-07-31.json
```

Every path above remains absent through `D6`. No future identity may reuse,
replace, rename, or hard-link any of them as an active artifact.

The exact `G9CB-5` residue state is:

```json
{
  "bytecode_cache": {
    "path": "results/.g9cb5-bytecode-cache-disabled",
    "state": "absent"
  },
  "publication_stages": {
    "glob": "results/.gross9_structural_clock_bundle_g9cb5_*.stage-*",
    "state": "absent"
  },
  "worker_stages": {
    "glob": "results/.gross9-structural-clock-g9cb5-worker-*",
    "state": "absent"
  }
}
```

## Required G9CB-5 prepublication-closure binding

`P6` must add a new exact singleton binding:

```text
bindings.failed_predecessor_prepublication_closures
```

It contains only `G9CB-5`. It is separate from:

- `failed_predecessor_preregistrations`, which remains the exact two-row
  historical `G9CB-1` list;
- `failed_predecessor_attempts`, which remains the exact two-row `G9CB-2` and
  `G9CB-3` list;
- `failed_predecessor_closures`, which remains the exact one-row `G9CB-4`
  pre-access/pre-sentinel closure list; and
- `successor_preregistrations`, which remains the exact three-row
  `G9CB-2`, `G9CB-3`, and `G9CB-4` list.

The new `G9CB-5` row has exactly these top-level keys:

```text
authority_decision
classification
failure
identity
permanently_absent_outputs
post_closure_incident
protocol_implementation
protocol_version
recovery_exposure
residue
status
topology
```

Its exact scalar classification is:

```json
{
  "classification": "pre_preregistration_publication_missing_runtime_input_bootstrap_failure",
  "identity": "G9CB-5",
  "protocol_version": "gross9_structural_clock_bundle_g9cb5_v1",
  "status": "historical_prepublication_closure_no_preregistration_no_attempt_no_clock_authority"
}
```

`authority_decision` is the exact `A5` binding above.
`protocol_implementation` is the exact `Q5` binding above.

`failure` is exactly:

```json
{
  "bytes_opened": 21801778,
  "exception": "FileNotFoundError: [Errno 2] No such file or directory: 'binance_um_aux_btc_2020_2026'",
  "exit_status": 1,
  "manifest_constructed": false,
  "normalized_invocation": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle",
  "observed_preregistration_invocations": 2,
  "official_production_invocations": null,
  "paths_opened": 16,
  "preregistration_published": false,
  "publication_capability_probe_started": false,
  "snapshot_final_recheck_completed": false,
  "status": "authorized_first_invocation_closed_identity"
}
```

`post_closure_incident` is exactly:

```json
{
  "bytes_opened": 105499876,
  "exception": "FileExistsError: Q5 exact results inventory differs",
  "exit_status": 1,
  "manifest_constructed": true,
  "metadata_json_decoded": true,
  "normalized_invocation": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle",
  "paths_opened": 57,
  "preregistration_published": false,
  "publication_capability_probe_started": false,
  "runtime_python_ast_parsed": true,
  "snapshot_final_recheck_completed": false,
  "source_model_or_history_values_decoded_or_loaded": false,
  "status": "unauthorized_post_closure_invocation_no_publication",
  "unauthorized_after_closure": true
}
```

`recovery_exposure` is exactly:

```json
{
  "candidate_or_economic_metric_computed": false,
  "frozen_open_interest_reconstruction": {
    "occurred_before_first_invocation": true,
    "output": {
      "columns": ["date", "open_interest"],
      "path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
      "rows": 674785,
      "sha256": "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
      "size_bytes": 19657777
    },
    "pandas_version": "2.3.3",
    "selected_columns": ["date", "open_interest"],
    "source": {
      "data_rows_traversed": 674785,
      "gzip_stream_decompressed": true,
      "header": ["date", "open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "tic", "day", "dxy", "kimchi_premium", "usdkrw", "btckrw", "dxy_available", "kimchi_available", "usdkrw_available", "external_any_available", "dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_zscore", "usdkrw_momentum", "open_interest", "open_interest_value", "cmc_circulating_supply", "open_interest_available"],
      "header_columns_decoded": 30,
      "logical_path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
      "resolved_path": "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
      "sha256": "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
      "size_bytes": 72898508
    },
    "transform": "pd.read_csv(source, usecols=['date', 'open_interest']).to_csv(target, index=False)"
  },
  "historical_filesystem_state": {
    "empty_mode_0700_directories": ["results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23", "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef"],
    "mode_0444_files": ["results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb3_access_claim_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_2026-07-31.json", "results/gross9_structural_clock_bundle_g9cb4_preregistration_2026-07-31.json", "results/gross9_structural_clock_bundle_preregistration_2026-07-31.json", "results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json"],
    "tracked_bytes_changed": false
  },
  "opaque_regular_file_restoration_destination_root": "/tmp/rllm-alpha-orthogonal-20260718",
  "opaque_regular_file_restoration_source_root": "/home/pakchu/rllm",
  "opaque_regular_file_restorations": [
    {"path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz", "sha256": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7", "size_bytes": 89326},
    {"path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz", "sha256": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7", "size_bytes": 1196481},
    {"path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz", "sha256": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c", "size_bytes": 66696659},
    {"path": "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl", "sha256": "bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7", "size_bytes": 1029745},
    {"path": "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl", "sha256": "b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371", "size_bytes": 1253048},
    {"path": "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl", "sha256": "07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0", "size_bytes": 6128620}
  ]
}
```

Those objects do not invent raw stdout/stderr files, timestamps, duration,
resource usage, or kernel-level traces that were not durably captured.

`topology` is exactly:

```json
{
  "g9cb5_authority_commit": "1ca718d9dab1077b041e753f3b011fbf5b23f047",
  "g9cb5_protocol_commit": "02c3c83a5253684057f44f51ee96bcb089b40b2f",
  "g9cb6_authority_commit": "<A6>",
  "preregistration_commit": null,
  "terminal_evidence_commit": null
}
```

`"<A6>"` is a metavariable replaced at `P6` by the exact lowercase 40-hex
commit that adds only this decision.

## G9CB-6 identity and paths

The exact active literals are:

```text
identity = G9CB-6
preregistration protocol = gross9_structural_clock_bundle_g9cb6_preregistration_v1
builder/publication protocol = gross9_structural_clock_bundle_g9cb6_v1
terminal exception = TerminalG9CB6Failure
terminal action = TERMINAL_G9CB6_ATTEMPT_CONSUMED_NO_RETRY
preregistration phase = Q6_PREREGISTRATION_PUBLICATION
fixed bytecode prefix = results/.g9cb6-bytecode-cache-disabled
worker stage prefix = results/.gross9-structural-clock-g9cb6-worker-
```

The exact active paths are:

| Role | Path |
|---|---|
| authority | `docs/gross9-structural-clock-bundle-g9cb6-successor-authority-decision-2026-07-31.md` |
| preregistration | `results/gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json` |
| access claim | `results/gross9_structural_clock_bundle_g9cb6_access_claim_2026-07-31.json` |
| attempt sentinel | `results/gross9_structural_clock_bundle_g9cb6_attempt_consumed_2026-07-31.json` |
| worker ledger pass 1 | `results/gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass1_2026-07-31.json` |
| worker ledger pass 2 | `results/gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass2_2026-07-31.json` |
| canonical CSV gzip | `results/gross9_structural_clock_bundle_g9cb6_2026-07-31.csv.gz` |
| final manifest | `results/gross9_structural_clock_bundle_g9cb6_manifest_2026-07-31.json` |

No `G9CB-5` active path may be reused.

## Exact results-inventory correction

The `Q5` defect computed tracked result names as only rows whose immediate
parent was exactly `results/`. That excluded legitimate tracked nested
top-level directories while `os.listdir(results)` included them.

The five authenticated tracked top-level directories that exposed the defect
are:

```text
results/bctp_mdp_run_2026-07-25
results/bctp_target_schedule_seals
results/cross_venue_volatility_shape_handoff_source_support_2026-07-30
results/event_horizon_candidates_ext_micro_c72_s2
results/token_combo_edge
```

`Q6` must derive tracked top-level inventory from every `git ls-files --
results` row by taking the first path component below `results/`. Thus a
direct tracked file contributes its filename and a tracked nested path
contributes its top-level directory name.

For a normalized tracked path `results/<entry>/...`, the contribution is
exactly `<entry>`. Empty, absolute, dot, dot-dot, backslash, or non-`results`
paths are rejected rather than normalized permissively.

Every preregistration and builder closed-phase inventory check must compare
the actual retained-descriptor entry set with exactly:

```text
tracked top-level results entries
union exact predecessor residue directory names
union exact phase-authorized active untracked leaves
```

No unrelated untracked or ignored entry is admitted. The correction does not
weaken exact inventory; it authenticates the complete Git tree shape that the
old direct-file-only projection omitted.

Both preregistration and builder implementations must use one shared logical
rule, with independent regressions proving direct files, nested tracked
directories, malformed paths, missing tracked top-level entries, and extra
untracked top-level entries.

## Frozen value-access and economic boundary

The OI recovery exposure above is historical, explicit, and immutable. `P6`
must include it in the `G9CB-5` prepublication-closure binding.

The `P6` creation-evidence counters remain process-local counters for the
single successful `P6` process. They must not erase, contradict, or be
interpreted as global historical counters. The historical recovery exposure
is reported separately and takes precedence for any question about whether
the frozen OI gzip was ever decompressed before `P6`.

No candidate identity or candidate artifact has been introduced. The
structural bundle remains candidate-independent. Comparator clocks remain
preseen by the research program but are not opened by `P6`, `C6`, or `D6`.

The exact domain, source bytes and hashes, feature logic, schedules, five
sleeves, order, configured weights, sides, holds, barriers, Rank7 model and
history bytes, two-pass worker-capability protocol, counters, serialization,
and economic prohibitions remain unchanged from `A5`.

## Authorized Q6 diff

`A6` adds only this file. `Q6` is the direct child of clean pushed `A6` and
may modify exactly these five paths:

```text
tests/test_build_gross9_structural_clock_bundle.py
tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
tests/test_preregister_gross9_structural_clock_bundle.py
training/build_gross9_structural_clock_bundle.py
training/preregister_gross9_structural_clock_bundle.py
```

`Q6` may make only the identity, topology, closure-binding, recovery-
disclosure, complete tracked-results inventory, and synthetic-regression
changes authorized here. It may not alter any frozen source byte, economic
object, structural clock rule, model, schedule, signal, side, hold, barrier,
weight, serializer, worker-capability ordering, or publication ordering.

No active `G9CB-6` artifact may exist at `Q6`.

## Required regression evidence before Q6

Before `Q6` is committed and pushed, regressions must prove at minimum:

1. the exact `Q5 -> A6 -> Q6` single-parent topology and exact diffs;
2. the `G9CB-5` closure binding contains the exact `A5`, `Q5`, first-failure,
   second-incident, recovery-exposure, permanent-absence, residue, and
   topology facts above;
3. every `G9CB-5` active path remains absent;
4. the existing predecessor lists remain unchanged and the new
   prepublication-closure list contains only `G9CB-5`;
5. the active protocol inventory includes exact `A6` and all inherited
   protocol paths once each;
6. both preregistration and builder accept legitimate tracked nested
   top-level result directories;
7. both reject a missing tracked top-level entry, extra untracked entry,
   malformed tracked path, or phase-unauthorized active leaf;
8. no symlink, hard-link alias, descriptor swap, Git pair mutation, directory
   substitution, or inventory drift is accepted;
9. every `g9cb5` active path, phase, exception, action, stage, and bytecode
   literal is historical-only and cannot become active authority;
10. all active outputs use only `g9cb6` paths;
11. synthetic `P6`, `C6`, `D6`, and committed-verifier topology succeeds;
12. every inherited authentication, boundary, isolation, serialization,
    two-pass, and publication regression remains passing; and
13. no test decodes an official source value or computes economics.

## Canonical commands and one-shot rule

After clean pushed `Q6`, the sole canonical preregistration command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
```

After `P6` is committed and pushed, the sole canonical claim command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
```

After `C6` is committed and pushed, the sole canonical production command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
```

After `D6` is committed and pushed, the metadata-only committed verifier is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
```

No rehearsal, official-source probe, partial production, retry, or second
invocation is authorized. Before the `G9CB-6` sentinel, any non-synthetic
protocol fault closes `G9CB-6` without same-identity repair. After the
sentinel, every failure is terminal under:

```text
TERMINAL_G9CB6_ATTEMPT_CONSUMED_NO_RETRY
```

Any continuation after either closure requires another new identity and a
new standalone successor authority decision.

## Stop conditions

Stop before `P6` unless all of the following are true:

- `A6` is the sole-change direct child of exact `Q5` and is pushed;
- `Q6` is the exact-five-change direct child of `A6` and is pushed;
- the full owned regression suite and independent review pass;
- the complete declared runtime-input set is present as exact regular files
  before launching the canonical process;
- the actual top-level `results/` inventory equals the corrected authenticated
  inventory;
- every `G9CB-5` and `G9CB-6` active artifact is absent; and
- the worktree and index are clean with `HEAD == @{upstream}`.

If any condition differs, do not invoke the canonical command.
