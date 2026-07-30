# Gross9 structural clock bundle runtime-isolation amendment — 2026-07-31

## Status

This candidate-independent amendment is `G9CB-1B`. It supplements
`G9CB-1A` and supersedes only the runtime-import, post-sentinel worker
authentication, staged-output/rebuild-receipt, selected physical-counter,
prohibited-output-placement, publication-allowlist, and completion-seal
clauses of:

- `docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md`; and
- `docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md`.

Every other requirement remains in force. In particular, `G9CB-1B` does not
change the domain, source hashes, five sleeves, weights, clocks, Rank7 learner,
Rank7 policy, Rank7 label formula, bundle files, frozen interpreter, package,
or lockfile environment, one-shot rule,
exactly two rebuilds, byte-identity gate, manifest-last publication, candidate
independence, or the ban on portfolio economics and overlap computation.

No source row, model array, history row, candidate row, comparator-clock row,
return, PnL, or metric was opened to create this amendment. It is based only
on authenticated source code, configuration metadata, the committed bundle
manifest metadata, and the two independent protocol reviews that identified
an import-boundary and replay-capability defect.

This amendment is not operative merely because it exists or is committed.
It must first be committed and pushed as a standalone metadata-only authority
commit. The later protocol commit `P`, preregistration, claim, sentinel, each
fresh worker, each per-pass core, and final manifest must bind both `G9CB-1A`
and this exact amendment before any generic import or value-row access.

The canonical ordered binding is:

```text
authority_amendments[0].identity = "G9CB-1A"
authority_amendments[1].identity = "G9CB-1B"
```

Each row has exactly:

```text
identity
path
path_type = "regular_file"
sha256
git_blob
git_mode = "100644"
authority_commit
```

The list order and complete row objects must be identical at:

```text
preregistration.bindings.authority_amendments
claim.authority_amendments
sentinel.authority_amendments
per_pass_core.authority_amendments
final_manifest.authority_amendments
```

The claim and sentinel therefore authenticate the amendment list directly,
not merely through the preregistration hash. A missing, additional, reordered,
renamed, or changed row is terminal before capability consumption, generic
import, or value-row access.

## Planned-path extension

`G9CB-1B` supersedes the exclusivity sentence in the `G9CB-1` “Planned files
and canonical artifacts” section only to append these exact paths:

| Stage | Path | Contract |
|---|---|---|
| Rank7 authority | `docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md` | Standalone metadata-only `G9CB-1A` authority |
| runtime-isolation authority | `docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md` | Standalone metadata-only `G9CB-1B` authority |
| isolated adapter primitives | `training/gross9_structural_clock_primitives.py` | Causal/structural functions only |
| isolated adapter tests | `tests/test_gross9_structural_clock_primitives.py` | Exact source-derived synthetic parity |
| isolated Rank7 facade | `execution/gross9_rank7_clock_runtime.py` | Narrow bundle/feature/scoring contract |
| isolated Rank7 tests | `tests/test_gross9_rank7_clock_runtime.py` | Exact differential synthetic parity |
| pass-1 consumption ledger | `results/gross9_structural_clock_bundle_worker_capability_consumed_pass1_2026-07-31.json` | Durable evidence that the slot-1 anonymous-pipe carrier was consumed |
| pass-2 consumption ledger | `results/gross9_structural_clock_bundle_worker_capability_consumed_pass2_2026-07-31.json` | Durable evidence that the slot-2 anonymous-pipe carrier was consumed |

The two implementation files and their tests are protocol files bound by `P`.
They are also authenticated through the executable import closure where
applicable. The two consumption ledgers are canonical terminal-attempt
artifacts: they are committed with the sentinel, CSV, and final manifest, but
neither ledger alone is clock or economic authority. Each worker may also
create the exact transient stage-local receipt name
`gross9_structural_clock_bundle_pass_receipt.json`; that receipt is embedded
by value and hash in the final manifest and is never a canonical or committed
sidecar. No other implementation, test, sidecar, cache, transient output, or
canonical artifact path is authorized.

The preregistration `output_paths` object and claim `canonical_outputs` object
each add the exact ordered field:

```text
worker_capability_consumption_ledgers = [
  "results/gross9_structural_clock_bundle_worker_capability_consumed_pass1_2026-07-31.json",
  "results/gross9_structural_clock_bundle_worker_capability_consumed_pass2_2026-07-31.json"
]
```

No per-pass core may include a pass-specific ledger observation because the
two core byte streams must be identical. Each core instead binds the common
sentinel, whose ordered capability rows contain both ledger paths. The parent
adds the two observed compact ledger bindings only to the final manifest after
both worker products and both ledgers authenticate.

## Contradictions and defects repaired

The original runtime inventory named:

```text
execution/portfolio_live.py
execution/rank7_runtime.py
execution/rex_llm_live.py
```

as metadata authority roots. The first implementation also imported all three
roots and several broad research/audit modules in each worker. Their static
closures contain dormant equity, return, PnL, drawdown, CAGR, ranking, and
overlap helpers. Importing those closures contradicts the permanent rule that
the builder must not import or call those helpers.

The first implementation also passed the same raw worker token through both
an environment variable and a command-line argument. A caller able to repeat
that invocation could replay a worker or create a third worker. That does not
satisfy “exactly two fresh rebuilds.”

Finally, several access counters were inferred from file size, manifest row
count, or loaded-object count after the operation. `G9CB-1A` requires counters
to increment at the operation that actually opens, decodes, computes, scores,
or compares the value.

## Isolated executable import closure

The three original runtime files remain authenticated direct-authority
metadata. They are not executable import roots for `G9CB-1B`.

Each official worker may freshly import only these two repository-local roots:

```text
execution/gross9_rank7_clock_runtime.py
training/gross9_structural_clock_primitives.py
```

The preregistration producer and production builder must discover the complete
static repository-local closure of those roots with the existing AST rule and
bind every closure member by path, SHA-256, Git blob, mode, and
package-initializer status. Production repeats the discovery before the
sentinel and requires exact equality.

The isolated closure may import external frozen-environment packages and
causal preprocessing modules. It must not import any candidate runner,
portfolio optimizer, audit/report module, economic-statistics module, or
helper that computes portfolio return, portfolio PnL, funding cash, equity,
drawdown, CAGR, economic rank, correlation, Jaccard, containment, or overlap.

`training/gross9_structural_clock_primitives.py` may contain only the exact
causal or structural primitives needed by the five bound clocks:

- market normalization and Binance auxiliary attachment;
- causal market, interest, Kimchi/FX, Markov, REX, nested-barrier, market-braid,
  and Rank7 state features;
- gate evaluation and immutable decision anchors;
- non-overlapping structural schedule walking;
- the frozen annual Rank7 fold masks, balanced `(year, source)` weights,
  deterministic ExtraTrees prediction, and frozen activation thresholds.

Causal feature ratios, log changes, volatility, range position, and state
quantiles are feature inputs, not portfolio returns or economic outputs. They
remain allowed only because they are exact parts of the already authenticated
clock feature contracts. The isolated module must not aggregate them into a
trade, sleeve, or portfolio economic result.

The existing in-memory parent authentication object remains authorized but is
not a sidecar or independent authority. Before sentinel publication the parent
constructs exactly:

```text
parent_authentication = {
  "environment": <exact preregistration.bindings.environment object>,
  "hashed_inputs": [
    {"path": <string>, "sha256": <lowercase hex>, "size_bytes": <int>},
    ...
  ],
  "runtime_import_closure": <exact preregistration.bindings.runtime_import_closure list>
}
```

The `hashed_inputs` membership algorithm walks only the canonical
preregistration JSON object, depth-first through mapping values in sorted-key
order and list elements in stored order. A mapping qualifies exactly when it
has a string `sha256` and at least one string path key from the ordered alias
list `("path", "logical_path", "repository_path")`. The digest must be exactly
64 lowercase hexadecimal characters. Every present alias key must have a
string value; a present non-string alias is terminal. If multiple path aliases
are present, their string values must be byte-identical; otherwise
authentication is terminal. The selected path is the first string-valued alias
in the stated order. A repository path must already be normalized relative
POSIX text with no empty, `.`, or `..` component; an absolute path must already
equal its no-symlink canonical absolute POSIX path. The observed regular-file
size and digest produce exactly `{"path", "sha256", "size_bytes"}`.

A mapping with a string `sha256` and at least one present alias key but no
string-valued alias is malformed and terminal, not silently ignored.

Qualifying rows are keyed only by the selected canonical path. Repeated rows
with the same path, digest, and observed size collapse to one row; the same
path with a different digest or alias value is terminal. If a qualifying row
declares `size_bytes`, it must be a non-boolean nonnegative integer equal to
the observed size; conflicting declared sizes are terminal. Two authenticated
observations of one path must also return the same size or authentication is
terminal. If two repeats both declare `path_type`, `git_blob`, or `git_mode`,
their corresponding values must agree; absence in one repeat is not a
conflict, while a declared mismatch is terminal. The final list must have
unique paths and is sorted
lexicographically by `path`. It is produced only after regular-file, size,
SHA-256, and applicable Git blob/mode authentication.

The parent-authentication object is serialized as compact canonical JSON with
sorted keys and no trailing LF; the sentinel contains the exact top-level
field `parent_authentication_sha256`, the lowercase SHA-256 of those bytes.
Each worker receives the same non-secret bytes only through
`--parent-auth-json`, requires canonical byte equality and the sentinel hash,
then independently recomputes all three fields before isolated import. Each
per-pass core contains the exact top-level `parent_authentication` object and
`parent_authentication_sha256`; the final manifest inherits both from the
byte-identical core. A different field, order, row, hash, transport, or
sidecar is forbidden.

All Git blob/mode/commit subprocess authentication is parent-only and finishes
before sentinel publication. A worker's independent recomputation uses
stdlib-only canonical-file path/type/size/SHA checks, closure discovery, and
environment-fingerprint checks; it compares the already parent-authenticated
Git metadata by canonical bytes and must not launch `git` or any other child
process.

## Isolated Rank7 runtime contract

`execution/gross9_rank7_clock_runtime.py` is a narrow, source-equivalent
runtime facade. Its ordered `__all__`, complete supported public API, and type
surface are frozen to exactly:

```text
constants:
  FEATURE_COLUMNS
  EXPECTED_SEEDS
  EXPECTED_MODEL_PARAMS
  SOURCE_COLUMNS
  SOURCE_PRIORITY
  NO_BARRIER_BPS
classes:
  Rank7BundleError
  Rank7FeatureError
  FrozenExtraTreesModel
  Rank7Decision
  Rank7Bundle
functions:
  apply_rank7_delay
  load_frozen_extra_trees
  rank7_manifest_hash
  rebuild_rank7_feature_context
  build_rank7_feature_context
  rank7_barrier_contract
  score_rank7_row
```

Names beginning with `_` are private and outside the public surface. The facade
must not expose `save_frozen_extra_trees` or any broad runtime, audit,
portfolio, candidate, or economic helper. The official builder may call only:

```text
FrozenExtraTreesModel.predict
rank7_manifest_hash
Rank7Bundle.load
build_rank7_feature_context
score_rank7_row
```

The synthetic differential suite may call every frozen public name. For each
constant, the Python value and concrete container/scalar type must equal the
original. Each facade class has a one-to-one corresponding original class:

```text
facade.Rank7BundleError       <-> original.Rank7BundleError
facade.Rank7FeatureError      <-> original.Rank7FeatureError
facade.FrozenExtraTreesModel  <-> original.FrozenExtraTreesModel
facade.Rank7Decision          <-> original.Rank7Decision
facade.Rank7Bundle            <-> original.Rank7Bundle
```

The three dataclasses must have identical field names, declaration order,
defaults/default factories, annotations after applying the class/module
correspondence above, `frozen` status, generated equality behavior, properties,
and method signatures. A successful operation must return the corresponding
implementation's class, not a dictionary or shared substitute;
`dataclasses.asdict` values are compared exactly. The two error classes must
have the same immediate base-class name and MRO shape. For every negative
fixture, a Rank7-specific failure must use the implementation's own
corresponding error class; a built-in or external-library failure must use the
same concrete exception class. In all cases the failing public operation and
exact message must match. Comparing only exception class-name strings is
insufficient.

Its bundle validation, portable model loading, model prediction order, feature
column order, medians, clipping, delay, hourly warm start, source identity,
thresholds, interaction gate, validity window, and source-routed barrier
metadata must be semantically identical to the authenticated
`execution/rank7_runtime.py` source contract.

The facade must not import `execution/rank7_runtime.py` at production time and
must not use module substitution, `runpy`, source rewriting, dynamic import
hooks, or preloaded fake modules. Its required helpers must come only from the
isolated authenticated closure.

Before `P`, synthetic differential tests must compare the isolated facade with
the original runtime without opening the official model arrays or
hourly-history values. The valid-fixture inventory is mandatory and contains:

1. one canonical synthetic bundle with five portable models, complete manifest,
   complete hourly history, valid fixture predictions, both sources, finite
   medians, exact clipping, and a non-empty validity interval;
2. portable-model one-dimensional and multi-row matrices, including leaf-only
   trees and both left/right traversal;
3. delay lengths `0`, shorter than the matrix, equal to the matrix, and longer
   than the matrix, with source-column restoration;
4. feature-context builds with and without hourly warm history and with both
   funding and premium source identity;
5. active and inactive score rows covering validity, decision clock, immutable
   anchor, source priority, score/risk thresholds, the funding interaction,
   and source-owned barrier metadata; and
6. manifest hashing and canonical JSON with the self-hash field present.

The malformed/adversarial fixture inventory is also mandatory. Starting from
the canonical valid fixture, the suite must independently mutate every
following contract and compare the exact disposition. If both implementations
accept a normalization the resulting value and type must match exactly; if
they reject, the failing public operation, corresponding exception type, and
exact message must match:

- manifest self-hash mismatch; unsupported `schema_version`; wrong
  `policy_type`, `strategy_id`, or cadence;
- missing, extra, reordered, or changed seeds; wrong tree count, prediction
  job count, portable-model format, or any exact ExtraTrees parameter;
- missing, extra, reordered, or changed feature columns; missing, extra,
  reordered, or changed source columns; missing, extra, reordered, or changed
  source priority; wrong delay, initial fill, cooldown, or no-overlap flag;
- absent or failed parity status, and each failed feature, prediction, or
  schedule parity gate;
- median wrong length or non-finite value; missing, non-finite, reversed, or
  wrong-length clip; invalid timestamp, annual cutoff mismatch, empty or
  reversed validity interval;
- threshold object missing or extra keys and each non-finite threshold;
- exits object missing or extra sources, wrong source-owned hold, or absent,
  non-finite, or nonnumeric take/stop barrier;
- model path absolute, escaping the bundle, symlinked within the bundle,
  missing, duplicated, or reordered; checksum mismatch; wrong model
  seed/count/format; missing, extra, malformed, wrong-dtype, wrong-shape,
  non-finite, invalid-offset, invalid-child, or incompatible portable-model
  arrays;
- fixture input/output wrong row count or shape, non-finite fixture value, and
  fixture prediction mismatch;
- hourly-history path absolute, escaping, symlinked within the bundle, or
  missing; checksum
  mismatch; wrong columns/order/dtypes; duplicate, unsorted, naive, invalid, or
  non-hourly-grid timestamps; non-finite numeric value; and declared/decoded
  row-count mismatch;
- market input missing a required column; duplicate, unsorted, naive,
  invalid, incomplete, or off-grid five-minute timestamps; one representative
  non-finite required numeric value; unavailable latest funding, premium, or
  open-interest observation; non-positive latest open-interest value; invalid
  `spot_rows` or `premium_rows`; and hourly warm-history overlap or gap; and
- score row wrong shape or non-finite; timestamp outside validity or off the
  hourly clock; absent immutable anchor; no source; both source bits set;
  funding-versus-premium priority; score below threshold; risk above cap;
  each side of the funding width/pullback interaction; and malformed barrier
  specification.

The literal expected case-ID tuple, in order, is:

```text
manifest_hash_mismatch
schema_version_wrong
policy_type_wrong
strategy_id_wrong
cadence_wrong
seeds_missing
seeds_extra
seeds_reordered
seeds_changed
trees_per_seed_wrong
prediction_n_jobs_wrong
model_format_wrong
param_max_depth_wrong
param_min_samples_leaf_wrong
param_max_features_wrong
param_bootstrap_wrong
feature_columns_missing
feature_columns_extra
feature_columns_reordered
feature_columns_changed
source_columns_missing
source_columns_extra
source_columns_reordered
source_columns_changed
source_priority_missing
source_priority_extra
source_priority_reordered
source_priority_changed
delay_bars_wrong
delay_initial_fill_wrong
anchor_cooldown_wrong
no_overlap_wrong
parity_missing
parity_status_failed
parity_feature_failed
parity_prediction_failed
parity_schedule_failed
medians_wrong_length
medians_nonfinite
clip_missing
clip_nonfinite
clip_reversed
clip_wrong_length
valid_from_invalid
valid_until_invalid
annual_cutoff_invalid
annual_cutoff_mismatch
validity_empty
validity_reversed
threshold_key_missing
threshold_key_extra
threshold_funding_score_nonfinite
threshold_premium_score_nonfinite
threshold_funding_risk_cap_nonfinite
threshold_premium_risk_cap_nonfinite
threshold_width_q20_nonfinite
threshold_pullback_q40_nonfinite
score_lambda_nonfinite
exits_missing_source
exits_extra_source
funding_hold_wrong
premium_hold_wrong
funding_take_missing
funding_take_nonfinite
funding_take_nonnumeric
funding_stop_missing
funding_stop_nonfinite
funding_stop_nonnumeric
premium_take_missing
premium_take_nonfinite
premium_take_nonnumeric
premium_stop_missing
premium_stop_nonfinite
premium_stop_nonnumeric
model_path_absolute
model_path_escape
model_path_internal_symlink
model_path_missing
model_path_duplicate
model_rows_reordered
model_checksum_mismatch
model_seed_wrong
model_declared_tree_count_wrong
model_row_format_wrong
model_declared_feature_width_wrong
model_declared_output_width_wrong
npz_array_missing
npz_array_extra
npz_container_malformed
npz_array_wrong_dtype
npz_array_wrong_shape
npz_array_nonfinite
npz_offsets_invalid
npz_child_invalid
npz_feature_incompatible
fixture_rows_wrong_count
fixture_rows_wrong_shape
fixture_rows_nonfinite
fixture_expected_wrong_shape
fixture_expected_nonfinite
fixture_prediction_mismatch
history_path_absolute
history_path_escape
history_path_internal_symlink
history_path_missing
history_checksum_mismatch
history_columns_wrong
history_dtype_coercion
history_timestamp_duplicate
history_timestamp_unsorted
history_timestamp_naive
history_timestamp_invalid
history_grid_gap
history_numeric_nonfinite
history_declared_row_count_mismatch
market_column_missing
market_timestamp_duplicate
market_timestamp_unsorted
market_timestamp_naive
market_timestamp_invalid
market_grid_gap
market_timestamp_off_grid
market_required_nonfinite
latest_open_interest_unavailable
latest_funding_unavailable
latest_premium_unavailable
latest_open_interest_nonpositive
spot_rows_wrong
premium_rows_wrong
hourly_overlap_mismatch
hourly_warm_start_gap
score_row_wrong_shape
score_row_nonfinite
score_before_validity
score_at_valid_until
score_off_clock
score_anchor_absent
score_source_absent
score_both_sources
score_source_priority
score_below_threshold
score_risk_above_cap
funding_width_pass
funding_pullback_pass
funding_interaction_fail
premium_ignores_funding_interaction
barrier_take_missing
barrier_stop_missing
barrier_take_nonnumeric
barrier_stop_nonnumeric
```

The tuple contains exactly 150 unique IDs.

This is a disposition-parity inventory, not permission for the official run
to use a symlink or a normalized malformed input: the separate protocol
path-type and source-authentication gates remain stricter and run first.
Every prose mutation above maps one-to-one to the same-semantic literal case
ID in the tuple above; no prose mutation may expand into an unlisted case and no case ID
may combine two mutations. Where one mutation is not reachable through a
public operation, the test must
construct the nearest public valid object and mutate only that one field; it
may not drop the case. Tests must freeze the inventory as parametrized case
IDs in a literal `EXPECTED_ADVERSARIAL_CASE_IDS` tuple, assert exact tuple
equality before running the cases, and emit one result for every ID. Deleting,
renaming, adding, or silently xfail/skipping a case is a test failure.

The synthetic differential verdict is exact, not tolerance-selected:

- exported constants, dictionaries, feature-column order, manifest hash
  strings, decision metadata, barrier metadata, and canonical JSON are equal
  by Python value and concrete type;
- NumPy outputs are equal with `numpy.array_equal(..., equal_nan=True)` and
  matching dtype and shape;
- pandas Series and DataFrames are equal with
  `pandas.testing.assert_series_equal` or `assert_frame_equal` using
  `check_exact=True`, `check_dtype=True`, and identical index/column order;
- timestamps, tuple/list distinctions, `None`, booleans, integer types, and
  floating-point bit patterns are identical;
- portable-model single-row and matrix predictions are bit-exact;
- function and method signatures, keyword-only parameters, defaults,
  properties, dataclass contracts, accepted valid fixtures, rejected malformed
  fixtures, exception operation, corresponding exception type, and exact
  message all match; and
- feature-context dictionaries have the exact same key set and each value is
  compared by its native exact rule above.

If an external library prevents bit-exact equality on the frozen environment,
that is a protocol failure; the test must not introduce a tolerance or round
values to obtain a pass. The official after-sentinel run remains the only
authorized proof on the sealed bundle, and its historical-annual versus
bundle-valid activation parity gate remains mandatory.

## Pure adapter orchestration

The builder keeps stdlib-only import-time and pre-sentinel paths. After a
worker consumes its capability, it imports the two isolated roots and
implements all `G9CB-1` orchestration itself.

The adapter must not import:

```text
training.long_regime_combo_scan
training.portfolio_opt_added_alpha_update
training.portfolio_opt_all_discovered_alpha_gross10
training.audit_fresh_kimchi_orthogonal_alpha
training.compare_expanding_extratrees_rank7_refit_cadence_pre2025
training.portfolio_opt_new_alpha_pool
execution.portfolio_live
execution.rank7_runtime
execution.rex_llm_live
```

Exact source-derived behavior must be locked with regression tests before
those broad imports are removed. The tests must cover market loading, Fresh
exclusive-side masks and stop-before-take schedule geometry, Markov state
mapping, both REX gate dialects, annual Rank7 fold purge, balanced weights,
deterministic prediction, source thresholds, funding interaction, and
non-overlap scheduling.

The only authorized economically derived replay remains the narrow Rank7 label
engine defined by `G9CB-1A`. Its trade object may expose only geometry plus:

```text
price_factor
funding_factor
funding_debit_factor
adverse_price_factor
```

It may not expose or compute a gross-return field, favorable-path factor,
stress cost, portfolio return, portfolio PnL, equity, CAGR, MDD, rank, or
overlap value. Ordinary Fresh and final Rank7 schedules remain OHLC-only
structural replays.

## Exactly two one-use worker capabilities

The worker authority is an in-kernel, genuinely consumable anonymous-pipe
carrier. A regular file, environment variable, command-line token, hard-link
marker, or deletable ledger is not enforcement authority.

Before sentinel publication, the parent reserves exactly two distinct
same-filesystem stage paths named:

```text
results/.gross9-structural-clock-worker-<exclusive-random-suffix>
```

Both paths must initially be absent. The parent exclusively creates only the
slot-1 directory with mode `0700` and requires it empty; the reserved slot-2
path must remain absent throughout slot 1. The parent also requires that both
canonical consumption-ledger paths are absent. For each ordered slot `1` then
`2`, the parent performs exactly this carrier preparation:

1. create one fresh anonymous kernel pipe with `os.pipe2(os.O_CLOEXEC)`; a
   numerically reused descriptor has no authority apart from the newly
   authenticated pipe identity;
2. require `stat.S_ISFIFO(os.fstat(read_fd).st_mode)`, record the read end's
   integer `st_dev` and `st_ino`, and require the two `(st_dev, st_ino)` pairs
   to be distinct;
3. generate exactly 32 random bytes into a mutable in-memory buffer, write all
   32 bytes to the pipe write end, then close the write end before sentinel
   publication;
4. retain only the read descriptor and the parent's mutable token buffer; do
   not duplicate the read descriptor or expose either object to another
   process; and
5. compute the lowercase hexadecimal SHA-256 of the 32 raw bytes for the
   sentinel binding.

The sentinel field is named exactly `worker_capabilities`. It is a two-element
list ordered by slot. Row `0` must map slot `1` to the pass-1 ledger and row `1`
must map slot `2` to the pass-2 ledger. Each row has exactly:

```text
slot
parent_pid
stage_directory
carrier_kind = "anonymous_pipe_v1"
carrier_device
carrier_inode
token_sha256
consumed_ledger_path
```

`parent_pid` is the one parent builder process's positive `os.getpid()` and is
identical in both rows. The two stage paths, two ledger paths, two
`(carrier_device, carrier_inode)` pairs, and two token hashes must each be
unique. The exact slot-to-ledger map is:

```text
slot 1 -> results/gross9_structural_clock_bundle_worker_capability_consumed_pass1_2026-07-31.json
slot 2 -> results/gross9_structural_clock_bundle_worker_capability_consumed_pass2_2026-07-31.json
```

No raw token, reversible token encoding, pipe content, or duplicate read
descriptor may appear in the sentinel, claim, preregistration,
parent-authentication JSON, process environment, command line, log, stage,
core, CSV, ledger, receipt, final manifest, or filesystem. The one-way
`token_sha256` and keyed completion HMAC are the only token-derived serialized
values. In addition to canonical path arguments and the defined
`--parent-auth-json`, a worker receives only its non-secret integer
read-descriptor number through `--worker-capability-fd` and the bound parent
PID through `--expected-parent-pid`. It also receives its own and the other
reserved canonical stage strings through `--output-dir` and
`--other-stage-directory` solely so isolation guards can be installed before
metadata authentication; the later sentinel check must match both strings
exactly. `subprocess.Popen` must use `close_fds=True` and
`pass_fds=(that_read_fd,)`, so the other slot's read descriptor is not
inherited. These are the only capability-specific invocation values; the
worker command begins exactly with
`[sys.executable, "-B", <canonical-builder-path>, "--internal-worker"]`.
The parent does not copy the ambient parent environment. It passes exactly
this environment mapping and no other name:

```text
BLIS_NUM_THREADS = "1"
CUDA_VISIBLE_DEVICES = ""
LANG = "C.UTF-8"
LC_ALL = "C.UTF-8"
MKL_NUM_THREADS = "1"
NUMEXPR_NUM_THREADS = "1"
OMP_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
PYTHONHASHSEED = "0"
PYTHONIOENCODING = "utf-8"
PYTHONNOUSERSITE = "1"
PYTHONDONTWRITEBYTECODE = "1"
PYTHONPATH = <canonical-repository-root>
PYTHONPYCACHEPREFIX = <canonical-repository-root>/results/.g9cb-bytecode-cache-disabled
PYTHONUNBUFFERED = "1"
PYTHONUTF8 = "1"
TZ = "UTC"
VECLIB_MAXIMUM_THREADS = "1"
```

The mapping's key order above is canonical for provenance, although process
environment lookup is unordered. The legacy worker token variable,
`PYTHONSTARTUP`, `G9CB_SYNTHETIC_TEST_ROOT`, dynamic-loader injection
variables, and every unlisted ambient name are therefore absent. The worker
requires exact `os.environ` equality only after the following parent
precondition: before sentinel publication the fixed `PYTHONPYCACHEPREFIX` path
is absent and no `*.pyc` file or `__pycache__` directory exists anywhere under
the repository root. It also requires `sys.dont_write_bytecode is True`,
`sys.pycache_prefix` equal to the fixed absent path, and the already frozen
interpreter/environment fingerprint before isolated import.

The guard rejects every repository-local or fixed-prefix `.pyc` file open.
The import recorder wraps both `SourceFileLoader` and
`SourcelessFileLoader`; a repository-local `SourcelessFileLoader` execution is
always terminal, while a repository-local source is path/SHA-authenticated
before execution and counted only after successful execution. Tests must
use a noncanonical synthetic root to place a malicious valid `.pyc` in an
ordinary repository `__pycache__`, prove the fixed absent prefix forces
authenticated source execution, separately prove official preflight rejects
that cache, require no
`.pyc`/`__pycache__` creation attempt, and require any injected bytecode-cache
read or write to terminate through the guard.

The preregistration's `bindings.environment` object adds the exact
`worker_process_environment` mapping above. Parent authentication carries that
complete environment object, and each per-pass core emits the same mapping at
`provenance.worker_process_environment`; byte inequality at any location is
terminal.

Except for the one authorized `Popen` handoff per slot, neither parent nor
worker may call `dup`, `dup2`, `fork`, `posix_spawn`, multiprocessing,
or subprocess/exec creation while a capability descriptor is live. Threads
do not create a new rebuild process and receive neither a duplicated
capability descriptor nor independent output authority; all inherited frozen
thread-count/model rules remain in force. Immediately after the parent-death
race check and before opening claim, preregistration, sentinel,
parent-authentication inputs, or other canonical metadata, the worker installs
a non-removable Python audit hook plus exact guards for `dup`, `dup2`, `fcntl`
descriptor duplication, `fork`, `forkpty`, `exec`, `posix_spawn`,
multiprocessing, and subprocess creation. Any guarded attempt is terminal.
The same hook rejects and records every `open`, directory-listing, link,
rename, create, remove, or stat-like access to the other slot's reserved stage
path. From protocol constants available before metadata authentication, it
also rejects every observation or mutation of the other slot's exact canonical
consumption-ledger path and records such an attempt in
`other_slot_ledger_access_events`. It also rejects every filesystem mutation
outside the worker's own three exact stage files and its exact
ledger-publication staging/canonical names,
plus every socket, network, named-pipe, shared-memory, or other IPC creation.
At installation the mutation allowlist is empty. Only after the sentinel,
slot, stage strings, and slot-to-ledger map authenticate may the guard add the
exact own ledger staging/canonical names. After one rebuild invocation has
returned and the complete CSV/core bytes exist only in memory, it may add the
exact CSV and core stage names; after those files reauthenticate and the
complete receipt bytes exist only in memory, it may add the exact receipt
stage name. No other mutation target is ever added.
Immediately after installation, the guard itself performs exactly one
no-follow `os.lstat(other_stage_directory)` absence check. Only this
guard-internal call, identified by a private non-reentrant state flag, is
exempt from the violation counter; it must raise `FileNotFoundError` with
`errno.ENOENT`, increments `other_stage_absence_checks` from `0` to `1`, and
cannot be invoked again. Any existing object, different error, reentry, or
external stat/list/open attempt is terminal and increments the violation
counter before failure.
Every access or path resolution under `/dev/fd` or any Linux procfs descriptor
namespace is forbidden. The procfs rule matches whole components for:

```text
/proc/self/fd or fdinfo
/proc/self/task/<tid>/fd or fdinfo
/proc/thread-self/fd or fdinfo
/proc/<pid>/fd or fdinfo
/proc/<pid>/task/<tid>/fd or fdinfo
```

where `<pid>` and `<tid>` are any decimal components, including the worker,
parent, and unrelated processes. The rule includes lexical aliases and
procfs/devfs magic-link targets. Any path-based open whose
no-follow type is FIFO or whose resolved descriptor identity is a FIFO is
forbidden. The worker may use only the already inherited bound capability FD
and may not reopen either slot's carrier by path.
Tests must inject every guarded process/descriptor family, a cross-stage
read/write/list/stat attempt, an unauthorized temporary-file write, and each
IPC family. They must also attempt `/proc/self/fd/<capability_fd>`,
`/proc/<expected_parent_pid>/fd/<other_slot_fd>`,
`/dev/fd/<capability_fd>`, a lexical alias of each, and a named FIFO both
immediately before metadata authentication and immediately before capability
reading, requiring terminal rejection without consuming either carrier.
Tests also cover `/proc/<pid>/task/<tid>/fd/<fd>` and both `fdinfo` forms.
Slot 1 tests an absent other-ledger stat and slot 2 tests read/stat/list/write
against the existing pass-1 ledger; every external attempt is terminal and
increments `other_slot_ledger_access_events`.

The guard inventory is exhaustive. “Descriptor/process family” means exactly:

```text
os.dup
os.dup2
fcntl.fcntl with F_DUPFD or F_DUPFD_CLOEXEC
os.chdir
os.fchdir
os.chroot
os.fork
os.forkpty
os.posix_spawn
os.posix_spawnp
os.execl
os.execle
os.execlp
os.execlpe
os.execv
os.execve
os.execvp
os.execvpe
os.spawnl
os.spawnle
os.spawnlp
os.spawnlpe
os.spawnv
os.spawnve
os.spawnvp
os.spawnvpe
os.system
os.popen
subprocess.Popen
subprocess.run
subprocess.call
subprocess.check_call
subprocess.check_output
multiprocessing.process.BaseProcess.start
concurrent.futures.ProcessPoolExecutor
audit events: os.fork, os.forkpty, os.posix_spawn, os.exec, subprocess.Popen
```

“Path observation family” means exactly `builtins.open`, `io.open`, `os.open`,
`os.stat`, `os.lstat`, `os.access`, `os.readlink`, `os.listdir`, `os.scandir`,
`os.walk`, `os.path.exists`, `os.path.lexists`, `os.path.isfile`,
`os.path.isdir`, `os.path.islink`, `os.path.getsize`, `os.path.realpath`, and
these `pathlib.Path` methods: `open`, `read_text`, `read_bytes`, `write_text`,
`write_bytes`, `stat`, `lstat`, `exists`, `is_file`, `is_dir`, `is_symlink`,
`iterdir`, `glob`, `rglob`, `mkdir`, `touch`, `chmod`, `unlink`, `rename`,
`replace`, `symlink_to`, and `hardlink_to`.
“Filesystem mutation family” means exactly:

```text
builtins/io open modes containing w, a, x, or +
os.open flags containing O_WRONLY, O_RDWR, O_CREAT, O_EXCL, O_TRUNC,
  O_APPEND, or O_TMPFILE
os.write, os.pwrite, os.writev, os.pwritev, os.copy_file_range, os.sendfile,
  os.splice, os.ftruncate, os.posix_fallocate, os.fchmod, os.fchown,
  os.fsync, os.fdatasync, os.setxattr, os.removexattr
os.mkdir, os.makedirs, os.mkfifo, os.mknod, os.link, os.symlink
os.rename, os.replace, os.remove, os.unlink, os.rmdir, os.removedirs
os.truncate, os.chmod, os.chown, os.utime
tempfile.TemporaryFile, NamedTemporaryFile, SpooledTemporaryFile,
  mkstemp, and mkdtemp
```

The guard records the canonical path and access mode for every descriptor it
opens. Descriptor mutations above are permitted only for a descriptor mapped
to the current exact allowlist target; writes to inherited stdout/stderr file
descriptors `1` and `2` are permitted only for token-free diagnostics.
“IPC family” means exactly:

```text
os.pipe, os.pipe2, os.openpty, os.mkfifo, os.mknod, os.memfd_create,
  os.eventfd, os.pidfd_open, pty.openpty
socket.socket, socket.socketpair, socket.fromfd
multiprocessing.Pipe, Queue, SimpleQueue, JoinableQueue, Manager, and Listener
multiprocessing.shared_memory.SharedMemory
mmap.mmap with a file descriptor and writable MAP_SHARED semantics
audit events: socket.__new__, socket.bind, socket.connect, socket.listen
```

After the one pre-guard `prctl` call, loading or invoking `ctypes`, `cffi`, a
native syscall bridge, or an unlisted process/path/mutation/IPC surface from
repository code is terminal. The authenticated isolated closure must contain
no such call site.

For every path-taking surface, normalization first applies `os.fspath`, decodes
bytes with the filesystem encoding and `surrogateescape`, rejects NUL, anchors
a relative path to the fixed canonical worker cwd, applies lexical
`normpath`, and compares whole path components against the forbidden prefixes
before any symlink following. Existing components are then walked with
no-follow `lstat`; each symlink target is re-anchored and subjected to the same
checks, with a loop or resolution error terminal. These exact normalized and
resolved forms, not an implementation-selected subset, define “lexical
alias.” The tests invoke every named callable/event at least once; an absent
platform callable is asserted absent and recorded by its exact name rather
than silently skipped.

Every guarded call rejects any non-`None` `dir_fd`, `src_dir_fd`, or
`dst_dir_fd` argument before filesystem access; integer `AT_FDCWD` is also
rejected rather than treated as default. Tests pass an unauthorized directory
descriptor to every listed API that supports one and prove that an otherwise
allowlisted relative path cannot be redirected. A path argument supplied as
an integer file descriptor is rejected unless that exact descriptor is
already present in the guard's operation-specific descriptor table.

The descriptor table has two directory-only exceptions. After metadata
authentication, the worker may open the canonical results directory and its
own stage directory with no-follow `O_RDONLY|O_DIRECTORY|O_CLOEXEC`. Those
descriptors authorize only `os.fsync`/`os.fdatasync` on the same captured
device/inode; they do not authorize `dir_fd` resolution, reads, writes,
renames, mode changes, or another path. This exact exception supplies the
required results-directory and stage-directory durability operations without
widening the file-target allowlist.

After the sentinel is atomically published and reauthenticated, the parent
launches slot `1` exactly once. The worker's first application operation,
before opening canonical metadata or reading the capability, must use Linux
`prctl(PR_SET_PDEATHSIG, SIGKILL)` and perform the standard race check:
`os.getppid()` immediately before and immediately after `prctl` must both equal
`--expected-parent-pid`, and that PID must equal the sentinel row. A mismatch
causes immediate self-termination before capability reading. If the parent
dies after `prctl`, the kernel kills the worker; if it died before setup, the
before/after parent-PID check fails.

Immediately after `Popen` returns successfully, and before `wait`, polling,
output inspection, or any other operation, the parent closes its copy of slot
1's read descriptor. The child is then the sole holder. A `Popen` error,
parent-close error, worker crash, signal, timeout, nonzero exit, or validation
failure is terminal; no launch retry is allowed.

Slot `2` is launched exactly once only after slot `1` exits successfully and
its ledger, three staged outputs, receipt HMAC, and observed `Popen.pid` have
all authenticated. The parent then retains the authenticated slot-1 CSV,
core, and receipt bytes only in private memory, unlinks the three slot-1 stage
files, `fsync`s and removes the slot-1 directory, requires the reserved slot-2
path is still absent, exclusively creates that directory with mode `0700`,
`fsync`s the results directory, and requires it empty. Only then may it launch
slot `2` with the same parent-death and immediate-parent-close rules. Slot 1
therefore cannot write to a pre-existing slot-2 directory, and slot 2 cannot
read a surviving slot-1 output. Neither worker receives the parent's retained
bytes.

A slot-1 failure requires closing slot 2's still-private read descriptor,
zeroing both retained token buffers on the controlled cleanup path, and
terminating without creating or launching slot 2. On every process-controlled
post-sentinel exception, the parent must terminate and wait for any live
authorized child, close every still-open capability descriptor, and zero every
application-owned mutable token buffer before raising the terminal action. On
`SIGKILL`, power loss, or other uncatchable termination, zeroization is not
claimed; kernel process teardown closes the parent's descriptors, and the
worker parent-death contract closes any transferred descriptor by killing the
worker. Tests must cover parent death before handoff, after handoff but before
capability read, during a forced partial read, and after complete consumption;
every case is terminal and leaves no usable carrier.

Before generic import or value access, a worker must:

1. establish `PR_SET_PDEATHSIG` and pass the two-parent-PID race check;
2. install all process, descriptor, path, write, and IPC guards using the
   invocation stage strings, with every evidence counter active from this
   point;
3. authenticate the canonical claim, preregistration, sentinel, ordered dual
   amendment bindings, exact `parent_authentication` bytes/hash, exact slot,
   both invocation stage strings, and exact slot-to-ledger map using
   stdlib-only metadata paths;
4. require an empty non-symlink own stage directory, exact
   `other_stage_absence_checks=1`, exact `other_stage_access_events=0`, and an
   exact `other_slot_ledger_access_events=0`, plus an absent own canonical
   ledger path without performing a second other-stage or any other-ledger
   operation;
5. `fstat` the received descriptor, require FIFO mode, and require exact
   `carrier_device` and `carrier_inode` equality with its sentinel row;
6. drain the carrier with raw `os.readv` calls into unwritten slices of one
   preallocated mutable 32-byte buffer; positive intermediate short chunks are
   accumulated, EOF before 32 total bytes is terminal, and after exactly 32
   bytes one additional `os.read(fd, 1)` must return EOF; then close the
   descriptor, treating an extra byte, read error, or close error as terminal;
7. require the lowercase SHA-256 of those 32 bytes to equal only its bound
   `token_sha256`; and
8. atomically publish its immutable consumption ledger before any repository
   import or value-row open.

The first successful pipe read that removes one or more token bytes is the
consumption linearization point. A complete read irreversibly drains the sole
carrier; a partial read followed by a crash also destroys replay authority
because the parent has already closed its copy and process death closes the
child's sole descriptor. If a worker dies before its first read, process death
still destroys the sole carrier and its unread bytes. Recreating an integer FD,
a pipe with a later reused inode, a token hash, a stage directory, or a deleted
ledger cannot recreate the simultaneously bound carrier identity and secret.
Thus deletion of a ledger cannot enable replay. Ledger durability is evidence
of successful consumption, not the mechanism that enforces one use.

The canonical consumption ledger is compact canonical JSON plus one trailing
LF, mode `0444`. The exact sequence is: create an exclusive no-follow regular
staging inode in the results directory; write all bytes; `fchmod(0444)`; file
`fsync`; hard-link with `follow_symlinks=False` to the absent exact canonical
path; results-directory `fsync`; unlink the staging name; results-directory
`fsync`; then reopen and reauthenticate the canonical path without following
symlinks. Its object has exactly:

```text
identity = "G9CB-1"
protocol_version = "gross9_structural_clock_bundle_v1"
slot
parent_pid
stage_directory
carrier_kind = "anonymous_pipe_v1"
carrier_device
carrier_inode
token_sha256
claim_hash
preregistration_manifest_hash
sentinel_manifest_hash
authority_amendments
status = "consumed_before_runtime_or_value_access"
```

The amendment list is the exact ordered list authenticated by the sentinel.
The worker reopens the ledger without following symlinks and verifies exact
bytes, mode, path type, SHA-256, canonical schema, and every field before the
first isolated import. The builder never unlinks, truncates, replaces, chmods,
or cleans a canonical ledger. A crash after carrier consumption but before
ledger publication leaves no ledger but remains terminal and non-replayable.

After ledger authentication, the worker imports its isolated closure and
rebuilds from the beginning. Immediately before output writing it requires
that the stage is still empty and the ledger is still the exact immutable
inode it published. It then writes exactly these three stage-local files,
through exclusive no-follow regular inodes, `fchmod(0400)`, file `fsync`, and
stage-directory `fsync` after each completed file, in this order, with the
receipt last:

```text
gross9_structural_clock_bundle.csv.gz
gross9_structural_clock_bundle_core.json
gross9_structural_clock_bundle_pass_receipt.json
```

No stage-local capability or consumption marker exists. The CSV and core are
the independently rebuilt products. The receipt is compact canonical JSON plus
one trailing LF and has exactly:

```text
identity = "G9CB-1"
protocol_version = "gross9_structural_clock_bundle_v1"
slot
parent_pid
worker_pid
stage_directory
consumed_ledger_path
consumed_ledger_sha256
rebuild_invocations_started = 1
rebuild_invocations_completed = 1
child_process_creation_events = 0
other_stage_access_events = 0
other_stage_absence_checks = 1
other_slot_ledger_access_events = 0
unauthorized_write_or_ipc_events = 0
csv_gzip_sha256
per_pass_core_sha256
completion_hmac_sha256
receipt_hash
```

`parent_pid` equals the sentinel and `--expected-parent-pid`. `worker_pid` is
`os.getpid()` and must equal the parent's observed `Popen.pid`; it must differ
from the parent PID and from the other successful worker PID. The rebuild
start counter increments at the single call boundary before opening the first
value source; the completion counter increments only after the complete CSV
and core bytes exist in memory, before the output-name allowlist expands. The
four zero event counters and one exact internal absence-check counter are
maintained by the non-removable process/cross-stage guards. Any value other
than exactly `1, 1, 0, 0, 1, 0, 0` is terminal before receipt creation.

The completion payload contains exactly the first seventeen receipt fields
through `per_pass_core_sha256`, serialized as compact canonical JSON with no
trailing LF. `completion_hmac_sha256` is lowercase
`HMAC-SHA256(raw_32_byte_token, completion_payload)`. `receipt_hash` is the
lowercase SHA-256 of compact canonical JSON containing the preceding eighteen
fields through `completion_hmac_sha256`, again with no trailing LF. The
receipt file then serializes all nineteen fields plus one LF.

The worker computes the CSV and core hashes by reopening the two completed
stage files, creates the receipt last, then overwrites its mutable token buffer
with zeros before exit. The parent independently reopens and hashes the exact
ledger, CSV, core, and receipt; verifies the observed child PID, receipt hash,
and HMAC with its retained slot token; then overwrites that mutable token
buffer with zeros. A copied pass-1 receipt cannot authenticate as slot 2
because slot, PID, stage, ledger hash, pipe token, and HMAC are all bound.

After both successful workers, the parent requires the two CSV byte streams to
be identical and the two core byte streams to be identical. The final manifest
contains both exact ordered evidence lists:

```text
worker_capability_consumption[0].slot = 1
worker_capability_consumption[1].slot = 2
rebuild_receipts[0].slot = 1
rebuild_receipts[1].slot = 2
```

Each `worker_capability_consumption` row has exactly `slot`, `parent_pid`,
`path`, `sha256`, `carrier_kind`, `carrier_device`, `carrier_inode`, and
`token_sha256`, all matching the sentinel and ledger. Each
`rebuild_receipts` row is the complete receipt object plus exactly
`pass_receipt_sha256`, the hash of the staged receipt file including its
trailing LF. Receipt rows intentionally differ; per-pass CSV and core bytes
must not. Slot-1 stage cleanup occurs at the exact pre-slot-2 isolation point
above after the parent retains authenticated bytes.
Slot-2 stage cleanup is allowed only after both ledgers authenticate and the
final manifest is durably published. Each cleanup may remove only its three
exact staged files and stage directory.

This explicitly supersedes the original phrase “deterministic rebuild
receipts.” The rebuilt CSV and per-pass core remain deterministic and
byte-identical. Slot, process, pipe identity, token digest, ledger digest, and
HMAC are intentionally pass-specific authentication evidence; their canonical
serialization and validation are deterministic, but their values are not
required to equal across passes.

The exact parent-death setup/race check, canonical metadata authentication,
process/cross-stage guard installation, carrier consumption, and ledger
publication are the only worker operations authorized between canonical
sentinel verification and the first isolated import. The parent's next
operation after sentinel verification is the one slot-1 `Popen`. Any unused
descriptor is closed and every retained mutable token buffer is zeroed on
every process-controlled exit path; no zeroization claim is made for abrupt
kernel or power termination.

## Physical event counters

Only the refinements enumerated in this section supersede prior counter
wording. Every unmentioned counter name, schema, per-sleeve/per-feature
increment rule, double-count distinction, zero assertion, and failure
contribution rule in `G9CB-1` and `G9CB-1A` remains operative.

Counters must be driven by the physical operation. Authentication hashing
performed before sentinel publication or repeated by a worker is metadata
evidence and is not included in post-capability decoded-source or loaded-model
access counters. Its exact path/SHA/size results are already represented by
the bound `parent_authentication.hashed_inputs`; this amendment does not claim
or emit a separate authentication-read byte counter.

- `bytes_read_by_logical_source` counts bytes returned by each post-capability
  authenticated binary source descriptor to its decoder. For `.gz` inputs it
  counts compressed file bytes before decompression; for plain CSV/JSONL it
  counts raw file bytes. Decoder seeks and rereads count again. Filesystem size,
  decompressed byte count, Unicode characters, and authentication-hash reads
  do not increment it. Its object has exactly the same ordered nine logical
  source keys listed below for `rows_decoded`, initialized to integer zero.
- `source_files_opened` increments immediately after a post-capability decoder
  successfully opens an authenticated market, funding, premium, open-interest,
  REX, or hourly-history descriptor. Authentication-only hash descriptors do
  not increment it.
- `rows_decoded.<source>` increments by the exact rows returned by a completed,
  non-streaming parser call. A header is not a row. Chunk inference or a
  post-hoc manifest row count is forbidden.
- `rows_used.causal_feature_rows_by_source` is an object with exactly the same
  nine ordered keys as `rows_decoded`:

  ```text
  market_5m
  funding
  premium
  open_interest
  rex_taker_train
  rex_taker_test
  rex_taker_eval
  rex_veto_source
  rank7_hourly_history
  ```

  Each parsed CSV row receives its parser-return ordinal; each REX object
  receives its non-empty physical-line ordinal. A per-source bitset starts
  empty. Immediately before a pure causal feature, attachment, state, gate, or
  warm-start primitive receives a frame/list, the adapter marks the exact
  source ordinals present in that argument and increments this counter only by
  newly marked ordinals. Passing the same row to another causal primitive does
  not increment it again. Rows decoded but never handed to a causal primitive
  remain unmarked and contribute zero. Market normalization, each auxiliary
  as-of attachment, each REX gate input, and Rank7 hourly warm start are
  separate instrumented handoff sites. The bitsets and increments occur at
  those handoffs, not by assigning `rows_decoded`, parser length, first-row
  counts, or output interval counts after reconstruction. A missing/additional
  key, delayed assignment, or value unequal to the bitset cardinality is
  terminal.
- `per_sleeve` has exactly the five preregistered sleeve keys and, for each,
  exactly `signal_rows_evaluated`, `intervals_emitted`, `long_intervals`,
  `short_intervals`, `fixed_horizon_exits`, `take_exits`, `stop_exits`, and
  `outcome_dependent_ohlc_rows_examined`. A signal counter increments at the
  actual timestamp/anchor evaluation call even when the gate rejects. Interval
  and side counters increment atomically when one validated structural interval
  is appended. Exactly one exit-kind counter increments with that append.
  Each stop/take replay increments both its sleeve OHLC counter and
  `rows_used.outcome_dependent_ohlc_rows_examined` on each actual OHLC row
  retrieval before barrier comparison; fixed-horizon-only schedules contribute
  zero OHLC examinations. The aggregate and sleeve increments occur together,
  not by summing or assigning after reconstruction. The additional Rank7
  replay/parity counters retain the exact `G9CB-1A` events.
- Each REX JSONL source is opened once as a binary descriptor and read to EOF
  from the beginning. Its byte counter increments by each raw read result.
  `rows_decoded` increments once for each non-empty physical line that
  successfully decodes as one duplicate-key-free JSON object; blank lines are
  not rows, and concatenated or multi-line JSON is forbidden.
- The completed-hourly-history gzip uses the same counted binary CSV path:
  compressed descriptor bytes increment
  `bytes_read_by_logical_source.rank7_hourly_history`, its successful decoder
  open increments `source_files_opened`, and its parsed data rows increment
  `rows_decoded.rank7_hourly_history`.
- `model_files_opened` increments immediately after `numpy.load` successfully
  opens one allowed portable model container. Bundle checksum/authentication
  reads do not increment it. Each of the five declared paths must generate
  exactly one event; duplicate, missing, reordered, or extra `numpy.load`
  operations are terminal.
- `prediction_rows_scored` increments only after an actual portable-model or
  annual-forest prediction call succeeds, by one for a one-dimensional row or
  by the first matrix dimension for a two-dimensional matrix. A failed call
  contributes zero.
- Before the two isolated root imports, the worker wraps
  `importlib.machinery.SourceFileLoader.exec_module` and
  `SourcelessFileLoader.exec_module`, then takes a `sys.modules` snapshot. A
  repository-local sourceless execution is terminal. Before a repository-local
  source execution, the wrapper resolves `module.__file__` without symlinks
  and authenticates its path and source SHA against the bound closure; after
  successful execution it appends the path to an execution recorder. A failed
  execution contributes zero. A duplicate
  path execution, an isolated root already present before the snapshot, a
  recorded module removed from `sys.modules`, or a newly present repository
  module absent from the execution recorder is terminal. A pre-snapshot
  repository path is allowed only when it is either an exact protocol file in
  `preregistration.bindings.protocol` or a package initializer loaded solely
  by those protocol files; the sorted exact allowed set is emitted at
  `provenance.preloaded_repository_paths`, and neither isolated root may occur
  there.
  The wrapper and reconciliation recorder remain active through both root
  imports, every function-local or lazy import during reconstruction, CSV
  serialization, and final counter capture. Immediately before core
  construction, the worker performs one final exact recorder/`sys.modules`
  reconciliation, freezes the sorted unique successful path set, and changes
  the wrapper into reject mode: any later repository-local module execution is
  terminal. It is never restored to permissive behavior in that worker.
  The per-pass core stores the frozen set at
  `provenance.runtime_import_paths`; both isolated root paths must occur
  exactly once. Despite its inherited name,
  `access_counters.file_access.runtime_modules_imported` is the integer count
  of successful unique repository-local module executions and equals
  `len(provenance.runtime_import_paths)`. It is not an import-statement count
  or the declared closure length.
- Rank7 label, activation, parity, and OHLC counters increment at the exact
  replay, computation, score, comparison, or examination event.

No counter may be estimated from `stat().st_size`, manifest row count, declared
model count, output interval count, or a post hoc multiplication.

## Amendment and prohibited-output placement

Each claim, sentinel, per-pass core, and final manifest must contain the
top-level canonical `authority_amendments` list defined in the Status section.

Within each per-pass core and final manifest, the permanent prohibited keys may
appear only at:

```text
evidence_boundary.prohibited_output_counters
```

The exact permanent key set is:

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

That object must contain exactly this set and every value must be
type `int` and equal zero; `false`, `0.0`, `"0"`, and missing or additional
keys are invalid. A prohibited key anywhere else inside a per-pass core or
final manifest—including provenance, parent authentication, embedded
claim/sentinel objects, Rank7 metadata, receipts, or any nested extension—is
terminal even when its value is zero.

This placement rule does not remove the separately required zero-access fields
from the preregistration, claim, or sentinel canonical schemas. Those artifacts
retain their inherited exact zero-access objects. The per-pass core and final
manifest bind claim and sentinel by compact path/SHA/hash records; they must
not embed the full claim or sentinel object.

The canonical CSV remains limited to the eight authorized structural columns.

## Publication allowlist and completion seal

This section explicitly supersedes the three-path publication allowlist in
`G9CB-1` “Commit, push, and future consumption” and the corresponding
three-artifact wording in `G9CB-1` “Completion condition.” It does not widen
any other commit.

The successful publication commit may add exactly these five previously absent
canonical files and may modify no tracked file:

```text
results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_manifest_2026-07-31.json
```

Let `D` be that publication commit. It must satisfy exactly:

```text
D has exactly one parent
first_parent(D) == C
diff(C, D) ==
  A results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_worker_capability_consumed_pass1_2026-07-31.json
  A results/gross9_structural_clock_bundle_worker_capability_consumed_pass2_2026-07-31.json
  A results/gross9_structural_clock_bundle_2026-07-31.csv.gz
  A results/gross9_structural_clock_bundle_manifest_2026-07-31.json
HEAD == D
HEAD == @{upstream}
the worktree and index are clean
```

The diff rows above are compared as a sorted exact set; no intervening commit,
merge parent, rename, deletion, or modification is permitted between `C` and
`D`.

The access claim remains in its earlier claim-only direct-child commit. The
sentinel remains the first canonical post-claim artifact. Each ledger is
published by its bound worker immediately after carrier consumption. Only
after both workers, both ledgers, both receipts, and byte identity authenticate
may the parent hard-link the canonical CSV. The final manifest is hard-linked
last and remains the publication commit point. A sentinel or ledger without
the final manifest is terminal-attempt evidence, not clock authority. A CSV
without the final manifest is an orphan. A final manifest without exact
sentinel, ledger, receipt, CSV, claim, preregistration, and dual-amendment
bindings is invalid.

No transient pass receipt or stage directory is committed. After successful
manifest-last publication and stage cleanup, the exact five paths above must
be the only untracked publication artifacts before the committed artifact
tests run.

`G9CB-1` is complete only when:

1. the original decision, `G9CB-1A`, and this `G9CB-1B` authority are each
   committed and pushed in their required standalone metadata-only commits;
2. the preregistration producer and tests are committed and pushed;
3. the canonical preregistration artifact and artifact test are committed and
   pushed;
4. the builder, isolated facade, isolated primitives, and all protocol tests
   are committed and pushed;
5. the claim-only direct-child commit is committed and pushed;
6. the sentinel with the exact ordered `worker_capabilities` and
   `authority_amendments` lists is atomically published before all runtime or
   value access;
7. exactly two fresh `Popen` workers consume their distinct anonymous-pipe
   carriers and publish their exact immutable ledgers without retry;
8. both pass-specific HMAC receipts authenticate distinct worker PIDs, slots,
   stages, ledgers, pipe-token hashes, exact `1,1,0,0,1,0,0`
   rebuild/isolation
   event counts, CSV hashes, and core hashes, with slot 1 removed before slot
   2 is created and no cross-pass file or IPC channel;
9. the two independently rebuilt CSV byte streams and the two independently
   rebuilt per-pass core byte streams are respectively byte-identical;
10. the two ledgers, canonical CSV, and final manifest are authenticated and
    the final manifest is published last;
11. all committed artifact validation passes without any protocol change; and
12. `D` is the exact direct child of `C`, adds exactly the five publication
    files above, is pushed, all transient stages are absent, the worktree and
    index are clean, and `HEAD == D == @{upstream}`.

Until all twelve conditions hold, no candidate may cite `G9CB-1` as a durable
Gross9 structural clock authority.

## Failure rule

Before the sentinel, an inability to authenticate either amendment, the pure
closure, capabilities, environment, sources, bundle metadata, or protocol
files stops without value access.

After the sentinel, every capability, import, source, model, prediction,
label, parity, schedule, counter, serialization, byte-identity, or publication
failure is terminal:

```text
TERMINAL_G9CB1_ATTEMPT_CONSUMED_NO_RETRY
```

## Decision

`G9CB-1B` repairs the executable import boundary, makes the two-pass authority
non-replayable, and requires honest physical counters without weakening any
economic or candidate-independence restriction. It authorizes no new alpha,
threshold, source, policy, side, hold, barrier, or weight.
