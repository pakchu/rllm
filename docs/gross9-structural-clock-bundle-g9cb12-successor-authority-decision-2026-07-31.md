# Gross9 structural clock bundle G9CB-12 successor authority decision — 2026-07-31

## Authority, separation, and decision

This document is the sole A12 authority for two mechanically separate acts:

1. T11 may seal the already-consumed, terminal G9CB-11 source-support attempt
   as historical evidence only; and
2. after successful T11, S12 may implement and invoke one fresh source-support
   identity, `G9CB-12-SOURCE-SUPPORT`, under the prospective rules below.

Historical G9CB-11 facts grant no source, retry, repair, resume, reuse, output,
or execution authority. Prospective G9CB-12 authority uses a distinct identity,
implementation, attempt hash, output inventory, replay guard, and one-shot
boundary. Treating any G9CB-11 fact as permission to rerun or consume G9CB-11
invalidates A12.

The operative G9CB-12 decision is **direct exact required-date membership**.
S12 selects Rank7 rows against the exact ordered required set
`market.date.tail(min(3000, len(market)))`. It does not construct a full-market
Rank7 relation and does not compare, normalize, align, round, snap, repair,
interpolate, resample, fill, or map timestamps by as-of or tolerance.

G9CB-12 through H12 is candidate-independent Gross9 clock infrastructure. A12,
T11, S12, M12, Q12, P12, C12, D12, V12, and H12 open or evaluate no candidate,
comparator, feature, schedule, signal, return, PnL, CAGR, MDD, drawdown, or other
economic value. The active alpha goal remains `incomplete`. Candidate-specific
work begins only after verified H12 and a fresh candidate-specific Ralplan.

## Exact A12 boundary

A12 has exact parent S11:

```text
S11 = 646fccbf6568bcf39fab12a47873f72da880ca01

first_parent(A12) == S11
diff(S11, A12) ==
  A 100644 docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md
```

No other tracked or untracked change belongs to A12. The document has worktree
mode `0644`. A12 is independently reviewed, committed alone, ordinarily
pushed, and must end with a clean worktree/index, zero repository bytecode, and
`HEAD == @{upstream}` before T11 starts.

This document does not predict its own future A12 commit, Git blob, SHA-256,
size, or any future T11, S12, M12, Q12, P12, C12, D12, V12, or H12 commit,
blob, artifact hash, self-hash, frame hash, size, or attempt hash. A12 opens no
raw source and infers no missing Rank7 count, location, timestamp, or value.

## Historical G9CB-11 facts only

Everything in this section is terminal history. Nothing in it authorizes a
G9CB-11 action.

### S11 Git and implementation binding

```text
S11               = 646fccbf6568bcf39fab12a47873f72da880ca01
first_parent(S11) = 7f5866be73e01e9531e585c7a13b19661906b05c  # T10
A11               = 189b5403c66ea0283e67b42b9fbc6ba909280a57
```

The exact S11 implementation pair is:

| Path | Size bytes | SHA-256 | Git blob | Git mode | Worktree mode |
|---|---:|---|---|---|---|
| `training/materialize_gross9_structural_clock_g9cb11_sources.py` | 98456 | `4e34ecbc9c812e4fe7d633110f2e06536ff491c553f4449d36fd8dca58bfb828` | `ab4459a9e6ae48047840787644a0839f474adc9c` | `100644` | `0644` |
| `tests/test_materialize_gross9_structural_clock_g9cb11_sources.py` | 147428 | `bf4a3716cd2dd612e7ac884df55298a4001b174d9dbdea39ab6d29907083d7de` | `4415d503f94ddb5e17bac3c115de550f8842b39b` | `100644` | `0644` |

### Sole official S11 execution

The exact command was invoked once:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb11_sources
```

Frozen result:

- official invocation count: `1`;
- exit status: `1`;
- phase: `transform`;
- exception class: `SourceSupportFailure`;
- exception message: `incomplete Rank7 projection coverage`;
- publication state: `post_sentinel_pre_other_output_failure`;
- failure reason: `rank7_all_history_coverage_mismatch`;
- traceback source-value excerpt emitted: `false`;
- Rank7 gap count disclosed: `false`;
- Rank7 gap location disclosed: `false`;
- Rank7 gap timestamp disclosed: `false`;
- Rank7 gap value disclosed: `false`;
- Rank7 gap detail disclosure count: `0`;
- retry allowed: `false`;
- resume allowed: `false`.

No descendant may report or infer any missing Rank7 count, location, timestamp,
or value, any physical Rank7 row count, or any required-tail count or proof from
the G9CB-11 failure.

### Immutable G9CB-11 attempt sentinel

```text
path       = results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json
type       = regular_file
worktree_mode = 0444
git_mode      = 100644  # when sealed by T11
link_count = 1
device     = 2096
inode      = 934842
size_bytes = 3056
sha256     = 128ad6213785ecfa360114eae6e3587254dda3b18e94108b9dd30a0f34533e31
attempt_hash = 6a6204b5074aee399f6a4e318d24764140cfb07aea9b6ebd01b021f7333038f1
identity   = G9CB-11-SOURCE-SUPPORT
repository_head   = 646fccbf6568bcf39fab12a47873f72da880ca01
repository_parent = 7f5866be73e01e9531e585c7a13b19661906b05c
one_shot = true
retry_allowed  = false
resume_allowed = false
raw_input_count = 7
opaque_bytes_hashed_before_publication = 190272610
```

The sentinel bytes, SHA-256, attempt hash, device/inode, type, mode, link count,
size, and path identity are immutable. T11 may add this existing inode to Git.
It may not rewrite, replace, rename, relink, chmod, truncate, remove, or
recreate it.

### Permanent G9CB-11 absences and prohibitions

The following four leaves were absent after the official failure and remain
permanently absent in this exact order:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz
configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json
```

G9CB-11 may never be rerun, repaired, resumed, cleaned up for reuse,
republished, completed, or consumed. No missing G9CB-11 output may be
synthesized. `M11`, `Q11`, `P11`, `C11`, `D11`, `V11`, and `H11` are
permanently forbidden. T11 is evidence-only and creates no G9CB-11 source
authority.

## Exact T11 two-file terminal seal

T11 starts only after clean pushed A12. It opens no official raw source,
performs no source-value inference, and tracks exactly:

1. the existing immutable G9CB-11 attempt sentinel; and
2. `results/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json`.

Its boundary is:

```text
first_parent(T11) == A12
diff(A12, T11) ==
  A 100644 results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json
  A 100644 results/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json
```

Both files are regular, single-link, worktree mode `0444`, Git mode `100644`,
duplicate-key-free canonical JSON plus exactly one LF. The existing sentinel
must retain its exact bytes, SHA-256, attempt hash, device/inode, type, mode,
link count, and size across add, commit, and push. No data leaf, partial output,
source value, physical Rank7 row count, required-tail count, required-tail
proof, or missing count/location/timestamp/value belongs to T11.

### Normative type and canonicalization rules

- `git_oid`: exactly 40 lowercase hexadecimal characters.
- `sha256`: exactly 64 lowercase hexadecimal characters.
- `repo_path`: a nonempty repository-relative POSIX path; absolute paths are
  invalid.
- `uint`: a JSON integer greater than or equal to zero; a boolean, float,
  string, or null is invalid.
- `string`: a JSON string.
- `boolean`: a JSON `true` or `false` literal.

Object key sets, exact literals, types, and normative array order are fixed by
the template below. Canonical JSON uses UTF-8, sorted object keys, separators
`(",", ":")`, `ensure_ascii=false`, `allow_nan=false`, duplicate-key
rejection, and no whitespace. Self-hash input has no trailing LF and omits only
`terminal_failure_hash`. Persisted JSON includes the resolved self-hash and
adds exactly one LF.

`<A12_COMMIT>` and `<T11_TERMINAL_FAILURE_HASH>` are schema metavariables,
never serialized strings. `A12_COMMIT` is learned only after A12 is committed
and authenticated. The terminal hash is learned only after the complete T11
object exists. There are no other placeholders and no future commit
prediction.

### Fully typed thirteen-key, thirty-four-access-key T11 ledger

The canonical sorted top-level object has exactly thirteen keys in this order:

```text
access, attempt_sentinel, authority, execution, failure, identity,
implementation, ledger_kind, output_state, schema_version, seal_authority,
status, terminal_failure_hash
```

The following sorted template is normative:

```json
{
  "access": {
    "attempt_sentinel_publication_count": 1,
    "cagr_evaluation_count": 0,
    "candidate_value_rows_opened": 0,
    "comparator_value_rows_opened": 0,
    "decode_pass_count": 9,
    "decode_passes": [
      "old_market",
      "replacement_market_date_scan",
      "replacement_market_tail",
      "funding",
      "premium",
      "old_open_interest",
      "binance_metrics_open_interest_date_scan",
      "binance_metrics_open_interest_selected_window",
      "rank7_spot_premium_5m"
    ],
    "drawdown_evaluation_count": 0,
    "economic_evaluation_count": 0,
    "economic_value_rows_opened": 0,
    "feature_value_rows_opened": 0,
    "generated_output_publication_count": 0,
    "generated_output_readback_count": 0,
    "global_metrics_alignment_comparison_count": 0,
    "mdd_evaluation_count": 0,
    "metrics_date_scan_count": 1,
    "metrics_overlap_row_count": 13,
    "metrics_selected_decode_count": 1,
    "metrics_selected_row_count": 120,
    "metrics_tail_row_count": 107,
    "non_selected_metrics_non_date_semantic_evaluation_count": 0,
    "off_grid_detail_disclosure_count": 0,
    "pnl_value_rows_opened": 0,
    "rank7_all_history_coverage_comparison_count": 1,
    "rank7_gap_detail_disclosure_count": 0,
    "rank7_spot_premium_5m_decode_count": 1,
    "rank7_tail_completeness_evaluation_count": 0,
    "raw_file_count": 7,
    "raw_file_open_count": 7,
    "replacement_market_date_scan_count": 1,
    "replacement_market_tail_decode_count": 1,
    "replacement_market_tail_selected_row_count": 107,
    "return_value_rows_opened": 0,
    "schedule_value_rows_opened": 0,
    "signal_value_rows_opened": 0
  },
  "attempt_sentinel": {
    "attempt_hash": "6a6204b5074aee399f6a4e318d24764140cfb07aea9b6ebd01b021f7333038f1",
    "device": 2096,
    "filesystem_type": "regular_file",
    "git_mode": "100644",
    "inode": 934842,
    "link_count": 1,
    "one_shot": true,
    "opaque_bytes_hashed_before_publication": 190272610,
    "path": "results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json",
    "raw_input_count": 7,
    "repository_head": "646fccbf6568bcf39fab12a47873f72da880ca01",
    "repository_parent": "7f5866be73e01e9531e585c7a13b19661906b05c",
    "resume_allowed": false,
    "retry_allowed": false,
    "sha256": "128ad6213785ecfa360114eae6e3587254dda3b18e94108b9dd30a0f34533e31",
    "size_bytes": 3056,
    "worktree_mode": "0444"
  },
  "authority": {
    "commit": "189b5403c66ea0283e67b42b9fbc6ba909280a57",
    "document_path": "docs/gross9-structural-clock-bundle-g9cb11-successor-authority-decision-2026-07-31.md"
  },
  "execution": {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb11_sources",
    "exit_status": 1,
    "invocation_count": 1,
    "one_shot": true,
    "resume_allowed": false,
    "retry_allowed": false
  },
  "failure": {
    "exception_class": "SourceSupportFailure",
    "exception_message": "incomplete Rank7 projection coverage",
    "failure_reason": "rank7_all_history_coverage_mismatch",
    "phase": "transform",
    "publication_state": "post_sentinel_pre_other_output_failure",
    "rank7_gap_count_disclosed": false,
    "rank7_gap_detail_disclosure_count": 0,
    "rank7_gap_location_disclosed": false,
    "rank7_gap_timestamp_disclosed": false,
    "rank7_gap_value_disclosed": false,
    "traceback_source_value_excerpt_emitted": false
  },
  "identity": "G9CB-11-SOURCE-SUPPORT",
  "implementation": {
    "commit": "646fccbf6568bcf39fab12a47873f72da880ca01",
    "files": [
      {
        "git_blob": "ab4459a9e6ae48047840787644a0839f474adc9c",
        "git_mode": "100644",
        "path": "training/materialize_gross9_structural_clock_g9cb11_sources.py",
        "sha256": "4e34ecbc9c812e4fe7d633110f2e06536ff491c553f4449d36fd8dca58bfb828",
        "size_bytes": 98456,
        "worktree_mode": "0644"
      },
      {
        "git_blob": "4415d503f94ddb5e17bac3c115de550f8842b39b",
        "git_mode": "100644",
        "path": "tests/test_materialize_gross9_structural_clock_g9cb11_sources.py",
        "sha256": "bf4a3716cd2dd612e7ac884df55298a4001b174d9dbdea39ab6d29907083d7de",
        "size_bytes": 147428,
        "worktree_mode": "0644"
      }
    ],
    "parent_commit": "7f5866be73e01e9531e585c7a13b19661906b05c"
  },
  "ledger_kind": "gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_v1",
  "output_state": {
    "downstream_consumable": false,
    "forbidden_stages": [
      "M11",
      "Q11",
      "P11",
      "C11",
      "D11",
      "V11",
      "H11"
    ],
    "permanently_absent_output_paths": [
      "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz",
      "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz",
      "configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json",
      "results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json"
    ],
    "source_authoritative": false,
    "terminal_evidence_paths": [
      "results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json"
    ]
  },
  "schema_version": 1,
  "seal_authority": {
    "commit": "<A12_COMMIT>",
    "document_path": "docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md"
  },
  "status": "terminal_rank7_all_history_coverage_failure",
  "terminal_failure_hash": "<T11_TERMINAL_FAILURE_HASH>"
}
```

The exact nested key sets are:

- `authority`: `commit`, `document_path`;
- `seal_authority`: `commit`, `document_path`;
- `implementation`: `commit`, `files`, `parent_commit`;
- each `implementation.files[]`: `git_blob`, `git_mode`, `path`, `sha256`,
  `size_bytes`, `worktree_mode`;
- `attempt_sentinel`: `attempt_hash`, `device`, `filesystem_type`, `git_mode`,
  `inode`, `link_count`, `one_shot`, `opaque_bytes_hashed_before_publication`,
  `path`, `raw_input_count`, `repository_head`, `repository_parent`,
  `resume_allowed`, `retry_allowed`, `sha256`, `size_bytes`, `worktree_mode`;
- `execution`: `command`, `exit_status`, `invocation_count`, `one_shot`,
  `resume_allowed`, `retry_allowed`;
- `failure`: `exception_class`, `exception_message`, `failure_reason`, `phase`,
  `publication_state`, `rank7_gap_count_disclosed`,
  `rank7_gap_detail_disclosure_count`, `rank7_gap_location_disclosed`,
  `rank7_gap_timestamp_disclosed`, `rank7_gap_value_disclosed`,
  `traceback_source_value_excerpt_emitted`;
- `access`: the exact thirty-four keys and values shown in the template;
- `output_state`: `downstream_consumable`, `forbidden_stages`,
  `permanently_absent_output_paths`, `source_authoritative`,
  `terminal_evidence_paths`.

The exact type rules are:

- `schema_version`, every `*_count`, every `*_rows*`, `size_bytes`,
  `link_count`, `device`, `inode`, `raw_input_count`,
  `opaque_bytes_hashed_before_publication`, `invocation_count`, and
  `exit_status` are `uint`;
- `one_shot`, `retry_allowed`, `resume_allowed`, every `*_disclosed`,
  `source_authoritative`, and `downstream_consumable` are `boolean`;
- authority, implementation, file-blob, sentinel-head, and sentinel-parent
  commits are `git_oid`;
- every `sha256`, `attempt_hash`, and resolved `terminal_failure_hash` is
  `sha256`;
- every `path`, `document_path`, and path-array member is `repo_path`;
- `git_mode`, `worktree_mode`, `filesystem_type`, `command`, `phase`,
  `exception_class`, `exception_message`, `publication_state`,
  `failure_reason`, `identity`, `ledger_kind`, and `status` are exact strings
  shown in the template;
- `implementation.files`, `access.decode_passes`,
  `output_state.terminal_evidence_paths`,
  `output_state.permanently_absent_output_paths`, and
  `output_state.forbidden_stages` are ordered arrays with the exact lengths,
  order, member types, and content shown above.

The `access` object proves seven opens and nine ordered logical passes;
replacement `1/1/107`; metrics `1/1/120/13/107`; one Rank7 decode; one invalid
all-history coverage comparison; zero Rank7 tail-completeness evaluation;
sentinel/generated publication/readback `1/0/0`; zero off-grid and Rank7 detail
disclosure; and all twelve existing economics counters at `0`. It records no
physical Rank7 row count, required-tail count/proof, or missing
count/location/timestamp/value.

## Prospective fresh G9CB-12 authority

This section becomes actionable only after successful, committed, pushed T11.
The fresh identity is exactly:

```text
G9CB-12-SOURCE-SUPPORT
```

S12 adds exactly:

```text
first_parent(S12) == T11
diff(T11, S12) ==
  A 100644 training/materialize_gross9_structural_clock_g9cb12_sources.py
  A 100644 tests/test_materialize_gross9_structural_clock_g9cb12_sources.py
```

Both files have worktree mode `0644`. The normative implementation-path order
inside access and replay ledgers is implementation first, tests second.

The exact sole official S12 command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb12_sources
```

This official command is **not a test**. It may run exactly once, only after
all named preflight tests and independent review/verification pass, S12 is
committed and pushed, the worktree/index are clean, `HEAD == @{upstream} ==
S12`, the direct parent and exact two-file diff authenticate, all five fresh
outputs are absent, the G11 terminal inventory is unchanged, and repository
bytecode count is zero. Any nonzero result consumes S12 permanently. Retry,
resume, repair, rerun, overwrite, or cleanup for reuse is forbidden.

### Exact five-output boundary and publication order

The exact ordered output list is:

1. `results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json`;
2. `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb12_complete.csv.gz`;
3. `data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb12_complete.csv.gz`;
4. `configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json`;
5. `results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json`.

All five must be absent before any official source opens. Publication uses a
private same-directory `O_TMPFILE`, complete write, `fchmod(0444)`, file fsync,
create-only `linkat` through `/proc/self/fd` with `AT_SYMLINK_FOLLOW`, directory
fsync, and linked device/inode identity verification. `AT_EMPTY_PATH`, named
temporary files, overwrite, rename, truncate, unlink, replacement, post-link
chmod, recovery, and `EEXIST` repair are forbidden. The immutable publication
order is sentinel, market, OI, manifest, support. Generated market and OI are
each read back exactly once; the process then reauthenticates all retained
inputs and all five outputs, rejects residue, and runs the final bytecode gate.

### Exact required-tail predicate

The required dates are exactly:

```text
market.date.tail(min(3000, len(market)))
```

The following rules are jointly normative:

- preserve the required dates in exact order;
- select by exact date membership only;
- selected dates equal the required dates, and every selected required date
  occurs exactly once;
- `rank7_required_tail_rows = min(3000, len(market))`, exactly `3000` for the
  frozen production market;
- `rank7_tail_exact_matches = min(3000, len(market))`, exactly `3000` for the
  frozen production market;
- `rank7_pre_tail_coverage_comparison_count = 0`;
- `rank7_gap_detail_disclosure_count = 0`;
- non-required rows before or after the required set are allowed;
- non-date values on non-required rows are ignored for admission, and poisoned
  non-date values on those rows must succeed;
- numeric, `latest`, `spot_rows`, and `premium_rows` checks apply only to
  selected rows;
- global invariants remain strict only for schema, date-token validity, UTC
  parse, uniqueness, monotonicity, and five-minute alignment;
- missing or duplicated required dates and any global invariant violation
  fail admission;
- no pre-tail comparison, missing-gap detail, or physical Rank7 row count is
  disclosed;
- full-history relation or comparison, full-market merge or join,
  `merge_asof`, tolerance mapping, fill, interpolation, resample, repair,
  timestamp transformation, and numeric conversion of non-selected rows are
  forbidden.

Direct exact membership is preferred. Exact index-selection/reindex without
timestamp transformation is compliant only when byte-equivalent to the exact
required-date set and when it never materializes a full-market relation.

### S12 replay and access binding

S12 adds exactly these replay/access members:

- `historical_s11`, copying authenticated G11 terminal evidence and all four
  permanent absences without authority transfer;
- `current_s12`, binding A12, T11, S12, the exact two S12 files, the exact
  required-tail proof, and the four Rank7 counters above;
- `replay_guard`, binding the exact stage order, paths, five-output set, and
  permanent G11 absences; and
- `access_ledger_hash`, self-binding the canonical access ledger.

No future commit prediction appears in T11 or S12. No candidate, comparator,
feature, schedule, signal, return, PnL, CAGR, MDD, drawdown, or economics value
is opened or evaluated.

## Orthogonal S12 terminal matrix and quarantine

`publication_state` is determined solely by the actual immutable ordered
output prefix. `failure_reason` identifies the failure category. Reason text
must never determine or override the prefix-derived state.

Exact publication states:

- `pre_sentinel_failure`;
- `post_sentinel_pre_other_output_failure`;
- `partial_publication_failure`.

Exact failure reasons:

- `preflight_or_binding_failure`;
- `rank7_tail_membership_mismatch`;
- `structural_or_schema_violation`;
- `zero_disclosure_breach`;
- `publication_or_readback_failure`;
- `final_reauthentication_failure`.

The following eight pairs are exhaustive:

| `publication_state` | `failure_reason` | Required actual prefix/failpoint class |
|---|---|---|
| `pre_sentinel_failure` | `preflight_or_binding_failure` | failure before any sentinel link |
| `pre_sentinel_failure` | `publication_or_readback_failure` | attempt-publication-before-link failure; no sentinel |
| `post_sentinel_pre_other_output_failure` | `rank7_tail_membership_mismatch` | sentinel only; before market link |
| `post_sentinel_pre_other_output_failure` | `structural_or_schema_violation` | sentinel only; before market link |
| `post_sentinel_pre_other_output_failure` | `zero_disclosure_breach` | sentinel only; before market link |
| `post_sentinel_pre_other_output_failure` | `publication_or_readback_failure` | sentinel only; before first non-sentinel link |
| `partial_publication_failure` | `publication_or_readback_failure` | ordered prefix length `2..5` |
| `partial_publication_failure` | `final_reauthentication_failure` | ordered prefix length `2..5` |

Every other pair is invalid and must be rejected.

Quarantine follows the actual prefix:

- **Pre-sentinel:** the official invocation is consumed; all five outputs are
  absent; a sentinel is never synthesized later. A later authority may track
  only a terminal invocation ledger recording sentinel absence.
- **Post-sentinel, pre-other-output:** the existing sentinel is immutable and
  present; market, OI, manifest, and support are absent. A later authority may
  track only the existing sentinel and a terminal ledger.
- **Partial publication:** the sentinel is immutable and present; present
  non-sentinel outputs form a nonempty ordered prefix, giving a total prefix
  length from `2` through `5`; every present non-sentinel output remains
  immutable, ignored/untracked, `published_non_authoritative`,
  `source_authoritative:false`, and `downstream_consumable:false`; the absent
  suffix remains absent. A later authority may track only the existing
  sentinel and a terminal ledger; no non-sentinel output may ever be tracked.

Every nonzero S12 result permanently forbids retry, repair, resume, republish,
cleanup for reuse, overwrite, M12, Q12, P12, C12, D12, V12, H12, economics,
and candidate work. Existing sentinels or partial outputs are never recovery
inputs.

## Success-only M12 through H12 topology

The committed chain is strictly sequential:

```text
S11 -> A12 -> T11 -> S12 -> M12 -> Q12 -> P12 -> C12 -> D12 -> H12
```

V12 is not a stage, file, artifact, or commit. The H12 supervisor invokes it
exactly once in memory. Every committed stage is ordinarily pushed and proves
clean `HEAD == @{upstream}`, exact direct parent, exact name-status, Git and
worktree modes, exact cardinality, authenticated predecessor bindings,
expected fresh-output absence, unchanged G11 terminal inventory, and zero
repository bytecode before its one-shot action or child starts.

### Exact parent, file-set, mode, and cardinality matrix

| Stage | Direct parent | Exact diff | Git/worktree modes | Cardinality |
|---|---|---|---|---:|
| `M12` | `S12` | add sentinel, source manifest, source support listed below | `100644/0444` each | 3 added |
| `Q12` | `M12` | modify build, preregister, and three tests listed below | `100644/0644` each | 5 modified |
| `P12` | `Q12` | add preregistration JSON | `100644/0444` | 1 added |
| `C12` | `P12` | add access-claim JSON | `100644/0444` | 1 added |
| `D12` | `C12` | add inherited attempt, two worker ledgers, canonical CSV.GZ, and manifest | `100644/0444` each | 5 added |
| `H12` | `D12` | add V12 handoff JSON | `100644/0444` | 1 added |

Exact M12 diff:

```text
first_parent(M12) == S12
diff(S12, M12) ==
  A 100644 results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json
  A 100644 configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json
  A 100644 results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json
```

M12 is metadata authentication only and has no shell command. The generated
market and OI outputs remain immutable, ignored, and untracked:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb12_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb12_complete.csv.gz
```

Exact Q12 diff:

```text
first_parent(Q12) == M12
diff(M12, Q12) ==
  M 100644 training/build_gross9_structural_clock_bundle.py
  M 100644 training/preregister_gross9_structural_clock_bundle.py
  M 100644 tests/test_build_gross9_structural_clock_bundle.py
  M 100644 tests/test_preregister_gross9_structural_clock_bundle.py
  M 100644 tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
```

Q12 is edit/test/commit only. **No Q12 publication command exists or is
permitted.**

Exact P12, C12, D12, and H12 diffs:

```text
first_parent(P12) == Q12
diff(Q12, P12) ==
  A 100644 results/gross9_structural_clock_bundle_g9cb12_preregistration_2026-07-31.json

first_parent(C12) == P12
diff(P12, C12) ==
  A 100644 results/gross9_structural_clock_bundle_g9cb12_access_claim_2026-07-31.json

first_parent(D12) == C12
diff(C12, D12) ==
  A 100644 results/gross9_structural_clock_bundle_g9cb12_attempt_consumed_2026-07-31.json
  A 100644 results/gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass1_2026-07-31.json
  A 100644 results/gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass2_2026-07-31.json
  A 100644 results/gross9_structural_clock_bundle_g9cb12_2026-07-31.csv.gz
  A 100644 results/gross9_structural_clock_bundle_g9cb12_manifest_2026-07-31.json

first_parent(H12) == D12
diff(D12, H12) ==
  A 100644 results/gross9_structural_clock_bundle_g9cb12_v12_handoff_2026-07-31.json
```

D12's five-file set is exactly the inherited attempt sentinel, the isolated
pass-1 and pass-2 worker capability-consumption ledgers, the canonical
byte-identical CSV.GZ publication, and its manifest. No sixth D12 file is
allowed.

The H12 supervisor attempt sentinel remains immutable, ignored, and untracked:

```text
results/gross9_structural_clock_bundle_g9cb12_h12_supervisor_attempt_consumed_2026-07-31.json
```

### Exact downstream commands and cardinality

| Stage | Exact command | Invocation/publication count |
|---|---|---:|
| `M12` | none; metadata authentication only | `0` |
| `Q12` | none; edit/test/commit only | `0` |
| `P12` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle` | exactly `1` |
| `C12` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim` | exactly `1` |
| `D12` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce` | exactly `1` |
| `V12` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication` | exactly `1`, nested in H12 only |
| `H12` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v12-handoff` | exactly `1`, after V12 success |

P12 preregisters no economics and opens no official values. C12 creates only
the claim. D12 uses two isolated workers and requires byte-identical canonical
CSV.GZ results. V12 publishes nothing, writes no file, creates no commit, and
computes no economics; operators never invoke it manually. H12 is V12's sole
caller and the one-file handoff's sole publisher.

## Exact canonical H12 handoff

H12 publishes one object with exactly ten canonical sorted top-level keys and
an ordered six-binding `predecessor_bindings` array. Angle-bracket values are
authority-stage metavariables only. H12 resolves them from already committed,
pushed, and authenticated stages and from canonical V12 stdout captured only
in memory. No placeholder is serialized and no future commit is predicted.

```json
{
  "active_alpha_goal": "incomplete",
  "identity": "G9CB-12-SOURCE-SUPPORT",
  "ledger_kind": "gross9_structural_clock_bundle_g9cb12_v12_handoff_v1",
  "next_workflow": "ralplan",
  "no_economics": true,
  "no_future_commit_prediction": true,
  "predecessor_bindings": [
    {
      "commit": "<S12_COMMIT>",
      "parent_commit": "<T11_COMMIT>",
      "stage": "S12",
      "tracked_paths": [
        "training/materialize_gross9_structural_clock_g9cb12_sources.py",
        "tests/test_materialize_gross9_structural_clock_g9cb12_sources.py"
      ]
    },
    {
      "commit": "<M12_COMMIT>",
      "parent_commit": "<S12_COMMIT>",
      "stage": "M12",
      "tracked_paths": [
        "results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json",
        "configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json"
      ]
    },
    {
      "commit": "<Q12_COMMIT>",
      "parent_commit": "<M12_COMMIT>",
      "stage": "Q12",
      "tracked_paths": [
        "training/build_gross9_structural_clock_bundle.py",
        "training/preregister_gross9_structural_clock_bundle.py",
        "tests/test_build_gross9_structural_clock_bundle.py",
        "tests/test_preregister_gross9_structural_clock_bundle.py",
        "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py"
      ]
    },
    {
      "commit": "<P12_COMMIT>",
      "parent_commit": "<Q12_COMMIT>",
      "stage": "P12",
      "tracked_paths": [
        "results/gross9_structural_clock_bundle_g9cb12_preregistration_2026-07-31.json"
      ]
    },
    {
      "commit": "<C12_COMMIT>",
      "parent_commit": "<P12_COMMIT>",
      "stage": "C12",
      "tracked_paths": [
        "results/gross9_structural_clock_bundle_g9cb12_access_claim_2026-07-31.json"
      ]
    },
    {
      "commit": "<D12_COMMIT>",
      "parent_commit": "<C12_COMMIT>",
      "stage": "D12",
      "tracked_paths": [
        "results/gross9_structural_clock_bundle_g9cb12_attempt_consumed_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass1_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass2_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb12_2026-07-31.csv.gz",
        "results/gross9_structural_clock_bundle_g9cb12_manifest_2026-07-31.json"
      ]
    }
  ],
  "schema_version": 1,
  "source_generation": "G9CB12",
  "v12_stdout_hash": "<V12_STDOUT_HASH>"
}
```

The exact H12 schema and binding rules are:

- the top-level key order is `active_alpha_goal`, `identity`, `ledger_kind`,
  `next_workflow`, `no_economics`, `no_future_commit_prediction`,
  `predecessor_bindings`, `schema_version`, `source_generation`,
  `v12_stdout_hash`;
- `schema_version` is JSON integer `1`, not a boolean;
- `ledger_kind`, `identity`, `active_alpha_goal`, `next_workflow`, and
  `source_generation` are the exact strings shown;
- `no_economics` and `no_future_commit_prediction` are JSON boolean `true`;
- `v12_stdout_hash` is the learned lowercase 64-hex SHA-256 of canonical V12
  stdout captured only in memory;
- `predecessor_bindings` has exact length six and exact stage order `S12`,
  `M12`, `Q12`, `P12`, `C12`, `D12`;
- each binding has exactly the sorted keys `commit`, `parent_commit`, `stage`,
  and `tracked_paths`;
- each `commit` and `parent_commit` resolves to an already learned lowercase
  40-hex Git OID;
- the parent chain is exactly S12/T11, M12/S12, Q12/M12, P12/Q12, C12/P12,
  and D12/C12;
- each ordered `tracked_paths` array is exactly the array shown in the
  template;
- V12 creates no file or commit and no official command rerun supplies its
  hash;
- persisted H12 is duplicate-key-free canonical UTF-8 JSON with sorted keys,
  compact separators, `ensure_ascii=false`, `allow_nan=false`, normative
  arrays, and exactly one trailing LF.

H12 rejects every missing, additional, renamed, retyped, reordered, malformed,
wrong-stage, wrong-parent, wrong-commit, wrong-path, unresolved, noncanonical,
or future-predicting value; false no-economics/future-prediction booleans; any
V12 file or commit; noncanonical V12 stdout; and any preexisting supervisor
sentinel or handoff.

## Verification commands and named test entry points

These are future verification requirements. Every pytest command must exit
`0`. Every Python/pytest command is bytecode-safe through both
`PYTHONDONTWRITEBYTECODE=1` and `python -B`. `compileall` is forbidden.

### Sentinel-safe preflight inventory

Named entry point:

```text
tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_preflight_inventory_sentinel_safe
```

Exact command:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_preflight_inventory_sentinel_safe
```

Exact inventory evidence:

- all five fresh G9CB-12 outputs listed above are absent;
- the G9CB-11 sentinel is present and unchanged at exact
  worktree-mode/link/device/inode/size `0444/1/2096/934842/3056`, SHA-256
  `128ad6213785ecfa360114eae6e3587254dda3b18e94108b9dd30a0f34533e31`,
  and attempt hash
  `6a6204b5074aee399f6a4e318d24764140cfb07aea9b6ebd01b021f7333038f1`;
- the four permanent G9CB-11 generated paths listed above are absent; and
- no other output absence is required or asserted.

### Required focused entry points

| Purpose | Named entry point | Exact command |
|---|---|---|
| Exact tail predicate | `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_tail_predicate_exact_membership` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_tail_predicate_exact_membership` |
| AST prohibition | `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_forbidden_ast_operations` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_forbidden_ast_operations` |
| Stage modes and direct parents | `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_stage_modes_and_parent_diff_cardinality` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_stage_modes_and_parent_diff_cardinality` |
| Bytecode residue | `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_bytecode_residue_scan` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_bytecode_residue_scan` |
| Synthetic materialize, no main | `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_synthetic_materialize_e2e_no_main` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_synthetic_materialize_e2e_no_main` |
| Terminal pair matrix | `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_terminal_pair_matrix` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py::test_terminal_pair_matrix` |
| Exact H12 handoff schema | `tests/test_build_gross9_structural_clock_bundle.py::test_g9cb12_h12_handoff_schema_exact` | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_build_gross9_structural_clock_bundle.py::test_g9cb12_h12_handoff_schema_exact` |

The dedicated H12 schema test uses synthetic canonical V12 stdout captured in
memory. It must not call `main()`, invoke any official command, create a V12
file or commit, or open an official source. It exhaustively rejects missing,
additional, renamed, retyped, reordered, wrong-binding, malformed-hash,
false-boolean, noncanonical JSON/stdout, and future-prediction mutations.

### Complete suites and static checks

Targeted suite:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_materialize_gross9_structural_clock_g9cb12_sources.py
```

Affected suite:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q tests/test_materialize_gross9_structural_clock_g9cb9_sources.py tests/test_materialize_gross9_structural_clock_g9cb10_sources.py tests/test_materialize_gross9_structural_clock_g9cb11_sources.py tests/test_materialize_gross9_structural_clock_g9cb12_sources.py tests/test_build_gross9_structural_clock_bundle.py tests/test_preregister_gross9_structural_clock_bundle.py tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
```

Static whitespace check:

```text
git diff --check
```

Fresh pass counts and exit status `0` are recorded before an official
one-shot boundary. Synthetic tests use isolated temporary roots, call only
`materialize(synthetic_config)`, leave `main()` and every official command
uncalled, and open no official raw source.

### Official S12 command — explicitly not a test

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb12_sources
```

This later monitored command runs exactly once only after committed/pushed
clean S12 and successful preflight. It is never used as test evidence. Any
nonzero exit terminally consumes S12 and activates the prefix-derived
quarantine; no retry, repair, resume, or rerun follows.

## Stop rules

Stop immediately and grant no downstream authority if any of the following
occurs:

- A12 is not the exact one-document `100644` child of S11
  `646fccbf6568bcf39fab12a47873f72da880ca01`, or A12 predicts its commit;
- A12 or T11 opens a raw source or reports or infers a missing Rank7 count,
  location, timestamp, value, physical row count, or required-tail proof;
- T11 is not the exact two-file child of A12, any T11 schema/type/literal/array
  or self-hash rule differs, or the G11 sentinel changes;
- a permanently absent G11 leaf appears or any G11 rerun, retry, repair,
  resume, reuse, synthesis, or forbidden M11-through-H11 stage is attempted;
- S12 is not the exact committed/pushed clean two-file child of T11, a fresh
  output preexists, or the sole official command would be invoked a second
  time;
- required dates differ from `market.date.tail(min(3000, len(market)))`, a
  required date is missing or duplicated, selected dates are not exact and in
  order, or a global schema/date/UTC/uniqueness/monotonicity/five-minute
  invariant fails;
- a non-required non-date value affects admission, a selected-only numeric
  check touches a non-selected row, any pre-tail comparison or gap detail is
  disclosed, or a full-market/all-history/as-of/tolerance/fill/interpolation/
  resample/repair/timestamp-transform relation appears;
- publication is not private, complete, create-only, immutable, file-fsynced,
  directory-fsynced, and same-inode verified;
- a terminal pair is not one of the eight allowed pairs or publication state
  is derived from reason text instead of actual prefix;
- any nonzero one-shot command or publication-prefix failure occurs;
- any M12-through-H12 parent, diff, path, mode, cardinality, command count,
  clean-push, predecessor, output-absence, G11-preservation, or bytecode
  invariant differs;
- Q12 has or invokes a publication command, D12 has other than its exact five
  outputs, V12 is invoked outside H12 or writes a file/commit, or the H12
  supervisor sentinel becomes tracked;
- H12 differs from its canonical ten-key/six-binding schema, predicts a future
  commit, or serializes an unresolved placeholder; or
- any candidate, comparator, feature, schedule, signal, return, PnL, CAGR,
  MDD, drawdown, or economics value is opened or evaluated through H12.

After a nonzero S12 result, preserve and quarantine the exact immutable prefix
and do not recover. After successful verified H12, stop G9CB-12 infrastructure
work and route to a fresh candidate-specific Ralplan. The active alpha goal is
still `incomplete`; economics, overlap, reproduction, commit, and push remain
future candidate-plan work.

## ADR

**Decision:** select Rank7 rows by direct exact membership in the frozen
required-date tail.

**Drivers:** exactness, zero pre-tail disclosure, minimal transformation
surface, one-shot safety, immutable G11 history, and inherited downstream
compatibility.

**Compliant alternative:** exact index-selection/reindex without timestamp
transformation, only when byte-equivalent to the required set and without any
full-market relation.

**Rejected alternatives:** full-history join or comparison, full-market merge,
as-of or tolerance mapping, fill, interpolation, resample, repair, and
timestamp transformation.

**Consequences:** missing or duplicated required dates and global
schema/date/UTC/uniqueness/monotonicity/five-minute invalidity terminally fail
S12. Non-required rows and their non-date poison are irrelevant. No economics
occurs through H12.

**Follow-up:** only after verified H12, start a candidate-specific Ralplan for
low-overlap alpha economics and reproduction while preserving the active goal
as incomplete until that later work is actually complete.
