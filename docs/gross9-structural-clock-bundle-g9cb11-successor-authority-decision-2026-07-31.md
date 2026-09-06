# Gross9 structural clock bundle G9CB-11 successor authority decision — 2026-07-31

## Authority, separation, and decision

This document is the sole A11 authority for two mechanically separate acts:

1. T10 may seal the already-consumed, terminal G9CB-10 source-support attempt
   as historical evidence only; and
2. after successful T10, S11 may implement and invoke one fresh source-support
   identity, `G9CB-11-SOURCE-SUPPORT`, under the prospective rules below.

Historical G9CB-10 facts grant no source, retry, repair, resume, reuse, output,
or execution authority. Prospective G9CB-11 authority uses a distinct identity,
implementation, attempt hash, output inventory, replay guard, and one-shot
boundary. Treating any G9CB-10 fact as permission to rerun or consume G9CB-10
invalidates A11.

The operative G9CB-11 decision is **exact structural-window selection**. S11
scans the complete Binance metrics descriptor for `create_time` only, selects
physical rows whose timestamps are exact members of the frozen 120-row grid,
and semantically decodes non-date fields only at those selected positions. It
does not normalize, align, round, snap, repair, interpolate, resample, or map
timestamps by tolerance.

G9CB-11 is candidate-independent Gross9 clock infrastructure. It is not an
alpha, signal, strategy, comparator, portfolio, economics, or overlap
authority. A11, T10, S11, and M11 open no candidate, comparator, return, PnL,
CAGR, MDD, drawdown, or economic value. No candidate-specific work begins
before successful H11 and a fresh candidate-specific Ralplan.

## Exact A11 boundary

A11 has exact parent S10:

```text
S10 = 1079c3575c7e7dced52eea15e1ef35ae0171a5dd

first_parent(A11) == S10
diff(S10, A11) ==
  A docs/gross9-structural-clock-bundle-g9cb11-successor-authority-decision-2026-07-31.md
```

No other tracked or untracked change belongs to A11. A11 is independently
reviewed, committed alone, ordinarily pushed, and must end with a clean
worktree/index, zero repository bytecode, and `HEAD == @{upstream}` before T10
starts. This document does not predict its own future commit, Git blob,
SHA-256, or size, and it does not predict any T10, S11, or M11 future commit,
blob, size, file hash, self-hash, frame hash, or attempt hash.

## Historical G9CB-10 facts only

Everything in this section is terminal history. Nothing in it authorizes a
G9CB-10 action.

### S10 Git and implementation binding

```text
S10               = 1079c3575c7e7dced52eea15e1ef35ae0171a5dd
first_parent(S10) = a3ce195b02598b139068294089695b5d5dcd5044  # T9
branch             = codex/gross9-structural-clock-bundle-20260731
```

The exact S10 diff adds only these two mode-`100644` files:

| Path | Size bytes | SHA-256 | Git blob | Git mode | Worktree mode |
|---|---:|---|---|---|---|
| `training/materialize_gross9_structural_clock_g9cb10_sources.py` | 73480 | `67e6ed5bbbc80e532c2fe706e4fd43ad3352400cc8e87669efa51363f0dd3915` | `a7bd897f79e3425e28adfcef28b40fb9eb373340` | `100644` | `0644` |
| `tests/test_materialize_gross9_structural_clock_g9cb10_sources.py` | 84505 | `765a3c73cb5ea836916463fc6b134378a811234fba374831bde36ba53d3a9c6e` | `6e7e5a9904e7f04145361df9814e31854ed378f2` | `100644` | `0644` |

### Sole official S10 execution

The exact command was invoked once:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb10_sources
```

Frozen result:

- official invocation count: `1`;
- exit status: `1`;
- phase: `transform`;
- exception class: `SourceSupportFailure`;
- exception message: `off-grid timestamps: binance_metrics_open_interest`;
- exact observed exception rendering:
  `SourceSupportFailure: off-grid timestamps: binance_metrics_open_interest`;
- traceback source-value excerpt emitted: `false`;
- off-grid location disclosed: `false`;
- off-grid count disclosed: `false`;
- off-grid timestamp disclosed: `false`;
- off-grid value disclosed: `false`;
- retry allowed: `false`;
- resume allowed: `false`.

The traceback proves that all seven retained inputs completed their configured
decode loop and that failure occurred at the global metrics five-minute
alignment check. It does not prove where or how many timestamps were off grid.
No descendant may report or infer any off-grid location, count, timestamp, or
value.

### Immutable G9CB-10 attempt sentinel

```text
path       = results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json
type       = regular_file
mode       = 0444
link_count = 1
device     = 2096
inode      = 656585
size_bytes = 3056
sha256     = 2f3ea59c815ec36c64bcacc4ddabac12165cb6b04e38ca300532a754e1522796
attempt_hash = ba52b19400d399ecedd09be6ceecc0605aeb959ec425df262ddbff590a0661c2
identity   = G9CB-10-SOURCE-SUPPORT
repository_head   = 1079c3575c7e7dced52eea15e1ef35ae0171a5dd
repository_parent = a3ce195b02598b139068294089695b5d5dcd5044
one_shot = true
retry_allowed  = false
resume_allowed = false
raw_input_count = 7
opaque_bytes_hashed_before_publication = 190272610
```

The sentinel bytes, device/inode, mode, link count, and path identity are
immutable. T10 may add this existing inode to Git. It may not rewrite,
replace, rename, relink, chmod, truncate, remove, or recreate it.

### Permanent G9CB-10 absences and prohibitions

The following four leaves were absent after the official failure and remain
permanently absent:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz
configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json
```

G9CB-10 may never be rerun, repaired, resumed, cleaned up for reuse,
republished, completed, or consumed. No missing G9CB-10 output may be
synthesized. `M10`, `Q10`, `P10`, `C10`, `D10`, `V10`, and `H10` are
permanently forbidden. T10 is evidence-only and creates no G9CB-10 source
authority.

## T9 predecessor evidence bound by A11

T9 is exact commit `a3ce195b02598b139068294089695b5d5dcd5044`. It tracks the
mode-`100644`, worktree-mode-`0444` pair:

| Path | Size | SHA-256 | Git blob |
|---|---:|---|---|
| `results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json` | 3049 | `aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c` | `e6f7485675d974d1ce8d20194ff3b715238464f2` |
| `results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json` | 2835 | `e2379760507306f9e810e8d504af37e3fd3aa2f58c545c72467a76904991289c` | `ee6a5ddea961c8fe605c5f9ec9104bbdc8bcfb7e` |

The G9CB-9 attempt hash is
`fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887`.
The T9 terminal failure hash is
`29ca03ab230e644499fd28704dffa75c38091ad597a72cd6c7e90ef7dfd3ef78`.
T9 remains immutable evidence and grants no authority to rerun S9.

## Exact T10 two-file terminal seal

T10 starts only after clean pushed A11 and opens no official raw source. It
performs no source-value inference. It tracks exactly:

1. the existing immutable G9CB-10 attempt sentinel; and
2. `results/gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_2026-07-31.json`.

Its boundary is:

```text
first_parent(T10) == A11
diff(A11, T10) ==
  A results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_2026-07-31.json
```

Both files are duplicate-key-free canonical JSON plus exactly one LF, regular,
single-link, worktree mode `0444`, and Git mode `100644`. The existing sentinel
must retain the exact bytes, SHA-256, device/inode, mode, link count, size, and
attempt hash stated above across add, commit, and push. No data leaf or partial
output belongs to T10.

### Normative type and canonicalization rules

- `git_oid`: exactly 40 lowercase hexadecimal characters.
- `sha256`: exactly 64 lowercase hexadecimal characters.
- `repo_path`: a nonempty repository-relative POSIX path.
- `repo_path_list`: an ordered JSON array of `repo_path` values.
- `uint`: a JSON integer greater than or equal to zero; a boolean, float,
  string, or null is invalid.
- `string`: a JSON string.
- `boolean`: a JSON `true` or `false` literal.

Object key sets and value types are exact. Canonical JSON uses UTF-8, sorted
object keys, separators `(",", ":")`, `ensure_ascii=false`, `allow_nan=false`,
duplicate-key rejection, and normative array order. Self-hashes use canonical
bytes without a trailing LF and omit only the named self-hash member. Stored
JSON adds exactly one LF.

Angle-bracket or all-caps forms below are schema metavariables, never
serialized strings. They are learned once from their named authenticated
stage. They are not predictions.

### Fully typed thirteen-key T10 terminal ledger

The T10 ledger has exactly these thirteen top-level keys and no others:

| Key | Type | Exact value or rule |
|---|---|---|
| `schema_version` | `uint` | literal `1` |
| `ledger_kind` | `string` | `gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_v1` |
| `identity` | `string` | `G9CB-10-SOURCE-SUPPORT` |
| `status` | `string` | `terminal_metrics_timestamp_alignment_failure` |
| `authority` | object | exact schema below |
| `seal_authority` | object | exact schema below |
| `implementation` | object | exact schema below |
| `attempt_sentinel` | object | exact schema below |
| `execution` | object | exact schema below |
| `failure` | object | exact schema below |
| `access` | object | exact schema below |
| `output_state` | object | exact schema below |
| `terminal_failure_hash` | `sha256` | `T10_TERMINAL_FAILURE_HASH`, derived below |

`authority` has exactly:

| Key | Type | Exact value |
|---|---|---|
| `commit` | `git_oid` | `6f9dd21554bc7b3282d0b2cbf7badee126e75c1a` |
| `document_path` | `repo_path` | `docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md` |

`seal_authority` has exactly:

| Key | Type | Exact value |
|---|---|---|
| `commit` | `git_oid` | `A11_COMMIT`, learned from the committed A11 stage |
| `document_path` | `repo_path` | `docs/gross9-structural-clock-bundle-g9cb11-successor-authority-decision-2026-07-31.md` |

`implementation` has exactly `commit`, `parent_commit`, and `files`:

- `commit`: `git_oid`, literal S10
  `1079c3575c7e7dced52eea15e1ef35ae0171a5dd`;
- `parent_commit`: `git_oid`, literal T9
  `a3ce195b02598b139068294089695b5d5dcd5044`;
- `files`: ordered array of exact length two, implementation first and tests
  second.

Each `files` member has exactly `path:repo_path`, `size_bytes:uint`,
`sha256:sha256`, `git_blob:git_oid`, `git_mode:string`, and
`worktree_mode:string`. The exact members are:

```json
[
  {"path":"training/materialize_gross9_structural_clock_g9cb10_sources.py","size_bytes":73480,"sha256":"67e6ed5bbbc80e532c2fe706e4fd43ad3352400cc8e87669efa51363f0dd3915","git_blob":"a7bd897f79e3425e28adfcef28b40fb9eb373340","git_mode":"100644","worktree_mode":"0644"},
  {"path":"tests/test_materialize_gross9_structural_clock_g9cb10_sources.py","size_bytes":84505,"sha256":"765a3c73cb5ea836916463fc6b134378a811234fba374831bde36ba53d3a9c6e","git_blob":"6e7e5a9904e7f04145361df9814e31854ed378f2","git_mode":"100644","worktree_mode":"0644"}
]
```

`attempt_sentinel` has exactly:

| Key | Type | Exact value |
|---|---|---|
| `path` | `repo_path` | exact G9CB-10 sentinel path |
| `filesystem_type` | `string` | `regular_file` |
| `worktree_mode` | `string` | `0444` |
| `git_mode` | `string` | `100644` |
| `link_count` | `uint` | `1` |
| `device` | `uint` | `2096` |
| `inode` | `uint` | `656585` |
| `size_bytes` | `uint` | `3056` |
| `sha256` | `sha256` | `2f3ea59c815ec36c64bcacc4ddabac12165cb6b04e38ca300532a754e1522796` |
| `attempt_hash` | `sha256` | `ba52b19400d399ecedd09be6ceecc0605aeb959ec425df262ddbff590a0661c2` |
| `repository_head` | `git_oid` | S10 literal |
| `repository_parent` | `git_oid` | T9 literal |
| `one_shot` | `boolean` | `true` |
| `retry_allowed` | `boolean` | `false` |
| `resume_allowed` | `boolean` | `false` |
| `raw_input_count` | `uint` | `7` |
| `opaque_bytes_hashed_before_publication` | `uint` | `190272610` |

`execution` has exactly:

| Key | Type | Exact value |
|---|---|---|
| `command` | `string` | exact official S10 command above |
| `invocation_count` | `uint` | `1` |
| `exit_status` | `uint` | `1` |
| `one_shot` | `boolean` | `true` |
| `retry_allowed` | `boolean` | `false` |
| `resume_allowed` | `boolean` | `false` |

`failure` has exactly:

| Key | Type | Exact value |
|---|---|---|
| `phase` | `string` | `transform` |
| `exception_class` | `string` | `SourceSupportFailure` |
| `exception_message` | `string` | `off-grid timestamps: binance_metrics_open_interest` |
| `traceback_source_value_excerpt_emitted` | `boolean` | `false` |
| `off_grid_location_disclosed` | `boolean` | `false` |
| `off_grid_count_disclosed` | `boolean` | `false` |
| `off_grid_timestamp_disclosed` | `boolean` | `false` |
| `off_grid_value_disclosed` | `boolean` | `false` |
| `off_grid_detail_disclosure_count` | `uint` | `0` |

`access` has exactly:

| Key | Type | Exact value |
|---|---|---:|
| `raw_file_count` | `uint` | 7 |
| `raw_file_open_count` | `uint` | 7 |
| `decode_pass_count` | `uint` | 8 |
| `decode_passes` | ordered string array | exact list below |
| `replacement_market_date_scan_count` | `uint` | 1 |
| `replacement_market_tail_decode_count` | `uint` | 1 |
| `replacement_market_tail_selected_row_count` | `uint` | 107 |
| `binance_metrics_open_interest_decode_count` | `uint` | 1 |
| `global_metrics_alignment_check_count` | `uint` | 1 |
| `attempt_sentinel_publication_count` | `uint` | 1 |
| `generated_output_publication_count` | `uint` | 0 |
| `generated_output_readback_count` | `uint` | 0 |
| `off_grid_detail_disclosure_count` | `uint` | 0 |
| `candidate_value_rows_opened` | `uint` | 0 |
| `comparator_value_rows_opened` | `uint` | 0 |
| `feature_value_rows_opened` | `uint` | 0 |
| `schedule_value_rows_opened` | `uint` | 0 |
| `signal_value_rows_opened` | `uint` | 0 |
| `return_value_rows_opened` | `uint` | 0 |
| `pnl_value_rows_opened` | `uint` | 0 |
| `cagr_evaluation_count` | `uint` | 0 |
| `mdd_evaluation_count` | `uint` | 0 |
| `drawdown_evaluation_count` | `uint` | 0 |
| `economic_value_rows_opened` | `uint` | 0 |
| `economic_evaluation_count` | `uint` | 0 |

The exact ordered T10 decode list is:

```json
["old_market","replacement_market_date_scan","replacement_market_tail","funding","premium","old_open_interest","binance_metrics_open_interest","rank7_spot_premium_5m"]
```

`output_state` has exactly:

- `terminal_evidence_paths`: ordered `repo_path_list` of exact length one,
  containing only the G9CB-10 attempt-sentinel path;
- `permanently_absent_output_paths`: ordered `repo_path_list` of exact length
  four, in the order stated in the permanent-absence section;
- `forbidden_stages`: ordered string array
  `["M10","Q10","P10","C10","D10","V10","H10"]`;
- `source_authoritative`: `boolean`, literal `false`;
- `downstream_consumable`: `boolean`, literal `false`.

`terminal_failure_hash` is SHA-256 of canonical no-LF JSON with only
`terminal_failure_hash` omitted. The persisted ledger adds exactly one LF.
T10 learns that hash from the completed object; A11 does not predict it.

## Prospective fresh G9CB-11 authority

This section becomes actionable only after successful, committed, pushed T10.
The fresh identity is exactly:

```text
G9CB-11-SOURCE-SUPPORT
```

S11 adds exactly:

```text
first_parent(S11) == T10
diff(T10, S11) ==
  A tests/test_materialize_gross9_structural_clock_g9cb11_sources.py
  A training/materialize_gross9_structural_clock_g9cb11_sources.py
```

Both Git modes are `100644`. The implementation-path order inside access and
replay ledgers is implementation first, tests second:

1. `training/materialize_gross9_structural_clock_g9cb11_sources.py`;
2. `tests/test_materialize_gross9_structural_clock_g9cb11_sources.py`.

The exact sole official command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb11_sources
```

### Fresh G9CB-11 output inventory

The exact ordered five-output list is:

1. `results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json`;
2. `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz`;
3. `data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz`;
4. `configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json`;
5. `results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json`.

All five must be absent before any source opens. Publication uses a private
same-directory `O_TMPFILE`, complete write, `fchmod(0444)`, file fsync,
create-only `linkat` through `/proc/self/fd` with `AT_SYMLINK_FOLLOW`, directory
fsync, and linked device/inode identity verification. `AT_EMPTY_PATH`, named
temporary files, overwrite, rename, truncate, unlink, replacement, post-link
chmod, recovery, and `EEXIST` repair are forbidden.

### Exact one-shot gate order

Before the command, targeted, affected, AST, static, whitespace, mutation, and
synthetic failpoint tests pass; independent code review and verification pass;
S11 is committed and pushed; and no official source was opened by those tests.

The official process then executes these gates in order:

1. validate the no-argument command, exact environment, and canonical root;
2. run the repository-bytecode absence gate as the first repository traversal;
3. require the exact branch, clean index/worktree, `HEAD == @{upstream} == S11`,
   direct parent T10, and exact S11 two-file diff;
4. authenticate A11, both immutable T9 files, both immutable T10 files, exact
   S10 Git metadata, the unchanged G9CB-10 sentinel, and all four permanent
   G9CB-10 absences without opening an official source value;
5. require all five G9CB-11 output leaves absent and reject path aliases;
6. open each of the exact seven raw files once with retained, distinct,
   no-follow descriptors and authenticate path/type/mode/link/size/hash without
   decoding a source value;
7. publish and authenticate the fresh immutable G9CB-11 attempt sentinel
   create-only; successful sentinel publication consumes the identity;
8. perform the exact nine logical passes and all structural validations below;
9. publish market, OI, manifest, and support in that order, with generated
   market and OI each read back exactly once;
10. reauthenticate all retained inputs and all five outputs, reject residue,
    run the final bytecode gate, and return success only after every check.

The command is invoked exactly once. Any nonzero exit terminally consumes the
official invocation regardless of whether the sentinel exists. Retry, resume,
repair, cleanup for reuse, or a second invocation is forbidden.

## Exact seven raw bindings without source-value access

A11 copies these seven descriptors literally from the authenticated G9CB-10
attempt sentinel. A11 and T10 do not open them. S11 opens each retained raw file
once and no G9CB-9 or G9CB-10 generated output is an S11 source.

| Name | Path | SHA-256 | Size bytes | Mode | Encoding |
|---|---|---|---:|---|---|
| `old_market` | `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz` | `a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c` | 66696659 | `0644` | gzip |
| `replacement_market` | `/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz` | `0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe` | 65420089 | `0644` | gzip |
| `funding` | `data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz` | `4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7` | 89326 | `0644` | gzip |
| `premium` | `data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz` | `b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7` | 1196481 | `0644` | gzip |
| `old_open_interest` | `/tmp/btcusdt_open_interest_5m_2020_2026.csv` | `e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31` | 19657777 | `0644` | plain CSV |
| `binance_metrics_open_interest` | `/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz` | `d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106` | 21440132 | `0644` | gzip |
| `rank7_spot_premium_5m` | `/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz` | `c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617` | 15772146 | `0644` | gzip |

The exact opaque total is `190272610` bytes. No symlink, hard-link alias,
non-regular file, link count other than one, mode drift, size drift, path-edge
drift, inode drift, descriptor drift, byte drift, short read, or reopening by
pathname is permitted.

## Frozen market transformation

The market transform remains the G9CB-10 structural partition:

```text
boundary   = 2026-05-31T15:00:00Z
domain_end = 2026-06-01T00:00:00Z
old_prefix = old_market[date <= boundary]
tail       = replacement_market[(date > boundary) & (date < domain_end)]
output     = old_prefix ⊎ tail
```

The old market and selected replacement tail have this exact ordered schema:

```json
["date","open","high","low","close","volume","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","tic","day","dxy","kimchi_premium","usdkrw","btckrw","dxy_available","kimchi_available","usdkrw_available","external_any_available","dxy_zscore","dxy_momentum","kimchi_premium_zscore","kimchi_premium_change","usdkrw_zscore","usdkrw_momentum"]
```

- old rows: `674785`;
- selected tail rows: `107`;
- output rows: `674892`;
- tail first/last: `2026-05-31T15:05:00Z` /
  `2026-05-31T23:55:00Z`;
- cadence: exactly 300 seconds;
- old and tail timestamp sets are disjoint;
- the first 674785 output rows are old-market exact;
- the final 107 output rows are selected-tail exact;
- replacement-prefix values are not compared;
- replacement-prefix non-date fields are not semantically evaluated.

Replacement uses one complete date-only scan and one selected 107-row tail
decode. No fill, interpolation, resampling, repair, or overlap-value comparison
is authorized.

## Exact metrics two-pass frozen-grid selection

The Binance metrics ordered schema is exactly:

```json
["create_time","symbol","sum_open_interest","sum_open_interest_value","count_toptrader_long_short_ratio","sum_toptrader_long_short_ratio","count_long_short_ratio","sum_taker_long_short_vol_ratio"]
```

The frozen selected grid has:

```text
start               = 2026-05-31T14:00:00Z
inherited_boundary  = 2026-05-31T15:00:00Z
required_last       = 2026-05-31T23:55:00Z
cadence_seconds     = 300
selected_grid_rows  = 120
overlap_rows        = 13
tail_rows           = 107
```

### Pass 1 — complete date scan

1. Duplicate the retained metrics descriptor.
2. Authenticate the exact eight-column header.
3. Parse only `create_time` from every physical row.
4. Require every date token valid and non-null and the complete date sequence
   globally unique and monotonic.
5. Do not require global five-minute alignment.
6. Select physical positions only when the parsed timestamp is an exact member
   of the frozen 120-row grid.
7. Require exactly one selected row per required timestamp and require selected
   timestamps to equal the frozen grid in exact order.
8. Reauthenticate the retained descriptor before pass 2.

The selector is a pure timestamp-membership filter over the frozen grid.
Wall-clock interval membership, nearest-neighbor distance, timestamp phase,
physical adjacency, or any non-date value may not influence selection.
Off-grid rows before, inside, or after the interval are irrelevant if every
required exact grid member exists and global uniqueness/monotonicity holds.

### Pass 2 — selected-position decode

1. Duplicate the same retained descriptor.
2. Frame all physical rows, but do not width-check or semantically evaluate any
   non-selected row's non-date fields.
3. At selected positions only, require exact row width and schema.
4. Semantically decode only `create_time`, `symbol`, and
   `sum_open_interest`.
5. Require every selected `symbol` to equal `BTCUSDT`.
6. Require selected dates to equal the pass-1 selected dates exactly.
7. Require the first 13 selected OI values, from 14:00 through 15:00 inclusive,
   to equal inherited OI exactly, including missing-value equality if any.
8. Require the final 107 selected OI values, from 15:05 through 23:55, to be
   finite and greater than zero.
9. Reauthenticate the retained descriptor.

The output OI frame is the exact inherited 674785-row frame followed by the
exact selected 107-row tail. It has exact schema
`["date","open_interest"]`, 674892 rows, and one continuous five-minute grid.
Inherited missingness outside the splice remains unchanged.

Global metrics alignment rejection; floor, ceil, round, snap, resample, fill,
interpolation, repair, `merge_asof`, tolerance mapping; semantic evaluation of
non-selected non-date fields; and reporting or inferring off-grid details are
forbidden.

The Rank7 source retains the exact ordered projection
`["date","spot_close","spot_rows","premium_index_1m_close","premium_rows"]`,
the exact-left one-to-one timestamp join, terminal validity, and 3000-row
tail-completeness rules. Funding and premium retain causal backward attachment
with exact tolerances `12h` and `2h`. These support checks may not construct a
feature, decision, signal, schedule, return, or economic value.

## Exact G9CB-11 access ledger

Every G9CB-11 access-bearing artifact contains exactly these top-level blocks:

```text
schema_version
ledger_kind
historical_s9
historical_s10
current_s11
process_local
replay_guard
access_ledger_hash
```

The exact top-level types are:

- `schema_version:uint`, literal `1`;
- `ledger_kind:string`, literal
  `gross9_structural_clock_bundle_g9cb11_access_v1`;
- `historical_s9:object`, exact schema below;
- `historical_s10:object`, exact schema below;
- `current_s11:object`, exact schema below;
- `process_local:object`, exact schema below;
- `replay_guard:object`, exact schema below;
- `access_ledger_hash:sha256`, learned at S11 finalization.

`access_ledger_hash` is SHA-256 of the complete canonical no-LF access ledger
with only `access_ledger_hash` omitted.

### Exact placeholder registry and authority stages

| Placeholder | JSON type | Exact authority stage |
|---|---|---|
| `A11_COMMIT` | `git_oid` | A11 commit |
| `T10_COMMIT` | `git_oid` | T10 commit |
| `T10_TERMINAL_LEDGER_SIZE_BYTES` | `uint` | authenticated T10 terminal-ledger file |
| `T10_TERMINAL_LEDGER_FILE_SHA256` | `sha256` | authenticated T10 terminal-ledger bytes, including final LF |
| `T10_TERMINAL_LEDGER_GIT_BLOB` | `git_oid` | T10 terminal-ledger Git blob |
| `T10_TERMINAL_FAILURE_HASH` | `sha256` | T10 terminal-ledger self-hash |
| `S11_COMMIT` | `git_oid` | S11 commit |
| `G9CB11_ATTEMPT_SENTINEL_SIZE_BYTES` | `uint` | authenticated S11 attempt sentinel |
| `G9CB11_ATTEMPT_SENTINEL_FILE_SHA256` | `sha256` | authenticated S11 attempt-sentinel bytes |
| `G9CB11_ATTEMPT_HASH` | `sha256` | S11 attempt-sentinel self-hash |
| `REPLAY_GUARD_HASH` | `sha256` | S11 replay guard finalization |
| `ACCESS_LEDGER_HASH` | `sha256` | complete S11 access-ledger finalization |
| `REPLACEMENT_MARKET_PHYSICAL_ROWS` | `uint` | completed S11 replacement-market date scan plus descriptor reauthentication |
| `FUNDING_PHYSICAL_ROWS` | `uint` | completed authenticated S11 funding decode |
| `PREMIUM_PHYSICAL_ROWS` | `uint` | completed authenticated S11 premium decode |
| `METRICS_PHYSICAL_ROWS` | `uint` | completed S11 metrics create-time scan plus descriptor reauthentication |
| `RANK7_PHYSICAL_ROWS` | `uint` | completed authenticated S11 Rank7 decode |
| `SOURCE_VALUE_ROWS_OPENED` | `uint` | S11 process-local finalization after all nine addends are final |

Every placeholder must resolve only from its named stage. Unresolved,
predicted, early-captured, wrong-stage, wrong-length, uppercase, retyped, or
same-primitive substituted values are invalid. Repeated occurrences of one
placeholder are byte-equal. A numerically equal uint from another pass does not
satisfy the authority-stage requirement.

### `historical_s9`

`historical_s9` has exactly these keys and concrete T9-bound values:

```json
{
  "identity": "G9CB-9-SOURCE-SUPPORT",
  "attempt_sentinel": {
    "path": "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
    "sha256": "aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c",
    "size_bytes": 3049,
    "attempt_hash": "fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887",
    "git_blob": "e6f7485675d974d1ce8d20194ff3b715238464f2",
    "git_mode": "100644",
    "seal_commit": "a3ce195b02598b139068294089695b5d5dcd5044"
  },
  "terminal_failure_ledger": {
    "path": "results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json",
    "sha256": "e2379760507306f9e810e8d504af37e3fd3aa2f58c545c72467a76904991289c",
    "size_bytes": 2835,
    "terminal_failure_hash": "29ca03ab230e644499fd28704dffa75c38091ad597a72cd6c7e90ef7dfd3ef78",
    "git_blob": "ee6a5ddea961c8fe605c5f9ec9104bbdc8bcfb7e",
    "git_mode": "100644",
    "seal_commit": "a3ce195b02598b139068294089695b5d5dcd5044"
  },
  "official_invocation_count": 1,
  "exit_status": 1,
  "raw_file_count": 7,
  "decode_pass_count": 7,
  "market_prefix_comparison_count": 1,
  "mismatch_fraction_percent": "0.00148",
  "traceback_value_excerpt_emitted": true,
  "traceback_value_excerpt_restated_in_ledger": false,
  "generated_readback_decode_count": 0,
  "candidate_rows_opened": 0,
  "comparator_clock_rows_opened": 0,
  "feature_signal_schedule_or_interval_values_computed": 0,
  "economic_or_overlap_values_computed": 0
}
```

This block is literal history. It is never merged into current S11 counters
and never authorizes S9 access or execution.

### `historical_s10`

`historical_s10` has exactly `identity`, `status`, `authority`,
`seal_authority`, `implementation`, `attempt_sentinel`, `terminal_ledger`,
`execution`, `failure`, `access`, and `output_state`.

- `identity` and `status` are literal copies of T10.
- `authority`, `seal_authority`, `implementation`, `attempt_sentinel`,
  `execution`, `failure`, `access`, and `output_state` are concrete deep literal
  copies of the corresponding T10 objects defined in this document. References,
  schema names, merged counters, or substituted objects are invalid.
- `terminal_ledger` has exactly:
  - `path:repo_path`, literal
    `results/gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_2026-07-31.json`;
  - `commit:git_oid`, `T10_COMMIT`;
  - `size_bytes:uint`, `T10_TERMINAL_LEDGER_SIZE_BYTES`;
  - `sha256:sha256`, `T10_TERMINAL_LEDGER_FILE_SHA256`;
  - `git_blob:git_oid`, `T10_TERMINAL_LEDGER_GIT_BLOB`;
  - `git_mode:string`, literal `100644`;
  - `worktree_mode:string`, literal `0444`;
  - `terminal_failure_hash:sha256`, `T10_TERMINAL_FAILURE_HASH`.

### `current_s11`

`current_s11` has exactly `identity`, `authority`, `implementation`,
`attempt_sentinel`, `execution`, and `access`.

- `identity:string` is literal `G9CB-11-SOURCE-SUPPORT`.
- `authority` has exactly:
  - `commit:git_oid` = `A11_COMMIT`;
  - `document_path:repo_path` = exact A11 path.
- `implementation` has exactly:
  - `commit:git_oid` = `S11_COMMIT`;
  - `parent_commit:git_oid` = `T10_COMMIT`;
  - `paths:repo_path_list`, exact length two and exact implementation-first
    order stated above.
- `attempt_sentinel` has exactly:
  - `path:repo_path`, exact G9CB-11 attempt-sentinel path;
  - `size_bytes:uint` = `G9CB11_ATTEMPT_SENTINEL_SIZE_BYTES`;
  - `sha256:sha256` = `G9CB11_ATTEMPT_SENTINEL_FILE_SHA256`;
  - `attempt_hash:sha256` = `G9CB11_ATTEMPT_HASH`.
- `execution` has exactly:
  - `command:string`, exact S11 command;
  - `invocation_count:uint` = `1`;
  - `one_shot:boolean` = `true`;
  - `retry_allowed:boolean` = `false`;
  - `resume_allowed:boolean` = `false`.

`current_s11.access` has exactly:

| Key | Type | Exact value |
|---|---|---:|
| `raw_file_count` | `uint` | 7 |
| `raw_file_open_count` | `uint` | 7 |
| `decode_pass_count` | `uint` | 9 |
| `decode_passes` | ordered string array | exact list below |
| `replacement_market_date_scan_count` | `uint` | 1 |
| `replacement_market_tail_decode_count` | `uint` | 1 |
| `replacement_market_tail_selected_row_count` | `uint` | 107 |
| `metrics_date_scan_count` | `uint` | 1 |
| `metrics_selected_decode_count` | `uint` | 1 |
| `metrics_selected_row_count` | `uint` | 120 |
| `metrics_overlap_row_count` | `uint` | 13 |
| `metrics_tail_row_count` | `uint` | 107 |
| `non_selected_metrics_non_date_semantic_evaluation_count` | `uint` | 0 |
| `global_metrics_alignment_comparison_count` | `uint` | 0 |
| `attempt_sentinel_publication_count` | `uint` | 1 |
| `generated_output_publication_count` | `uint` | 4 |
| `generated_output_readback_count` | `uint` | 2 |
| `off_grid_detail_disclosure_count` | `uint` | 0 |
| `candidate_value_rows_opened` | `uint` | 0 |
| `comparator_value_rows_opened` | `uint` | 0 |
| `feature_value_rows_opened` | `uint` | 0 |
| `schedule_value_rows_opened` | `uint` | 0 |
| `signal_value_rows_opened` | `uint` | 0 |
| `return_value_rows_opened` | `uint` | 0 |
| `pnl_value_rows_opened` | `uint` | 0 |
| `cagr_evaluation_count` | `uint` | 0 |
| `mdd_evaluation_count` | `uint` | 0 |
| `drawdown_evaluation_count` | `uint` | 0 |
| `economic_value_rows_opened` | `uint` | 0 |
| `economic_evaluation_count` | `uint` | 0 |

The exact ordered S11 pass list is:

```json
["old_market","replacement_market_date_scan","replacement_market_tail","funding","premium","old_open_interest","binance_metrics_open_interest_date_scan","binance_metrics_open_interest_selected_window","rank7_spot_premium_5m"]
```

### `process_local`

`process_local` has exactly:

| Key | Type | Exact value or binding |
|---|---|---|
| `stage` | `string` | `S11` |
| `slot` | `uint` | 0 |
| `invocation_count` | `uint` | 1 |
| `raw_file_count` | `uint` | 7 |
| `raw_file_open_count` | `uint` | 7 |
| `decode_pass_count` | `uint` | 9 |
| `old_market_rows_opened` | `uint` | 674785 |
| `replacement_market_date_rows_scanned` | `uint` | `REPLACEMENT_MARKET_PHYSICAL_ROWS` |
| `replacement_market_tail_rows_opened` | `uint` | 107 |
| `funding_rows_opened` | `uint` | `FUNDING_PHYSICAL_ROWS` |
| `premium_rows_opened` | `uint` | `PREMIUM_PHYSICAL_ROWS` |
| `old_open_interest_rows_opened` | `uint` | 674785 |
| `binance_metrics_open_interest_date_rows_scanned` | `uint` | `METRICS_PHYSICAL_ROWS` |
| `binance_metrics_open_interest_selected_window_rows_opened` | `uint` | 120 |
| `rank7_spot_premium_5m_rows_opened` | `uint` | `RANK7_PHYSICAL_ROWS` |
| `source_value_rows_opened` | `uint` | `SOURCE_VALUE_ROWS_OPENED` |
| `non_selected_metrics_non_date_semantic_evaluation_count` | `uint` | 0 |
| `global_metrics_alignment_comparison_count` | `uint` | 0 |
| `generated_output_readback_count` | `uint` | 2 |
| `off_grid_detail_disclosure_count` | `uint` | 0 |
| `candidate_value_rows_opened` | `uint` | 0 |
| `comparator_value_rows_opened` | `uint` | 0 |
| `feature_value_rows_opened` | `uint` | 0 |
| `schedule_value_rows_opened` | `uint` | 0 |
| `signal_value_rows_opened` | `uint` | 0 |
| `return_value_rows_opened` | `uint` | 0 |
| `pnl_value_rows_opened` | `uint` | 0 |
| `cagr_evaluation_count` | `uint` | 0 |
| `mdd_evaluation_count` | `uint` | 0 |
| `drawdown_evaluation_count` | `uint` | 0 |
| `economic_value_rows_opened` | `uint` | 0 |
| `economic_evaluation_count` | `uint` | 0 |

The exact nine-addend formula is:

```text
SOURCE_VALUE_ROWS_OPENED =
    674785
  + REPLACEMENT_MARKET_PHYSICAL_ROWS
  + 107
  + FUNDING_PHYSICAL_ROWS
  + PREMIUM_PHYSICAL_ROWS
  + 674785
  + METRICS_PHYSICAL_ROWS
  + 120
  + RANK7_PHYSICAL_ROWS
```

The addends bind in order to old market, replacement date scan, replacement
tail, funding, premium, old OI, metrics date scan, metrics selected window, and
Rank7. Each row touch is counted once per logical pass. Headers, descriptor
duplication, evidence reads, and generated readbacks are excluded.
`SOURCE_VALUE_ROWS_OPENED` cannot resolve before all nine stage-authoritative
addends are final. Omission, duplication, reordering of provenance,
substitution, early binding, or plus/minus-one mutation is invalid.

## Exact replay guard

`replay_guard` has exactly:

```text
a11
t10
s11
identities
attempt_hashes
terminal_failure_hashes
expected_output_paths
pairwise_output_intersections
identities_pairwise_distinct
attempt_hashes_pairwise_distinct
replay_guard_hash
```

`a11` has exactly:

- `commit:git_oid` = `A11_COMMIT`;
- `authority_document_path:repo_path` = exact A11 path.

`t10` has exactly:

- `commit:git_oid` = `T10_COMMIT`;
- `attempt_sentinel_path:repo_path` = exact G9CB-10 sentinel path;
- `attempt_hash:sha256` = literal
  `ba52b19400d399ecedd09be6ceecc0605aeb959ec425df262ddbff590a0661c2`;
- `terminal_ledger_path:repo_path` = exact T10 terminal-ledger path;
- `terminal_ledger_sha256:sha256` = `T10_TERMINAL_LEDGER_FILE_SHA256`;
- `terminal_failure_hash:sha256` = `T10_TERMINAL_FAILURE_HASH`.

`s11` has exactly:

- `commit:git_oid` = `S11_COMMIT`;
- `parent_commit:git_oid` = `T10_COMMIT`;
- `implementation_paths:repo_path_list`, exact length two, exact
  implementation-first order stated above;
- `attempt_sentinel_path:repo_path` = exact G9CB-11 sentinel path;
- `attempt_hash:sha256` = `G9CB11_ATTEMPT_HASH`.

`identities` has exactly:

```json
{"historical_s9":"G9CB-9-SOURCE-SUPPORT","historical_s10":"G9CB-10-SOURCE-SUPPORT","current_s11":"G9CB-11-SOURCE-SUPPORT"}
```

`attempt_hashes` has exactly:

- `historical_s9:sha256` =
  `fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887`;
- `historical_s10:sha256` =
  `ba52b19400d399ecedd09be6ceecc0605aeb959ec425df262ddbff590a0661c2`;
- `current_s11:sha256` = `G9CB11_ATTEMPT_HASH`.

`terminal_failure_hashes` has exactly:

- `historical_s9:sha256` =
  `29ca03ab230e644499fd28704dffa75c38091ad597a72cd6c7e90ef7dfd3ef78`;
- `historical_s10:sha256` = `T10_TERMINAL_FAILURE_HASH`.

`expected_output_paths` has exactly `historical_s9`, `historical_s10`, and
`current_s11`. Each value is an ordered `repo_path_list` of exact length five:

```text
historical_s9:
1. results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json
2. data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
3. data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
4. configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json
5. results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json

historical_s10:
1. results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json
2. data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz
3. data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz
4. configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json
5. results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json

current_s11:
1. results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json
2. data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz
3. data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz
4. configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json
5. results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json
```

`pairwise_output_intersections` has exactly:

```json
{"historical_s9_historical_s10":[],"historical_s9_current_s11":[],"historical_s10_current_s11":[]}
```

- `identities_pairwise_distinct:boolean` is literal `true`;
- `attempt_hashes_pairwise_distinct:boolean` is literal `true`;
- `replay_guard_hash:sha256` is `REPLAY_GUARD_HASH`.

`REPLAY_GUARD_HASH` omits only `replay_guard_hash` from canonical no-LF JSON.
The two S11 implementation-path arrays are byte-equal. Reordering, merging,
swapping, collapsing, cross-generation substitution, nonempty intersection, or
recomputed hashes over a mutated schema do not legitimize a mutation.

## Exact S11 failure classes and quarantine

Let `O11` be the ordered four-output list after the attempt sentinel: market,
OI, manifest, support. Every nonzero S11 exit resolves to exactly one class.

### `pre_sentinel_failure`

- The official invocation is consumed.
- The sentinel and all `O11` members are absent.
- A sentinel is never synthesized later.
- A later authority may track only a terminal invocation ledger recording
  `sentinel_state` as absent.

### `post_sentinel_pre_other_output_failure`

- The existing sentinel is immutable and present.
- Every `O11` member is absent.
- A later authority may track only the existing sentinel and a terminal
  ledger.

### `partial_publication_failure`

- The existing sentinel is immutable and present.
- Present `O11` members form a nonempty ordered prefix of length one through
  four.
- Each present member remains immutable, ignored/untracked,
  `published_non_authoritative`, `source_authoritative:false`, and
  `downstream_consumable:false`.
- Absent suffix members remain absent.
- A later authority may track only the existing sentinel and terminal ledger.
  No `O11` member may ever be tracked.

All three classes permanently forbid retry, repair, resume, republish, cleanup
for reuse, M11, Q11, P11, C11, D11, V11, H11, economics, and candidate work.
An existing sentinel or partial output is never accepted as recovery.

## Success-only M11 through H11 topology

The success topology is strictly sequential:

```text
S10 -> A11 -> T10 -> S11 -> M11 -> Q11 -> P11 -> C11 -> D11 -> H11
```

V11 is command-only. It is invoked exactly once only by the one-shot H11
supervisor. Every committed stage is ordinarily pushed and proves a clean
`HEAD == @{upstream}` before its child starts.

The exact direct-parent and tracked-diff topology is:

```text
first_parent(M11) == S11
diff(S11, M11) ==
  A results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json
  A configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json

first_parent(Q11) == M11
diff(M11, Q11) ==
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py

first_parent(P11) == Q11
diff(Q11, P11) ==
  A results/gross9_structural_clock_bundle_g9cb11_preregistration_2026-07-31.json

first_parent(C11) == P11
diff(P11, C11) ==
  A results/gross9_structural_clock_bundle_g9cb11_access_claim_2026-07-31.json

first_parent(D11) == C11
diff(C11, D11) ==
  A results/gross9_structural_clock_bundle_g9cb11_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb11_worker_capability_consumed_pass1_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb11_worker_capability_consumed_pass2_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb11_2026-07-31.csv.gz
  A results/gross9_structural_clock_bundle_g9cb11_manifest_2026-07-31.json

first_parent(H11) == D11
diff(D11, H11) ==
  A results/gross9_structural_clock_bundle_g9cb11_v11_handoff_2026-07-31.json
```

### M11

M11 exists only after status-zero S11 and complete authentication of all five
outputs. It tracks exactly:

1. `results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json`;
2. `configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json`;
3. `results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json`.

The exact ignored and untracked generated set is:

1. `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz`;
2. `data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz`.

### Q11

Q11 modifies exactly:

1. `training/build_gross9_structural_clock_bundle.py`;
2. `training/preregister_gross9_structural_clock_bundle.py`;
3. `tests/test_build_gross9_structural_clock_bundle.py`;
4. `tests/test_preregister_gross9_structural_clock_bundle.py`;
5. `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py`.

It substitutes only `market_5m` and `open_interest`, binds A11, both T10 files,
both S11 files, all three M11 files, and both generated files, and preserves
the historical S9 and S10 failure blocks without erasure, merging, counter
summing, or authority transfer.

### P11, C11, D11, V11, and H11

P11 adds only:

```text
results/gross9_structural_clock_bundle_g9cb11_preregistration_2026-07-31.json
```

C11 adds only:

```text
results/gross9_structural_clock_bundle_g9cb11_access_claim_2026-07-31.json
```

D11 adds only:

```text
results/gross9_structural_clock_bundle_g9cb11_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb11_worker_capability_consumed_pass1_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb11_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb11_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb11_manifest_2026-07-31.json
```

H11 adds only:

```text
results/gross9_structural_clock_bundle_g9cb11_v11_handoff_2026-07-31.json
```

The H11 supervisor attempt sentinel remains immutable, ignored, and untracked:

```text
results/gross9_structural_clock_bundle_g9cb11_h11_supervisor_attempt_consumed_2026-07-31.json
```

Exact commands:

```text
P11: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
C11: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
D11: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
V11: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
H11: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v11-handoff
```

P11 preregisters no economics and opens no official values. C11 creates only
the claim. D11 uses two isolated workers and requires byte-identical outputs.
V11 publishes nothing and computes no economics. Operators never invoke V11
manually; the H11 supervisor invokes it once and captures its canonical stdout
in memory. H11 records `active_alpha_goal=incomplete` and routes automatically
to a fresh candidate-specific Ralplan.

Strict full-calendar CAGR/MDD, same-gross comparison, causal one-bar shifts,
funding cash accounting, and overlap thresholds remain frozen in the existing
Gross9 protocol, but G9CB-11 does not execute or evaluate them as infrastructure.

## Verification rules

### A11 and T10

1. Authenticate S10/T9 topology and exact S10 file metadata without opening an
   official raw source.
2. Reauthenticate the G9CB-10 sentinel bytes, SHA, device/inode, type, mode,
   link count, size, attempt hash, and one-shot fields.
3. Prove all four G9CB-10 generated leaves absent.
4. Prove A11 is the exact one-file child of S10 and predicts no future value.
5. Independently review A11, commit it alone, push, and prove clean
   `HEAD == @{upstream}`.
6. Construct T10 without source access; recursively validate all exact keys,
   types, literals, list orders, booleans versus integers, canonical bytes, and
   self-hash.
7. Prove T10 adds exactly sentinel plus ledger and preserves sentinel identity
   and bytes across add/commit/push.

### S11 synthetic and official gates

1. Synthetic tests cover exact market partition, two-pass metrics selection,
   missing/shifted/duplicate/conflicting grid rows, poisoned non-selected
   values, descriptor drift, publication failpoints, all three terminal
   classes, and no recovery.
2. Recursive schema mutations reject missing, additional, renamed, retyped,
   reordered, merged, swapped, collapsed, substituted, wrong-stage, early, or
   unresolved values even after enclosing hashes are recomputed.
3. AST and runtime-open guards independently prove zero candidate, comparator,
   feature, schedule, signal, return, PnL, CAGR/MDD, drawdown, and economics
   access.
4. Canonical JSON, CSV, gzip, frame-hash, create-only publication, fd closure,
   failpoint, and same-inode vectors pass in synthetic roots only.
5. Targeted and affected tests, static checks, whitespace checks, independent
   code review, and independent verifier approval pass before S11 commit.
6. S11 is the exact pushed two-file child of T10; worktree/index are clean;
   bytecode count is zero; all historical evidence authenticates; five G9CB-11
   outputs are absent; and four G9CB-10 absences persist.
7. Invoke the exact S11 command once. Never retry.
8. On success, authenticate all five immutable outputs, exact market/OI row
   counts, schemas, bounds, hashes, frame hashes, modes, links, attempt/access/
   replay/support hashes, output-set completeness, nine passes, row-touch
   formula, and zero forbidden access before M11.
9. On failure, classify exactly once under the matching terminal class and stop
   all downstream work.

### Downstream

At each M11-through-H11 stage, prove exact direct parent, exact diff, clean
status, zero bytecode, predecessor bindings, preserved terminal history, and
`HEAD == @{upstream}` before the child. M11 tracks no generated data. Q11
substitutes only the two source paths. D11 workers are isolated and
byte-identical. V11 is command-only and no-economics. H11 is the sole V11
caller and sole one-file handoff publisher.

## Stop rules

Stop immediately and grant no downstream authority if any of the following
occurs:

- A11 is not the exact one-file child of S10;
- T10 is not the exact two-file child of A11;
- the G9CB-10 sentinel changes in path, bytes, SHA, device/inode, type, mode,
  link count, size, or attempt hash;
- a permanently absent G9CB-10 leaf appears;
- any G9CB-10 rerun, repair, resume, reuse, inference, or downstream stage is
  attempted;
- T10 or A11 opens an official raw source or infers an undisclosed off-grid
  detail;
- S11 is not the exact two-file child of T10, is not clean and pushed, or any
  fresh output preexists;
- a required metrics grid timestamp is missing, duplicated, shifted, or out of
  order; a selected symbol differs; the 13-row overlap is not exact; or a
  107-row tail OI value is null, non-finite, zero, or negative;
- metrics selection depends on phase, proximity, interval membership,
  adjacency, or a non-date value; or any forbidden normalization, as-of,
  tolerance, fill, interpolation, resampling, repair, or global alignment check
  is used;
- a non-selected metrics non-date value is semantically evaluated or an
  off-grid detail is disclosed;
- publication is not private, complete, create-only, immutable, file-fsynced,
  directory-fsynced, and same-inode verified;
- any access/replay member, type, literal, order, authority stage, hash, or
  nine-addend provenance binding differs;
- any candidate, comparator, feature, signal, schedule, return, PnL, CAGR/MDD,
  drawdown, or economic value is opened or evaluated before the post-H11 fresh
  candidate authority;
- the official S11 command exits nonzero; or
- any later topology, exact diff, one-shot, clean-push, V11-supervision, or
  no-economics invariant fails.

After nonzero S11, preserve the exact terminal-class state and do not recover.
After successful H11, stop G9CB-11 infrastructure work and route to a fresh
candidate-specific Ralplan with the active alpha goal still incomplete.

## ADR

**Decision:** use exact structural-window selection under a fresh G9CB-11
identity after sealing terminal G9CB-10.

**Drivers:** exact timestamp semantics, mechanical authority separation,
bounded non-date value access, deterministic publication, and a one-shot
failure boundary.

**Rejected alternatives:** causal backward-asof changes OI semantics and
availability obligations; a checksum-verified archive rebuild requires a new
external source-rebuild authority; floor, ceil, round, or snap transforms are
ungrounded and create collision or leakage risk.

**Consequences:** all metrics timestamps are parsed, but only 120 selected rows
have symbol/OI semantics. Missing or conflicting required rows terminate the
identity. Historical S9/S10 blocks remain literal and non-authoritative.

**Follow-up:** on success, complete only the exact M11-through-H11
infrastructure path, then start a fresh candidate-specific plan. On failure,
seal the terminal class under a later standalone authority; do not retry.
