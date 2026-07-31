# Gross9 structural clock bundle G9CB-10 successor authority decision — 2026-07-31

## Authority and decision

This document is the sole A10 authority for a fresh, candidate-independent
Gross9 clock source-support identity after the terminal S9 failure.  It
authorizes one structural source successor, `G9CB-10-SOURCE-SUPPORT`, whose
market construction preserves the authenticated old market through
`2026-05-31T15:00:00Z` and appends only the exact replacement tail strictly
after that boundary and strictly before `2026-06-01T00:00:00Z`.

The decision is **old-prefix plus replacement-tail disjoint union**.  No
replacement overlap value may be normalized, validated, selected, compared,
concatenated, output-hashed, serialized, or reported.  The replacement source
is scanned once for timestamps and decoded once only at the 107 selected tail
physical positions.  All other source transformations remain the already
authorized G9CB-9 transformations.

G9CB-10 is verification infrastructure, not an alpha, signal, strategy,
portfolio, economics authority, or overlap authority.  It has no
`EA10`, `QE10`, `E10`, or `EE10` stage.  Even after a successful H10, the active
alpha-discovery goal remains incomplete and must continue under a fresh,
separately named candidate-specific Ralplan, defaulting to
`CRSB-G9CB10-336`.

## Exact A10 seal

A10 is created and ordinarily pushed before T9 begins.  Its direct parent is
the already-pushed S9 implementation commit and its complete tracked diff is
this one added file:

```text
first_parent(A10) == fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e
diff(fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e, A10) ==
  A docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md
```

The A10 worktree and index must otherwise be clean, repository bytecode count
must be zero, and the ordinary push must end with `HEAD == @{upstream}`.  A10
does not claim its own future commit, Git blob, SHA-256, or size.

## Frozen pre-existing evidence

The following facts existed before A10 and may be authenticated without
opening official source values:

| Evidence | Frozen binding |
|---|---|
| A9 authority commit | `98fe1e95708ad095cf0727363c32a89e7d03ead6` |
| A9 authority path | `docs/gross9-structural-clock-bundle-g9cb9-successor-authority-decision-2026-07-31.md` |
| A9 authority SHA-256 / size / blob / Git mode | `bda665d4b0c5c6d4ce7846f6b5d5665146d67a43ccf5f0cca4366ef18b923096` / `64733` / `266ab363adcf03c760e307ec9ef3c54dbbe92ce5` / `100644` |
| T8 terminal-evidence commit | `4188f35caa2c491f7b12f400d0815ea3a1a6144b` |
| S9 implementation commit | `fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e` |
| S9 implementation path | `training/materialize_gross9_structural_clock_g9cb9_sources.py` |
| S9 implementation SHA-256 / size / blob / Git mode | `af48353f865b2fa3568d8f1981e0a8b40f964a97f387f33915b6b6921209010f` / `51569` / `71c2532dd14914b9b4c5d15be2be4a3f975e2302` / `100644` |
| S9 synthetic-test path | `tests/test_materialize_gross9_structural_clock_g9cb9_sources.py` |
| S9 synthetic-test SHA-256 / size / blob / Git mode | `0eefaac0b76a26c8ba04a50b775f22d0b611542379cf6c29beb77be48921af5b` / `40717` / `b815da42fd90e440fb788c435698e8eafc0a1e7d` / `100644` |

The branch is exactly
`codex/gross9-structural-clock-bundle-20260731`.  S9 has direct parent T8, and
T8 has direct parent A9.

## Terminal S9 disclosure and prohibition

The official S9 command was invoked exactly once:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb9_sources
```

It returned exit status `1` with
`SourceSupportFailure: market logical prefix differs`.  The sole retained
mismatch label is `kimchi_premium`, and the already disclosed aggregate
mismatch fraction is `0.00148%`.  The traceback did emit a source-value
excerpt.  This authority deliberately does not reproduce, encode, summarize,
separately hash, or infer that excerpt or any constituent value.

S9 decoded all seven pre-existing sources and performed one market-prefix
comparison.  It performed zero generated-output readbacks and computed no
feature, signal, schedule, interval, candidate, comparator, return, PnL,
funding cash, CAGR, MDD, economic, or overlap value.  S9 may never be rerun,
repaired, resumed, or republished.  M9 and Q9 are permanently forbidden.

The sole published S9 evidence is:

```text
path        = results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json
path_type   = regular_file
mode        = 0444
link_count  = 1
size_bytes  = 3049
sha256      = aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c
attempt_hash= fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887
```

These four G9CB-9 outputs are permanently absent and may never be reconstructed
or consumed:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json
```

## Exact raw-source bindings

S10 binds exactly seven distinct regular files.  Each descriptor is retained,
authenticated before decode, identity-checked during use, and rehashed after
decode.  No symlink, hard link, path replacement, size drift, mode drift, or
byte drift is permitted.

| Name | Path | SHA-256 | Size bytes | Mode | Encoding |
|---|---|---|---:|---|---|
| `old_market` | `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz` | `a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c` | 66696659 | `0644` | gzip |
| `replacement_market` | `/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz` | `0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe` | 65420089 | `0644` | gzip |
| `funding` | `data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz` | `4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7` | 89326 | `0644` | gzip |
| `premium` | `data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz` | `b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7` | 1196481 | `0644` | gzip |
| `old_open_interest` | `/tmp/btcusdt_open_interest_5m_2020_2026.csv` | `e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31` | 19657777 | `0644` | plain CSV |
| `binance_metrics_open_interest` | `/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz` | `d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106` | 21440132 | `0644` | gzip |
| `rank7_spot_premium_5m` | `/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz` | `c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617` | 15772146 | `0644` | gzip |

The exact total authenticated opaque input size recorded before source decode
is `190272610` bytes.

## Exact source schemas and inherited transforms

The old market and selected replacement tail use this exact ordered 26-column
schema:

```json
["date","open","high","low","close","volume","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","tic","day","dxy","kimchi_premium","usdkrw","btckrw","dxy_available","kimchi_available","usdkrw_available","external_any_available","dxy_zscore","dxy_momentum","kimchi_premium_zscore","kimchi_premium_change","usdkrw_zscore","usdkrw_momentum"]
```

The materialized open-interest schema is exactly:

```json
["date","open_interest"]
```

The Binance metrics source schema is exactly:

```json
["create_time","symbol","sum_open_interest","sum_open_interest_value","count_toptrader_long_short_ratio","sum_toptrader_long_short_ratio","count_long_short_ratio","sum_taker_long_short_vol_ratio"]
```

The Rank7 projection schema is exactly:

```json
["date","spot_close","spot_rows","premium_index_1m_close","premium_rows"]
```

S10 changes only the market-partition transform and the fresh identity/bindings.
It otherwise inherits without semantic change: the exact OI common-timestamp
comparison, 13-row splice window and 107-row tail; the exact Rank7 projection,
join, latest-validity and 3000-row tail-completeness checks; causal backward
funding attachment with `12h` tolerance; causal backward premium attachment
with `2h` tolerance; deterministic CSV (`LF`, `%.17g`, empty NA, minimal
quoting), deterministic gzip (`compresslevel=9`, empty embedded filename,
`mtime=0`), frame hashing over canonical uncompressed CSV bytes, inherited
nine-row manifest construction, and private same-directory O_TMPFILE
create-only publication/readback rules.

The S10 support object has exactly these top-level keys:

```text
access_ledger
attempt_sentinel
identity
materialized_sources
raw_sources
source_manifest
source_support_commit
support_hash
validation
version
```

Relative to S9, `access` is replaced by the exact `access_ledger` contract in
this authority, and `validation.market_prefix_exact` is replaced by the exact
`validation.market_partition` object.  No compatibility alias is permitted.

## Exact S10 attempt sentinel and predecode order

The official identity and version are exactly:

```text
identity = G9CB-10-SOURCE-SUPPORT
version  = gross9_structural_clock_bundle_g9cb10_source_support_v1
```

The S10 attempt sentinel has exactly the following top-level and nested schema.
`raw_inputs` is the ordered seven-element projection of the binding table above,
with exactly `mode_octal`, `name`, `path`, `path_type`, `sha256`, and
`size_bytes` per element.

```json
{
  "attempt_hash": "<S10_ATTEMPT_HASH:sha256>",
  "branch": "codex/gross9-structural-clock-bundle-20260731",
  "expected_outputs": [
    "results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
    "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
    "configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json"
  ],
  "identity": "G9CB-10-SOURCE-SUPPORT",
  "one_shot": true,
  "raw_inputs": [
    {"mode_octal":"0644","name":"old_market","path":"data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c","size_bytes":66696659},
    {"mode_octal":"0644","name":"replacement_market","path":"/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz","path_type":"regular_file","sha256":"0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe","size_bytes":65420089},
    {"mode_octal":"0644","name":"funding","path":"data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7","size_bytes":89326},
    {"mode_octal":"0644","name":"premium","path":"data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7","size_bytes":1196481},
    {"mode_octal":"0644","name":"old_open_interest","path":"/tmp/btcusdt_open_interest_5m_2020_2026.csv","path_type":"regular_file","sha256":"e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31","size_bytes":19657777},
    {"mode_octal":"0644","name":"binance_metrics_open_interest","path":"/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106","size_bytes":21440132},
    {"mode_octal":"0644","name":"rank7_spot_premium_5m","path":"/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617","size_bytes":15772146}
  ],
  "repository_head": "<S10:git_oid>",
  "repository_parent": "<T9:git_oid>",
  "resume_allowed": false,
  "retry_allowed": false,
  "source_access_at_publication": {
    "opaque_bytes_hashed": 190272610,
    "preexisting_sources_decoded": 0,
    "source_rows_decoded": 0
  },
  "status": "source_support_attempt_consumed_before_source_decode",
  "topology": {
    "authority_commit": "<A10:git_oid>",
    "implementation_commit": "<S10:git_oid>",
    "terminal_evidence_commit": "<T9:git_oid>"
  },
  "version": "gross9_structural_clock_bundle_g9cb10_source_support_v1"
}
```

`attempt_hash` is SHA-256 over canonical JSON with only `attempt_hash` omitted.
The persisted sentinel is canonical JSON plus exactly one trailing LF, regular,
single-link, mode `0444`, and published create-only before any source decode.

The normative S10 order is:

1. validate the no-argument command and canonical repository root;
2. run the repository-bytecode gate before any other repository traversal;
3. validate branch, clean index/worktree, `HEAD == @{upstream}`, S10 direct
   parent T9, and every exact ancestor/diff binding;
4. authenticate A10 and both immutable T9 files and prove every fresh G9CB-10
   output absent;
5. retain/open/authenticate all seven source descriptors without decoding any
   source value;
6. publish and authenticate the immutable S10 attempt sentinel create-only;
7. perform eight decode passes: `old_market`,
   `replacement_market_date_scan`, `replacement_market_tail`, `funding`,
   `premium`, `old_open_interest`, `binance_metrics_open_interest`, and
   `rank7_spot_premium_5m`;
8. publish market, then OI, each create-only;
9. decode/read back each generated source exactly once and verify canonical
   logical frames;
10. publish the source manifest, then publish support last; and
11. reauthenticate all retained inputs and all five outputs, reject residue,
    rerun the bytecode gate, and only then return success.

Any nonzero official S10 exit permanently consumes the identity, regardless of
which leaves already exist.  There is no repair, resume, cleanup for reuse,
republish, or retry.



## Operative decision record

### Principles

1. Seal terminal evidence before forward motion.
2. Preserve the old market exactly; append only the authenticated tail.
3. One identity, one parent, one create-only publication path.
4. Minimize new value exposure; planning uses sealed facts only.
5. Candidate/economics work remains downstream of H10 and a fresh candidate
   authority; G9CB-10 itself never becomes a candidate.

### Decision drivers

1. Causal correctness and exact calendar coverage.
2. Minimum additional source exposure.
3. Deterministic single-parent publication and auditability.

### Options

#### Chosen — old-prefix plus replacement-tail splice

- Keep all 674,785 old rows and values unchanged.
- Select replacement rows only when
  `old_last < date < 2026-06-01T00:00:00Z`.
- Require exactly 107 unique, monotonic, five-minute-aligned rows from
  `15:05` through `23:55` and a continuous 674,892-row combined grid.
- Never compare, choose, reconcile, or disclose replacement overlap values.

Pros: smallest safe transform, preserves canonical history, no new source.
Cons: strict boundary proof; any mismatch is terminal.

#### Alternative — authenticate a new full replacement source

Pros: uniform source provenance. Cons: new discovery, broader access, more risk.
Invalid unless the splice is structurally impossible under a new authority.

#### Alternative — abandon this lane

Pros: no more source risk. Cons: fails the active objective despite a safe
structural splice. Valid only if the splice or a new full source is impossible.

## Deliberate pre-mortem

1. **Boundary drift** — a pre-boundary row is replaced or a tail row is lost.
   Mitigate with exact partition predicates, 107-row count, fixed bounds, and
   combined-grid tests.
2. **Hidden overlap dependence** — code reintroduces value comparison.
   Mitigate with a transform that slices before concatenation, AST/source tests
   forbidding old/replacement value equality, and adversarial differing-prefix
   synthetic success.
3. **Topology/publication contamination** — extra files, wrong parent, retry,
   or mutable output. Mitigate with exact-diff Git gates, immutable O_TMPFILE
   publication, failpoint tests, and HEAD/upstream checks.
4. **Infrastructure mislabeled as alpha** — a downstream step treats Gross9
   clocks as a candidate or opens economics without a candidate authority.
   Mitigate with command-only V10, the exact H10 routing ledger, explicit
   rejection of G9CB-10 economics paths, and a fresh candidate-specific Ralplan.

## Exact topology

```text
S9 -> A10 -> T9 -> S10 -> M10 -> Q10 -> P10 -> C10 -> D10 -> H10
```

- `S9 = fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e`.
- A10: exactly
  `docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md`.
  Its direct parent is S9 and its diff is exactly that one added file.
- T9: exactly two files: the existing immutable S9 attempt sentinel and one new
  canonical terminal-failure ledger. No source data file is tracked.
- S10: exactly one implementation and one synthetic test file.
- M10: exactly the S10 attempt sentinel, source manifest, and support artifact;
  generated market/OI data remain ignored and untracked.
- Q10: exactly the five approved protocol implementation/test files.
- P10: preregistration artifact only.
- C10: access-claim artifact only.
- D10: exactly the attempt sentinel, pass-1 ledger, pass-2 ledger, canonical
  result, and final manifest under the frozen worker protocol.
- V10 is not a commit. It is a command-only verification gate on clean pushed
  D10, creates no file, leaves `HEAD == D10 == @{upstream}`, and leaves the
  worktree/index clean.
- H10 is the sole post-V10 G9CB-10 commit. It records verification metadata only
  and adds exactly one immutable canonical JSON handoff ledger. Before invoking
  V10, the H10 supervisor publishes one immutable ignored/untracked attempt
  sentinel; H10 binds it, but it is never added to the index.

Exact A10 seal:

```text
first_parent(A10) == fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e
diff(S9, A10) ==
  A docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md
```

### T9 permanence

T9 tracks exactly:

1. `results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json`;
2. `results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json`.

```text
first_parent(T9) == A10
diff(A10, T9) ==
  A results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json
```

Both are Git mode `100644` and worktree mode `0444`. Every descendant through
D10, the V10 command, and H10 prove both Git blobs, file hashes, self-hashes,
and modes unchanged. T9 is
evidence-only, never source authority, and no descendant may modify, delete,
rename, replace, re-seal, or supersede either file.

### Exact M10 boundary

M10 exists only after S10 returns exit status zero and all five S10 outputs pass
final authentication.

`tracked_set(M10)` is exactly:

1. `results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json`;
2. `configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json`;
3. `results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json`.

`generated_untracked_set(M10)` is exactly:

1. `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz`;
2. `data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz`.

The tracked files have Git mode `100644` and worktree mode `0444`. The data
files are regular, single-link, worktree mode `0444`, ignored, absent from the
index, and bound in support by path, SHA-256, size, schema, rows, bounds, and
frame hash.

### Exact S10/Q10/P10/C10/D10 diffs

```text
first_parent(S10) == T9
diff(T9, S10) ==
  A training/materialize_gross9_structural_clock_g9cb10_sources.py
  A tests/test_materialize_gross9_structural_clock_g9cb10_sources.py

first_parent(M10) == S10
diff(S10, M10) ==
  A results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json
  A configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json

first_parent(Q10) == M10
diff(M10, Q10) ==
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py

first_parent(P10) == Q10
diff(Q10, P10) ==
  A results/gross9_structural_clock_bundle_g9cb10_preregistration_2026-07-31.json

first_parent(C10) == P10
diff(P10, C10) ==
  A results/gross9_structural_clock_bundle_g9cb10_access_claim_2026-07-31.json

first_parent(D10) == C10
diff(C10, D10) ==
  A results/gross9_structural_clock_bundle_g9cb10_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb10_worker_capability_consumed_pass1_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb10_worker_capability_consumed_pass2_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb10_2026-07-31.csv.gz
  A results/gross9_structural_clock_bundle_g9cb10_manifest_2026-07-31.json
```

Each stage is committed and ordinarily pushed before its child starts. Every
stage preflight requires `HEAD == @{upstream}`, the exact direct parent/diff,
and a clean worktree/index except for the create-only outputs of the command
that is about to be sealed.

### Exact Q10 consumable bindings

Q10 binds exactly A10, both immutable T9 files, both S10 implementation files,
all three M10 tracked files, and both M10-bound generated files. Tracked-file
bindings use exactly `path`, `sha256`, `size_bytes`, `git_blob`, `git_mode`, and
`seal_commit`. Generated-file bindings use exactly `path`, `sha256`,
`size_bytes`, `filesystem_mode_octal`, `frame_hash`, `rows`, `first_timestamp`,
`last_timestamp`, and `schema`.

Q10 substitutes only `market_5m` and `open_interest` with M10-bound G9CB-10
paths. Other manifest rows remain inherited. S9/T9 are authentication-only;
Q10 may not decode G9CB-9 output, direct old/replacement inputs, or quarantined
S10 output.

## S9 terminal facts to seal

- Official S9 command invocation count: exactly one.
- Exit status: `1`; retry/resume are permanently false.
- Failure class/message: `SourceSupportFailure: market logical prefix differs`.
- The sole disclosed mismatch label is `kimchi_premium`; no rows or values are
  restated or inferred.
- S9 attempt sentinel: regular, mode `0444`, link count one, size 3049,
  SHA-256 `aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c`,
  attempt hash `fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887`.
- Other four G9CB-9 outputs remain absent.
- S9 decoded seven pre-existing sources and performed one market-prefix
  comparison. It produced zero generated readbacks, features, schedules,
  candidates, comparator rows, economics, CAGR/MDD, or overlap.
- M9 and Q9 are permanently absent and forbidden.

## A10 requirements

A10 must freeze:

1. exact terminal facts and T9 two-file seal;
2. exact G9CB-10 names, paths, schemas, typed runtime metavariables, counters,
   and commit topology;
3. no-overlap-comparison transform;
4. S10 one-shot command and terminal failure rule;
5. exact historical S9-access disclosure propagated into S10 support, Q10
   preregistration, P10 claim, and production evidence;
6. unchanged OI splice, Rank7 projection, funding/premium causal attachment,
   serialization, frame hash, publication, and Q/P/C/D/V rules from A9, except
   fresh G9CB-10 bindings and the market transform; and
7. the exact H10 verification handoff contract and the prohibition on treating
   G9CB-10 as a candidate or economics authority.

A10 freezes only facts that pre-exist it: A9/T8/S9 commits, the S9 sentinel,
the seven raw bindings, schemas, constants, paths, commands, and future artifact
schemas. It must not guess any T9/S10/M10-produced hash, size, frame hash, Git
blob, or commit.

The following are typed runtime metavariables, not A10 facts:

```text
<S10_ATTEMPT_HASH:sha256>           <S10_ATTEMPT_SHA256:sha256>
<S10_ATTEMPT_SIZE_BYTES:int>        <S10_MARKET_SHA256:sha256>
<S10_MARKET_SIZE_BYTES:int>         <S10_MARKET_FRAME_HASH:sha256>
<S10_OI_SHA256:sha256>              <S10_OI_SIZE_BYTES:int>
<S10_OI_FRAME_HASH:sha256>          <S10_MANIFEST_SHA256:sha256>
<S10_MANIFEST_SIZE_BYTES:int>       <S10_SUPPORT_SHA256:sha256>
<S10_SUPPORT_SIZE_BYTES:int>        <S10_SUPPORT_HASH:sha256>
```

`sha256` is exactly 64 lowercase hexadecimal characters; `int` is a
non-negative JSON integer. S10 learns each value once from create-only
publication/authenticated readback. M10 seals them in committed bytes. Q10
copies them literally and may authenticate equality but may not guess,
substitute, or treat recomputation as authority.

## S10 exact market-partition invariant

Freeze:

- `boundary = 2026-05-31T15:00:00Z`;
- `domain_end = 2026-06-01T00:00:00Z`;
- `old_prefix = old_market[date <= boundary]`;
- `tail = replacement_market[(date > boundary) & (date < domain_end)]`;
- `output = old_prefix ⊎ tail` (normative disjoint union).

Mechanically prove:

1. every old-prefix timestamp is `<= boundary`;
2. every tail timestamp is `> boundary` and `< domain_end`;
3. `set(old_prefix.date) ∩ set(tail.date) == ∅`;
4. output timestamps equal the union of the two sets;
5. `len(output) == len(old_prefix) + len(tail) == 674892`;
6. `output[:674785]` is row/value-exact with old-prefix;
7. `output[674785:]` is row/value-exact with tail;
8. tail count is 107, first `15:05`, last `23:55`, with one continuous
   five-minute grid.

Replacement selection is timestamp-first and uses two descriptor-duplicate
passes after sentinel publication: one ordered `date`-only scan to identify
physical tail rows, then one exact-schema decode restricted to those 107 row
positions. Replacement rows at or before the boundary may be framed and skipped
but no non-date field from them may be normalized, validated, selected,
compared, concatenated, hashed as output, serialized, or reported. The complete
replacement descriptor is still authenticated and postdecode-rehashed. This is
seven distinct raw files and eight decode passes (replacement date scan plus
replacement tail decode).

Support records exactly:

```json
{"boundary_inclusive_old":"2026-05-31T15:00:00Z","disjoint_union":true,"domain_end_exclusive":"2026-06-01T00:00:00Z","old_prefix_rows":674785,"output_rows":674892,"overlap_value_comparison_count":0,"replacement_prefix_non_date_values_semantically_evaluated":0,"replacement_prefix_rows_selected":0,"replacement_tail_rows":107,"timestamp_intersection_rows":0}
```

## S10 outputs

Fresh names use `g9cb10` and never reuse G9CB-9 leaves:

1. `results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json`;
2. `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz`;
3. `data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz`;
4. `configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json`;
5. `results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json`.

All use the existing canonical JSON/CSV/gzip/frame-hash and O_TMPFILE
create-only publication rules. The official command is invoked once only after
clean pushed S10; any failure terminates G9CB-10 without repair or retry.

## Failed-S10 quarantine

Any nonzero official S10 exit terminally consumes G9CB-10 even if all five
leaves were published. Retry, repair, resume, cleanup for reuse, M10, and Q10 are
forbidden. Present leaves remain immutable, non-authoritative, and unusable
downstream. They are never removed, overwritten, renamed, republished, or
tracked by M10. A later standalone authority may seal only the attempt sentinel
and a terminal ledger; it must never track a partial market, OI, manifest, or
support file. Every expected leaf is classified with exact keys `path`, `state`
(`absent` or `published_non_authoritative`), `sha256`, `size_bytes`,
`filesystem_mode_octal`, `source_authoritative:false`, and
`downstream_consumable:false`.

## Immutable access-ledger and replay contract

T9 publishes
`results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json`.
The traceback emitted a source-value excerpt. The ledger records that disclosure
fact but does not reproduce, encode, summarize, separately hash, or infer its
contents. The only retained mismatch facts are the column label
`kimchi_premium` and the already disclosed aggregate fraction `0.00148%`.

The terminal ledger has exactly this schema and no additional member:

```json
{
  "schema_version": 1,
  "ledger_kind": "gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_v1",
  "identity": "G9CB-9-SOURCE-SUPPORT",
  "status": "terminal_market_prefix_mismatch",
  "authority": {
    "commit": "98fe1e95708ad095cf0727363c32a89e7d03ead6",
    "path": "docs/gross9-structural-clock-bundle-g9cb9-successor-authority-decision-2026-07-31.md"
  },
  "seal_authority": {
    "commit": "<A10:git_oid>",
    "path": "docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md"
  },
  "implementation": {
    "commit": "fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e",
    "parent_commit": "4188f35caa2c491f7b12f400d0815ea3a1a6144b",
    "files": [
      "training/materialize_gross9_structural_clock_g9cb9_sources.py",
      "tests/test_materialize_gross9_structural_clock_g9cb9_sources.py"
    ]
  },
  "attempt_sentinel": {
    "path": "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
    "path_type": "regular_file",
    "filesystem_mode_octal": "0444",
    "link_count": 1,
    "size_bytes": 3049,
    "sha256": "aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c",
    "attempt_hash": "fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887"
  },
  "execution": {
    "official_command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb9_sources",
    "official_invocation_count": 1,
    "exit_status": 1,
    "retry_allowed": false,
    "resume_allowed": false
  },
  "failure": {
    "class": "SourceSupportFailure",
    "message": "market logical prefix differs",
    "disclosed_mismatch_columns": ["kimchi_premium"],
    "mismatch_fraction_percent": "0.00148",
    "traceback_value_excerpt_emitted": true,
    "traceback_value_excerpt_restated_in_ledger": false
  },
  "access": {
    "raw_file_count": 7,
    "decode_pass_count": 7,
    "decoded_preexisting_sources": [
      "old_market",
      "replacement_market",
      "funding",
      "premium",
      "old_open_interest",
      "binance_metrics_open_interest",
      "rank7_spot_premium_5m"
    ],
    "market_prefix_comparison_count": 1,
    "generated_readback_decode_count": 0,
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "model_history_or_rex_values_opened": 0,
    "pre2025_anchor_value_rows_opened": 0,
    "feature_signal_schedule_or_interval_values_computed": 0,
    "economic_or_overlap_values_computed": 0
  },
  "output_state": {
    "published_terminal_evidence": [
      "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json"
    ],
    "permanently_absent_outputs": [
      "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
      "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
      "configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
      "results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json"
    ],
    "generated_source_readback_count": 0,
    "m9_permitted": false,
    "q9_permitted": false
  },
  "terminal_failure_hash": "<T9_TERMINAL_FAILURE_HASH:sha256>"
}
```

Canonicalization is UTF-8 JSON with sorted keys, separators `(",", ":")`,
`ensure_ascii=false`, `allow_nan=false`, duplicate-key rejection, and no
trailing LF for hashing. `terminal_failure_hash` is SHA-256 over the canonical
object with only that member omitted. The persisted file is the complete
canonical object plus exactly one trailing LF.

T9 learns and seals the following runtime values; A10 does not predict them:

```text
<T9_LEDGER_SHA256:sha256>             <T9_LEDGER_SIZE_BYTES:int>
<T9_TERMINAL_FAILURE_HASH:sha256>     <T9_LEDGER_GIT_BLOB:git_oid>
<T9_SENTINEL_GIT_BLOB:git_oid>        <T9:git_oid>
```

The ledger cannot contain its own future T9 commit or Git blob. S10 and later
stages bind those facts externally.

### Exact G9CB-10 access ledger

Every G9CB-10 access-bearing artifact contains an `access_ledger` with exactly
the following nested key schema:

```json
{
  "schema_version": 1,
  "ledger_kind": "gross9_structural_clock_bundle_g9cb10_access_v1",
  "historical_s9": {
    "identity": "G9CB-9-SOURCE-SUPPORT",
    "attempt_sentinel": {
      "path": "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
      "sha256": "aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c",
      "size_bytes": 3049,
      "attempt_hash": "fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887",
      "git_blob": "<T9_SENTINEL_GIT_BLOB:git_oid>",
      "git_mode": "100644",
      "seal_commit": "<T9:git_oid>"
    },
    "terminal_failure_ledger": {
      "path": "results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json",
      "sha256": "<T9_LEDGER_SHA256:sha256>",
      "size_bytes": "<T9_LEDGER_SIZE_BYTES:int>",
      "terminal_failure_hash": "<T9_TERMINAL_FAILURE_HASH:sha256>",
      "git_blob": "<T9_LEDGER_GIT_BLOB:git_oid>",
      "git_mode": "100644",
      "seal_commit": "<T9:git_oid>"
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
  },
  "current_s10": {
    "identity": "G9CB-10-SOURCE-SUPPORT",
    "attempt_sentinel": {
      "path": "results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json",
      "sha256": "<S10_ATTEMPT_SHA256:sha256>",
      "size_bytes": "<S10_ATTEMPT_SIZE_BYTES:int>",
      "attempt_hash": "<S10_ATTEMPT_HASH:sha256>"
    },
    "official_invocation_count": 1,
    "raw_file_count": 7,
    "decode_pass_count": 8,
    "decode_passes": [
      "old_market",
      "replacement_market_date_scan",
      "replacement_market_tail",
      "funding",
      "premium",
      "old_open_interest",
      "binance_metrics_open_interest",
      "rank7_spot_premium_5m"
    ],
    "replacement_date_scan_count": 1,
    "replacement_tail_decode_count": 1,
    "generated_readback_decode_count": 2,
    "replacement_prefix_rows_selected": 0,
    "replacement_prefix_non_date_values_semantically_evaluated": 0,
    "replacement_tail_rows_selected": 107,
    "overlap_value_comparison_count": 0,
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "feature_signal_schedule_or_interval_values_computed": 0,
    "economic_or_overlap_values_computed": 0
  },
  "process_local": {
    "stage": "<PROCESS_LOCAL_STAGE:string>",
    "slot": "<PROCESS_LOCAL_SLOT:int>",
    "invocation_count": "<PROCESS_LOCAL_INVOCATION_COUNT:int>",
    "source_files_opened": "<PROCESS_LOCAL_SOURCE_FILES_OPENED:int>",
    "source_value_rows_opened": "<PROCESS_LOCAL_SOURCE_VALUE_ROWS_OPENED:int>",
    "candidate_rows_opened": "<PROCESS_LOCAL_CANDIDATE_ROWS_OPENED:int>",
    "comparator_clock_rows_opened": "<PROCESS_LOCAL_COMPARATOR_CLOCK_ROWS_OPENED:int>",
    "model_files_opened": "<PROCESS_LOCAL_MODEL_FILES_OPENED:int>",
    "runtime_modules_imported": "<PROCESS_LOCAL_RUNTIME_MODULES_IMPORTED:int>",
    "pre2025_anchor_value_rows_opened": "<PROCESS_LOCAL_PRE2025_ANCHOR_VALUE_ROWS_OPENED:int>",
    "feature_signal_schedule_or_interval_values_computed": "<PROCESS_LOCAL_FEATURE_SIGNAL_SCHEDULE_OR_INTERVAL_VALUES_COMPUTED:int>",
    "portfolio_economic_values_computed": "<PROCESS_LOCAL_PORTFOLIO_ECONOMIC_VALUES_COMPUTED:int>",
    "economic_or_overlap_values_computed": "<PROCESS_LOCAL_ECONOMIC_OR_OVERLAP_VALUES_COMPUTED:int>"
  },
  "replay_guard": {
    "prior_identity": "G9CB-9-SOURCE-SUPPORT",
    "current_identity": "G9CB-10-SOURCE-SUPPORT",
    "prior_attempt_hash": "fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887",
    "current_attempt_hash": "<S10_ATTEMPT_HASH:sha256>",
    "prior_terminal_failure_hash": "<T9_TERMINAL_FAILURE_HASH:sha256>",
    "authority_commit": "<A10:git_oid>",
    "authority_path": "docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md",
    "terminal_evidence_commit": "<T9:git_oid>",
    "terminal_failure_ledger_path": "results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json",
    "implementation_commit": "<S10:git_oid>",
    "implementation_paths": [
      "training/materialize_gross9_structural_clock_g9cb10_sources.py",
      "tests/test_materialize_gross9_structural_clock_g9cb10_sources.py"
    ],
    "prior_expected_output_paths": [
      "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
      "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
      "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
      "configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
      "results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json"
    ],
    "current_expected_output_paths": [
      "results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json",
      "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
      "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
      "configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json",
      "results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json"
    ],
    "replay_guard_hash": "<REPLAY_GUARD_HASH:sha256>"
  },
  "access_ledger_hash": "<ACCESS_LEDGER_HASH:sha256>"
}
```

Angle-bracket `string`/`int` metavariables are schema notation only. `string`
is emitted as a JSON string and `int` is emitted as a JSON integer, never as a
quoted placeholder. `replay_guard_hash` omits only itself from its
canonical object; `access_ledger_hash` omits only itself from the complete
ledger. Prior/current identities and attempt hashes must differ, and the two
expected-output path sets must have an empty intersection.

After M10, descendants copy `historical_s9`, `current_s10`, and `replay_guard`
literally. Only `process_local` may change before recomputing
`access_ledger_hash`; blocks or counters may never be summed, substituted,
swapped, collapsed, or treated as authority for another invocation. Historical
disclosure grants no authority to rerun S9 and contains no raw rows or values.

### Exact process-local stage table

`process_local` always has the exact keys shown above. All slots, invocation
counts, and counters are non-negative JSON integers.

| Stage | Exact creation and seal rule |
|---|---|
| A10 | No access ledger; schema and pre-existing bindings only. |
| T9 | Creates the standalone historical terminal ledger without reopening an official source; T9 seals it. |
| S10 | Successful sole run emits `stage="S10"`, `slot=0`, `invocation_count=1`, `source_files_opened=7`, zero candidate/comparator/model/runtime-module/pre-2025-anchor/feature/economic counters, and the exact runtime `source_value_rows_opened`; M10 seals it. |
| M10 | Creates no counters; seals the S10 ledger and all learned S10 bindings. |
| Q10 | Persists no process ledger and opens no official values; copies M10 literals into the five protocol files. |
| P10 | Emits `stage="P10"`, `slot=0`, `invocation_count=1`; all ten access/computation counters are zero; P10 seals it. |
| C10 | Emits `stage="C10"`, `slot=0`, `invocation_count=1`; all ten access/computation counters are zero; C10 seals it. |
| D10 pass 1 | Emits `stage="D10_PASS1"`, `slot=1`, `invocation_count=1`; each runtime counter is recorded once as a typed `<D10_PASS1_FIELD:int>` and sealed by the pass-1 worker ledger. |
| D10 pass 2 | Emits `stage="D10_PASS2"`, `slot=2`, `invocation_count=1`; each runtime counter is recorded once as a typed `<D10_PASS2_FIELD:int>` and sealed by the pass-2 worker ledger. |
| V10 | Emits no ledger, file, or commit; verification only. |
| H10 | Emits only the canonical V10 handoff ledger below; opens no official source, candidate, comparator, return, or economic value and creates no `access_ledger`. |

## Exact canonical commands

```text
S10: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb10_sources
P10: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
C10: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
D10: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
V10: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
H10: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v10-handoff
```

S10 is one-shot. H10 has exactly one consumed attempt: no-effect preparation
before the supervisor-attempt sentinel link may be repeated only while both
canonical leaves remain absent and V10 invocation count remains zero; after
that link, no retry or resume exists. P10/C10/D10/V10 retain A9's exact command
strings; Q10 changes the bound G9CB-10 identity/paths and adds the exact H10 supervisor
entry point. Operators do not invoke V10 separately. The sole consumed H10
command performs all preflight capture, invokes the exact V10 command once as a
child process, captures its stdout as raw in-memory bytes, recaptures post-V10
evidence, and only then constructs and publishes H10. No named evidence file,
environment-value transport, shell substitution, or manually transcribed JSON
is permitted.

## V10 and H10 handoff boundary

The one-shot H10 supervisor first proves both its attempt-sentinel leaf and its
canonical H10 leaf absent while `HEAD == D10 == @{upstream}`, the worktree and
index are clean, and the repository bytecode count is zero. It then publishes
and authenticates the attempt sentinel create-only before capturing the V10
before-state. Successful sentinel linking is the consumed-attempt boundary.
The immutable ignored sentinel is present for every subsequent step and makes
a fresh supervisor process reject before V10. In the same original process the
supervisor invokes V10 exactly once using the exact V10 argv, cwd, and
environment above, with stdout connected to a raw byte pipe and stderr kept
separate. V10 creates no file, access ledger, artifact, or commit; it verifies
committed D10 bytes, bindings, modes, predecessor evidence, residue, the exact
authorized ignored sentinel, and source materialization and performs no
economics. After success, `HEAD` remains D10. Any existing attempt-sentinel or
H10 leaf at supervisor start, even byte-valid and immutable, causes terminal
rejection before V10 is invoked.

V10 prints exactly one duplicate-key-free canonical JSON object followed by one
LF. Its payload has exactly these keys, learned from the successful run:

```json
{
  "claim_commit": "<C10:git_oid>",
  "claim_hash": "<C10_CLAIM_HASH:sha256>",
  "csv_gzip_sha256": "<D10_CSV_GZIP_SHA256:sha256>",
  "final_manifest_hash": "<D10_FINAL_MANIFEST_HASH:sha256>",
  "head": "<D10:git_oid>",
  "identity": "G9CB-10",
  "interval_count": "<V10_INTERVAL_COUNT:int>",
  "preregistration_manifest_hash": "<P10_MANIFEST_HASH:sha256>",
  "preregistration_seal_commit": "<P10:git_oid>",
  "protocol_implementation_commit": "<Q10:git_oid>",
  "protocol_version": "gross9_structural_clock_bundle_g9cb10_v1",
  "publication_commit": "<D10:git_oid>",
  "sentinel_manifest_hash": "<D10_SENTINEL_MANIFEST_HASH:sha256>"
}
```

The V10 payload has exactly thirteen keys. `claim_hash` is required in both the
printed V10 payload and `H10.verification.stdout_payload`; all twelve other
members and their behavior remain unchanged. The payload hash, exact stdout
hash, byte count, and H10 handoff hash cover the complete thirteen-key payload.

H10 is authorized only after that sole V10 invocation returns status zero.
Before V10, capture `HEAD`, upstream, clean index/worktree, repository bytecode
count, and the duplicate-key-free canonical `git status --porcelain=v1 -z
--untracked-files=all` byte hash. After V10, recapture the same facts. H10 adds
exactly:

```text
results/gross9_structural_clock_bundle_g9cb10_v10_handoff_2026-07-31.json

first_parent(H10) == D10
diff(D10, H10) ==
  A results/gross9_structural_clock_bundle_g9cb10_v10_handoff_2026-07-31.json
```

The M10-bound generated ignored/untracked set remains exactly the materialized
market and OI paths frozen above. The exact **additional H10-supervisor**
ignored/untracked delta throughout V10, H10 publication, H10 commit, and H10
push is:

```text
results/gross9_structural_clock_bundle_g9cb10_h10_supervisor_attempt_consumed_2026-07-31.json
```

Therefore the complete G9CB-10 authority-bound ignored/untracked set at V10 and
H10 is exactly these three paths, with no fourth member:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz
results/gross9_structural_clock_bundle_g9cb10_h10_supervisor_attempt_consumed_2026-07-31.json
```

The supervisor-attempt sentinel has exactly this schema and no additional
member:

```json
{
  "attempt_hash": "<H10_SUPERVISOR_ATTEMPT_HASH:sha256>",
  "branch": "codex/gross9-structural-clock-bundle-20260731",
  "expected_handoff_path": "results/gross9_structural_clock_bundle_g9cb10_v10_handoff_2026-07-31.json",
  "identity": "G9CB-10-H10-SUPERVISOR",
  "one_shot": true,
  "repository_head": "<D10:git_oid>",
  "repository_parent": "<C10:git_oid>",
  "resume_allowed": false,
  "retry_allowed": false,
  "source_access_at_publication": {
    "candidate_or_economic_values_opened": 0,
    "official_sources_opened": 0,
    "v10_invocation_count": 0
  },
  "status": "h10_supervisor_attempt_consumed_before_v10",
  "supervisor_command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v10-handoff",
  "v10_command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication",
  "version": "gross9_structural_clock_bundle_g9cb10_h10_supervisor_attempt_v1"
}
```

`attempt_hash` omits only itself from the canonical object. The persisted
sentinel is canonical JSON plus one LF, regular, single-link, mode `0444`,
ignored and absent from the index. Its create-only publisher uses the same
private same-directory O_TMPFILE/procfd-link/file-fsync/directory-fsync/
same-inode protocol as H10. A failure before its link leaves it absent and
invokes no V10; this is the only retryable no-effect preparation region. A
successful link permanently consumes the only H10 supervisor attempt,
regardless of every later outcome. It is never removed, rewritten,
renamed, relinked, chmodded, tracked, or reused.

The H10 ledger has exactly this schema and no additional member:

```json
{
  "schema_version": 1,
  "ledger_kind": "gross9_structural_clock_bundle_g9cb10_v10_handoff_v1",
  "identity": "G9CB-10",
  "stage": "H10",
  "verification": {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication",
    "official_invocation_count": 1,
    "exit_status": 0,
    "stdout_payload": {
      "claim_commit": "<C10:git_oid>",
      "claim_hash": "<C10_CLAIM_HASH:sha256>",
      "csv_gzip_sha256": "<D10_CSV_GZIP_SHA256:sha256>",
      "final_manifest_hash": "<D10_FINAL_MANIFEST_HASH:sha256>",
      "head": "<D10:git_oid>",
      "identity": "G9CB-10",
      "interval_count": "<V10_INTERVAL_COUNT:int>",
      "preregistration_manifest_hash": "<P10_MANIFEST_HASH:sha256>",
      "preregistration_seal_commit": "<P10:git_oid>",
      "protocol_implementation_commit": "<Q10:git_oid>",
      "protocol_version": "gross9_structural_clock_bundle_g9cb10_v1",
      "publication_commit": "<D10:git_oid>",
      "sentinel_manifest_hash": "<D10_SENTINEL_MANIFEST_HASH:sha256>"
    },
    "stdout_payload_sha256": "<V10_STDOUT_PAYLOAD_SHA256:sha256>",
    "stdout_sha256": "<V10_STDOUT_WITH_LF_SHA256:sha256>",
    "stdout_size_bytes": "<V10_STDOUT_SIZE_BYTES:int>",
    "head_before": "<D10:git_oid>",
    "head_after": "<D10:git_oid>",
    "upstream_before": "<D10:git_oid>",
    "upstream_after": "<D10:git_oid>",
    "head_equals_upstream_before": true,
    "head_equals_upstream_after": true,
    "worktree_clean_before": true,
    "worktree_clean_after": true,
    "index_clean_before": true,
    "index_clean_after": true,
    "git_status_porcelain_z_sha256_before": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "git_status_porcelain_z_sha256_after": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "repository_bytecode_count_before": 0,
    "repository_bytecode_count_after": 0,
    "filesystem_publications_created": 0,
    "published_paths": []
  },
  "publication": {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v10-handoff",
    "official_invocation_count": 1,
    "attempt_sentinel": {
      "path": "results/gross9_structural_clock_bundle_g9cb10_h10_supervisor_attempt_consumed_2026-07-31.json",
      "path_type": "regular_file",
      "filesystem_mode_octal": "0444",
      "link_count": 1,
      "sha256": "<H10_SUPERVISOR_ATTEMPT_SHA256:sha256>",
      "size_bytes": "<H10_SUPERVISOR_ATTEMPT_SIZE_BYTES:int>",
      "attempt_hash": "<H10_SUPERVISOR_ATTEMPT_HASH:sha256>"
    },
    "v10_capture_channel": "in_memory_stdout_pipe",
    "v10_child_command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication",
    "v10_child_invocation_count": 1,
    "link_source": "/proc/self/fd/{tmpfd}",
    "linkat_flag": "AT_SYMLINK_FOLLOW",
    "cross_process_recovery_allowed": false
  },
  "routing": {
    "active_alpha_goal": "incomplete",
    "g9cb10_role": "candidate_independent_gross9_clock_infrastructure",
    "next_automatic_workflow": "fresh_candidate_specific_ralplan",
    "default_successor_candidate": "CRSB-G9CB10-336",
    "successor_mechanism_doc": "docs/circle-reserve-schema-bridge-mechanism-decision-2026-07-30.md",
    "successor_terminal_predecessor_doc": "docs/circle-reserve-schema-bridge-terminal-gross9-authority-failure-2026-07-31.md"
  },
  "handoff_hash": "<H10_HANDOFF_HASH:sha256>"
}
```

Angle-bracket integer metavariables become JSON integers. The payload hash is
over canonical `stdout_payload` without LF; `stdout_sha256` is over the exact
printed payload plus its one LF. `handoff_hash` omits only itself from the
complete canonical ledger. The persisted H10 file is canonical JSON plus one
LF, regular, single-link, worktree mode `0444`, and Git mode `100644`. H10 does
not contain its own future commit or Git blob; the next candidate authority
binds those externally.

Q10 implements the H10 supervisor, canonical constructor, validator, and
publication primitive inside
`training/build_gross9_structural_clock_bundle.py` and covers them in
`tests/test_build_gross9_structural_clock_bundle.py`. V10 itself and production
workers never invoke the publisher. The supervisor is the sole publisher and
sole evidence channel: it captures the one successful V10 stdout and
before/after evidence in memory and passes those exact bytes directly to the
constructor. No H10 construction, authentication, failure-seal, or commit path
may invoke V10 again. There is no H10 recovery entry point.

The supervisor-attempt sentinel is published and authenticated before the V10
subprocess is launched. V10 and every later H10 step reopen it read-only and
authenticate its path type, mode, link count, canonical bytes, SHA-256,
`attempt_hash`, D10/C10 topology, commands, and zero-at-publication counters.
Its ignored status is exact and is the only authorized output-directory delta
before the H10 handoff leaf is linked.

The direct `linkat(tmpfd, "", ..., AT_EMPTY_PATH)` form is forbidden. A
pre-A10 unprivileged runtime probe under uid/euid `1000`, effective capability
mask zero, and the active Linux filesystem returned `ENOENT` for that form.
The same private O_TMPFILE probe succeeded with
`linkat(AT_FDCWD, "/proc/self/fd/<tmpfd>", dirfd, name,
AT_SYMLINK_FOLLOW)`, yielding the same device/inode, mode `0444`, and link
count one. H10 therefore freezes the already-audited unprivileged procfd form,
not a capability-dependent `AT_EMPTY_PATH` form.

Let `dirfd` identify `results/`, and let `name` be
`gross9_structural_clock_bundle_g9cb10_v10_handoff_2026-07-31.json`. The
create-only H10 publisher executes exactly:

1. construct the complete canonical H10 JSON plus one LF in memory, recompute
   `stdout_payload_sha256`, `stdout_sha256`, `stdout_size_bytes`, and
   `handoff_hash`, and reject any mismatch;
2. create a private same-directory inode with
   `openat(dirfd, ".", O_TMPFILE|O_RDWR|O_CLOEXEC, 0600)`;
3. write all final bytes while handling short writes, then `pread` them back
   and verify exact size/bytes, duplicate-key-free canonical round-trip, every
   embedded hash, and the handoff self-hash;
4. `fchmod(tmpfd, 0444)`, verify exact mode `0444`, and permit no later write or
   truncate through `tmpfd`;
5. `fsync(tmpfd)`;
6. atomically create the final leaf with
   `linkat(AT_FDCWD, "/proc/self/fd/<tmpfd>", dirfd, name,
   AT_SYMLINK_FOLLOW)`; `EEXIST` is never repaired, replaced, removed, renamed
   over, or retried;
7. `fsync(dirfd)`; and
8. open the linked leaf only with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`, prove it is
   the same `st_dev`/`st_ino` as `tmpfd`, regular, single-link, mode `0444`, and
   byte/hash/canonical-form exact, then close every descriptor.

The output path is never opened writable and is never created through a named
temporary file, `creat`, rename, replacement, truncation, or post-link chmod.
At every externally observable stop it is therefore absent or the complete
immutable H10 ledger; a partial or writable canonical leaf is impossible.

Successful attempt-sentinel `linkat` is the irreversible supervisor-attempt
boundary. Any later supervisor invocation sees the sentinel and terminates
before V10, even when the H10 leaf is absent. Successful H10 `linkat` is the
physical handoff publication boundary, but authority is granted only if the
same original supervisor invocation retains `tmpfd`, successfully completes
directory fsync and linked-identity verification, closes all descriptors, and
returns status zero. Any failure or failpoint before the H10 link leaves the
handoff leaf absent but the consumed sentinel present and terminally forbids
another H10/V10 attempt and all candidate continuation. Any failure, failpoint,
or process termination after the H10 link leaves the complete mode-`0444` leaf
in place but permanently classifies it as published non-authoritative evidence.
It may be inspected read-only only under a later standalone failure-seal
authority; it may not be staged as H10, committed, pushed, consumed, rebuilt,
rewritten, relinked, replaced, deleted, or chmodded. Cross-process recovery is
forbidden. An existing attempt sentinel or handoff leaf at command start is
never accepted as recovery and never causes a V10 invocation. Only a
status-zero original H10 supervisor with the same sentinel binding permits Git
staging, the exact one-file H10 commit, and ordinary push. The ignored sentinel
remains immutable and untracked. V10 is never rerun.

V10 is terminal on any nonzero exit: no H10, rerun, repair, economics, or
candidate consumption is permitted. A later standalone authority may seal the
read-only verification failure, but it may not reuse this V10 identity.

H10 is not an economics authority and opens no official source, candidate,
comparator, return, PnL, funding cash, CAGR, MDD, economic, or overlap value.
There is no EA10/QE10/E10/EE10 in G9CB-10. The active alpha goal remains
incomplete after H10 and automatically enters a fresh candidate-specific
Ralplan. The default candidate is the separately named
`CRSB-G9CB10-336`, whose plan must bind H10 and the already-frozen CRSB
mechanism without treating the retired CRSB-336 identity as reusable.

## Acceptance criteria

1. A10 is independently reviewed and committed/pushed as the sole child diff.
2. T9 tracks exactly sentinel + ledger, both Git mode `100644`, worktree `0444`.
3. S10 implementation/test receive independent review; all synthetic and
   affected tests pass with no repository bytecode.
4. Sole S10 run either succeeds exactly once or permanently closes the identity.
5. On success, M10 seals exactly three metadata artifacts and no data files.
6. Q10/P10/C10/D10 and command-only V10 preserve causal/full-calendar/
   same-gross infrastructure rules and exact topology.
7. H10 has the exact one-file diff, carries the complete thirteen-key V10
   payload plus the exact H10 publication block and supervisor-attempt binding,
   and follows the frozen one-shot supervisor, pre-V10 create-only ignored
   sentinel, private O_TMPFILE, canonical verification,
   `fchmod(0444)`, file-fsync, procfd `AT_SYMLINK_FOLLOW` create-only link,
   directory-fsync, and same-process linked-identity protocol. Every failure
   leaves the leaf absent before link or complete, immutable, non-authoritative,
   and uncommittable after link.
8. The active alpha goal remains incomplete after H10. Candidate economics may
   begin only under a separately named successor and a fresh candidate-specific
   Ralplan, defaulting to `CRSB-G9CB10-336` and binding the pre-existing CRSB
   mechanism and H10 without reusing retired CRSB-336.

## Stop rules

Stop the active branch if S9 is rerun/modified, M9/Q9 appears, overlap value
comparison is needed, the tail is not exactly 107 rows, any official S10 or V10
failure occurs, topology/diff cleanliness breaks, publication cannot remain
create-only, H10 cannot be sealed after successful V10, or further planning
requires source-value inspection. After S10 failure, all present partial
outputs are quarantined permanently and no downstream stage may authenticate
or consume them. After V10 failure, H10 and candidate consumption are forbidden.
After the supervisor-attempt sentinel is linked, any V10 or H10 failure
permanently forbids H10 and candidate continuation. The sentinel mechanically
blocks a fresh V10 invocation. A post-link handoff failure leaves immutable
non-authoritative evidence that cannot be recovered, committed, pushed, or
consumed. V10 and H10 are never retried.

## ADR

**Decision:** choose the old-prefix plus replacement-tail splice.

**Drivers:** causal full-calendar correctness, minimum extra source exposure,
and deterministic auditable publication.

**Alternatives considered:** authenticate a new full replacement source, or
abandon the lane. Both remain fallback-only for the invalidation conditions in
the options analysis.

**Why:** it preserves canonical history, minimizes exposure, and proves only the
structural facts needed to close the terminal coverage gap.

**Consequences:** the new branch must disclose S9 history, forbid overlap value
comparison, and treat any structural mismatch as terminal.

**Follow-up:** if structurally invalidated, require another standalone authority
for a newly authenticated full source or abandon this lane. If H10 succeeds,
start the separately named candidate-specific Ralplan; do not invent G9CB-10
economics.
