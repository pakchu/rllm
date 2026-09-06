# Gross9 structural clock bundle G9CB-9 successor authority decision — 2026-07-31

## Decision

Freeze `G9CB-9` as a new candidate-independent structural-clock
infrastructure identity.

`G9CB-9` is not a retry, resume, repair, amendment, v2, or completion of
`G9CB-8`. The sole canonical `G9CB-8` production invocation published its
attempt-consumed sentinel and pass-1 capability-consumed ledger, consumed only
the pass-1 worker capability, and then failed terminally because the frozen
primary market source ended before the frozen full-calendar terminal boundary.
`G9CB-8` is permanently closed with no clock authority.

The existing ignored/untracked `G9CB-8` sentinel and pass-1 ledger are
immutable terminal evidence at clean pushed `C8`. They must be force-added
together, and only together, at `T8`. The retained empty pass-1 stage and
absent pass-2 stage are permanent residue, not artifacts and not authority.

The successor preserves the full half-open domain
`[2023-06-01T00:00:00Z, 2026-06-01T00:00:00Z)`. It does not shorten the final
calendar, bridge a gap geometrically, fabricate a market or open-interest row,
forward-fill either source, or weaken the physical-grid guard. Instead it
authorizes one separately reviewed source-support implementation and one
source-only materialization run. That run must prove both that the replacement
market is a strict logical append of the frozen market and that the official
Binance USD-M five-minute metrics source supplies an exact, positive,
timestamp-aligned OI observation for every appended market open through
`2026-05-31T23:55:00Z`. It must also authenticate and validate the pre-existing
five-minute spot/completed-premium source whose exact-timestamp projection
supplies Rank7's `spot_close`, `spot_rows`, `premium_index_1m_close`, and
`premium_rows` inputs, and it must prove the inherited funding and hourly
premium sources remain causally available through the new terminal row under
their unchanged tolerances. Rank7 continues to apply its already-frozen causal
one-complete-bar OI delay; no signal, feature, or decision behavior is changed.

The exact successor topology is:

```text
C8 -> A9 -> T8 -> S9 -> M9 -> Q9 -> P9 -> C9 -> D9
```

This decision inherits every compatible mechanic and prohibition frozen by
all prior Gross9 structural-clock authority and amendment documents through
`A8`. Signal rules, features, models, histories, sleeves, weights, sides,
holds, barriers, Rank7 behavior, causal availability flags, two-pass equality,
worker isolation, capability consumption, counters, serialization,
manifest-last publication, candidate independence, and the post-publication
economic/overlap boundary remain unchanged except for the exact source
coverage correction and predecessor closure stated here.

This file becomes operative only when committed and pushed alone as `A9`, the
direct child of exact clean pushed `C8`. No terminal-evidence seal, source
materialization, protocol change, preregistration, claim, or production action
is authorized before that seal.

## Evidence boundary

This decision distinguishes:

1. Git- and filesystem-authenticated metadata facts;
2. the supplied sole canonical D8 execution observation;
3. the post-failure timestamp-only forensic source access stated below; and
4. deductions from committed code and those facts.

It does not claim an unrecoverable runtime count, unstated source value,
candidate value, comparator value, or economic result.

### Exact committed predecessor chain

```text
A8 = 33a5aad98c19cec29aba253933145d76b893be93
Q8 = 6a1ec54b2218adb2b46f0072cfd5a5991ce63aee
P8 = 3b4c628a18fba4e24d2e742b59cdbecc2a1b62a7
C8 = 3c8696905bad0bb36e79f759e06299a4148e62eb
```

The direct-parent chain and diffs are exact:

```text
first_parent(A8) == C7
first_parent(Q8) == A8
first_parent(P8) == Q8
first_parent(C8) == P8

diff(A8, Q8) ==
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py

diff(Q8, P8) ==
  A results/gross9_structural_clock_bundle_g9cb8_preregistration_2026-07-31.json

diff(P8, C8) ==
  A results/gross9_structural_clock_bundle_g9cb8_access_claim_2026-07-31.json
```

The operative `A8` document is a tracked mode-`100644` regular file:

```json
{
  "authority_commit": "33a5aad98c19cec29aba253933145d76b893be93",
  "git_blob": "3adc4c8e04901a001b0ada47b273756b63128e60",
  "git_mode": "100644",
  "path": "docs/gross9-structural-clock-bundle-g9cb8-successor-authority-decision-2026-07-31.md",
  "path_type": "regular_file",
  "sha256": "8b2ced344ef3e40fbdba68427a8f3467abedab2c57edbd6afe84b8da6691aec0",
  "size_bytes": 21309
}
```

The operative `P8` preregistration is:

```json
{
  "filesystem_mode_octal": "0444",
  "git_blob": "d231fab022c2d3901b42e7d3b2dd786e7c85bde9",
  "git_mode": "100644",
  "manifest_hash": "9f33fff41b63722808e670eb037f1aba454be0c30ccf86097405bc49d754de33",
  "path": "results/gross9_structural_clock_bundle_g9cb8_preregistration_2026-07-31.json",
  "path_type": "regular_file",
  "protocol_implementation_commit": "6a1ec54b2218adb2b46f0072cfd5a5991ce63aee",
  "protocol_version": "gross9_structural_clock_bundle_g9cb8_preregistration_v1",
  "seal_commit": "3b4c628a18fba4e24d2e742b59cdbecc2a1b62a7",
  "sha256": "3d9a453e27efd9ae1136bc7a6d35396c95fa049248735c59d8c89b4294ddebb5",
  "size_bytes": 66807
}
```

The operative `C8` access claim is:

```json
{
  "claim_hash": "785c011f74e2829870030d81faac6da7ff75358af84f97736ab8da208c9f296b",
  "filesystem_mode_octal": "0444",
  "git_blob": "a14b549295744a9141bc3c39b17452c4a24e1031",
  "git_mode": "100644",
  "path": "results/gross9_structural_clock_bundle_g9cb8_access_claim_2026-07-31.json",
  "path_type": "regular_file",
  "protocol_parent_commit": "3b4c628a18fba4e24d2e742b59cdbecc2a1b62a7",
  "seal_commit": "3c8696905bad0bb36e79f759e06299a4148e62eb",
  "sha256": "5a9363ff266523640b1f0618e0b9e4f6ba9ec111f0b263e8dc4f5f115fd5a239",
  "size_bytes": 17698
}
```

The exact `Q8` protocol file bindings are:

| Path | SHA-256 | Git blob | Size |
|---|---|---|---:|
| `tests/test_build_gross9_structural_clock_bundle.py` | `d7e70d13e07bd82e934bf99459e8a774a271078f6362f4b9e9e1a9a3228e6431` | `fb316c2c12952be514a52518cdf7de2582aee7ce` | 352066 |
| `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py` | `70c8c635dbd319c6ddc5bfc08696452c2e88dc38ef1d2b2b241fe5c7f78c1390` | `c7f331d96526127f8c6744c6d35344c4969d4d33` | 21371 |
| `tests/test_preregister_gross9_structural_clock_bundle.py` | `d7bef358c21c054fb87f63c45530a480a3e2debd7646753cf9ca5958188b56d4` | `d7e0bcf484f7d808c19a5bb528253762cb7a88e6` | 103219 |
| `training/build_gross9_structural_clock_bundle.py` | `7b6f05753c719a0bd8decd2b5b41725fe5a0a6eb0fd0f2a55fa093278c1a9778` | `e6670c39c8adbc078ff070174679149f6b6aba13` | 420663 |
| `training/preregister_gross9_structural_clock_bundle.py` | `faed1d9eaa5d0284e9e741882f56c4cbac9fe643c034ceb68efdf60894f755cd` | `a3b2512129fca5a055ff707e7710ff75e49d57f8` | 230568 |

All Git modes are `100644`.

## Immutable G9CB-8 terminal finding

The sole official production command was exactly:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
```

It was invoked once and never retried. It exited with status `1` after
`1:08.18`, with maximum resident set size `1,527,368` KB. The exact worker
message was:

```text
TerminalG9CB8Failure: generic market ends before the canonical terminal boundary
```

The exact parent message was:

```text
TerminalG9CB8Failure: fresh worker failed with PID 581091 and status 1
```

No separate durable raw stdout/stderr capture was created. The tool transcript
is diagnostic evidence, not a production input. Exact raw stderr bytes and
hash are unrecoverable and must not be invented. The command emitted no
successful-result stdout.

### Durable terminal evidence

The attempt-consumed sentinel is canonical JSON, worktree mode `0444`, and has:

```json
{
  "claim_commit": "3c8696905bad0bb36e79f759e06299a4148e62eb",
  "filesystem_mode_octal": "0444",
  "git_blob": "7459f27563c36a8a2cf2141e4add6b7c6b8cbb4e",
  "git_mode": "100644",
  "manifest_hash": "205b6e934d318a42f608a8a4fb16461a620f63a463ac6a229341a065cefeec92",
  "parent_authentication_sha256": "695f5f0f392f776a8ec805cdbf2570d6112a8f1a1c8e11d3c9b7e104780ea2e5",
  "path": "results/gross9_structural_clock_bundle_g9cb8_attempt_consumed_2026-07-31.json",
  "path_type": "regular_file",
  "protocol_version": "gross9_structural_clock_bundle_g9cb8_v1",
  "resume_allowed": false,
  "retry_allowed": false,
  "sha256": "024c89a4ec6590f656f0b0e092e49997e1661dede37d23932ee2cf3822f09ffe",
  "size_bytes": 3654,
  "status": "attempt_consumed_before_runtime_or_value_access"
}
```

The pass-1 capability-consumed ledger is canonical JSON, worktree mode `0444`,
and has:

```json
{
  "carrier_device": 14,
  "carrier_inode": 22975562,
  "carrier_kind": "anonymous_pipe_v1",
  "claim_hash": "785c011f74e2829870030d81faac6da7ff75358af84f97736ab8da208c9f296b",
  "filesystem_mode_octal": "0444",
  "git_blob": "98ed78849c31dc26dc2f420aa43807a7ba75e5ad",
  "git_mode": "100644",
  "identity": "G9CB-8",
  "parent_pid": 580524,
  "path": "results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass1_2026-07-31.json",
  "path_type": "regular_file",
  "preregistration_manifest_hash": "9f33fff41b63722808e670eb037f1aba454be0c30ccf86097405bc49d754de33",
  "protocol_version": "gross9_structural_clock_bundle_g9cb8_v1",
  "sentinel_manifest_hash": "205b6e934d318a42f608a8a4fb16461a620f63a463ac6a229341a065cefeec92",
  "sha256": "070baca2b4f04f61216e08c60a2a1176fef6b0d6fa9c9a87e6a5bf6058d0cf4d",
  "size_bytes": 1766,
  "slot": 1,
  "stage_directory": "results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761",
  "status": "consumed_before_runtime_or_value_access",
  "token_sha256": "8ac5623c7cf431758d6d91ef03e79e8c5e3f497bdfbc4811b98e328c287608aa"
}
```

The sentinel and ledger zero-access/status fields are facts at their respective
publication boundaries only. They are not false claims that the worker stayed
at zero access afterward.

### Post-ledger exposure

The traceback location and committed control flow establish:

```text
worker capabilities consumed: pass1=1, pass2=0
worker ledgers published: pass1=1, pass2=0
declared isolated runtime roots imported: 2 of 2
market decoded and handed off: yes
funding decoded and handed off: yes
premium decoded and handed off: yes
open interest decoded and handed off: yes
exact runtime decoded/handoff row counts recoverable: no
REX JSONL opened: no
Rank7 model/history opened: no
features constructed: no
schedules reached: no
sleeve intervals reached: no
candidate rows opened: 0
comparator clock rows opened: 0
pre-2025 anchor rows opened: 0
portfolio return/PnL values computed: 0
funding cash values computed: 0
CAGR/MDD values computed: 0
overlap values computed: 0
```

The pass-1 stage is an empty, non-symlink, mode-`0700` directory:

```text
results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761
```

The pass-2 stage is absent:

```text
results/.gross9-structural-clock-g9cb8-worker-dcb23c75d25376df58352acb
```

These canonical G9CB-8 outputs are permanently absent:

```text
results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass2_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb8_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_g9cb8_manifest_2026-07-31.json
```

The fixed bytecode prefix, all probe paths, all staged CSV/core/receipt files,
and every other G9CB-8 stage are absent. The retained pass-1 stage must remain
empty and mode `0700`; the pass-2 stage and permanent outputs must remain
absent through `D9`. Neither stage may be removed, populated, renamed, reused,
committed, or treated as authority.

## Root cause and forensic source disclosure

The frozen market binding was:

```json
{
  "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
  "sha256": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
  "size_bytes": 66696659
}
```

After G9CB-8 was terminally closed, a timestamp-focused forensic read decoded
the old market and OI sources. This was real source-value access and is not
zero-access evidence. It opened no candidate, comparator, model, history, PnL,
CAGR, MDD, or overlap value.

The old normalized market and OI both end at the physical open:

```text
2026-05-31T15:00:00Z
```

The timestamp-focused forensic read established `674785` normalized market
rows, but this authority does not assert or depend on an exact physical gzip
line number. The inherited five-minute continuity guard passed before the
terminal check. Therefore the old source's implied next boundary is
`2026-05-31T15:05:00Z`, while the frozen full-calendar terminal boundary is
`2026-06-01T00:00:00Z` and requires the physical open
`2026-05-31T23:55:00Z`.

The exact shortfall is:

```text
32100 seconds = 8 hours 55 minutes = 107 missing five-minute opens
```

This is a false source-coverage premise, not a reason to weaken the boundary
model. OI is left-joined and cannot extend or remove market rows. Funding and
premium attachment preserve the market row set. No source hash mismatch at
parent authentication is alleged.

The earlier G9CB-4 boundary correction correctly separated physical opens from
the geometry-only terminal boundary but had no official-source coverage
certificate. Its synthetic early-end rejection was correct. Q8 was forbidden
to change source bytes, paths, hashes, or domain. The G9CB-8 guard therefore
failed correctly on a false inherited premise.

## Authorized source-only successor preparation

### Frozen replacement sources and prior exposure

The market extension candidate is the pre-existing regular file:

```json
{
  "absolute_path": "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz",
  "filesystem_mode_octal": "0644",
  "path_type": "regular_file",
  "sha256": "0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe",
  "size_bytes": 65420089
}
```

The OI extension candidate is the official Binance USD-M daily metrics archive
already materialized as one five-minute gzip:

```json
{
  "absolute_path": "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz",
  "filesystem_mode_octal": "0644",
  "path_type": "regular_file",
  "sha256": "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106",
  "size_bytes": 21440132
}
```

Its frozen ordered schema is exactly:

```json
["create_time","symbol","sum_open_interest","sum_open_interest_value","count_toptrader_long_short_ratio","sum_toptrader_long_short_ratio","count_long_short_ratio","sum_taker_long_short_vol_ratio"]
```

The predecessor OI input remains:

```json
{
  "absolute_path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
  "filesystem_mode_octal": "0644",
  "path_type": "regular_file",
  "sha256": "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
  "size_bytes": 19657777
}
```

Rank7 additionally requires the pre-existing completed five-minute spot and
one-minute premium-index aggregate source:

```json
{
  "absolute_path": "/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz",
  "filesystem_mode_octal": "0644",
  "path_type": "regular_file",
  "sha256": "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617",
  "size_bytes": 15772146
}
```

A9 does not guess this file's complete raw schema. S9 learns its ordered raw
schema during the sole authorized decode, rejects duplicate column names, and
freezes only this required ordered projection:

```json
["date","spot_close","spot_rows","premium_index_1m_close","premium_rows"]
```

Additional raw columns are permitted but cannot be joined, counted as Rank7
inputs, or used by G9CB-9. The projection is exact-left-joined one-to-one on
UTC-naive `date` after the materialized market, funding, premium, and OI
attachments. It is not copied into either materialized data output.

S9 also authenticates and decodes the two inherited production auxiliary
sources so terminal compatibility is proved against actual bytes rather than a
synthetic assumption:

```json
[
  {"mode_octal":"0644","name":"funding","path":"data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7","size_bytes":89326},
  {"mode_octal":"0644","name":"premium","path":"data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7","size_bytes":1196481}
]
```

These are not replacements and their two inherited source-manifest rows remain
byte-for-byte identical. S9 may use them only through the existing
`attach_binance_um_aux_frames` behavior with exact inherited tolerances
`funding=12h` and `premium=2h`, solely to certify availability on the completed
market tail.

This decision makes no pristine-source claim. The current G9CB-9 preparation
authenticated the replacement files only as opaque bytes before A9, but the
repository contains earlier committed research that decoded sources at the
same logical paths, including the spot/completed-premium path. Those earlier
reports did not establish a current G9CB-9 source-support certificate. The
predecessor OI was also decoded during historical recovery and the terminal
G9CB-8 worker. S9 therefore records present-byte source support, not novelty or
absence of all historical exposure.

The exact initially absent, ignored materialized destinations are:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
```

The successor trusts neither filename nor presumed coverage. S9 must prove both
extensions and terminal Rank7/auxiliary compatibility from the authenticated
bytes and the frozen rules below.

### S9 source-support implementation

`S9` is the direct child of clean pushed `T8` and adds exactly:

```text
training/materialize_gross9_structural_clock_g9cb9_sources.py
tests/test_materialize_gross9_structural_clock_g9cb9_sources.py
```

The implementation may use existing project dependencies only. It must have a
synthetic test suite and an independent code review before S9 is committed and
pushed. The suite must cover success; raw hash/size/type/mode drift; symlinks;
hard-link or inode aliasing; schema drift; market-prefix drift; OI-overlap
conflict; missing splice-anchor or tail OI; non-positive tail OI; missing or
invalid Rank7 spot/premium projection columns; duplicate or off-grid
spot/premium timestamps; incomplete Rank7 projection coverage; invalid recent
Rank7 row counts; missing, stale, duplicate, or terminally unavailable funding
or hourly-premium rows; duplicates; ordering; internal gaps; early/late
terminal coverage; output preexistence; deterministic serialization; canonical
JSON bytes; exact counters; retained-descriptor identity drift; and every
publication failpoint frozen below. Temporary private inodes must be closed,
while a canonical create-exclusive output already published after attempt
consumption must remain immutable evidence.

The one no-argument official command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb9_sources
```

It may be invoked exactly once, only at clean pushed S9. Any official failure
closes this source-support identity without repair, comparison relaxation,
second run, Q9, or G9CB-9 production. A later continuation would require a new
standalone authority.

### Exact S9 predecode order and attempt sentinel

The command must execute these gates in this order:

1. validate the no-argument parent command shape and canonical repository root;
2. run the shared repository-bytecode absence check as the first repository
   filesystem gate;
3. require the exact branch, `HEAD == @{upstream} == S9`, direct parent `T8`,
   exact two-file S9 diff, and otherwise clean index/worktree;
4. authenticate the indivisible T8 evidence pair and exact permanent G9CB-8
   residue/absence state;
5. require all five canonical output paths below to be absent, non-symlink, and
   non-hard-link aliases of every input;
6. open all seven pre-existing source inputs with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`,
   retain every descriptor, authenticate path/type/mode/size/link-count/SHA-256
   from those descriptors, and prove distinct device/inode identities without
   decompressing them;
7. create-exclusively publish and directory-fsync the source-support attempt
   sentinel at mode `0444`; and only then
8. decompress or decode any source field or value.

The attempt sentinel path is:

```text
results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json
```

Its exact top-level schema is the following canonical JSON object; angle-bracket
metavariables are filled from already-authenticated Git or filesystem facts:

```json
{
  "attempt_hash": "<SHA256_CANONICAL_OBJECT_WITHOUT_ATTEMPT_HASH>",
  "branch": "codex/gross9-structural-clock-bundle-20260731",
  "expected_outputs": [
    "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
    "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
    "configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json"
  ],
  "identity": "G9CB-9-SOURCE-SUPPORT",
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
  "repository_head": "<S9>",
  "repository_parent": "<T8>",
  "resume_allowed": false,
  "retry_allowed": false,
  "source_access_at_publication": {"opaque_bytes_hashed":190272610,"preexisting_sources_decoded":0,"source_rows_decoded":0},
  "status": "source_support_attempt_consumed_before_source_decode",
  "topology": {"authority_commit":"<A9>","implementation_commit":"<S9>","terminal_evidence_commit":"<T8>"},
  "version": "gross9_structural_clock_bundle_g9cb9_source_support_v1"
}
```

`attempt_hash` is SHA-256 of canonical UTF-8 JSON after omitting the
`attempt_hash` member entirely. Canonical JSON uses sorted object keys, list
order as written, `separators=(",", ":")`, `ensure_ascii=False`,
`allow_nan=False`, and no trailing newline for hashing. The published file is
those canonical bytes plus one LF. No field other than `attempt_hash` is
excluded.

The seven authenticated input descriptors remain open through final
postcondition validation. Every decode starts at offset zero through a duplicate
of the corresponding retained descriptor; no input pathname may be reopened
after gate 6. S9 records `fstat` identity before hashing, immediately before
decode, immediately after decode, and at final postcondition. Device, inode,
regular-file type, permission bits, size, and `st_nlink == 1` must remain exact.
It hashes the complete bytes from the retained descriptor before sentinel
publication and again after decode; both digests and byte counts must equal the
frozen binding. Descriptor aliasing, short reads, non-seekability, identity
drift, or pathname replacement is terminal.

No canonical output pathname is opened for writing. For each output, S9 opens
an unnamed same-directory `O_TMPFILE` regular inode at mode `0600`, writes all
bytes, `fchmod(0444)`, file-`fsync`s, verifies exact bytes/hash/size/type/mode and
`st_nlink == 0` by the same descriptor, and only then uses the inherited
create-only `linkat("/proc/self/fd/<fd>", parent_fd, leaf,
AT_SYMLINK_FOLLOW)` primitive to publish the completed inode at the canonical
leaf. It directory-`fsync`s, requires the descriptor and no-follow canonical
stat to identify the same mode-`0444`, single-link inode, and closes the
descriptor. Absence of `O_TMPFILE`, `/proc/self/fd`, or the link capability,
`EEXIST`, or any pre-link/post-link identity mismatch is terminal. Thus a crash
can expose no partial or writable canonical output; a canonical leaf, if
present, already contains the complete immutable bytes.

The exact durable checkpoint order is:

1. finish repository/raw-input authentication while retaining all seven input
   descriptors;
2. prepare and verify the attempt sentinel on an unnamed inode, create-only
   link it at the canonical leaf, directory-`fsync`, and verify the linked inode;
3. decode all seven inputs through retained-descriptor duplicates and complete
   every raw-source/transform/coverage validation;
4. prepare, verify, create-only link, directory-`fsync`, and verify the
   materialized market;
5. publish the materialized OI through the same unnamed-inode sequence;
6. no-follow reopen and decode each generated data output exactly once, prove
   byte binding and exact logical-frame equality, then close the readbacks;
7. publish the source manifest through the same unnamed-inode sequence;
8. publish the support artifact last through the same unnamed-inode sequence;
   and
9. reauthenticate every retained input descriptor and every output, require no
   unauthorized output, temporary file, or repository bytecode, and only then
   return success.

The synthetic suite injects a failure before and after every create-only link,
after each numbered checkpoint, and after each descriptor identity check. A
failure never removes or mutates a durable canonical output already published.
The official identity remains consumed after checkpoint 2 and cannot be
resumed.

### Exact source transforms and acceptance gates

S9 decodes exactly seven pre-existing sources plus exactly two generated-output
readbacks. It opens no other source value.

For the market pair, S9 parses rows in physical CSV order, performs the
inherited UTC-naive conversion, rejects null or duplicate normalized
timestamps, and sorts with explicit stable `kind="mergesort"` before applying
`drop_duplicates("date", keep="last")`. Because duplicates are prohibited, the
drop is an asserted identity and no pandas default tie ordering is relied on.
It requires both ordered schemas to equal exactly
`["date","open","high","low","close","volume","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","tic","day","dxy","kimchi_premium","usdkrw","btckrw","dxy_available","kimchi_available","usdkrw_available","external_any_available","dxy_zscore","dxy_momentum","kimchi_premium_zscore","kimchi_premium_change","usdkrw_zscore","usdkrw_momentum"]`;
require the complete normalized old frame to equal the replacement prefix
through `2026-05-31T15:00:00Z` in every timestamp and value; filter the
replacement to timestamps strictly before
`2026-06-01T00:00:00Z`; and require one unique continuous 300-second grid ending
at `2026-05-31T23:55:00Z`. Both old and materialized in-domain frames must have
respectively `674785` and `674892` rows, so the append is exactly `107` rows.

For OI, the old two-column frame must have exact schema
`["date","open_interest"]`, `674785` unique monotonic rows, and final timestamp
`2026-05-31T15:00:00Z`. The metrics input must have the exact eight-column
schema above, unique monotonic five-minute-aligned `create_time`, constant
`symbol == "BTCUSDT"`, and finite positive `sum_open_interest` wherever used.
The only authorized mapping is:

```text
date = UTC-naive(create_time)
open_interest = numeric(sum_open_interest)
```

There is no as-of join, interpolation, forward fill, tolerance, resampling, or
choice among alternative transforms. On every timestamp common to old OI and
metrics, parsed `float64` values must compare exactly with zero tolerance and
matching missingness. The exact 13-timestamp splice window from
`2026-05-31T14:00:00Z` through `2026-05-31T15:00:00Z`, inclusive, must be
present in both sources and equal. Every one of the exact 107 timestamps from
`2026-05-31T15:05:00Z` through `2026-05-31T23:55:00Z` must occur exactly once
in metrics with positive finite OI. The materialized OI is the unchanged old
logical frame followed by those 107 mapped rows, for exactly `674892` rows.

For the Rank7 spot/completed-premium source, S9 learns the actual ordered raw
schema, records its canonical schema hash and column count, requires unique
column names, and requires the exact
five-column projection frozen above to exist. It converts `date` to UTC-naive,
rejects null, duplicate, non-monotonic, off-five-minute, or out-of-order
timestamps, and exact-left-joins the four value columns one-to-one onto the
complete materialized market grid without fill, tolerance, resampling, or
as-of matching. Every materialized market timestamp must have exactly one
source row. In the final 3,000 market rows, `spot_rows` and `premium_rows` must
both be finite and exactly `5`; the latest `spot_close` must be finite and
positive and the latest `premium_index_1m_close` finite. Other projected values
remain the authenticated source values and are not repaired. S9 records the
projection row count, ordered projection schema, first/last timestamp, and
these acceptance facts in support; it computes no Rank7 feature or decision.

For funding and hourly premium, S9 requires the columns accepted by
`normalise_funding_history_frame` and `normalise_premium_index_frame`, rejects
null or duplicate normalized timestamps, and invokes the current inherited
normalizers plus `_merge_aux` itself against the complete materialized market:
`value_cols=["funding_rate"]`, backward direction, tolerance `12h`; then
`value_cols=["premium_index"]`, backward direction, tolerance `2h`. It does not
invoke rolling z-scores, changes, Rank7 reconstruction, or any signal. Both
joins must preserve all `674892` market timestamps exactly. Every one of the
107 appended rows, including `2026-05-31T23:55:00Z`, must have availability
`1.0` for both sources and finite attached values. The run records decoded and
normalized source counts plus exact tail-availability counts; no stale value
outside the inherited tolerance is accepted.

This exact-timestamp OI is a raw source observation, not a same-bar decision
feature. The unchanged Rank7 code shifts raw OI and availability by one complete
five-minute bar in `_rank7_build_braid_state`, while its other interest features
are shifted by `_rank7_live_decision_features`. Q9 must prove this unchanged
causal behavior in the integrated regression below. Funding and premium retain
their existing causal backward joins and tolerances.

Both generated CSVs use UTF-8, a header, `index=False`, LF line endings, date
format `%Y-%m-%d %H:%M:%S`, `float_format="%.17g"`, `na_rep=""`, and CSV minimal
quoting. Each gzip member uses compression level `9`, `mtime=0`, and an empty
embedded filename. Each destination is create-exclusive, non-symlink,
single-link, mode `0444`, and published only from a completed file-fsynced
unnamed inode followed by directory fsync. S9 then reopens
both no-follow and decodes each once to prove exact logical-frame equality.
Repeated construction is exercised only with synthetic destinations in tests;
the official destinations are written once.

For each generated data file, `frame_hash` is SHA-256 of the exact
uncompressed CSV byte stream emitted by the serialization rule above,
including the header and final LF and before gzip compression. It is not a
pandas-object hash, gzip hash, decompressed-and-reserialized hash, or hash of
canonical JSON. The synthetic test vector whose exact bytes are
`date,value\n2026-01-01 00:00:00,1\n` has `frame_hash`
`8e362a6177525d72ecae23994c231bc666be1c61995aaddbf1afcaedc805b433`.
The separately reported file `sha256` is over the complete deterministic gzip
bytes.

### Exact five-output publication and M9 seal

After the sentinel, the official command creates exactly these remaining four
outputs, for five filesystem outputs total:

```text
data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz
configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json
results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json
```

The two data files remain ignored/untracked. The source manifest preserves all
eight inherited rows and every non-OI/non-market row value from
`configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json`.
It changes exactly `as_of` to `2026-07-31`, `market_5m.path/sha256`, and
`open_interest.path/sha256`, then inserts exactly this ninth row immediately
after `open_interest` and before `rex_taker_train`:

```json
{"name":"rank7_spot_premium_5m","path":"/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz","sha256":"c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617"}
```

The resulting ordered source-name list has exactly nine unique members;
`schema_version` remains integer `1`. The manifest is canonical JSON and is
published before the support artifact.

The final support artifact has exactly these top-level keys and nested keys;
the shown list orders and member names are part of the schema:

```json
{
  "access": {
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "decoded_generated_readbacks": ["materialized_market","materialized_open_interest"],
    "decoded_preexisting_sources": ["old_market","replacement_market","funding","premium","old_open_interest","binance_metrics_open_interest","rank7_spot_premium_5m"],
    "economic_or_overlap_values_computed": 0,
    "feature_signal_schedule_or_interval_values_computed": 0,
    "model_history_or_rex_values_opened": 0,
    "pre2025_anchor_value_rows_opened": 0,
    "raw_source_decode_count": 7,
    "readback_decode_count": 2
  },
  "attempt_sentinel": {"attempt_hash":"<ATTEMPT_HASH>","path":"results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json","sha256":"<SENTINEL_FILE_SHA256>","size_bytes":"<INTEGER>"},
  "identity": "G9CB-9-SOURCE-SUPPORT",
  "materialized_sources": {
    "market_5m": {"filesystem_mode_octal":"0444","first_timestamp":"<UTC_Z>","frame_hash":"<SHA256>","gzip":{"compresslevel":9,"embedded_filename":"","mtime":0},"last_timestamp":"2026-05-31T23:55:00Z","path":"data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz","path_type":"regular_file","rows":674892,"schema":["date","open","high","low","close","volume","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","tic","day","dxy","kimchi_premium","usdkrw","btckrw","dxy_available","kimchi_available","usdkrw_available","external_any_available","dxy_zscore","dxy_momentum","kimchi_premium_zscore","kimchi_premium_change","usdkrw_zscore","usdkrw_momentum"],"sha256":"<SHA256>","size_bytes":"<INTEGER>"},
    "open_interest": {"filesystem_mode_octal":"0444","first_timestamp":"<UTC_Z>","frame_hash":"<SHA256>","gzip":{"compresslevel":9,"embedded_filename":"","mtime":0},"last_timestamp":"2026-05-31T23:55:00Z","path":"data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz","path_type":"regular_file","rows":674892,"schema":["date","open_interest"],"sha256":"<SHA256>","size_bytes":"<INTEGER>"}
  },
  "raw_sources": [
    {"decoded_rows":674785,"mode_octal":"0644","name":"old_market","normalized_rows":674785,"path":"data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c","size_bytes":66696659},
    {"decoded_rows":"<INTEGER>","mode_octal":"0644","name":"replacement_market","normalized_rows":"<INTEGER>","path":"/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz","path_type":"regular_file","sha256":"0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe","size_bytes":65420089},
    {"decoded_rows":"<INTEGER>","mode_octal":"0644","name":"funding","normalized_rows":"<INTEGER>","path":"data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7","size_bytes":89326},
    {"decoded_rows":"<INTEGER>","mode_octal":"0644","name":"premium","normalized_rows":"<INTEGER>","path":"data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7","size_bytes":1196481},
    {"decoded_rows":674785,"mode_octal":"0644","name":"old_open_interest","normalized_rows":674785,"path":"/tmp/btcusdt_open_interest_5m_2020_2026.csv","path_type":"regular_file","sha256":"e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31","size_bytes":19657777},
    {"decoded_rows":"<INTEGER>","mode_octal":"0644","name":"binance_metrics_open_interest","normalized_rows":"<INTEGER>","path":"/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106","size_bytes":21440132},
    {"decoded_rows":"<INTEGER>","mode_octal":"0644","name":"rank7_spot_premium_5m","normalized_rows":"<INTEGER>","path":"/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617","size_bytes":15772146}
  ],
  "source_manifest": {"path":"configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json","sha256":"<SHA256>","size_bytes":"<INTEGER>"},
  "source_support_commit": "<S9>",
  "support_hash": "<SHA256_CANONICAL_OBJECT_WITHOUT_SUPPORT_HASH>",
  "validation": {
    "appended_market_rows": 107,
    "appended_open_interest_rows": 107,
    "domain_end_exclusive": "2026-06-01T00:00:00Z",
    "funding_attachment_tolerance": "12h",
    "funding_tail_available_rows": 107,
    "latest_funding_available_after_causal_attachment": true,
    "latest_open_interest_positive_after_exact_join": true,
    "latest_premium_available_after_causal_attachment": true,
    "market_grid_seconds": 300,
    "market_prefix_exact": true,
    "metrics_common_timestamp_values_exact": true,
    "oi_grid_seconds": 300,
    "oi_splice_window_exact_rows": 13,
    "oi_tail_exact_rows": 107,
    "old_last_timestamp": "2026-05-31T15:00:00Z",
    "premium_attachment_tolerance": "2h",
    "premium_tail_available_rows": 107,
    "rank7_spot_premium_exact_join_rows": 674892,
    "rank7_spot_premium_first_timestamp": "<UTC_Z>",
    "rank7_spot_premium_last_timestamp": "<UTC_Z>",
    "rank7_spot_premium_latest_values_valid": true,
    "rank7_spot_premium_projection_schema": ["date","spot_close","spot_rows","premium_index_1m_close","premium_rows"],
    "rank7_spot_premium_raw_column_count": "<INTEGER>",
    "rank7_spot_premium_raw_schema_sha256": "<SHA256>",
    "rank7_spot_premium_tail_complete_rows": 3000,
    "required_last_timestamp": "2026-05-31T23:55:00Z"
  },
  "version": "gross9_structural_clock_bundle_g9cb9_source_support_v1"
}
```

Each `raw_sources` record has exactly the eight keys shown; every
`"<INTEGER>"` metavariable is replaced by a JSON integer learned during the
sole run, never retained as a string, and no other key is allowed.
`rank7_spot_premium_raw_schema_sha256` is SHA-256 of canonical UTF-8 JSON for
the actual ordered raw column-name array, using the same canonical JSON options
and no trailing LF. Its column-count field must equal that array length; Q9
binds both learned values literally from M9.
`support_hash` is SHA-256 of canonical JSON with only `support_hash` omitted.
For the synthetic two-key payload
`{"identity":"G9CB-9","version":"gross9_structural_clock_bundle_g9cb9_source_support_v1"}`
the canonical hash test vector is
`2d9cb35565c002c9b56f7c6236913e01870b6f5cae6835197c4d9a5ab9e21f7a`.

The source manifest and support artifact use the same canonical JSON rules as
the sentinel and are published with one trailing LF through the frozen
unnamed-inode create-only-link sequence at mode `0444`. The support artifact is
published last.

`M9` is the direct child of S9 and force-adds exactly the sentinel, source
manifest, and support artifact. These three files have Git mode `100644` and
worktree mode `0444`. M9 tracks no data file. The data hashes, sizes, schemas,
row counts, bounds, and frame hashes are first learned by S9, sealed by M9, and
bound literally by Q9; A9 does not guess those bytes.

## Q9 protocol

`Q9` is the direct child of clean pushed `M9` and changes exactly the inherited
five protocol files:

```text
tests/test_build_gross9_structural_clock_bundle.py
tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
tests/test_preregister_gross9_structural_clock_bundle.py
training/build_gross9_structural_clock_bundle.py
training/preregister_gross9_structural_clock_bundle.py
```

Q9 may change only:

1. active identity/protocol/path/exception/action/stage/cache names from G9CB8
   to G9CB9;
2. exact `A9`, `T8`, `S9`, `M9`, and predecessor topology/bindings;
3. append the exact G9CB-8 terminal row to the existing ordered
   `failed_predecessor_attempts` container after G9CB-2 and G9CB-3;
4. authenticate the G9CB-8 sentinel and pass-1 ledger as an indivisible T8
   pair before any source-value open;
5. enforce the G9CB-8 permanent absences and exact slot1/slot2 residue states;
6. bind the S9 implementation/test and all three M9 source-support artifacts;
7. replace exactly the active `market_5m` and `open_interest` source logical
   paths/hashes with the two M9-bound complete materialized sources;
8. require the exact ninth `rank7_spot_premium_5m` manifest row bound above,
   load only its frozen five-column projection, and exact-left-join its four
   Rank7 value columns one-to-one by UTC-naive timestamp after funding,
   premium, and OI attachment, with no fill, tolerance, resampling, or as-of
   match;
9. add `rank7_spot_premium_5m` exactly once to source-name validation, file/row
   counters, causal handoff accounting, bootstrap declarations, parent
   authentication, and P9/C9/D9 counter contracts; and
10. add exact regression coverage for these changes.

The production logical-source key order is exactly:

```json
["market_5m","funding","premium","open_interest","rank7_spot_premium_5m","rex_taker_train","rex_taker_test","rex_taker_eval","rex_veto_source","rank7_hourly_history"]
```

That exact ten-key order is used by `rows_decoded`,
`file_access.bytes_read_by_logical_source`,
`rows_used.causal_feature_rows_by_source`, and P9's
`access_counter_names.rows_decoded`. `source_files_opened` remains a separate
scalar counting every actual source-file open, including repeated opens of one
logical source; `model_files_opened` and `runtime_modules_imported` remain
separate scalars. `rows_decoded[name]` sums parser-return rows over every decode
of that logical source, while `bytes_read_by_logical_source[name]` sums bytes
actually returned across those opens. The spot/completed-premium CSV is decoded
exactly once. Its complete parser-return frame is handed to the exact attachment
helper before projection filtering, so every parser ordinal, including an
allowed row outside the materialized market grid, is conservatively entered
once in `causal_feature_rows_by_source["rank7_spot_premium_5m"]`; therefore that
counter equals its `rows_decoded` value. It is distinct from the separately
certified `674892` exact joined market rows. No eleventh key or collapsed
five-source map is permitted.

Q9 must not weaken or alter the terminal-boundary check. It must not change the
domain, old in-domain market or OI logical-prefix values, any other source row,
features, models, histories, signals, schedules, sleeves, weights, holds,
barriers, sides, causal availability semantics, counters, two-pass rule, worker
guard, publication algorithm, candidate/economic prohibition, or overlap
prohibition.

The source-support implementation/test, A9, T8 pair, three M9 artifacts, and all
prior authority/protocol paths must be included in the active authenticated
inventory. Initial bootstrap declarations and the P9 manifest must bind exactly
the same unique path set with sorted two-sided mismatch diagnostics.

Q9 must add one integrated synthetic full-adapter regression. Its fixture has a
complete predecessor market/OI prefix, exactly 107 appended market rows,
exactly 107 positive exact-timestamp OI rows, representative funding and
premium histories, and a complete synthetic spot/completed-premium projection.
The test must run the actual market normalization, causal funding/premium
attachment, `attach_open_interest`, the actual new exact projection-join helper,
Rank7 context path, complete production-grid assertion, and two isolated
synthetic worker passes. Injecting the four Rank7 columns directly into the
market fixture is forbidden. The test must
assert all 107 tail rows have `open_interest_available == 1`, latest OI is
positive, recent Rank7 row counts equal `5`, latest spot/premium values satisfy
the frozen checks, changing OI at bar `t` cannot change any Rank7 decision input
at the same bar, the returned Rank7 and production grids equal the complete
market grid, both fresh workers have the exact ten-key logical-source schema
and exact synthetic counter values, and both emitted CSV/core bytes are
identical. A test that stubs away Rank7
normalization, any source attachment, grid equality, or worker isolation does
not satisfy this gate.

P9 must disclose the earlier S9 source-support access explicitly. The existing
frozen-OI preclaim members remain unchanged, and
`source_preclaim_disclosures` gains exactly one additional member named
`g9cb9_source_support` with this schema:

```json
{
  "attempt_sentinel": {"attempt_hash":"<ATTEMPT_HASH>","path":"results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json","sha256":"<SHA256>","size_bytes":"<INTEGER>"},
  "historical_access": {
    "decoded_generated_readbacks": ["materialized_market","materialized_open_interest"],
    "decoded_preexisting_sources": ["old_market","replacement_market","funding","premium","old_open_interest","binance_metrics_open_interest","rank7_spot_premium_5m"],
    "normalized_rows": {"binance_metrics_open_interest":"<INTEGER>","funding":"<INTEGER>","old_market":674785,"old_open_interest":674785,"premium":"<INTEGER>","rank7_spot_premium_5m":"<INTEGER>","replacement_market":"<INTEGER>"},
    "prohibited_computations": {"candidate_comparator_or_anchor_values":0,"economic_or_overlap_values":0,"feature_signal_schedule_or_interval_values":0,"model_history_or_rex_values":0},
    "raw_source_decode_count": 7,
    "readback_decode_count": 2
  },
  "p9_process_access_at_creation": {"candidate_rows_opened":0,"comparator_clock_rows_opened":0,"model_files_opened":0,"pre2025_anchor_value_rows_opened":0,"runtime_modules_imported":0,"source_files_opened":0,"source_value_rows_opened":0},
  "source_manifest": {"path":"configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json","sha256":"<SHA256>","size_bytes":"<INTEGER>"},
  "support_artifact": {"path":"results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json","sha256":"<SHA256>","size_bytes":"<INTEGER>","support_hash":"<SUPPORT_HASH>"}
}
```

Every `"<INTEGER>"` is replaced by a JSON integer and every hash metavariable
by the exact lowercase digest from M9. No additional key is permitted inside
`g9cb9_source_support`. The active P9 process still starts at zero; those
process-local counters do not erase the separately bound historical S9 decode.

Treating S9 as opaque-only after it performs the authorized comparison,
copying only P9's zero counters, or omitting the historical source-value
disclosure is terminal.

All P9/C9/D9 entry points retain the shared strict repository-bytecode
preflight as the first filesystem gate after argument/worker discrimination
and root canonicalization. Q9 tests may authenticate source/support bytes and
metadata opaquely but must not decode an official source value.

Before Q9 is committed: the exact affected suite, materialization synthetic
suite, AST parsing, diff checks, Ruff/Pyright added-line checks, bootstrap/
manifest parity, active-path absence, bytecode absence, and independent review
must pass.

## Exact seals

All commits use branch:

```text
codex/gross9-structural-clock-bundle-20260731
```

### A9

```text
first_parent(A9) == C8

diff(C8, A9) ==
  A docs/gross9-structural-clock-bundle-g9cb9-successor-authority-decision-2026-07-31.md

HEAD == A9 == @{upstream}
worktree and index clean
```

### T8

```text
first_parent(T8) == A9

diff(A9, T8) ==
  A results/gross9_structural_clock_bundle_g9cb8_attempt_consumed_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass1_2026-07-31.json

HEAD == T8 == @{upstream}
worktree and index clean
```

Both files must reproduce the exact bytes, hashes, prospective Git blobs,
internal sentinel hash, fields, and worktree mode `0444` above. Git mode is
`100644`. Adding either alone or adding any other G9CB-8 output is forbidden.

### S9

```text
first_parent(S9) == T8

diff(T8, S9) ==
  A tests/test_materialize_gross9_structural_clock_g9cb9_sources.py
  A training/materialize_gross9_structural_clock_g9cb9_sources.py

HEAD == S9 == @{upstream}
worktree and index clean
```

### M9

```text
first_parent(M9) == S9

diff(S9, M9) ==
  A configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json
  A results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json

HEAD == M9 == @{upstream}
worktree and index clean
```

### Q9

```text
first_parent(Q9) == M9

diff(M9, Q9) ==
  M tests/test_build_gross9_structural_clock_bundle.py
  M tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
  M tests/test_preregister_gross9_structural_clock_bundle.py
  M training/build_gross9_structural_clock_bundle.py
  M training/preregister_gross9_structural_clock_bundle.py

HEAD == Q9 == @{upstream}
worktree and index clean
```

### P9, C9, D9, V9

The canonical commands are exactly:

```text
P9: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle
C9: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim
D9: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce
V9: PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication
```

The active paths use `g9cb9` and date `2026-07-31`. P9 adds only the
preregistration, C9 adds only the access claim, and successful D9 adds exactly
the sentinel, two worker ledgers, canonical CSV gzip, and final manifest.
Each is a clean pushed direct child in order. V9 runs only after successful D9
is committed and pushed.

No candidate, comparator, pre-2025 allocation anchor, portfolio return, PnL,
funding cash, CAGR, MDD, economic metric, or overlap metric may be opened or
computed before successful committed V9. The sole exception is the exact S9
source-support run, whose access is limited to the seven pre-existing sources,
two generated readbacks, and strict equality/coverage/materialization facts
above.

## Required G9CB-8 predecessor row

At P9, the exact G9CB-8 row is the third and final element of ordered
`bindings.failed_predecessor_attempts`, after G9CB-2 and G9CB-3. `<A9>`, `<T8>`,
`<S9>`, `<M9>`, and the M9-derived metavariables below are replaced by exact
lowercase commits, hashes, blobs, integer sizes, schemas, and paths. The row has
exactly the following top-level members and nested schema; no member is optional
and no additional member is permitted:

```json
{
  "access_claim": {
    "claim_hash": "785c011f74e2829870030d81faac6da7ff75358af84f97736ab8da208c9f296b",
    "filesystem_mode_octal": "0444",
    "git_blob": "a14b549295744a9141bc3c39b17452c4a24e1031",
    "git_mode": "100644",
    "path": "results/gross9_structural_clock_bundle_g9cb8_access_claim_2026-07-31.json",
    "path_type": "regular_file",
    "protocol_parent_commit": "3b4c628a18fba4e24d2e742b59cdbecc2a1b62a7",
    "seal_commit": "3c8696905bad0bb36e79f759e06299a4148e62eb",
    "sha256": "5a9363ff266523640b1f0618e0b9e4f6ba9ec111f0b263e8dc4f5f115fd5a239",
    "size_bytes": 17698
  },
  "authority_decision": {
    "authority_commit": "33a5aad98c19cec29aba253933145d76b893be93",
    "git_blob": "3adc4c8e04901a001b0ada47b273756b63128e60",
    "git_mode": "100644",
    "path": "docs/gross9-structural-clock-bundle-g9cb8-successor-authority-decision-2026-07-31.md",
    "path_type": "regular_file",
    "sha256": "8b2ced344ef3e40fbdba68427a8f3467abedab2c57edbd6afe84b8da6691aec0",
    "size_bytes": 21309
  },
  "classification": "terminal_market_source_coverage_shortfall_after_pass1_source_decode_before_features_or_model_access",
  "diagnostic": {
    "elapsed_wall_clock": "1:08.18",
    "exit_status": 1,
    "max_rss_kb": 1527368,
    "parent_terminal_message": "fresh worker failed with PID 581091 and status 1",
    "raw_stderr_capture": null,
    "raw_stdout_capture": null,
    "worker_pid": 581091,
    "worker_terminal_message": "generic market ends before the canonical terminal boundary"
  },
  "exposure": {
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "decoded_and_handed_off": ["market","funding","premium","open_interest"],
    "exact_decoded_and_handoff_counts_recoverable": false,
    "features_constructed": false,
    "funding_cash_values_computed": 0,
    "isolated_runtime_roots_imported": 2,
    "overlap_metric_values_computed": 0,
    "portfolio_economic_values_computed": 0,
    "pre2025_anchor_value_rows_opened": 0,
    "rank7_model_or_history_opened": false,
    "rex_jsonl_opened": false,
    "schedules_reached": false,
    "sleeve_intervals_reached": false,
    "worker_capabilities_consumed": {"pass1":1,"pass2":0},
    "worker_ledgers_published": {"pass1":1,"pass2":0}
  },
  "identity": "G9CB-8",
  "permanently_absent_outputs": [
    "results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass2_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb8_2026-07-31.csv.gz",
    "results/gross9_structural_clock_bundle_g9cb8_manifest_2026-07-31.json"
  ],
  "preregistration": {
    "filesystem_mode_octal": "0444",
    "git_blob": "d231fab022c2d3901b42e7d3b2dd786e7c85bde9",
    "git_mode": "100644",
    "manifest_hash": "9f33fff41b63722808e670eb037f1aba454be0c30ccf86097405bc49d754de33",
    "path": "results/gross9_structural_clock_bundle_g9cb8_preregistration_2026-07-31.json",
    "path_type": "regular_file",
    "protocol_implementation_commit": "6a1ec54b2218adb2b46f0072cfd5a5991ce63aee",
    "protocol_version": "gross9_structural_clock_bundle_g9cb8_preregistration_v1",
    "seal_commit": "3b4c628a18fba4e24d2e742b59cdbecc2a1b62a7",
    "sha256": "3d9a453e27efd9ae1136bc7a6d35396c95fa049248735c59d8c89b4294ddebb5",
    "size_bytes": 66807
  },
  "protocol_implementation": {
    "commit": "6a1ec54b2218adb2b46f0072cfd5a5991ce63aee",
    "files": [
      {"git_blob":"fb316c2c12952be514a52518cdf7de2582aee7ce","git_mode":"100644","path":"tests/test_build_gross9_structural_clock_bundle.py","sha256":"d7e70d13e07bd82e934bf99459e8a774a271078f6362f4b9e9e1a9a3228e6431","size_bytes":352066},
      {"git_blob":"c7f331d96526127f8c6744c6d35344c4969d4d33","git_mode":"100644","path":"tests/test_gross9_structural_clock_bundle_preregistration_artifact.py","sha256":"70c8c635dbd319c6ddc5bfc08696452c2e88dc38ef1d2b2b241fe5c7f78c1390","size_bytes":21371},
      {"git_blob":"d7e0bcf484f7d808c19a5bb528253762cb7a88e6","git_mode":"100644","path":"tests/test_preregister_gross9_structural_clock_bundle.py","sha256":"d7bef358c21c054fb87f63c45530a480a3e2debd7646753cf9ca5958188b56d4","size_bytes":103219},
      {"git_blob":"e6670c39c8adbc078ff070174679149f6b6aba13","git_mode":"100644","path":"training/build_gross9_structural_clock_bundle.py","sha256":"7b6f05753c719a0bd8decd2b5b41725fe5a0a6eb0fd0f2a55fa093278c1a9778","size_bytes":420663},
      {"git_blob":"a3b2512129fca5a055ff707e7710ff75e49d57f8","git_mode":"100644","path":"training/preregister_gross9_structural_clock_bundle.py","sha256":"faed1d9eaa5d0284e9e741882f56c4cbac9fe643c034ceb68efdf60894f755cd","size_bytes":230568}
    ],
    "parent_commit": "33a5aad98c19cec29aba253933145d76b893be93"
  },
  "protocol_version": "gross9_structural_clock_bundle_g9cb8_v1",
  "residue": {
    "bytecode_cache": {"path":"results/.g9cb8-bytecode-cache-disabled","state":"absent"},
    "slot1_stage": {"committed":false,"filesystem_mode_octal":"0700","path":"results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761","staged_core_state":"absent","staged_csv_state":"absent","staged_receipt_state":"absent","state":"empty_directory"},
    "slot2_stage": {"committed":false,"path":"results/.gross9-structural-clock-g9cb8-worker-dcb23c75d25376df58352acb","state":"absent"}
  },
  "root_cause": {
    "domain_end_exclusive": "2026-06-01T00:00:00Z",
    "missing_five_minute_opens": 107,
    "missing_seconds": 32100,
    "old_market_last_open": "2026-05-31T15:00:00Z",
    "old_market_source": {"path":"data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz","sha256":"a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c","size_bytes":66696659},
    "old_open_interest_last_timestamp": "2026-05-31T15:00:00Z",
    "required_last_open": "2026-05-31T23:55:00Z",
    "source_hash_mismatch_at_parent_authentication": false
  },
  "source_successor": {
    "attempt_sentinel": {"attempt_hash":"<ATTEMPT_HASH>","git_blob":"<M9_SENTINEL_GIT_BLOB>","git_mode":"100644","path":"results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json","path_type":"regular_file","seal_commit":"<M9>","sha256":"<M9_SENTINEL_SHA256>","size_bytes":"<INTEGER>"},
    "materialized_market": {"filesystem_mode_octal":"0444","frame_hash":"<M9_MARKET_FRAME_HASH>","path":"data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz","path_type":"regular_file","rows":674892,"sha256":"<M9_MARKET_SHA256>","size_bytes":"<INTEGER>"},
    "materialized_open_interest": {"filesystem_mode_octal":"0444","frame_hash":"<M9_OI_FRAME_HASH>","path":"data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz","path_type":"regular_file","rows":674892,"sha256":"<M9_OI_SHA256>","size_bytes":"<INTEGER>"},
    "rank7_spot_premium_5m": {"filesystem_mode_octal":"0644","path":"/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz","path_type":"regular_file","sha256":"c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617","size_bytes":15772146},
    "source_manifest": {"git_blob":"<M9_SOURCE_MANIFEST_GIT_BLOB>","git_mode":"100644","path":"configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json","path_type":"regular_file","seal_commit":"<M9>","sha256":"<M9_SOURCE_MANIFEST_SHA256>","size_bytes":"<INTEGER>"},
    "support_artifact": {"git_blob":"<M9_SUPPORT_GIT_BLOB>","git_mode":"100644","path":"results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json","path_type":"regular_file","seal_commit":"<M9>","sha256":"<M9_SUPPORT_SHA256>","size_bytes":"<INTEGER>","support_hash":"<SUPPORT_HASH>"}
  },
  "status": "historical_terminal_attempt_consumed_no_clock_authority",
  "terminal_evidence": {
    "attempt_sentinel": {"claim_commit":"3c8696905bad0bb36e79f759e06299a4148e62eb","filesystem_mode_octal":"0444","git_blob":"7459f27563c36a8a2cf2141e4add6b7c6b8cbb4e","git_mode":"100644","manifest_hash":"205b6e934d318a42f608a8a4fb16461a620f63a463ac6a229341a065cefeec92","path":"results/gross9_structural_clock_bundle_g9cb8_attempt_consumed_2026-07-31.json","path_type":"regular_file","protocol_version":"gross9_structural_clock_bundle_g9cb8_v1","resume_allowed":false,"retry_allowed":false,"seal_commit":"<T8>","sha256":"024c89a4ec6590f656f0b0e092e49997e1661dede37d23932ee2cf3822f09ffe","size_bytes":3654,"status":"attempt_consumed_before_runtime_or_value_access"},
    "pass1_worker_ledger": {"filesystem_mode_octal":"0444","git_blob":"98ed78849c31dc26dc2f420aa43807a7ba75e5ad","git_mode":"100644","path":"results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass1_2026-07-31.json","path_type":"regular_file","seal_commit":"<T8>","sha256":"070baca2b4f04f61216e08c60a2a1176fef6b0d6fa9c9a87e6a5bf6058d0cf4d","size_bytes":1766,"slot":1,"status":"consumed_before_runtime_or_value_access"}
  },
  "topology": {
    "g9cb8_authority_commit": "33a5aad98c19cec29aba253933145d76b893be93",
    "g9cb8_claim_commit": "3c8696905bad0bb36e79f759e06299a4148e62eb",
    "g9cb8_preregistration_commit": "3b4c628a18fba4e24d2e742b59cdbecc2a1b62a7",
    "g9cb8_protocol_commit": "6a1ec54b2218adb2b46f0072cfd5a5991ce63aee",
    "g9cb9_authority_commit": "<A9>",
    "source_support_implementation_commit": "<S9>",
    "source_support_materialization_commit": "<M9>",
    "terminal_evidence_commit": "<T8>"
  }
}
```

Every `"<INTEGER>"` is replaced by a JSON integer. Publication-time zero fields
remain publication-time facts and do not replace the separate post-ledger
`exposure` object. P9 must construct and compare this literal object, not merely
check a subset of fields.

## Completion and terminal rule

G9CB-9 is complete only when every commit from A9 through D9 has the exact
single-parent topology and diff above; the source materialization and support
certificate are exact; both fresh workers produce byte-identical outputs; the
final manifest is published last; V9 passes against committed bytes; all
predecessor evidence and residue remain exact; both materialized sources remain
exact; and `HEAD == @{upstream}` with a clean worktree/index and no repository
bytecode.

The official G9CB-9 production command may be invoked exactly once, only after
clean pushed C9. No official-source rehearsal, production probe, partial run,
retry, resume, or second invocation is authorized.

After a G9CB-9 sentinel exists, every failure is terminal:

```text
TERMINAL_G9CB9_ATTEMPT_CONSUMED_NO_RETRY
```

Any future continuation after such a failure requires another new identity and
standalone authority. This decision authorizes no alpha promotion by itself.
Economic and overlap evaluation begins only after clean pushed verified D9.
