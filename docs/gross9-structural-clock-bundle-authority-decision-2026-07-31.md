# Gross9 structural clock bundle authority decision — 2026-07-31

## Decision

Freeze `G9CB-1` as candidate-independent infrastructure completed before any
new alpha identity exists.

`G9CB-1` may publish one standalone, exact five-sleeve Gross9 structural clock
bundle over the half-open UTC domain:

```text
[2023-06-01T00:00:00Z, 2026-06-01T00:00:00Z)
```

The bundle contains signed structural intervals plus provenance and access
counters. It is not an economic result. It must not compute or publish
portfolio return, PnL, funding cash, an economic rank, CAGR, MDD, a candidate
metric, or an overlap metric.

This decision does not authorize production access. No market, open-interest,
funding, premium, model-history, source, return, PnL, or clock row was opened
to create this document. Before this authority was drafted, an earlier shell
inspection accidentally opened opaque bytes from the compressed frozen
open-interest cache without decompression, parsing, header decoding, or row
decoding; the exact disclosure appears below. The pre-2025 Gross9 anchor was
authenticated only by filesystem SHA-256 and Git blob identity; its JSON
values were not opened.

## Independence boundary

`G9CB-1` is a new authority with no alpha candidate, candidate family,
candidate configuration, candidate clock, or candidate outcome in scope.

It binds the generic Gross9 anchor, portfolio configs, sleeve configs, runtime
modules, Rank7 bundle, environment locks, and transitive source manifest
directly. It must not import, call, parse, require, or derive authority from:

- Ethereum Settlement Demand Impulse source support;
- an Ethereum Settlement Demand Impulse preregistration;
- an Ethereum Settlement Demand Impulse source, novelty, Gross9, or economics
  command;
- any public or private Ethereum Settlement Demand Impulse module, runtime
  entry point, function, method, callback, subprocess command, or dynamically
  resolved callable;
- `_reconstruct_gross9_runtime_clocks`;
- any `results/ethereum_settlement_demand_impulse_*` canonical output; or
- any helper whose production precondition is an Ethereum Settlement Demand
  Impulse stage.

The new adapter may import generic repository runtime modules only after every
bound path, SHA-256, Git blob, import closure, environment lock, package
version, and source-manifest hash has authenticated. Authentication must use
stdlib-only code in the new builder. The durable attempt-consumed sentinel
must then be published before the first generic runtime import or value-row
read.

Tests must reject any production import, path constant, subprocess target, or
artifact dependency that crosses this boundary.

## Planned files and canonical artifacts

Only the following implementation and artifact paths are planned:

| Stage | Path | Contract |
|---|---|---|
| preregistration producer | `training/preregister_gross9_structural_clock_bundle.py` | Metadata/hash/Git/environment authentication only; no value-row access |
| preregistration tests | `tests/test_preregister_gross9_structural_clock_bundle.py` | Synthetic and repository-identity tests |
| preregistration artifact | `results/gross9_structural_clock_bundle_preregistration_2026-07-31.json` | Write-once canonical JSON authority |
| artifact test | `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py` | Reproduces and validates the committed preregistration artifact |
| production builder | `training/build_gross9_structural_clock_bundle.py` | Creates the claim or performs the single claimed build |
| builder tests | `tests/test_build_gross9_structural_clock_bundle.py` | Synthetic-only row tests, sentinel tests, two-pass determinism, and publication tests |
| pre-access claim | `results/gross9_structural_clock_bundle_access_claim_2026-07-31.json` | Earlier immutable claim-only direct-child commit |
| attempt sentinel/ledger | `results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json` | Atomically published immediately before first runtime/value access |
| canonical interval bytes | `results/gross9_structural_clock_bundle_2026-07-31.csv.gz` | Deterministic gzip containing only signed intervals |
| final manifest | `results/gross9_structural_clock_bundle_manifest_2026-07-31.json` | Manifest-last authority, including claim, sentinel, provenance, counters, and dual-rebuild receipts |

Temporary same-filesystem staging files are not artifacts and must never be
accepted as authority.

## Frozen Gross9 identity

Sleeve order is semantic and must remain exactly:

| Order | Sleeve | Configured weight | Side authority |
|---:|---|---:|---|
| 0 | `cand_rex_veto_7` | `1.6` | Exact REX decision: long `1`, short `-1` |
| 1 | `fresh_kimchi_fx` | `2.0` | Exclusive long/short gate: long `1`, short `-1` |
| 2 | `frozen_annual_rank7` | `3.0` | Long only: `1` |
| 3 | `markov_transition_long` | `2.0` | Long only: `1` |
| 4 | `rex_taker_low_range_position` | `0.4` | Exact REX decision: long `1`, short `-1` |

The configured weights sum to exactly `9.0`. Weight is provenance only. It
must not be converted into quantity, allocated equity, leverage, return, PnL,
or a portfolio path.

Side `0`, `AUTO`, `FLAT`, null side, and any side other than integer `1` or
`-1` are forbidden in the canonical CSV. `fresh_kimchi_fx` emits an interval
only when exactly one of its frozen long and short gates is active. Both-active
and neither-active states emit no interval.

## Canonical interval geometry

The structural clock uses half-open intervals:

```text
[entry_time_utc, exit_time_utc)
```

Every entry and exit must:

- use exact UTC-second text `YYYY-MM-DDTHH:MM:SSZ`;
- fall on a Unix-epoch multiple of `300` seconds;
- satisfy `domain_start <= entry_time_utc < exit_time_utc <= domain_end`;
- be sorted strictly by entry within its sleeve; and
- have no duplicate entry within its sleeve.

Within each sleeve, the next interval must satisfy:

```text
next.entry_time_utc >= previous.exit_time_utc
```

Touching half-open intervals are allowed. Cross-sleeve overlap is allowed and
must not be removed: non-overlap is a per-sleeve scheduling rule, not a
portfolio-wide rule.

Signal decisions use completed causal data. Entry is the next 5-minute open,
one bar after the signal. Fixed-hold intervals end at:

```text
entry bar index + hold_bars
```

The frozen fixed holds are:

| Sleeve | Hold |
|---|---:|
| `cand_rex_veto_7` | `144` five-minute bars |
| `markov_transition_long` | `576` five-minute bars |
| `rex_taker_low_range_position` | `144` five-minute bars |

`fresh_kimchi_fx` uses a maximum hold of `288` five-minute bars, a `400` bps
take, a `250` bps stop, and conservative stop-before-take resolution when both
barriers touch in one bar.

`frozen_annual_rank7` preserves the exact source-routed exits in its bound
bundle:

- funding leg: maximum hold `576`, take `400` bps, no enabled stop;
- premium leg: maximum hold `144`, stop `300` bps, no enabled take; and
- long side only.

For an OHLC-determined barrier exit, `exit_time_utc` is the first 5-minute
boundary after the first occupied bar that touches the frozen barrier. This
keeps a same-entry-bar exit positive and represents the occupied structural
bar as `[bar_open, bar_close)`. At a fixed horizon with no barrier touch,
`exit_time_utc` is the horizon boundary. The CSV never contains entry price,
exit price, high, low, barrier level, return, or PnL.

Signals that cannot produce a complete interval ending no later than the
exclusive domain end are omitted. Causal warm-up rows before the domain start
may be read when required by the frozen feature contract, but they must be
counted and cannot produce an output interval whose entry precedes the domain.

## Canonical CSV

The decompressed CSV columns and order are exactly:

```text
identity,sleeve,sleeve_order,configured_weight,interval_index,entry_time_utc,exit_time_utc,side
```

Field rules are:

- `identity` is always `G9CB-1`;
- `sleeve` is one of the five exact names above;
- `sleeve_order` is the canonical integer `0` through `4`;
- `configured_weight` is respectively `1.6`, `2.0`, `3.0`, `2.0`, or `0.4`;
- `interval_index` starts at `0` independently for each sleeve and increments
  by one;
- timestamps use the exact UTC form above; and
- `side` is the unquoted decimal integer `1` or `-1`.

Rows are ordered by `sleeve_order`, then `interval_index`. The ordering must
also be equivalent to sorting by `sleeve_order`, `entry_time_utc`,
`exit_time_utc`, and `side`; any disagreement is terminal.

CSV serialization is UTF-8 without BOM, RFC-4180 quoting with comma delimiter,
LF line endings, no blank lines, and one final LF. Gzip serialization is one
member, compression level `9`, empty original filename, `mtime=0`, no comment,
no extra field, XFL `2`, and OS byte `255`. The exact gzip prefix is:

```text
1f 8b 08 00 00 00 00 00 02 ff
```

The two independent passes must produce byte-identical decompressed CSV,
compressed CSV, and SHA-256.

## Canonical JSON

The preregistration, pre-access claim, attempt sentinel, per-pass core, and
final manifest use one JSON rule:

```text
UTF-8
object keys sorted lexicographically
separators ",", ":"
ensure_ascii=true
allow_nan=false
no insignificant whitespace
one trailing LF in the file
```

For an object carrying `manifest_hash`, compute:

```text
SHA256(canonical compact JSON of the object with manifest_hash excluded,
       without the trailing LF)
```

Then serialize the complete object by the rule above. `claim_hash` follows the
same rule with `claim_hash` excluded. File SHA-256 always hashes the complete
canonical file bytes, including the final LF.

Numbers must be integers or exact finite JSON decimals fixed by this decision.
Runtime-derived floating-point values are forbidden from every canonical
artifact.

## Direct generic authority

The following path/SHA/Git-blob records were verified at repository commit.
Config and manifest metadata was inspected only where needed to freeze this
contract; the pre-2025 anchor remained hash-only:

```text
91b41254319686f8b64bba797708f8e637aeddd3
```

Every record is mode `100644`. The preregistration artifact must reproduce the
same path, SHA-256, and Git blob directly. A later protocol commit may have a
different commit ID only if every listed blob remains identical.

| Authority | Path | SHA-256 | Git blob |
|---|---|---|---|
| Gross9 portfolio | `configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json` | `006f82e1f0affad9f96a08a6c600542feec4a0e1198ed99b8630627de4913450` | `a78173a3bd43a0c072e5e157d19579391bc10e29` |
| base portfolio | `configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json` | `3f6c929f6b03797093b8b81f50ede533176aa169f5f81a4bb5f616d31afd24ff` | `d8a2403f7e22dbe2c440c7ca031bc42e8557a86f` |
| portfolio runtime | `execution/portfolio_live.py` | `5edd4e9aa749e538d7de6a9990e31b94fbcb444b7e1498714cea82036962863d` | `801fae922f196c3d819b207045ba3f8d8c9f85d5` |
| Rank7 runtime | `execution/rank7_runtime.py` | `1ba1ab8f0af7cee0bac4885836776d50f2aff9dd30319d47e9a322f82f36c0dc` | `10294fe2b763de22c8928d061374600a2c90a1f8` |
| REX runtime | `execution/rex_llm_live.py` | `2e0de376e967b237afb711dd44503ec45dbb9b6548f575219c1cf93cc2de9c48` | `a4ab48081786f979ad20da03db39410e8545aaac` |
| transitive source manifest | `configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json` | `27a5095b18acaf10c9f5aa68c2ddac1ab1ebe4f506828e1fcfec34c414eb3ba6` | `9ff9d3efb3fcd0688fbce1a1694417089edc63df` |
| pre-2025 anchor, hash-only | `results/gross9_pre2025_authoritative_anchor_2026-07-28.json` | `329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875` | `f0f73d05b666ebc86adb2b068e0d6369c57c8da2` |
| REX-veto config | `configs/live/rex_veto_7_candidate.json` | `36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db` | `067a43c69b5433185c8c4a79e16e5d59597c9c0e` |
| fresh-kimchi config | `configs/shadow/fresh_kimchi_fx_2026-07-16.json` | `f3e764d5d065643905105ae1c46668a22684569289c3781b79fc6b2efcc5154f` | `310e65980b9e3987054fa6bc04e5abbab36d8cda` |
| Rank7 config | `configs/shadow/frozen_annual_rank7_2026-07-16.json` | `b75621bb604266d1cd2529a29f8bdb6aec3b1f2c14ff00d88673ef007362526d` | `29ec02f4dab2f49fc09360f65ff1c510347a7847` |
| Rank7 bundle manifest | `artifacts/rank7/frozen_annual_rank7_2026/manifest.json` | `2c45484dce48658ef7d342df7a3bb8e83cd0f31d4728bbb72fd38e612ec3b7a9` | `bd375546e6a273e59f14a58aea19f725a5aeb0ad` |
| Markov config | `configs/shadow/markov_transition_long_2026-07-16.json` | `ebfec66715428b2fffead13e17229fb4369816daeeeab2c02cf0115e7110b755` | `5f92d86cee2c617c590656c10ff05530196fc150` |
| REX-taker config | `configs/shadow/rex_taker_low_range_position_2026-07-16.json` | `d4c56a6f1659189876c1d3f2e519a3dbc2608c754720c5cd1f65a02adb5589e4` | `ede2d9d632f57eda2a4369a05d12916ef1f5ac5c` |
| project lock | `pyproject.toml` | `972713ffd03a621c8e3a5acf61b8aa5f7aa68d573d68415bfab34a5b68304e90` | `fa8a6907c7e965f588216f23a4e6e51e270bbea0` |
| resolution lock | `uv.lock` | `ff965ca88c9eb9f17efe74a6d550ab99d093b44cda2467cee6f5738fb60f770a` | `e4d529eca8110a530c362eb7883430bb81893140` |

The pre-2025 anchor is identity metadata only. Neither preregistration nor
production may parse its object, keys, counts, statistics, or any other value.
Its permanent counters are:

```text
pre2025_anchor_bytes_hashed = true
pre2025_anchor_git_blob_authenticated = true
pre2025_anchor_json_parsed = false
pre2025_anchor_value_rows_opened = 0
```

## Rank7 transitive bundle closure

The Rank7 manifest internal hash remains:

```text
06211697e4717f15db2c796da606c3785bfc25cac8ffa417fb3274063cb6ac8d
```

The following manifest-declared files must authenticate before use:

| Path | SHA-256 | Git blob |
|---|---|---|
| `artifacts/rank7/frozen_annual_rank7_2026/state/completed_hourly_history.csv.gz` | `8d3ef5bae39c36e9955caf8c30bc20deedf375aa2876da9070a32a3fbd0f2f08` | `e767b3edf7b9186c4d73566216c013573e30fb44` |
| `artifacts/rank7/frozen_annual_rank7_2026/models/seed_7.npz` | `b1f1c529cccabdd24465be995f9156fe211e5ad07792b0298ad42c2f1d4ddfb4` | `ae2bab7b43254b75d26778e6c0189c1d9a9f9d8b` |
| `artifacts/rank7/frozen_annual_rank7_2026/models/seed_71.npz` | `df53e7b99090171b87c7e9fe4ef14b3f2a318e371df7d9735bd4d16b89eac5f9` | `80a392b3f51122c22c385a71ea07265869a13db7` |
| `artifacts/rank7/frozen_annual_rank7_2026/models/seed_715.npz` | `ab9dff0aea41e4d55cd5c1a709c7ce061140891845e4596db30caf1505aaacf2` | `20de028a609a3a766c6720cdaab0f45d40166058` |
| `artifacts/rank7/frozen_annual_rank7_2026/models/seed_2026.npz` | `5938b411a04b8b34b2cbed97778da8be33a1dcf574b6ff63480b594ab94fd51a` | `7da6edb5b7d474699b5bd603ba64668248ea60c1` |
| `artifacts/rank7/frozen_annual_rank7_2026/models/seed_71515.npz` | `de955a31433722a61f18038195bdaad39efdb0a2cbfed3f6fe10dcd4a1ed63a5` | `3caf8ee1c83223c3a96d84733b8517b6a595c701` |

The builder may open model/history values only after the sentinel. It must
count model files loaded, hourly-history rows decoded, and prediction rows
scored. No model score or feature value may be emitted.

## Source closure

The committed transitive source manifest directly binds this exact ordered
inventory:

| Name | Logical path | SHA-256 |
|---|---|---|
| `market_5m` | `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz` | `a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c` |
| `funding` | `data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz` | `4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7` |
| `premium` | `data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz` | `b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7` |
| `open_interest` | `/tmp/btcusdt_open_interest_5m_2020_2026.csv` | `e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31` |
| `rex_taker_train` | `data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl` | `07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0` |
| `rex_taker_test` | `data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl` | `b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371` |
| `rex_taker_eval` | `data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl` | `bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7` |
| `rex_veto_source` | `data/rex_event_reasoning_policy_sft_20260712.jsonl` | `2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48` |

Hashing these bytes is metadata/hash access, not value-row access. Decoding a
header, record, field, array, or cell is value access and is forbidden before
the sentinel.

Two different open-interest byte artifacts must not be conflated.

Before any claim, the command-line inspection named
`data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz`, a
symlink to the home-repository frozen gzip. Opaque compressed bytes were
opened and displayed, but the stream was not decompressed or parsed. A later
metadata-only hash pass read the complete artifact. The preregistration and
pre-access claim must disclose:

```text
frozen_open_interest_gzip_logical_path
  data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz
frozen_open_interest_gzip_resolved_path
  /home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz
frozen_open_interest_gzip_size_bytes
  72898508
frozen_open_interest_gzip_sha256
  dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192
frozen_open_interest_gzip_opaque_bytes_opened_preclaim
  true
frozen_open_interest_gzip_decompressed_preclaim
  false
frozen_open_interest_gzip_headers_decoded_preclaim
  0
frozen_open_interest_gzip_rows_decoded_preclaim
  0
frozen_open_interest_gzip_fields_or_values_opened_preclaim
  0
```

Separately, the source manifest's operative open-interest input is the regular
CSV at `/tmp`. Its complete bytes were read only for metadata hashing:

```text
open_interest_logical_path
  /tmp/btcusdt_open_interest_5m_2020_2026.csv
open_interest_artifact_size_bytes
  19657777
open_interest_artifact_bytes_read_for_sha256_preclaim
  19657777
open_interest_sha256_preclaim
  e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31
open_interest_headers_decoded_preclaim
  0
open_interest_rows_decoded_preclaim
  0
open_interest_fields_or_values_opened_preclaim
  0
```

Neither byte count is a row count, and neither grants authority to decode an
artifact before the sentinel.

The current checkout is not production-ready: the logical market, funding, and
premium paths are symlinks, and the generic source manifest binds
open-interest to `/tmp`. These facts were established from filesystem metadata
without decoding rows. The claim-only commit is forbidden until every logical
source path is an exact-hash regular file backed durably enough for both fresh
rebuild processes. Replacing a symlink with identical regular-file bytes at
the same bound logical path is restoration; changing a logical path or hash is
not restoration and requires a new infrastructure identity. If the `/tmp`
open-interest authority cannot satisfy the durable two-pass gate unchanged,
`G9CB-1` stops before the sentinel.

The official run requires every source to be a readable regular file with the
exact bound SHA-256. A symlink, directory, socket, device, missing path, path
substitution, decompression drift, schema drift, duplicate timestamp, or hash
mismatch stops before the sentinel. The builder must not create, reconstruct,
download, repair, relink, or rewrite a missing source. Network access is
forbidden.

After the sentinel, production may open causal market, open-interest, funding,
premium, model, and REX source rows needed to reproduce the five intervals.
It may examine outcome-dependent OHLC paths solely to determine the exact
barrier exit boundary. Funding rows are causal features only; the builder must
not apply a funding cash flow.

## Runtime and environment closure

Before preregistration, the producer must discover the complete static
repository-local import closure reachable from:

```text
execution/portfolio_live.py
execution/rank7_runtime.py
execution/rex_llm_live.py
```

It must also discover the complete repository-local closure of every generic
module the new adapter plans to import. The preregistration artifact records
each closure member by path, SHA-256, Git blob, and package-initializer status.
The exact sorted path inventory is then immutable. Production independently
repeats AST discovery with stdlib-only code and requires exact set equality
before the sentinel. New, missing, reordered, dynamically substituted, or
untracked code is terminal.

The adapter may use generic functions only from that authenticated closure.
It must implement its own `G9CB-1` orchestration and interval serializer. It
must not import an existing candidate-specific Gross9 marginal runner as the
bundle authority.

The frozen environment is:

```text
Python implementation       CPython
Python version              3.10.10
platform                    Linux
machine                     x86_64
libc                        glibc 2.39
zlib compile/runtime        1.3 / 1.3
NumPy                       2.2.6
pandas                      2.3.3
SciPy                       1.15.3
scikit-learn                1.7.2
torch                       2.9.0
transformers                5.7.0.dev0
peft                        0.18.1
datasets                    4.6.1
trl                         0.29.0
websockets                  15.0.1
SQLAlchemy                  absent
installed distributions     108
distribution inventory SHA  a5b435e485426d7254ed222692bf3b9c6444ae992e582084398dc57b960549dc
```

The distribution inventory is a normalized
`[-_.]+ -> "-"`, lowercase name-to-version object hashed as canonical compact
JSON. A selected-package mismatch, full-inventory mismatch, ABI mismatch,
lockfile mismatch, or zlib mismatch stops before the sentinel. Production
must not install, remove, or upgrade a package.

## Preregistration artifact

The preregistration producer is metadata-only. It may:

- read this decision, implementation files, configs, manifests, package
  metadata, Git metadata, filesystem metadata, and bytes for hashing;
- discover and parse Python source only for static import closure;
- validate source path type and hash without decoding source records; and
- record exact path/SHA/Git-blob inventories.

It may not import a generic Gross9 runtime, parse the pre-2025 anchor, decode a
source header or row, open a model array/history row, reconstruct a clock, or
compute an outcome.

The artifact freezes:

- identity `G9CB-1` and this complete decision;
- the exact domain, sleeve order, weights, side rules, and interval geometry;
- the exact CSV and JSON byte contracts;
- every protocol, authority, runtime-closure, adapter-closure, Rank7, lock,
  environment, and source binding;
- the pre-access claim and sentinel schemas;
- the two-pass and manifest-last publication protocol;
- all access-counter names;
- all forbidden computations;
- `candidate_identity_present=false`;
- `candidate_artifacts_opened=false`;
- `comparator_clock_rows_opened=0`;
- `comparator_clocks_preseen_by_research_program=true`; and
- one-shot/no-retry/no-repair status.

Its creation-time evidence boundary is exactly:

```text
source_bytes_hashed                         = true
source_value_rows_opened                    = 0
pre2025_anchor_value_rows_opened            = 0
runtime_modules_imported                    = 0
esdi_runtime_or_private_invocations         = 0
model_files_loaded                          = 0
model_or_history_rows_opened                = 0
market_rows_opened                          = 0
open_interest_rows_opened                   = 0
funding_rows_opened                         = 0
premium_rows_opened                         = 0
outcome_dependent_ohlc_rows_opened          = 0
gross9_clock_rows_opened                    = 0
candidate_rows_opened                       = 0
comparator_clock_rows_opened                = 0
portfolio_return_or_pnl_computed            = false
funding_cash_computed                       = false
economic_metric_computed                    = false
candidate_or_overlap_metric_computed        = false
```

## Clean pushed Git seal and pre-access claim

Let `P` be the clean, pushed protocol commit containing:

- this decision;
- the preregistration producer and its tests;
- the committed preregistration artifact and its artifact test;
- the production builder and its tests; and
- no canonical access claim, sentinel, bundle CSV, or final manifest.

Every bound protocol and authority path must be tracked, byte-identical to
`P`, and represented by its recorded Git blob.

The builder's claim-only mode creates the pre-access claim without importing a
runtime or decoding a value row. Let `C` be the commit that adds that claim.
Production is authorized only if all of the following are true:

```text
C has exactly one parent
first_parent(C) == P
diff(P, C) == A results/gross9_structural_clock_bundle_access_claim_2026-07-31.json
HEAD == C
HEAD == @{upstream}
the worktree and index are clean
the claim is tracked and byte-identical to HEAD
all protocol and authority blobs remain byte-identical to P
the sentinel, CSV, and final manifest do not exist
```

The claim records `P`, `C`-independent parent bindings, every planned protocol
blob, the preregistration SHA/manifest, exact output paths, zero value access,
one-shot status, and `retry_allowed=false`. Because the claim cannot contain
its own commit ID before commit, production derives `C` from `HEAD`, proves
the direct-child and claim-only relation, and binds `C` into the sentinel and
final manifest.

The claim remains immutable after commit. It is not rewritten with production
timestamps, counters, status, or output hashes.

## Durable attempt-consumed sentinel

The canonical sentinel path is:

```text
results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json
```

After all stdlib-only authentication succeeds, the builder constructs the
sentinel bytes in memory. The sentinel binds:

- identity and protocol version;
- `P` and claim commit `C`;
- access-claim path, file SHA-256, and `claim_hash`;
- preregistration path, file SHA-256, and `manifest_hash`;
- exact canonical CSV and final-manifest paths;
- `status="attempt_consumed_before_runtime_or_value_access"`;
- `one_shot=true`;
- `retry_allowed=false`;
- `resume_allowed=false`;
- zero runtime imports and zero value-row access; and
- its own `manifest_hash`.

Publish it atomically in the destination directory:

1. create a unique same-directory regular staging file with exclusive create;
2. write the complete canonical bytes, `fsync` the file, and set mode `0444`;
3. hard-link the complete staging inode to the absent canonical path, which
   atomically fails if the canonical path already exists;
4. `fsync` the destination directory;
5. remove the staging name and `fsync` the directory again; and
6. reopen the canonical sentinel with no symlink following and verify exact
   bytes, SHA-256, `manifest_hash`, mode, and link target.

The next operation after successful sentinel verification must be the first
authenticated generic-runtime import or first value-row open. No unrelated
preparation may occur between them.

If the canonical sentinel already exists, production must not restart,
resume, replay, or infer completion from a partial output. If the process
raises, is killed, crashes, loses power, or fails any check after the
canonical sentinel is published, the sentinel remains and `G9CB-1` is
terminal with no retry.

A crash before the canonical hard-link exists is not a consumed value-access
attempt because runtime import and value access remain forbidden. A leftover
staging inode must be treated as a blocked pre-access condition and inspected
without opening source values; it cannot be silently reused as the canonical
sentinel.

## Production access and honest counters

Each rebuild pass must keep exact integer counters. The final core contains:

```text
file_access:
  bytes_read_by_logical_source
  source_files_opened
  model_files_opened
  runtime_modules_imported

rows_decoded:
  market_5m
  funding
  premium
  open_interest
  rex_taker_train
  rex_taker_test
  rex_taker_eval
  rex_veto_source
  rank7_hourly_history

rows_used:
  causal_feature_rows_by_source
  prediction_rows_scored
  outcome_dependent_ohlc_rows_examined

per_sleeve:
  signal_rows_evaluated
  intervals_emitted
  long_intervals
  short_intervals
  fixed_horizon_exits
  take_exits
  stop_exits
  outcome_dependent_ohlc_rows_examined
```

Counters increment at decode/examination time, not only when a row creates an
interval. A row used by two sleeves is counted in each applicable per-sleeve
counter and once in its physical source counter. The manifest documents this
distinction. No counter may be estimated from output length after the fact.

The permanent prohibited-output counters are:

```text
pre2025_anchor_value_rows_opened       = 0
candidate_rows_opened                  = 0
comparator_clock_rows_opened           = 0
portfolio_return_values_computed       = 0
portfolio_pnl_values_computed          = 0
funding_cash_values_computed           = 0
cagr_values_computed                   = 0
mdd_values_computed                    = 0
economic_rank_values_computed          = 0
candidate_metric_values_computed       = 0
overlap_metric_values_computed         = 0
```

The builder must not import or call an equity curve, return, PnL, funding-cash,
drawdown, CAGR, ranking, correlation, Jaccard, containment, or overlap helper.
Synthetic tests must fail if any prohibited key appears in the CSV, per-pass
core, or final manifest.

## Two independent rebuild passes

One consumed attempt performs exactly two rebuild passes. They are independent
fresh subprocesses:

- each starts a fresh Python interpreter after the sentinel;
- each imports the authenticated generic runtime closure independently;
- each reads the same hash-sealed durable inputs from the beginning;
- each uses a separate fresh same-filesystem staging directory;
- neither receives frames, arrays, clocks, caches, model objects, temporary
  files, or serialized intermediate state from the other; and
- neither reads the other pass's output.

Each pass emits only staged:

```text
gross9_structural_clock_bundle.csv.gz
gross9_structural_clock_bundle_core.json
```

The per-pass core contains the exact domain, schema, sleeve contract,
provenance inventories, claim binding, sentinel path/SHA/manifest, access
counters, evidence boundary, CSV byte length, decompressed CSV SHA-256, and
compressed CSV SHA-256. It contains no pass number, wall-clock time, process
ID, hostname, temporary path, random nonce, or other nondeterministic field.

Before publication, the parent requires:

```text
pass_1.csv.gz bytes == pass_2.csv.gz bytes
pass_1.core.json bytes == pass_2.core.json bytes
```

It then independently reparses and validates the CSV and core. Any mismatch,
empty required sleeve, schema violation, source drift, counter drift, or
validation failure is terminal because the sentinel has already consumed the
attempt.

## Manifest-last publication

After both passes match:

1. copy one verified CSV byte sequence to a same-directory exclusive staging
   file;
2. `fsync`, set mode `0444`, and hard-link that complete staging inode to the
   absent
   `results/gross9_structural_clock_bundle_2026-07-31.csv.gz`, failing
   atomically if the canonical path exists;
3. `fsync` the results directory;
4. construct the final manifest from the byte-identical core;
5. add two pass receipts carrying the identical CSV/core hashes;
6. bind the immutable access claim by path, SHA-256, `claim_hash`, `P`, and
   `C`;
7. bind the immutable sentinel by path, file SHA-256, and `manifest_hash`;
8. compute the final `manifest_hash`;
9. write the manifest to a same-directory exclusive staging file, `fsync`,
   set mode `0444`, and hard-link it to the absent canonical final-manifest
   path, failing atomically if that path exists; and
10. `fsync` the results directory.

After each successful canonical hard-link, remove only its staging name and
`fsync` the results directory again. The canonical inode remains unchanged.

The final manifest is the publication commit point and must be published last.
The CSV without the final manifest is an orphan, not authority. The final
manifest without an exact CSV, claim, and sentinel match is invalid.

The final manifest emits only:

- interval schema and hashes;
- exact authority, runtime, environment, Rank7, and source provenance;
- claim and sentinel provenance;
- deterministic rebuild receipts;
- access and interval counters;
- zero/prohibited-computation assertions;
- candidate absence; and
- the research-context disclosure that comparator clocks were pre-seen.

It must not embed source rows, feature values, model scores, prices, returns,
PnL, funding cash, economic statistics, candidate data, comparator data, or
overlap results.

## Commit, push, and future consumption

After successful local publication, run the committed artifact tests without
changing protocol code. The publication commit may add only:

```text
results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json
results/gross9_structural_clock_bundle_2026-07-31.csv.gz
results/gross9_structural_clock_bundle_manifest_2026-07-31.json
```

The access claim remains in its earlier claim-only commit. The publication
commit and branch must be pushed, the worktree must be clean, and
`HEAD == @{upstream}` before any future candidate may bind the artifact.

A future candidate must be separately named and preregistered after this
publication. Its preregistration must bind:

- the pushed publication commit;
- Git blobs and SHA-256 hashes of the CSV and final manifest;
- the final `manifest_hash`;
- the exact `G9CB-1` identity and domain;
- `candidate_identity_absent_during_bundle_creation=true`;
- `comparator_clocks_preseen_by_research_program=true`; and
- the fact that `G9CB-1` contains structural clocks only and grants no
  economic conclusion.

A future candidate may read the committed bundle only after its own
preregistration is committed and pushed. It must not claim that Gross9 or
comparator clocks were unseen, pristine, or candidate-blind at the research
program level.

## Exact stop and no-repair rules

Before sentinel publication, a metadata-only authentication failure stops
without value access. The only permitted recovery is restoring the exact
already-sealed bytes, Git state, source path type, or frozen environment.
Changing a path, hash, blob, package, schema, sleeve rule, domain, counter,
serializer, or runtime is not restoration and requires a new identity.

After sentinel publication, any failure is terminal. `G9CB-1` must not be:

- retried, resumed, restarted, checkpointed, or replayed;
- repaired by changing or wrapping a runtime;
- repaired by substituting a source, symlink, worktree, model, config, anchor,
  manifest, environment, or lock resolution;
- repaired by dropping, clipping, shifting, merging, or deduplicating an
  interval after seeing output;
- repaired by changing a side, hold, stride, barrier, same-bar rule, domain,
  order, weight, CSV field, counter, gzip option, or JSON rule;
- repaired by accepting one pass, a partial pass, an orphan CSV, or a
  nonidentical dual rebuild;
- repaired by deleting or modifying the sentinel;
- repaired by creating the final manifest after a crash;
- repaired by importing an Ethereum Settlement Demand Impulse helper or
  canonical output; or
- repaired under a new alpha candidate while retaining `G9CB-1`.

The terminal post-sentinel action is:

```text
TERMINAL_G9CB1_ATTEMPT_CONSUMED_NO_RETRY
```

Any successor must use a new infrastructure identity, a new decision, a new
preregistration, and a new claim. It must disclose the failed `G9CB-1`
attempt and any rows exposed before failure.

## Completion condition

`G9CB-1` is complete only when:

1. this decision is committed and pushed;
2. the preregistration producer/tests are committed and pushed;
3. the canonical preregistration artifact and artifact test are committed and
   pushed;
4. the builder/tests are committed and pushed;
5. the claim-only direct-child commit is committed and pushed;
6. the sentinel is atomically published before all runtime/value access;
7. two fresh-process rebuilds produce byte-identical CSV and core bytes;
8. the CSV and final manifest are published manifest-last;
9. artifact validation passes without protocol changes; and
10. the sentinel, CSV, and final manifest are committed and pushed with a
    clean `HEAD == @{upstream}` seal.

Until all ten conditions hold, no candidate may cite `G9CB-1` as a durable
Gross9 structural clock authority.
