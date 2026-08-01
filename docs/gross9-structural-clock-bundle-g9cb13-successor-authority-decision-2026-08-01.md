# Gross9 structural clock bundle G9CB-13 successor authority decision — 2026-08-01

Status: Authoritative execution artifact. Execution is authorized by the fresh repair RALPLAN Architect→Critic consensus recorded in `.omx/plans/g9cb13-ralplan-consensus-handoff.json`; this is not a draft planning artifact.

Context: The exact direct-parent historical context is Q12 commit `e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54`. The retained prior five-iteration failed Ralplan at `.omx/plans/g9cb13-ralplan-max-iteration-outcome.md` remains audit evidence only and does not determine this authority's status. The fresh approved reviews are `.omx/plans/g9cb13-repair-ralplan-architect-review-iteration2.md` (`APPROVE`) followed by `.omx/plans/g9cb13-repair-ralplan-critic-review-iteration2.md` (`APPROVE`), with `execution_handoff_authorized=true`.

Approved sources: `.omx/plans/prd-g9cb13-p12-terminal-successor.md`, `.omx/plans/test-spec-g9cb13-p12-terminal-successor.md`, `.omx/plans/g9cb13-ralplan-consensus-handoff.json`, and `.omx/context/g9cb12-p12-terminal-successor-20260801T084940Z.md`, with the prior A12 authority document used for established authority style.

Revision mode: fresh repair RALPLAN iteration 2, approved and promoted to execution authority.

## Requirements summary

This authority requires:

1. preserve the failed G9CB12 P12 invocation as historical evidence only;
2. keep the G9CB8 empty mode-0700 residue immutable and unrepurposed;
3. make Q13 mandatory on the adopted path, with no M13 artifact on that path;
4. freeze every future stage boundary, direct parent, command, mode, and count;
5. define exact T12 and H13 canonical schemas, including canonicalization and rejection rules;
6. introduce a two-layer preflight plan that does not consume the future one-shot;
7. keep candidate economics downstream of a verified H13 handoff;
8. terminate this infrastructure execution at candidate-specific Ralplan after verified H13, without opening candidate implementation or economics first;
9. analyze a fresh S13/M13 source-support path as a rejected, non-executable alternative only;
10. require a fresh Ralplan/authority if the exact adoption gate fails before A13; and
11. keep the planning artifacts strictly file-local to `.omx/plans/` and `.omx/context/`.
12. define exact P13/C13/D13/V13/H13 command-output evidence and the in-process H13 ordering mechanism, with no external H13 log path.

The mandatory topology is `Q12→A13→T12→Q13→P13→C13→D13→H13→candidate-specific Ralplan`. A13 has exact direct parent Q12 `e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54`. There is no M13, no automatic S13 fallback, and no authority to rerun S12 or the G9CB12 P12 command, call C12/D12/V12/H12, alter or reuse the G9CB8 residue, or open economics before a verified H13 handoff. Future commit OIDs are never predicted; future bindings are learned only by the exact procedures in the H13 successor-binding contract.

## Evidence summary

The names-only inventory digest is `sha256(LF.join(sorted(top_level_results_names)) + LF)` over UTF-8 bytes.
At every lifecycle point:

```text
observed = tracked_at_authenticated_head ∪ exact_three_residues ∪ active_untracked_prefix
```

The authoritative pre-T12 state is:

- tracked at Q12: count `1353`, digest `402ec9c2d47fb3abd355255fd898433adaf662dd2d299155e2d93e82d029189d`;
- observed: count `1356`, digest `28b65b05303434fccd35837a5b72214f30d205f9a5ab1a526e52af85b17dcd30`;
- active untracked prefix: empty;
- exact immutable residues, each an empty non-symlink untracked directory with mode `0700`:
  1. `results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef`;
  2. `results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23`;
  3. `results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761`.

All later exact counts, digests, and active-prefix transitions are frozen once in the successor lifecycle contract under the T12 schema. Residues are never members of `active_untracked_prefix`; they are the separate `exact_three_residues` set.

## Adopted-source evidence from the G9CB12 source-support artifact

| Field | Value |
| --- | --- |
| identity | `G9CB-12-SOURCE-SUPPORT` |
| version | `gross9_structural_clock_bundle_g9cb12_source_support_v1` |
| source_support_commit | `807c62b656476c1d5e3d47e3b949686371de3b0b` |
| support_hash | `67b234ed27b8be671597ac996d8dd674e0aa446fd0403333eaa102cd41ff5ab1` |
| source_manifest.path | `configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json` |
| source_manifest.sha256 | `e1104f952415f2867acc60ccfad874a47b89564b951306eb852b10be12c834a6` |
| source_manifest.size_bytes | `1718` |
| attempt_sentinel.path | `results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json` |
| attempt_sentinel.sha256 | `f7f4348095a287643155ebc6fc181d878a9b048ee61055514703941c1a422fc6` |
| attempt_sentinel.size_bytes | `3056` |

The exact `materialized_sources`, `raw_sources`, and `stage_bindings` adopted by A13 are serialized in the A13 authority schema below.

The following `adopted_source_bindings` object is normative and exact. Its canonical compact sorted-key UTF-8 bytes, using `ensure_ascii=false`, `allow_nan=false`, and no trailing LF, are exactly `5,186` bytes with SHA-256 `f33c061e7f555725621b3ad8b0bdd257ddf211997162ae9da32a4815cd674c4a`:

```json
{
  "materialized_sources": {
    "market_5m": {
      "filesystem_mode_octal": "0444",
      "first_timestamp": "2019-12-31T15:00:00Z",
      "frame_hash": "e59b0955c53933cb7f877ba5b1b9f925be339c91c405cec5c83df63a3fa0f6e2",
      "gzip": {
        "compresslevel": 9,
        "embedded_filename": "",
        "mtime": 0
      },
      "last_timestamp": "2026-05-31T23:55:00Z",
      "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb12_complete.csv.gz",
      "path_type": "regular_file",
      "rows": 674892,
      "schema": [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "tic",
        "day",
        "dxy",
        "kimchi_premium",
        "usdkrw",
        "btckrw",
        "dxy_available",
        "kimchi_available",
        "usdkrw_available",
        "external_any_available",
        "dxy_zscore",
        "dxy_momentum",
        "kimchi_premium_zscore",
        "kimchi_premium_change",
        "usdkrw_zscore",
        "usdkrw_momentum"
      ],
      "sha256": "b5d6f4b2edb4d3a7f68dc84097a5c3caf6947b2cf77ceb6a3e621d220068a282",
      "size_bytes": 69874443
    },
    "open_interest": {
      "filesystem_mode_octal": "0444",
      "first_timestamp": "2019-12-31T15:00:00Z",
      "frame_hash": "75cacfd52f48b91804097b9e5cbc478983a0be68ce3d0d1419bac67823c3bdab",
      "gzip": {
        "compresslevel": 9,
        "embedded_filename": "",
        "mtime": 0
      },
      "last_timestamp": "2026-05-31T23:55:00Z",
      "path": "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb12_complete.csv.gz",
      "path_type": "regular_file",
      "rows": 674892,
      "schema": [
        "date",
        "open_interest"
      ],
      "sha256": "dc73ba1d080bd4c4a493e315467ca998399a55914041c67c6ae08db84266ded8",
      "size_bytes": 5043062
    }
  },
  "raw_sources": [
    {
      "decoded_rows": 674785,
      "mode_octal": "0644",
      "name": "old_market",
      "normalized_rows": 674785,
      "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
      "path_type": "regular_file",
      "sha256": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
      "size_bytes": 66696659
    },
    {
      "decoded_rows": 684910,
      "mode_octal": "0644",
      "name": "replacement_market",
      "normalized_rows": 107,
      "path": "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz",
      "path_type": "regular_file",
      "sha256": "0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe",
      "size_bytes": 65420089
    },
    {
      "decoded_rows": 7029,
      "mode_octal": "0644",
      "name": "funding",
      "normalized_rows": 7029,
      "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
      "path_type": "regular_file",
      "sha256": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
      "size_bytes": 89326
    },
    {
      "decoded_rows": 56232,
      "mode_octal": "0644",
      "name": "premium",
      "normalized_rows": 56232,
      "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
      "path_type": "regular_file",
      "sha256": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
      "size_bytes": 1196481
    },
    {
      "decoded_rows": 674785,
      "mode_octal": "0644",
      "name": "old_open_interest",
      "normalized_rows": 674785,
      "path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
      "path_type": "regular_file",
      "sha256": "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
      "size_bytes": 19657777
    },
    {
      "decoded_rows": 604166,
      "mode_octal": "0644",
      "name": "binance_metrics_open_interest",
      "normalized_rows": 604166,
      "path": "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz",
      "path_type": "regular_file",
      "sha256": "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106",
      "size_bytes": 21440132
    },
    {
      "mode_octal": "0644",
      "name": "rank7_spot_premium_5m",
      "path": "/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz",
      "path_type": "regular_file",
      "sha256": "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617",
      "size_bytes": 15772146
    }
  ],
  "stage_bindings": [
    {
      "commit": "807c62b656476c1d5e3d47e3b949686371de3b0b",
      "parent_commit": "87c9d32df28f4b8c157d78e2d88145d6bfbb92c0",
      "stage": "S12",
      "tracked_files": [
        {
          "git_blob": "95d1e6f6ca91b7c7d7ac433c4103e301f3e70ce2",
          "git_mode": "100644",
          "path": "training/materialize_gross9_structural_clock_g9cb12_sources.py",
          "sha256": "83c5735f31e4382516f11b6809bffa808c06baa82203a9cf280e142268a06474",
          "size_bytes": 102387,
          "worktree_mode": "0644"
        },
        {
          "git_blob": "16c69bc9a1204635f3d4d49f15bac48f14a29543",
          "git_mode": "100644",
          "path": "tests/test_materialize_gross9_structural_clock_g9cb12_sources.py",
          "sha256": "5deda72f5e8384a6021ee4dcf5c5c4d6a1cccb01e34af301adcd6fc7b8c29fe2",
          "size_bytes": 41534,
          "worktree_mode": "0644"
        }
      ]
    },
    {
      "commit": "e9e8d7b0425b943c1def39553b861e3659b52a11",
      "parent_commit": "807c62b656476c1d5e3d47e3b949686371de3b0b",
      "stage": "M12",
      "tracked_files": [
        {
          "git_blob": "a80e238b94b46c7aa33e6008dc46d24aeba76393",
          "git_mode": "100644",
          "path": "results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json",
          "sha256": "f7f4348095a287643155ebc6fc181d878a9b048ee61055514703941c1a422fc6",
          "size_bytes": 3056,
          "worktree_mode": "0444"
        },
        {
          "git_blob": "8c1462de820b52cefbb1cbe9d49d832336a5dddf",
          "git_mode": "100644",
          "path": "configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json",
          "sha256": "e1104f952415f2867acc60ccfad874a47b89564b951306eb852b10be12c834a6",
          "size_bytes": 1718,
          "worktree_mode": "0444"
        },
        {
          "git_blob": "19337b117087f2c97ea5108d6befd0f1835a4423",
          "git_mode": "100644",
          "path": "results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json",
          "sha256": "b76532be19288da334aa53931555f763ef71bca03bb0aaf04f999ec65e0f2f04",
          "size_bytes": 18274,
          "worktree_mode": "0444"
        }
      ]
    }
  ]
}
```

## Known historical bindings from repository artifacts

### A12

- Commit: `a533ec5ec6bb01d0eeed8ab66a37a3a10f1dba5d`
- Parent: `646fccbf6568bcf39fab12a47873f72da880ca01`

| Path | Git blob | Git mode | SHA-256 | Size bytes | Worktree mode |
| --- | --- | --- | ---: | ---: | ---: |
| `docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md` | `f27653975eb4a1b7fd2ce057034fc26ad447a0ff` | `100644` | `1c10d085d9e38aad9568f8769795de38d9d8729bf41334db70c839723d64ba6f` | `44054` | `0644` |

### S12

- Commit: `807c62b656476c1d5e3d47e3b949686371de3b0b`
- Parent: `87c9d32df28f4b8c157d78e2d88145d6bfbb92c0`

| Path | Git blob | Git mode | SHA-256 | Size bytes | Worktree mode |
| --- | --- | --- | ---: | ---: | ---: |
| `training/materialize_gross9_structural_clock_g9cb12_sources.py` | `95d1e6f6ca91b7c7d7ac433c4103e301f3e70ce2` | `100644` | `83c5735f31e4382516f11b6809bffa808c06baa82203a9cf280e142268a06474` | `102387` | `0644` |
| `tests/test_materialize_gross9_structural_clock_g9cb12_sources.py` | `16c69bc9a1204635f3d4d49f15bac48f14a29543` | `100644` | `5deda72f5e8384a6021ee4dcf5c5c4d6a1cccb01e34af301adcd6fc7b8c29fe2` | `41534` | `0644` |

### M12

- Commit: `e9e8d7b0425b943c1def39553b861e3659b52a11`
- Parent: `807c62b656476c1d5e3d47e3b949686371de3b0b`

| Path | Git blob | Git mode | SHA-256 | Size bytes | Worktree mode |
| --- | --- | --- | ---: | ---: | ---: |
| `results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json` | `a80e238b94b46c7aa33e6008dc46d24aeba76393` | `100644` | `f7f4348095a287643155ebc6fc181d878a9b048ee61055514703941c1a422fc6` | `3056` | `0444` |
| `configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json` | `8c1462de820b52cefbb1cbe9d49d832336a5dddf` | `100644` | `e1104f952415f2867acc60ccfad874a47b89564b951306eb852b10be12c834a6` | `1718` | `0444` |
| `results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json` | `19337b117087f2c97ea5108d6befd0f1835a4423` | `100644` | `b76532be19288da334aa53931555f763ef71bca03bb0aaf04f999ec65e0f2f04` | `18274` | `0444` |

### Q12

- Commit: `e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54`
- Parent: `e9e8d7b0425b943c1def39553b861e3659b52a11`

| Path | Git blob | Git mode | SHA-256 | Size bytes | Worktree mode |
| --- | --- | --- | ---: | ---: | ---: |
| `training/build_gross9_structural_clock_bundle.py` | `5965d0f82323de96fbca4c2380549f7f04f4c5c9` | `100644` | `d095d52afb21f4bd7cd703a2aa46ec75291649748f98f17d9bbfcf87a19fbfd4` | `461964` | `0644` |
| `training/preregister_gross9_structural_clock_bundle.py` | `22f591b1913db243151f232935e6ac0450235ce3` | `100644` | `92197f0e86aea86af6fefef69d8a5879f0335921042a6fb7b504edf0b5365ca3` | `257603` | `0644` |
| `tests/test_build_gross9_structural_clock_bundle.py` | `374a2b1f573caf500efd8dad74d85802682873f4` | `100644` | `5eaa9a7ee38337955d715d7d7ff6499aa5d80bc975036a00ccc9eeebd9d223a3` | `365210` | `0644` |
| `tests/test_preregister_gross9_structural_clock_bundle.py` | `94f0de0e08a23cb56fdfdfabfc8edaa8cecd7fc7` | `100644` | `0d5cc4b181411e3daac94f74d5da3c9d6a6ed224cab6f9fb93b34c9ee1246404` | `113115` | `0644` |
| `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py` | `83b6beca0f7950db79333ced00c51784ddff3f5e` | `100644` | `89284d8baa45a3cf817d29ebc5e34e29178d3d4dbfc98db8b6d35f7f35d01475` | `21395` | `0644` |

## Selected executable topology

```text
Q12→A13→T12→Q13→P13→C13→D13→H13→candidate-specific Ralplan
```

This topology is mandatory. A13 is an additive one-file authority commit whose exact direct parent is Q12 `e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54`.

## Rejected alternative topology

```text
Q12 -> A13 -> T12 -> S13 -> M13 -> Q13 -> P13 -> C13 -> D13 -> H13 -> candidate-specific Ralplan
```

The fresh-source path is analyzed only as a rejected alternative. It is not executable under this authority. If the exact adoption gate fails before A13, stop and require a new Ralplan/authority; do not auto-fallback.

## RALPLAN-DR summary

### Principles

1. Terminal failure evidence stays historical and is never rewritten into recovery.
2. Authenticated historical source bytes may be consumed downstream, but G13 must never claim to have produced, replayed, or rewritten them.
3. Q13 is mandatory on the adopted path, and no M13 artifact exists on that path.
4. Exact inventory must be names-only, digestable, and non-consuming before any future one-shot boundary.
5. Candidate economics remain downstream of a verified H13 handoff only.

### Decision drivers

1. Preserve the P12 failure and G9CB8 residue exactly as historical evidence.
2. Make the adopted path serializable without placeholders, especially around T12, H13, V13, and the adoption bindings.
3. Keep the next one-shot boundary safe by proving inventory equality and non-consumption before any publication.

### Viable options

#### Option A — adopted committed S12/M12 evidence, then Q13

Treat the already committed S12/M12 artifacts as historical evidence under the successor authority. Freeze T12, require Q13 as the first code-changing stage, publish P13/C13/D13/H13 only after exact preflight, and route to candidate-specific Ralplan after verified H13.

Pros:
- recovery-free;
- smallest admissible delta;
- preserves successful source-support history;
- keeps Q13 mandatory without adding an adoption-path M13 artifact;
- matches the current no-economics stop rule.

Cons:
- requires careful wording so adoption remains evidence-only;
- demands an exact inventory gate that cannot consume the future one-shot;
- must keep future OIDs as learned values rather than guesses.

#### Option B — fresh S13/M13 source-support path, then Q13

Create a new S13/M13 source-support chain before Q13.

Pros:
- strongest provenance separation;
- no cross-identity reuse ambiguity;
- cleanest lineage if adoption is invalidated.

Cons:
- reruns source-support work;
- expands the one-shot surface;
- adds scope and delay;
- is non-executable under this authority because the adopted path is the only selected topology.

### Rejection rationale for Option B

Option B is fairly analyzed but rejected. This authority requires exact adoption of already committed S12/M12 evidence plus a mandatory Q13 repair boundary. If the adoption gate fails before A13, a fresh Ralplan/authority is required; this authority does not authorize an automatic fallback into S13/M13.

### Fresh consensus execution record

The durable fresh consensus record is `.omx/plans/g9cb13-ralplan-consensus-handoff.json`. Review order is exact: Architect iteration 2 `APPROVE`, then Critic iteration 2 `APPROVE`. The record has `ralplan_architect_review.complete=true`, `ralplan_critic_review.complete=true`, `ralplan_consensus_gate.complete=true`, and `ralplan_consensus_gate.execution_handoff_authorized=true`. The retained prior failed run has `execution_authorized=false` and is audit history only.

## Exact stage matrix

| Stage | Direct parent | Exact diff | Worktree / Git mode | Cardinality | Exact command | Invocation count |
| --- | --- | --- | --- | --- | --- | --- |
| A13 | Q12 | add one new authority document at `docs/gross9-structural-clock-bundle-g9cb13-successor-authority-decision-2026-08-01.md` | 0644 / 100644 | 1 added | none; metadata/authorship only | 0 |
| T12 | A13 | add one terminal-failure ledger at `results/gross9_structural_clock_bundle_g9cb12_p12_terminal_failure_2026-08-01.json` | 0444 / 100644 | 1 added | none; metadata seal only | 0 |
| Q13 | T12 | modify exactly five files: `training/build_gross9_structural_clock_bundle.py`, `training/preregister_gross9_structural_clock_bundle.py`, `tests/test_build_gross9_structural_clock_bundle.py`, `tests/test_preregister_gross9_structural_clock_bundle.py`, `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py` | 0644 / 100644 each | 5 modified | none; helper tests only | 0 |
| P13 | Q13 | add one preregistration JSON at `results/gross9_structural_clock_bundle_g9cb13_preregistration_2026-08-01.json` | 0444 / 100644 | 1 added | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle` | 1 |
| C13 | P13 | add one access-claim JSON at `results/gross9_structural_clock_bundle_g9cb13_access_claim_2026-08-01.json` | 0444 / 100644 | 1 added | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim` | 1 |
| D13 | C13 | add five outputs: `results/gross9_structural_clock_bundle_g9cb13_attempt_consumed_2026-08-01.json`, `results/gross9_structural_clock_bundle_g9cb13_worker_capability_consumed_pass1_2026-08-01.json`, `results/gross9_structural_clock_bundle_g9cb13_worker_capability_consumed_pass2_2026-08-01.json`, `results/gross9_structural_clock_bundle_g9cb13_2026-08-01.csv.gz`, `results/gross9_structural_clock_bundle_g9cb13_manifest_2026-08-01.json` | 0444 / 100644 each | 5 added | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce` | 1 |
| H13 | D13 | publish the supervisor sentinel first at `results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_attempt_consumed_2026-08-01.json`, run nested V13 once in memory, then add the verified handoff JSON at `results/gross9_structural_clock_bundle_g9cb13_v13_handoff_2026-08-01.json`; commit only the handoff JSON and leave the supervisor sentinel ignored/untracked | supervisor sentinel `0444` and ignored/untracked; handoff JSON `0444 / 100644` | 1 tracked add + 1 untracked sentinel | `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v13-handoff` | 1 |

## Exact T12 canonical terminal-failure schema

T12 is a duplicate-key-free canonical JSON object with exactly these top-level keys in this order:

```text
active_alpha_goal, failure, identity, inventory, ledger_kind, output_state,
predecessor_bindings, schema_version, status, terminal_failure_hash
```

### Field contract

- `schema_version` is JSON integer `1`.
- `active_alpha_goal` is the exact string `incomplete`.
- `identity` is the exact string `G9CB-12-P12-TERMINAL-FAILURE`.
- `ledger_kind` is the exact string `gross9_structural_clock_bundle_g9cb13_p12_terminal_failure_v1`.
- `status` is the exact string `historical_pre_sentinel_failure`.
- `terminal_failure_hash` is the SHA-256 self-hash of the canonical JSON object with that key omitted from the hash input; the hash input uses no trailing-LF bytes.
- Persisted JSON uses UTF-8, sorted keys, compact separators, `ensure_ascii=false`, `allow_nan=false`, and exactly one trailing LF.
- Duplicate keys, noncanonical key order, missing keys, extra keys, reordered arrays, retyped literals, invented access/decode counters, later-phase claims, and any unresolved future OIDs are rejected.
- The persisted ledger may never invent access/decode counters, sentinel counts, or later-phase publication claims.

### Failure object

`failure` is an object with exactly these keys in this order:

```text
command, exception_class, exception_message, exit_status, failure_location,
invocation_count, one_shot, phase, publication_state, publication_write_count,
resume_allowed, retry_allowed, sentinel_present
```

- `command` is the exact P12 command:
  `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle`
- `exception_class` is the exact string `FileExistsError`.
- `exception_message` is the exact string `Q12 exact results inventory differs`.
- `exit_status` is the exact integer `1`.
- `failure_location` is the exact string `training/preregister_gross9_structural_clock_bundle.py:6068`.
- `invocation_count` is the exact integer `1`.
- `one_shot` is JSON boolean `true`.
- `phase` is the exact string `prepublication_inventory_gate`.
- `publication_state` is the exact string `pre_sentinel_failure`.
- `publication_write_count` is the exact integer `0`.
- `resume_allowed` is JSON boolean `false`.
- `retry_allowed` is JSON boolean `false`.
- `sentinel_present` is JSON boolean `false`.
- No later phase may be serialized or implied.

### Inventory

The persisted T12 `inventory` records only the historical state at the failed P12 boundary. It never serializes future G9CB13 events. Its exact lexically ordered keys are:

```text
active_untracked_prefix, exact_three_residues, observed, tracked_at_authenticated_head
```

- `active_untracked_prefix` is the exact empty JSON array `[]`; P12 failed before publishing any result.
- `exact_three_residues` is an ordered array of exactly three rows. Every row has these lexically ordered keys and types:
  `empty` (boolean), `mode` (string), `name` (string), `path` (string), `symlink` (boolean), `tracked` (boolean).
- The exact rows, in order, are:

```json
[
  {
    "empty": true,
    "mode": "0700",
    "name": ".gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef",
    "path": "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef",
    "symlink": false,
    "tracked": false
  },
  {
    "empty": true,
    "mode": "0700",
    "name": ".gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23",
    "path": "results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23",
    "symlink": false,
    "tracked": false
  },
  {
    "empty": true,
    "mode": "0700",
    "name": ".gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761",
    "path": "results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761",
    "symlink": false,
    "tracked": false
  }
]
```

- `observed` has exactly the lexically ordered keys `count`, `digest`; values are integer `1356` and string `28b65b05303434fccd35837a5b72214f30d205f9a5ab1a526e52af85b17dcd30`.
- `tracked_at_authenticated_head` has exactly the lexically ordered keys `commit`, `count`, `digest`, `stage`; values are string `e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54`, integer `1353`, string `402ec9c2d47fb3abd355255fd898433adaf662dd2d299155e2d93e82d029189d`, and string `Q12`.
- Any extra, missing, reordered, renamed, nonempty, symlinked, tracked, or mode-drifted residue fails closed.

### Successor inventory lifecycle contract — not serialized in T12

Each lifecycle row is a verification expectation, not historical T12 content. A row has these lexically ordered keys:
`active_untracked_prefix`, `authenticated_head`, `lifecycle_point`, `observed`.
`authenticated_head` has keys `count` (integer), `digest` (string), `stage` (string); `observed` has keys `count` (integer), `digest` (string). Active prefixes contain only currently untracked stage outputs and never contain the three residues.

| Lifecycle point | Authenticated tracked head: stage / count / digest | Exact active untracked prefix | Observed count / digest |
| --- | --- | --- | --- |
| pre-T12 | Q12 / 1353 / `402ec9c2d47fb3abd355255fd898433adaf662dd2d299155e2d93e82d029189d` | `[]` | 1356 / `28b65b05303434fccd35837a5b72214f30d205f9a5ab1a526e52af85b17dcd30` |
| T12 active before commit | Q12 / 1353 / `402ec9c2d47fb3abd355255fd898433adaf662dd2d299155e2d93e82d029189d` | `[T12 ledger]` | 1357 / `e48ff41bfbee2e6c9a6fbdddd97ca6a76410a54a1e0392a556810f89d45beddf` |
| post-T12 commit | T12 / 1354 / `e3429999be21f32c53b7bfc274557f832d0a98ae03fdb92545497b9a7f708e1f` | `[]` | 1357 / `e48ff41bfbee2e6c9a6fbdddd97ca6a76410a54a1e0392a556810f89d45beddf` |
| pre-P13 after clean Q13 commit | Q13 / 1354 / `e3429999be21f32c53b7bfc274557f832d0a98ae03fdb92545497b9a7f708e1f` | `[]` | 1357 / `e48ff41bfbee2e6c9a6fbdddd97ca6a76410a54a1e0392a556810f89d45beddf` |
| P13 active before commit | Q13 / 1354 / `e3429999be21f32c53b7bfc274557f832d0a98ae03fdb92545497b9a7f708e1f` | `[P13]` | 1358 / `2f2d590ec572ef1cdcadec6d63908297f634c0fe60cd180a4bb2bef8e54c61a1` |
| post-P13 commit | P13 / 1355 / `6398b0b23922c39e6ba80a07ab19280c46281f6073e3d7a05e4ca45ddf68223d` | `[]` | 1358 / `2f2d590ec572ef1cdcadec6d63908297f634c0fe60cd180a4bb2bef8e54c61a1` |
| C13 active before commit | P13 / 1355 / `6398b0b23922c39e6ba80a07ab19280c46281f6073e3d7a05e4ca45ddf68223d` | `[C13]` | 1359 / `cfe2568f5c0ebc69766f27bc6fce9d88113fba6706aef89fb53a0c6ee60c1be6` |
| post-C13 commit | C13 / 1356 / `bf344bb677ff0fcb9dad150b0ab2c69771f309d2738be744b778ccdfbbac1502` | `[]` | 1359 / `cfe2568f5c0ebc69766f27bc6fce9d88113fba6706aef89fb53a0c6ee60c1be6` |
| D13 attempt | C13 / 1356 / `bf344bb677ff0fcb9dad150b0ab2c69771f309d2738be744b778ccdfbbac1502` | `[attempt]` | 1360 / `cd36815c72de184438712d51b966aa14b0b719598bc1c2554649037e4e14a99f` |
| D13 pass1 | C13 / 1356 / `bf344bb677ff0fcb9dad150b0ab2c69771f309d2738be744b778ccdfbbac1502` | `[attempt, pass1]` | 1361 / `9d6f64fee268a560f116b6d8515db5d4753899690f31b2cb552b09ac8923b8b0` |
| D13 pass2 | C13 / 1356 / `bf344bb677ff0fcb9dad150b0ab2c69771f309d2738be744b778ccdfbbac1502` | `[attempt, pass1, pass2]` | 1362 / `196f28a63887968d3e725e88e38c3897784aebc5f66df512a95c7fc4c976c51b` |
| D13 csv | C13 / 1356 / `bf344bb677ff0fcb9dad150b0ab2c69771f309d2738be744b778ccdfbbac1502` | `[attempt, pass1, pass2, csv]` | 1363 / `488e6255940d415e09baba0721ee78b40451f20ecb20e2a19e3d70758ee66476` |
| D13 manifest | C13 / 1356 / `bf344bb677ff0fcb9dad150b0ab2c69771f309d2738be744b778ccdfbbac1502` | `[attempt, pass1, pass2, csv, manifest]` | 1364 / `1a6871ed316cd4efbe4ea6865c679fe0aede78547372036eec395191b8a88420` |
| post-D13 commit | D13 / 1361 / `7d48ae557c4a2953b28acf6eb48db327c19d5f2fe1d7c38a163298b0bba46e06` | `[]` | 1364 / `1a6871ed316cd4efbe4ea6865c679fe0aede78547372036eec395191b8a88420` |
| H13 supervisor published | D13 / 1361 / `7d48ae557c4a2953b28acf6eb48db327c19d5f2fe1d7c38a163298b0bba46e06` | `[results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_attempt_consumed_2026-08-01.json]` | 1365 / `7938cf1198ca1667d53579a029609ac284b35723b5e156da804ad8844c04c8ab` |
| H13 handoff published before commit | D13 / 1361 / `7d48ae557c4a2953b28acf6eb48db327c19d5f2fe1d7c38a163298b0bba46e06` | `[results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_attempt_consumed_2026-08-01.json, results/gross9_structural_clock_bundle_g9cb13_v13_handoff_2026-08-01.json]` | 1366 / `c5b7ffea9018ef9ad042eb4ed1e9692edbe657d07e3f2383a53ba117e8131bb5` |
| post-H13 handoff commit | H13 / 1362 / `8fad688add4ce74b5096e5549aca023e4120ca2cdbcbee15972739026ad05d87` | `[results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_attempt_consumed_2026-08-01.json]` | 1366 / `c5b7ffea9018ef9ad042eb4ed1e9692edbe657d07e3f2383a53ba117e8131bb5` |

The symbolic active-prefix labels `H13 supervisor` and `H13 handoff` expand to the exact paths in the stage matrix, in that same order: `results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_attempt_consumed_2026-08-01.json` first and `results/gross9_structural_clock_bundle_g9cb13_v13_handoff_2026-08-01.json` second. The supervisor must be published before nested V13; the handoff is published only after V13 succeeds; only then is the handoff committed. Already committed T12/P13/C13/D13/H13 paths are excluded from every later active prefix.

### Output state

`output_state` is an exact object with these keys in this order:

```text
downstream_consumable, forbidden_g9cb12_commands, forbidden_g9cb12_stages,
permanently_absent_paths, permitted_g9cb13_commands, resume_allowed,
retry_allowed
```

- `downstream_consumable` is JSON boolean `false`.
- `resume_allowed` is JSON boolean `false`.
- `retry_allowed` is JSON boolean `false`.
- `permanently_absent_paths` is the exact ordered list of nine permanently absent G9CB12 paths:
  1. `results/gross9_structural_clock_bundle_g9cb12_preregistration_2026-07-31.json`
  2. `results/gross9_structural_clock_bundle_g9cb12_access_claim_2026-07-31.json`
  3. `results/gross9_structural_clock_bundle_g9cb12_attempt_consumed_2026-07-31.json`
  4. `results/gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass1_2026-07-31.json`
  5. `results/gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass2_2026-07-31.json`
  6. `results/gross9_structural_clock_bundle_g9cb12_2026-07-31.csv.gz`
  7. `results/gross9_structural_clock_bundle_g9cb12_manifest_2026-07-31.json`
  8. `results/gross9_structural_clock_bundle_g9cb12_v12_handoff_2026-07-31.json`
  9. `results/gross9_structural_clock_bundle_g9cb12_h12_supervisor_attempt_consumed_2026-07-31.json`
- `forbidden_g9cb12_stages` is the exact ordered list `S12`, `P12`, `C12`, `D12`, `V12`, `H12`.
- `forbidden_g9cb12_commands` is an exact ordered array of stage-scoped records with lexically ordered keys `command`, `generation`, `invocation_scope`, `stage`:

```json
[
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb12_sources",
    "generation": "G9CB12",
    "invocation_scope": "historical_only",
    "stage": "S12"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle",
    "generation": "G9CB12",
    "invocation_scope": "historical_only",
    "stage": "P12"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim",
    "generation": "G9CB12",
    "invocation_scope": "historical_only",
    "stage": "C12"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce",
    "generation": "G9CB12",
    "invocation_scope": "historical_only",
    "stage": "D12"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication",
    "generation": "G9CB12",
    "invocation_scope": "historical_only",
    "stage": "V12"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v12-handoff",
    "generation": "G9CB12",
    "invocation_scope": "historical_only",
    "stage": "H12"
  }
]
```

- `permitted_g9cb13_commands` is an exact ordered array of stage-scoped records with lexically ordered keys `command`, `generation`, `invocation_scope`, `stage`:

```json
[
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.preregister_gross9_structural_clock_bundle",
    "generation": "G9CB13",
    "invocation_scope": "adopted_path",
    "stage": "P13"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --create-claim",
    "generation": "G9CB13",
    "invocation_scope": "adopted_path",
    "stage": "C13"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v uv run python -B -m training.build_gross9_structural_clock_bundle --produce",
    "generation": "G9CB13",
    "invocation_scope": "adopted_path",
    "stage": "D13"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication",
    "generation": "G9CB13",
    "invocation_scope": "nested_once_inside_h13_only",
    "stage": "V13"
  },
  {
    "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v13-handoff",
    "generation": "G9CB13",
    "invocation_scope": "outer_h13_only",
    "stage": "H13"
  }
]
```

- Textually identical executable strings are distinguished by authenticated generation, stage, path constants, and head topology; no G9CB12 artifact, path, or identity may be targeted by a permitted G9CB13 invocation.
- `output_state` rejects any repurposing, rename, retry, or resume of these paths.

### Predecessor bindings

`predecessor_bindings` is the exact ordered array `[A12, Q12]`. Each stage row has lexically ordered keys `commit`, `parent_commit`, `stage`, `tracked_files`; each tracked-file row has lexically ordered keys `git_blob`, `git_mode`, `path`, `sha256`, `size_bytes`, `worktree_mode`.

```json
[
  {
    "commit": "a533ec5ec6bb01d0eeed8ab66a37a3a10f1dba5d",
    "parent_commit": "646fccbf6568bcf39fab12a47873f72da880ca01",
    "stage": "A12",
    "tracked_files": [
      {
        "git_blob": "f27653975eb4a1b7fd2ce057034fc26ad447a0ff",
        "git_mode": "100644",
        "path": "docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md",
        "sha256": "1c10d085d9e38aad9568f8769795de38d9d8729bf41334db70c839723d64ba6f",
        "size_bytes": 44054,
        "worktree_mode": "0644"
      }
    ]
  },
  {
    "commit": "e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54",
    "parent_commit": "e9e8d7b0425b943c1def39553b861e3659b52a11",
    "stage": "Q12",
    "tracked_files": [
      {
        "git_blob": "5965d0f82323de96fbca4c2380549f7f04f4c5c9",
        "git_mode": "100644",
        "path": "training/build_gross9_structural_clock_bundle.py",
        "sha256": "d095d52afb21f4bd7cd703a2aa46ec75291649748f98f17d9bbfcf87a19fbfd4",
        "size_bytes": 461964,
        "worktree_mode": "0644"
      },
      {
        "git_blob": "22f591b1913db243151f232935e6ac0450235ce3",
        "git_mode": "100644",
        "path": "training/preregister_gross9_structural_clock_bundle.py",
        "sha256": "92197f0e86aea86af6fefef69d8a5879f0335921042a6fb7b504edf0b5365ca3",
        "size_bytes": 257603,
        "worktree_mode": "0644"
      },
      {
        "git_blob": "374a2b1f573caf500efd8dad74d85802682873f4",
        "git_mode": "100644",
        "path": "tests/test_build_gross9_structural_clock_bundle.py",
        "sha256": "5eaa9a7ee38337955d715d7d7ff6499aa5d80bc975036a00ccc9eeebd9d223a3",
        "size_bytes": 365210,
        "worktree_mode": "0644"
      },
      {
        "git_blob": "94f0de0e08a23cb56fdfdfabfc8edaa8cecd7fc7",
        "git_mode": "100644",
        "path": "tests/test_preregister_gross9_structural_clock_bundle.py",
        "sha256": "0d5cc4b181411e3daac94f74d5da3c9d6a6ed224cab6f9fb93b34c9ee1246404",
        "size_bytes": 113115,
        "worktree_mode": "0644"
      },
      {
        "git_blob": "83b6beca0f7950db79333ced00c51784ddff3f5e",
        "git_mode": "100644",
        "path": "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
        "sha256": "89284d8baa45a3cf817d29ebc5e34e29178d3d4dbfc98db8b6d35f7f35d01475",
        "size_bytes": 21395,
        "worktree_mode": "0644"
      }
    ]
  }
]
```

A13 is not a T12 predecessor row: A13 is T12's direct Git parent and is bound later in H13 as a successor-stage row. No future commit is predicted inside T12.

### Exact G13 supervisor schema

The persisted G13 supervisor sentinel is a duplicate-key-free canonical JSON object with exactly these top-level keys in this order:

```text
attempt_hash, capability_sha256, expected_handoff_path, h13_command, identity,
one_shot, repository_head, repository_parent, resume_allowed, retry_allowed,
supervisor_pid, uv_executable, uv_executable_sha256, v13_command, zero_economics
```

### Field contract

- `attempt_hash` is the SHA-256 of the canonical compact JSON bytes of the 14-key core with `attempt_hash` omitted and no trailing LF bytes included in the hash input.
- `capability_sha256` is the lowercase SHA-256 of the fresh 32-byte capability learned at H13.
- `expected_handoff_path` is the exact H13 handoff path `results/gross9_structural_clock_bundle_g9cb13_v13_handoff_2026-08-01.json`.
- `h13_command` is the exact `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --publish-v13-handoff` command.
- `identity` is the exact string `G9CB-13-H13-SUPERVISOR`.
- `one_shot` is JSON boolean `true`.
- `repository_head` is the learned exact D13 commit.
- `repository_parent` is the learned exact C13 commit and the direct parent of D13.
- `resume_allowed` is JSON boolean `false`.
- `retry_allowed` is JSON boolean `false`.
- `supervisor_pid` is the positive runtime integer PID.
- `uv_executable` is the exact path `/home/pakchu/.local/bin/uv`.
- `uv_executable_sha256` is `085e6be0fbb5f63c7ba39829703a7229cd62d2bd0b78ae145da9bf897e0fc007`.
- `v13_command` is the exact `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication` command.
- `zero_economics` is JSON boolean `true`.
- Persisted supervisor bytes are canonical UTF-8 sorted compact JSON with exactly one trailing LF, are write-once at mode `0444`, carry empty economics, and are never serialized inside `successor_bindings`.
- Reject candidate/comparator/economic_result/economic-result keys and any noncanonical or duplicated key layout.
- The supervisor sentinel remains ignored/untracked and is not part of `successor_bindings`.

### Command evidence contract

The exact command-output contract is:

- `P13`: invocation `1`, exit `0`, stdout exactly one UTF-8 line: `created results/gross9_structural_clock_bundle_g9cb13_preregistration_2026-08-01.json <manifest_hash>\n`; stderr `0` bytes. `manifest_hash` is lowercase 64 hex and equals the persisted P13 `manifest_hash`.
- `C13`: invocation `1`, exit `0`, stdout `0` bytes, stderr `0` bytes; only the access-claim namespace add occurs.
- `D13`: invocation `1`, exit `0`, Python stdout `0` bytes, and Python stderr `0` bytes. Because the official command uses `/usr/bin/time -v`, outer stderr must consist only of the GNU-time telemetry block, include `Command being timed` and `Exit status: 0`, and contain no bytes before that anchored block. Any preamble before the block is treated as Python stderr and must be empty. The namespace grows in exact `attempt → pass1 → pass2 → csv → manifest` order.
- nested `V13`: invocation `1` inside H13 only, exit `0`, stdout exact canonical 13-key JSON followed by one LF, stderr `0` bytes, and zero file/HEAD/namespace changes.
- H13 outer: invocation `1`, exit `0`, stdout `0` bytes, stderr `0` bytes; supervisor + handoff namespace adds only.
- `A13`/`T12`/`Q13` official command count `0`.
- The ordering mechanism is the in-process state machine `PRE_SUPERVISOR -> SUPERVISOR_LINKED -> V13_VERIFIED -> HANDOFF_LINKED`, with only that transition order legal. After supervisor link, the namespace digest is taken; before/after nested V13 the digests must match; H13 handoff prelink recheck requires `V13_VERIFIED` and the exact supervisor-only namespace; commit occurs only after `HANDOFF_LINKED`.
- Add exactly the twelve named tests, remove none, and add no others. Those planned tests instrument the publish-helper/subprocess boundary and assert the exact event trace tuple without executing the official one-shot. The persisted supervisor self-hash plus the H13 `v13_stdout_hash` plus the state-machine test are the ordering evidence. No external H13 log path exists.

### Canonicalization and rejection rules

`canonicalization` must explicitly reject:

- duplicate keys;
- noncanonical JSON serialization;
- reordered keys or arrays;
- retyped literals;
- future commit prediction;
- recursively reject any nested `candidate`, `comparator`, `economic_result`, or `economic-result` key set and any value reachable beneath those keys; the required `no_economics=true` guard remains required;
- any supervisor sentinel in tracked bindings;
- any attempt to treat the successor handoff as final approval rather than a verified infrastructure handoff;
- any attempt to serialize the T12 failure as recoverable or as a later-phase success.

## Exact H13 canonical schema

The H13 handoff is a duplicate-key-free canonical JSON object with exactly these top-level keys in this order:

```text
active_alpha_goal, adopted_source_bindings, adopted_source_generation, identity,
ledger_kind, next_workflow, no_economics, no_future_commit_prediction,
schema_version, source_adoption_mode, successor_bindings,
successor_generation, t12_persisted_sha256, t12_terminal_failure_hash,
v13_stdout_hash
```

### Field contract

- `schema_version` is JSON integer `1`.
- `active_alpha_goal` is the exact string `incomplete`.
- `identity` is the exact string `G9CB-13-SOURCE-SUCCESSOR`.
- `ledger_kind` is the exact string `gross9_structural_clock_bundle_g9cb13_v13_handoff_v1`.
- `next_workflow` is the exact string `ralplan`.
- `no_economics` is JSON boolean `true`.
- `no_future_commit_prediction` is JSON boolean `true`.
- `adopted_source_generation` is the exact string `G9CB12`.
- `successor_generation` is the exact string `G9CB13`.
- `source_adoption_mode` is the exact string `authenticated_g9cb12_source_bytes_consumable_by_g9cb13_no_republication_v1`.
- `t12_persisted_sha256` is the SHA-256 of the complete persisted one-LF T12 file bytes.
- `t12_terminal_failure_hash` is the internal self-hash of the canonical T12 object with `terminal_failure_hash` omitted and no LF bytes included in that hash input.
- `v13_stdout_hash` is the SHA-256 of canonical nested V13 stdout bytes.
- Persisted JSON uses UTF-8, sorted keys, compact separators, `ensure_ascii=false`, `allow_nan=false`, and exactly one trailing LF.
- Duplicate keys, noncanonical serialization, reordered arrays, unresolved placeholders, future commit prediction, and any recursively forbidden `candidate`, `comparator`, `economic_result`, or `economic-result` key set are rejected; the required `no_economics=true` guard remains required.
- H13 is a verified infrastructure handoff conveying no candidate approval, never a draft artifact.
- Any explicit nested `candidate`, `comparator`, `economic_result`, or `economic-result` field set is recursively rejected; the required `no_economics` guard stays present.

### Adopted-source bindings

`adopted_source_bindings` is an exact object with these nested keys in lexical order: `materialized_sources`, `raw_sources`, `stage_bindings`.

- Its value must be canonical-JSON identical to the complete 5,186-byte UTF-8 object printed under **Exact A13 authority-document schema**; canonical bytes use sorted keys, compact separators, `ensure_ascii=false`, `allow_nan=false`, and no trailing LF for this equality check.
- The exact canonical object SHA-256 is `f33c061e7f555725621b3ad8b0bdd257ddf211997162ae9da32a4815cd674c4a`; any byte, key, type, value, row-order, or cardinality drift fails H13.
- That object contains exactly two full materialized-source rows, seven raw-source rows (six 8-key rows plus the 6-key `rank7_spot_premium_5m` row), and two full stage-binding rows `S12` then `M12`, including every tracked-file row shown there.
- `S12` and `M12` remain evidence-only authenticated historical bindings. Q13 and D13 may consume the immutable bytes, but G13 never claims to have produced, replayed, or rewritten them.

### Successor bindings

`successor_bindings` is an exact ordered array with six rows: `A13`, `T12`, `Q13`, `P13`, `C13`, `D13`.

Each row uses exactly these keys in this order: `commit`, `parent_commit`, `stage`, `tracked_files`.

#### Tracked-file row schema and cardinalities

The `tracked_files[]` arrays inside successor bindings are ordered arrays of exact tracked-file rows. Each tracked-file row uses exactly these keys in this order: `git_blob`, `git_mode`, `path`, `sha256`, `size_bytes`, `worktree_mode`.

Per-stage cardinalities are exact:

- `A13`: 1 tracked file, learned only from the committed A13 artifact
- `T12`: 1 tracked file, learned only from the committed T12 artifact
- `Q13`: 5 tracked files, learned only from the five exact Q13 paths
- `P13`: 1 tracked file, learned only from the committed P13 artifact
- `C13`: 1 tracked file, learned only from the committed C13 artifact
- `D13`: 5 tracked files, learned only from the committed D13 artifact

The learned-at-stage constraint is strict: the exact tracked-file rows are serialized only from the committed artifact for that stage, never by prediction or cross-stage copying. For each stage, the executor must:

1. bind `commit = git rev-parse <stage_commit>` and `parent_commit = git rev-parse <stage_commit>^`;
2. require `parent_commit` to equal the immediately preceding stage commit in the stage matrix;
3. derive the exact ordered path set with `git diff-tree --no-commit-id --name-status -r <parent_commit> <commit>` and require the declared cardinality and status;
4. derive `git_blob` and `git_mode` with `git ls-tree <commit> -- <path>`;
5. derive `size_bytes` with `git cat-file -s <git_blob>` and `sha256` from `git show <commit>:<path>` bytes;
6. require the checked-out worktree bytes to match the Git bytes and derive `worktree_mode` with `stat`;
7. reject a missing, extra, reordered, renamed, mode-drifted, or cross-stage path before H13 serialization.

- `A13` points at `docs/gross9-structural-clock-bundle-g9cb13-successor-authority-decision-2026-08-01.md`; the implementation must populate its commit-bound row from the committed A13 artifact before T12 serialization.
- `T12` points at `results/gross9_structural_clock_bundle_g9cb12_p12_terminal_failure_2026-08-01.json`.
- `Q13` points at exactly the five existing build/preregister/test files listed in the stage matrix.
- `P13` points at `results/gross9_structural_clock_bundle_g9cb13_preregistration_2026-08-01.json`.
- `C13` points at `results/gross9_structural_clock_bundle_g9cb13_access_claim_2026-08-01.json`.
- `D13` points at the five exact output files listed in the stage matrix.
- The H13 supervisor sentinel is intentionally not tracked, not serialized, and not part of `successor_bindings`.

### V13 nested verification contract

V13 is not a top-level stage. It is a nested verification call that occurs exactly once inside H13 and produces no files, commits, or publications.

- Exact command: `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.build_gross9_structural_clock_bundle --verify-publication`
- Command count: `1`
- Side effects: zero file writes, zero commits, zero publication artifacts, empty stderr
- Canonical stdout: exact 13-key JSON object with the exact 13-key order listed below, with G13 identity/protocol/topology values
- Exact ordered keys:
  1. `claim_commit`
  2. `claim_hash`
  3. `csv_gzip_sha256`
  4. `final_manifest_hash`
  5. `head`
  6. `identity`
  7. `interval_count`
  8. `preregistration_manifest_hash`
  9. `preregistration_seal_commit`
  10. `protocol_implementation_commit`
  11. `protocol_version`
  12. `publication_commit`
  13. `sentinel_manifest_hash`
- Type / learned-vs-literal constraints:

| Field | Type | Constraint |
| --- | --- | --- |
| `claim_commit` | string | learned exact commit from the claim stage |
| `claim_hash` | string | learned exact SHA-256 of the claim artifact |
| `csv_gzip_sha256` | string | learned exact SHA-256 of the D13 CSV |
| `final_manifest_hash` | string | learned exact SHA-256 of the D13 manifest |
| `head` | string | learned exact commit and must equal `publication_commit` |
| `identity` | string | literal `G9CB-13-SOURCE-SUCCESSOR` |
| `interval_count` | integer | learned exact non-negative row count |
| `preregistration_manifest_hash` | string | learned exact SHA-256 of the P13 preregistration |
| `preregistration_seal_commit` | string | learned exact P13 commit |
| `protocol_implementation_commit` | string | learned exact Q13 commit |
| `protocol_version` | string | literal `gross9_structural_clock_bundle_g9cb13_v1` |
| `publication_commit` | string | learned exact D13 commit |
| `sentinel_manifest_hash` | string | learned exact lowercase 64-hex `manifest_hash` embedded in `results/gross9_structural_clock_bundle_g9cb13_attempt_consumed_2026-08-01.json`; it is the D13 attempt sentinel self-hash computed over canonical compact JSON with `manifest_hash` omitted and no trailing LF, and is neither the H13 supervisor `attempt_hash` nor a file-byte SHA-256 |

## Two-layer preflight plan

### Layer 1 — synthetic helper tests before Q13 commit

Before the Q13 commit is created, the future five-file boundary is guarded by synthetic helper tests that only inspect deterministic fixtures and exact schemas. The exact tests are:

- `tests/test_preregister_gross9_structural_clock_bundle.py::test_q13_t12_terminal_failure_schema_exact`
- `tests/test_preregister_gross9_structural_clock_bundle.py::test_q13_t12_false_history_and_overclaim_rejected`
- `tests/test_preregister_gross9_structural_clock_bundle.py::test_q13_t12_adoption_exactness_and_drift_rejected`
- `tests/test_preregister_gross9_structural_clock_bundle.py::test_q13_t12_residual_active_g12_literals_rejected`
- `tests/test_preregister_gross9_structural_clock_bundle.py::test_q13_exact_adoption_gate_failure_requires_new_authority`
- `tests/test_build_gross9_structural_clock_bundle.py::test_q13_topology_and_frozen_paths_exact`
- `tests/test_build_gross9_structural_clock_bundle.py::test_q13_independent_inventory_oracle_rejects_production_oracle_copy`
- `tests/test_build_gross9_structural_clock_bundle.py::test_q13_real_worktree_inventory_preflight_exact`
- `tests/test_build_gross9_structural_clock_bundle.py::test_q13_v13_schema_and_direct_call_rejection_exact`
- `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py::test_q13_h13_schema_and_supervisor_leakage_exact`
- `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py::test_q13_preregistration_artifact_exact`
- `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py::test_q13_h13_verified_handoff_not_approval_and_no_economics_exact`

### Layer 2 — real-worktree pytest preflight after Q13 is committed and clean

After Q13 is committed, pushed, and the worktree is clean, run the exact real-worktree preflight under an explicit gate:

```text
G9CB13_REAL_WORKTREE_PREFLIGHT=1 PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error tests/test_build_gross9_structural_clock_bundle.py::test_q13_real_worktree_inventory_preflight_exact
```

This preflight has official one-shot invocation count `0` and does not consume the later publication command. P13 repeats the same implementation check immediately before publication.

### Exact five-path Q13 additive-test contract

The pre-Q13 baseline is exactly `705 passed, 1 skipped`. Q13 modifies exactly these five paths, with mode `0644` / `100644` for each, and no sixth path:

1. `training/build_gross9_structural_clock_bundle.py`
2. `training/preregister_gross9_structural_clock_bundle.py`
3. `tests/test_build_gross9_structural_clock_bundle.py`
4. `tests/test_preregister_gross9_structural_clock_bundle.py`
5. `tests/test_gross9_structural_clock_bundle_preregistration_artifact.py`

The twelve Layer 1 test names are absent before Q13. Add exactly those twelve named absent tests, remove none, add no others, and introduce no new skip. The exact five-path command is:

```text
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m pytest -q -W error training/build_gross9_structural_clock_bundle.py training/preregister_gross9_structural_clock_bundle.py tests/test_build_gross9_structural_clock_bundle.py tests/test_preregister_gross9_structural_clock_bundle.py tests/test_gross9_structural_clock_bundle_preregistration_artifact.py
```

Its required result is exactly `717 passed, 1 skipped`. The dedicated real-worktree gate remains exactly `1 passed` and consumes no official one-shot invocation. The test inventory oracle must independently derive tracked top-level names from Git, merge only the frozen three-name residue allowlist, mutate production declarations for negative checks, and reject drift, duplication, rename, symlink, and mode changes; it must not import or copy the production expected-inventory object.

## Acceptance criteria

1. A13, T12, Q13, P13, C13, D13, and H13 each have exact artifact paths, modes, and parent diffs that match the stage matrix.
2. Stage-parent name-status and summary audits match the declared stage parents, stage commits, and cardinalities.
3. P13, C13, D13, nested V13, and H13 outer each have recorded command counts, exit statuses, stdout/stderr behavior, and ordering evidence that match the stage matrix and command evidence contract.
4. Inventory snapshots match the exact tracked-head rows, residue rows, lifecycle points, active prefixes, and observed digests.
5. The five-path Q13 boundary (two training modules plus three test files) reaches `717 passed, 1 skipped` only if exactly 12 tests are additive with zero removals or new skips.
6. Bytecode-disabled commands are the only focused commands in the verification path.
7. The tree is clean at handoff, the downstream commit is created, `git push` succeeds, and `HEAD` matches `@{u}`.
8. H13 ordering is exact supervisor-sentinel publication first at `results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_attempt_consumed_2026-08-01.json`, nested V13 second, exact handoff publication third at `results/gross9_structural_clock_bundle_g9cb13_v13_handoff_2026-08-01.json`, and handoff commit last, with only the supervisor sentinel remaining untracked.
9. No economics open before verified H13, and the no-economics guard remains explicit and true.
10. The rejected fresh-source fallback remains non-executable.
11. No external H13 log path exists; ordering proof comes from the persisted supervisor self-hash, the H13 `v13_stdout_hash`, and the state-machine event trace.

## Premortem

| Scenario | Trigger | Detection | Containment | Owner | Stop |
| --- | --- | --- | --- | --- | --- |
| Adoption gate fails before A13 | Exact adoption evidence cannot be bound without guessing | A13 binding audit | Stop the plan and require a new Ralplan/authority; do not auto-fallback | planner, architect | No successor handoff |
| Stale G12 literals leak into active G13 fields | A G9CB12 label appears in an active G13 identity, path, protocol version, command scope, or successor binding rather than in authenticated historical/adopted evidence | Generation/stage-scoped schema and text audit | Preserve required historical G9CB12 evidence; reject or replace only the stale literal in the active G13 field | verifier | No handoff |
| Terminal failure overclaim appears | The T12 artifact claims a later phase or a sentinel that was never present | Failure-object audit | Keep `publication_state=pre_sentinel_failure`, `sentinel_present=false`, `publication_write_count=0`, `one_shot=true`, `retry_allowed=false`, `resume_allowed=false` | critic, verifier | No T12 approval |
| Inventory digest drifts | Name set, count, or digest changes unexpectedly | Top-level name digest audit | Fail closed; do not advance to publication | test-engineer | No publication command |
| Production/test oracle duplication appears | Tests copy the production inventory object instead of deriving their own | Source review and mutation test | Keep the oracle independent and mutation-based | test-engineer, verifier | No test approval |
| V13 direct invocation leaks | Nested V13 is exposed as a stage or file-producing command | Command-path audit | Restrict V13 to nested H13 execution only | executor, verifier | No H13 publish |
| H13 sentinel leaks into tracked state | Sentinel appears in Git or in serialized bindings | Inventory and H13 schema checks | Keep sentinel ignored/untracked and out of bindings | executor, verifier | No H13 publish |
| Economics open before H13 | Candidate/comparator/economic-result fields appear early | Schema and review checks | Fail closed; no economics until verified H13 | architect, verifier | No economics work |
| One-shot command is repeated | A second terminal one-shot is attempted | Invocation counter audit | Preserve the one-shot rule and stop immediately | planner, executor | No retry |

## Risks and mitigations

- **Risk: identity adoption is over-claimed.** Mitigation: keep adoption as evidence-only, not as a replay or rewrite.
- **Risk: the exact inventory preflight still misses a residue class.** Mitigation: use the exact three-directory allowlist and fail on any extra, missing, renamed, symlinked, or mode-drifted path.
- **Risk: the handoff leaks economics too early.** Mitigation: hard-code `no_economics=true`, require verified H13, and keep candidate-specific work separate.
- **Risk: future commit OIDs are guessed.** Mitigation: keep all future OIDs as learned exact values and reject any prediction.
- **Risk: this authority treats the rejected alternative as a fallback.** Mitigation: mark Option B non-executable and require a fresh Ralplan/authority if adoption fails before A13.

## Staffing roster and guidance

### Available agent types in this workspace

`explore`, `analyst`, `planner`, `architect`, `debugger`, `executor`, `team-executor`, `test-engineer`, `verifier`, `code-reviewer`, `dependency-expert`, `designer`, `writer`, `git-master`, `code-simplifier`, `researcher`, `prometheus-strict-metis`, `prometheus-strict-momus`, `prometheus-strict-oracle`, `critic`, `scholastic`, `vision`

### Sequential staffing dependencies

1. `researcher` confirms any remaining upstream doc or API details.
2. `planner` / `architect` lock the topology, schemas, and stop rules.
3. `executor` drafts the exact successor artifacts.
4. `test-engineer` builds the synthetic helper tests and the non-consuming real-worktree preflight.
5. `verifier` checks hashes, counts, path sets, and residue allowlists.
6. `critic` or `code-reviewer` challenges the adoption-vs-fallback decision before handoff.
7. `git-master` finalizes history hygiene only after the above evidence is stable.

### Follow-up staffing guidance

#### `$ultragoal`

Recommended default follow-up for durable goal execution after this handoff.

Suggested lane mix:

- 1 leader-owned ultragoal thread
- 1 `executor` for the successor authority and ledger drafts
- 1 `test-engineer` for inventory, schema, and preflight coverage
- 1 `verifier` for hashes, counts, and residue proof

Suggested reasoning levels:

- leader: high
- executor: medium
- test-engineer: medium
- verifier: high

Why this lane exists: it keeps the durable goal ledger in one place while still letting the implementation and validation threads move independently.

#### `$team`

Use when parallel ownership is materially useful, especially during authorized execution and the successor artifacts are being implemented in parallel.

Suggested headcount: 4 workers.

Suggested roles inside the single launch:

- 1 `executor` — draft the successor authority and terminal-failure ledger artifacts
- 1 `test-engineer` — build the exact inventory/preflight tests
- 1 `verifier` — validate file-line evidence, hashes, and stage topology
- 1 `critic` — challenge the adoption-vs-fallback branch and the no-economics boundary

Suggested reasoning levels:

- executor/test-engineer: medium
- verifier/critic: high

Launch hint:

```text
omx team 4 "inside the team run, assign 1 executor, 1 test-engineer, 1 verifier, and 1 critic; draft the G9CB13 successor authority, terminal-failure ledger, inventory preflight, hash checks, and no-economics review"
```

If the CLI cannot encode mixed roles explicitly, launch once and assign the four internal roles inside the team session; do not issue four separate launches.

#### `$ralph` fallback

Use only if a persistent single-owner completion/verification loop is intentionally preferred over Ultragoal + Team.

This is not the default path here.

### Team verification path

Before shutdown, Team must prove:

1. the exact topology and direct-parent claims are internally consistent;
2. the real-worktree preflight does not invoke the official one-shot command;
3. the three-directory 0700 residue allowlist is exact and exhaustive;
4. the H13 handoff schema blocks economics until verified H13 exists;
5. the adopted path never falls back automatically into S13/M13;
6. the supervisor sentinel is published before nested V13 and remains the only untracked H13 residue.

Ultragoal should checkpoint the Team evidence plus a fresh goal snapshot only after those conditions pass.

### Goal-mode follow-up suggestions

- `$ultragoal` — default follow-up for durable goal execution.
- `$autoresearch-goal` — only when the next work is a research deliverable with evaluator-backed evidence.
- `$performance-goal` — only when the next work is a measurable performance, latency, or throughput optimization project.

## ADR

**Decision:** prefer adoption of already committed S12/M12 evidence, require Q13 on the adoption path, and do not introduce an M13 artifact on that path.

**Drivers:** recovery-free continuity, exact inventory closure, preserved terminal evidence, and minimal successor delta.

**Alternatives considered:** fresh S13/M13 source-support rebuild; hybrid adoption plus fresh source rerun; no successor at all.

**Why chosen:** the successful source-support artifacts already exist in Git, the failure occurred at the preregistration inventory boundary, and Q13 can repair the successor topology without replaying source support.

**Consequences:** this authority must keep the adopted path explicit and guard it with a review gate. If adoption is rejected, a fresh Ralplan/authority is required instead of an automatic fallback.

**Follow-ups:** publish the terminal-failure ledger, verify the exact inventory preflight, and hand off to candidate-specific Ralplan only after the successor handoff is committed and verified.
