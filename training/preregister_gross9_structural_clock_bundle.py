"""Create the metadata-only G9CB-1 structural-clock preregistration.

This module deliberately has a stdlib-only import surface.  It authenticates
opaque bytes, Git metadata, permitted JSON metadata, static Python imports,
and the installed-distribution inventory.  It never imports a repository
runtime and never decodes an anchor, source, model, or history value.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zlib


PROTOCOL_VERSION = "gross9_structural_clock_bundle_preregistration_v1"
IDENTITY = "G9CB-1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_gross9_structural_clock_bundle.py"
)
PREREGISTRATION_TEST = Path(
    "tests/test_preregister_gross9_structural_clock_bundle.py"
)
ARTIFACT_TEST = Path(
    "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py"
)
BUILDER_SOURCE = Path("training/build_gross9_structural_clock_bundle.py")
BUILDER_TEST = Path("tests/test_build_gross9_structural_clock_bundle.py")
AUTHORITY_DECISION_PATH = Path(
    "docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md"
)
RANK7_AUTHORITY_AMENDMENT_PATH = Path(
    "docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md"
)
RUNTIME_ISOLATION_AMENDMENT_PATH = Path(
    "docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md"
)
PRIMITIVES_SOURCE = Path("training/gross9_structural_clock_primitives.py")
PRIMITIVES_TEST = Path("tests/test_gross9_structural_clock_primitives.py")
RANK7_FACADE_SOURCE = Path("execution/gross9_rank7_clock_runtime.py")
RANK7_FACADE_TEST = Path("tests/test_gross9_rank7_clock_runtime.py")
PREREGISTRATION_PATH = Path(
    "results/gross9_structural_clock_bundle_preregistration_2026-07-31.json"
)
ACCESS_CLAIM_PATH = Path(
    "results/gross9_structural_clock_bundle_access_claim_2026-07-31.json"
)
ATTEMPT_SENTINEL_PATH = Path(
    "results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json"
)
BUNDLE_PATH = Path("results/gross9_structural_clock_bundle_2026-07-31.csv.gz")
FINAL_MANIFEST_PATH = Path(
    "results/gross9_structural_clock_bundle_manifest_2026-07-31.json"
)
WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS = (
    Path(
        "results/"
        "gross9_structural_clock_bundle_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    Path(
        "results/"
        "gross9_structural_clock_bundle_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
)

AUTHORITY_DECISION_COMMIT = "e6ff406444a95068100cfacf617a3a23bcf918e3"
AUTHORITY_DECISION_SHA256 = (
    "e00ed64151fa6db292bf4ee8242f30a29dfab77c26c66eab751685bfa1c1b23a"
)
AUTHORITY_DECISION_GIT_BLOB = "709cba7091ad7010fa2619ba856c260cf4ddc2fd"
RANK7_AUTHORITY_AMENDMENT_COMMIT = (
    "f1ae4e68bfb0d0b861cd9979762f87e51a55f69d"
)
RANK7_AUTHORITY_AMENDMENT_SHA256 = (
    "a99b1a2b3d738ecc1cea8595eed2d88759c9b5fa7faf751a53b643fcc1a808cb"
)
RANK7_AUTHORITY_AMENDMENT_GIT_BLOB = (
    "0c7781ebe25178c592bb526ac51ee00c5ba840e2"
)
RUNTIME_ISOLATION_AMENDMENT_COMMIT = (
    "2550e0b8ee348b4217744a73d9781dba1e1e91a3"
)
RUNTIME_ISOLATION_AMENDMENT_SHA256 = (
    "354ae3870dd6dedf738b38bdd266d85b24389fe5de10d1fa0b3dbdde18d1c2de"
)
RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB = (
    "c2da15ff249e46a8fac2040d67f531a683b7fd7e"
)
DIRECT_AUTHORITY_VERIFICATION_COMMIT = "91b41254319686f8b64bba797708f8e637aeddd3"
EXPECTED_BRANCH = "codex/gross9-structural-clock-bundle-20260731"

PROTOCOL_PATHS = (
    RANK7_AUTHORITY_AMENDMENT_PATH,
    RUNTIME_ISOLATION_AMENDMENT_PATH,
    AUTHORITY_DECISION_PATH,
    PREREGISTRATION_SOURCE,
    PREREGISTRATION_TEST,
    ARTIFACT_TEST,
    BUILDER_SOURCE,
    BUILDER_TEST,
    PRIMITIVES_SOURCE,
    PRIMITIVES_TEST,
    RANK7_FACADE_SOURCE,
    RANK7_FACADE_TEST,
)
RUNTIME_IMPORT_ROOTS = (
    RANK7_FACADE_SOURCE,
    PRIMITIVES_SOURCE,
)

DIRECT_AUTHORITY_BINDINGS = (
    (
        "gross9_portfolio",
        "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json",
        "006f82e1f0affad9f96a08a6c600542feec4a0e1198ed99b8630627de4913450",
        "a78173a3bd43a0c072e5e157d19579391bc10e29",
    ),
    (
        "base_portfolio",
        "configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json",
        "3f6c929f6b03797093b8b81f50ede533176aa169f5f81a4bb5f616d31afd24ff",
        "d8a2403f7e22dbe2c440c7ca031bc42e8557a86f",
    ),
    (
        "portfolio_runtime",
        "execution/portfolio_live.py",
        "5edd4e9aa749e538d7de6a9990e31b94fbcb444b7e1498714cea82036962863d",
        "801fae922f196c3d819b207045ba3f8d8c9f85d5",
    ),
    (
        "rank7_runtime",
        "execution/rank7_runtime.py",
        "1ba1ab8f0af7cee0bac4885836776d50f2aff9dd30319d47e9a322f82f36c0dc",
        "10294fe2b763de22c8928d061374600a2c90a1f8",
    ),
    (
        "rex_runtime",
        "execution/rex_llm_live.py",
        "2e0de376e967b237afb711dd44503ec45dbb9b6548f575219c1cf93cc2de9c48",
        "a4ab48081786f979ad20da03db39410e8545aaac",
    ),
    (
        "transitive_source_manifest",
        "configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json",
        "27a5095b18acaf10c9f5aa68c2ddac1ab1ebe4f506828e1fcfec34c414eb3ba6",
        "9ff9d3efb3fcd0688fbce1a1694417089edc63df",
    ),
    (
        "pre2025_anchor_hash_only",
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json",
        "329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875",
        "f0f73d05b666ebc86adb2b068e0d6369c57c8da2",
    ),
    (
        "rex_veto_config",
        "configs/live/rex_veto_7_candidate.json",
        "36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db",
        "067a43c69b5433185c8c4a79e16e5d59597c9c0e",
    ),
    (
        "fresh_kimchi_config",
        "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
        "f3e764d5d065643905105ae1c46668a22684569289c3781b79fc6b2efcc5154f",
        "310e65980b9e3987054fa6bc04e5abbab36d8cda",
    ),
    (
        "rank7_config",
        "configs/shadow/frozen_annual_rank7_2026-07-16.json",
        "b75621bb604266d1cd2529a29f8bdb6aec3b1f2c14ff00d88673ef007362526d",
        "29ec02f4dab2f49fc09360f65ff1c510347a7847",
    ),
    (
        "rank7_bundle_manifest",
        "artifacts/rank7/frozen_annual_rank7_2026/manifest.json",
        "2c45484dce48658ef7d342df7a3bb8e83cd0f31d4728bbb72fd38e612ec3b7a9",
        "bd375546e6a273e59f14a58aea19f725a5aeb0ad",
    ),
    (
        "markov_config",
        "configs/shadow/markov_transition_long_2026-07-16.json",
        "ebfec66715428b2fffead13e17229fb4369816daeeeab2c02cf0115e7110b755",
        "5f92d86cee2c617c590656c10ff05530196fc150",
    ),
    (
        "rex_taker_config",
        "configs/shadow/rex_taker_low_range_position_2026-07-16.json",
        "d4c56a6f1659189876c1d3f2e519a3dbc2608c754720c5cd1f65a02adb5589e4",
        "ede2d9d632f57eda2a4369a05d12916ef1f5ac5c",
    ),
    (
        "project_lock",
        "pyproject.toml",
        "972713ffd03a621c8e3a5acf61b8aa5f7aa68d573d68415bfab34a5b68304e90",
        "fa8a6907c7e965f588216f23a4e6e51e270bbea0",
    ),
    (
        "resolution_lock",
        "uv.lock",
        "ff965ca88c9eb9f17efe74a6d550ab99d093b44cda2467cee6f5738fb60f770a",
        "e4d529eca8110a530c362eb7883430bb81893140",
    ),
)

RANK7_BUNDLE_MANIFEST_PATH = Path(
    "artifacts/rank7/frozen_annual_rank7_2026/manifest.json"
)
RANK7_BUNDLE_ROOT = RANK7_BUNDLE_MANIFEST_PATH.parent
RANK7_BUNDLE_MANIFEST_HASH = (
    "06211697e4717f15db2c796da606c3785bfc25cac8ffa417fb3274063cb6ac8d"
)
RANK7_FILE_BINDINGS = (
    (
        "state/completed_hourly_history.csv.gz",
        "8d3ef5bae39c36e9955caf8c30bc20deedf375aa2876da9070a32a3fbd0f2f08",
        "e767b3edf7b9186c4d73566216c013573e30fb44",
    ),
    (
        "models/seed_7.npz",
        "b1f1c529cccabdd24465be995f9156fe211e5ad07792b0298ad42c2f1d4ddfb4",
        "ae2bab7b43254b75d26778e6c0189c1d9a9f9d8b",
    ),
    (
        "models/seed_71.npz",
        "df53e7b99090171b87c7e9fe4ef14b3f2a318e371df7d9735bd4d16b89eac5f9",
        "80a392b3f51122c22c385a71ea07265869a13db7",
    ),
    (
        "models/seed_715.npz",
        "ab9dff0aea41e4d55cd5c1a709c7ce061140891845e4596db30caf1505aaacf2",
        "20de028a609a3a766c6720cdaab0f45d40166058",
    ),
    (
        "models/seed_2026.npz",
        "5938b411a04b8b34b2cbed97778da8be33a1dcf574b6ff63480b594ab94fd51a",
        "7da6edb5b7d474699b5bd603ba64668248ea60c1",
    ),
    (
        "models/seed_71515.npz",
        "de955a31433722a61f18038195bdaad39efdb0a2cbfed3f6fe10dcd4a1ed63a5",
        "3caf8ee1c83223c3a96d84733b8517b6a595c701",
    ),
)

SOURCE_MANIFEST_PATH = Path(
    "configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json"
)
SOURCE_BINDINGS = (
    (
        "market_5m",
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
    ),
    (
        "funding",
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
        "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
    ),
    (
        "premium",
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
        "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
    ),
    (
        "open_interest",
        "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
        "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
    ),
    (
        "rex_taker_train",
        "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
        "07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0",
    ),
    (
        "rex_taker_test",
        "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
        "b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371",
    ),
    (
        "rex_taker_eval",
        "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl",
        "bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7",
    ),
    (
        "rex_veto_source",
        "data/rex_event_reasoning_policy_sft_20260712.jsonl",
        "2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48",
    ),
)

FROZEN_OPEN_INTEREST_GZIP_PATH = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
)
FROZEN_OPEN_INTEREST_GZIP_RESOLVED_PATH = Path(
    "/home/pakchu/rllm/data/"
    "cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
)
FROZEN_OPEN_INTEREST_GZIP_SIZE = 72_898_508
FROZEN_OPEN_INTEREST_GZIP_SHA256 = (
    "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
)
OPEN_INTEREST_PATH = Path("/tmp/btcusdt_open_interest_5m_2020_2026.csv")
OPEN_INTEREST_SIZE = 19_657_777
OPEN_INTEREST_SHA256 = (
    "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31"
)

FROZEN_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.10.10",
    "platform": "Linux",
    "machine": "x86_64",
    "libc": "glibc 2.39",
    "zlib_compile": "1.3",
    "zlib_runtime": "1.3",
    "selected_distributions": {
        "datasets": "4.6.1",
        "numpy": "2.2.6",
        "pandas": "2.3.3",
        "peft": "0.18.1",
        "scikit-learn": "1.7.2",
        "scipy": "1.15.3",
        "sqlalchemy": "absent",
        "torch": "2.9.0",
        "transformers": "5.7.0.dev0",
        "trl": "0.29.0",
        "websockets": "15.0.1",
    },
    "distribution_count": 108,
    "distribution_inventory_sha256": (
        "a5b435e485426d7254ed222692bf3b9c6444ae992e582084398dc57b960549dc"
    ),
}
WORKER_PROCESS_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": REPOSITORY_ROOT.as_posix(),
    "PYTHONPYCACHEPREFIX": (
        REPOSITORY_ROOT / "results/.g9cb-bytecode-cache-disabled"
    ).as_posix(),
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}

SLEEVES = (
    {
        "order": 0,
        "name": "cand_rex_veto_7",
        "configured_weight": 1.6,
        "side_rule": "exact_rex_decision_integer_1_or_minus_1",
        "hold_bars": 144,
        "entry_delay_bars": 1,
    },
    {
        "order": 1,
        "name": "fresh_kimchi_fx",
        "configured_weight": 2.0,
        "side_rule": "exclusive_long_short_gate_integer_1_or_minus_1",
        "maximum_hold_bars": 288,
        "take_bps": 400,
        "stop_bps": 250,
        "same_bar_policy": "stop_before_take",
        "entry_delay_bars": 1,
    },
    {
        "order": 2,
        "name": "frozen_annual_rank7",
        "configured_weight": 3.0,
        "side_rule": "long_only_integer_1",
        "funding_exit": {"maximum_hold_bars": 576, "take_bps": 400, "stop": None},
        "premium_exit": {"maximum_hold_bars": 144, "take": None, "stop_bps": 300},
        "entry_delay_bars": 1,
    },
    {
        "order": 3,
        "name": "markov_transition_long",
        "configured_weight": 2.0,
        "side_rule": "long_only_integer_1",
        "hold_bars": 576,
        "entry_delay_bars": 1,
    },
    {
        "order": 4,
        "name": "rex_taker_low_range_position",
        "configured_weight": 0.4,
        "side_rule": "exact_rex_decision_integer_1_or_minus_1",
        "hold_bars": 144,
        "entry_delay_bars": 1,
    },
)

CREATION_EVIDENCE_BOUNDARY = {
    "source_bytes_hashed": True,
    "source_value_rows_opened": 0,
    "pre2025_anchor_value_rows_opened": 0,
    "runtime_modules_imported": 0,
    "esdi_runtime_or_private_invocations": 0,
    "model_files_loaded": 0,
    "model_or_history_rows_opened": 0,
    "market_rows_opened": 0,
    "open_interest_rows_opened": 0,
    "funding_rows_opened": 0,
    "premium_rows_opened": 0,
    "outcome_dependent_ohlc_rows_opened": 0,
    "gross9_clock_rows_opened": 0,
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "portfolio_return_or_pnl_computed": False,
    "funding_cash_computed": False,
    "economic_metric_computed": False,
    "candidate_or_overlap_metric_computed": False,
}

PERMANENT_PROHIBITED_COUNTERS = {
    "pre2025_anchor_value_rows_opened": 0,
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "portfolio_return_values_computed": 0,
    "portfolio_pnl_values_computed": 0,
    "funding_cash_values_computed": 0,
    "cagr_values_computed": 0,
    "mdd_values_computed": 0,
    "economic_rank_values_computed": 0,
    "candidate_metric_values_computed": 0,
    "overlap_metric_values_computed": 0,
}

SOURCE_COUNTER_NAMES = (
    "market_5m",
    "funding",
    "premium",
    "open_interest",
    "rex_taker_train",
    "rex_taker_test",
    "rex_taker_eval",
    "rex_veto_source",
    "rank7_hourly_history",
)
PER_SLEEVE_COUNTER_NAMES = (
    "signal_rows_evaluated",
    "intervals_emitted",
    "long_intervals",
    "short_intervals",
    "fixed_horizon_exits",
    "take_exits",
    "stop_exits",
    "outcome_dependent_ohlc_rows_examined",
)


def repository_path(
    path: str | os.PathLike[str], repository_root: Path = REPOSITORY_ROOT
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return candidate


def canonical_json_bytes(value: Any, *, trailing_lf: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_lf else b"")


def canonical_hash(
    value: Mapping[str, Any], excluded_key: str = "manifest_hash"
) -> str:
    body = dict(value)
    body.pop(excluded_key, None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def sha256_file(
    path: str | os.PathLike[str], repository_root: Path = REPOSITORY_ROOT
) -> str:
    digest = hashlib.sha256()
    with repository_path(path, repository_root).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_type(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "character_device"
    return "other"


def validate_file(
    path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    expected_type: str = "regular_file",
    expected_size: int | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    absolute = repository_path(path, repository_root)
    actual_type = _path_type(absolute)
    if actual_type != expected_type:
        raise ValueError(f"{path}: expected {expected_type}, found {actual_type}")
    size = absolute.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {size}")
    digest = sha256_file(absolute)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch")
    return {
        "path": str(path),
        "path_type": actual_type,
        "size_bytes": size,
        "sha256": digest,
    }


def _run_git(
    arguments: Sequence[str], repository_root: Path = REPOSITORY_ROOT
) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def git_blob(
    path: str | os.PathLike[str], repository_root: Path = REPOSITORY_ROOT
) -> tuple[str, str]:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(f"{path}: not a repository path") from error
    output = _run_git(
        ["ls-tree", "HEAD", "--", candidate.as_posix()], repository_root
    )
    fields = output.split(None, 3)
    if len(fields) != 4 or fields[1] != "blob":
        raise ValueError(f"{candidate}: not a tracked Git blob at HEAD")
    return fields[2], fields[0]


def validate_git_blob(
    path: str | os.PathLike[str],
    expected_blob: str,
    *,
    expected_mode: str = "100644",
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    blob, mode = git_blob(path, repository_root)
    if blob != expected_blob:
        raise ValueError(f"{path}: Git blob mismatch")
    if mode != expected_mode:
        raise ValueError(f"{path}: expected Git mode {expected_mode}, found {mode}")
    return {"git_blob": blob, "git_mode": mode}


def _tracked_binding(
    path: str | os.PathLike[str],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    expected_sha256: str | None = None,
    expected_blob: str | None = None,
) -> dict[str, Any]:
    absolute = repository_path(path, repository_root)
    if _path_type(absolute) != "regular_file":
        raise ValueError(f"{path}: tracked binding must be a regular file")
    digest = sha256_file(absolute)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch")
    blob, mode = git_blob(path, repository_root)
    if expected_blob is not None and blob != expected_blob:
        raise ValueError(f"{path}: Git blob mismatch")
    if mode != "100644":
        raise ValueError(f"{path}: expected Git mode 100644, found {mode}")
    return {
        "path": Path(path).as_posix(),
        "path_type": "regular_file",
        "sha256": digest,
        "git_blob": blob,
        "git_mode": mode,
    }


def validate_git_seal(
    repository_root: Path = REPOSITORY_ROOT,
    expected_branch: str = EXPECTED_BRANCH,
) -> dict[str, Any]:
    branch = _run_git(["branch", "--show-current"], repository_root)
    if branch != expected_branch:
        raise ValueError(f"expected branch {expected_branch}, found {branch}")
    upstream_name = _run_git(
        ["rev-parse", "--abbrev-ref", "@{upstream}"], repository_root
    )
    head = _run_git(["rev-parse", "HEAD"], repository_root)
    upstream = _run_git(["rev-parse", "@{upstream}"], repository_root)
    if head != upstream:
        raise ValueError("HEAD does not equal upstream")
    if _run_git(["status", "--porcelain=v1"], repository_root):
        raise ValueError("worktree or index is not clean")
    return {
        "expected_branch": expected_branch,
        "expected_upstream": f"origin/{expected_branch}",
        "required_head_equals_upstream": True,
        "required_worktree_and_index_clean": True,
        "observed_upstream_name": upstream_name,
    }


def normalized_distribution_inventory() -> dict[str, str]:
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ValueError("installed distribution has no Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = distribution.version
        previous = inventory.get(name)
        if previous is not None and previous != version:
            raise ValueError(f"conflicting installed versions for {name}")
        inventory[name] = version
    return dict(sorted(inventory.items()))


def environment_inventory() -> dict[str, Any]:
    inventory = normalized_distribution_inventory()
    libc_name, libc_version = platform.libc_ver()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "libc": f"{libc_name} {libc_version}",
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "selected_distributions": {
            name: inventory.get(name, "absent")
            for name in FROZEN_ENVIRONMENT["selected_distributions"]
        },
        "distribution_count": len(inventory),
        "distribution_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(inventory)
        ).hexdigest(),
        "distribution_inventory": inventory,
    }


def worker_process_environment(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    environment = dict(WORKER_PROCESS_ENVIRONMENT)
    root = repository_root.resolve()
    environment["PYTHONPATH"] = root.as_posix()
    environment["PYTHONPYCACHEPREFIX"] = (
        root / "results/.g9cb-bytecode-cache-disabled"
    ).as_posix()
    return environment


def validate_environment(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    actual = environment_inventory()
    for key, expected in FROZEN_ENVIRONMENT.items():
        if actual[key] != expected:
            raise ValueError(
                f"environment mismatch for {key}: expected {expected!r}, "
                f"found {actual[key]!r}"
            )
    actual["worker_process_environment"] = worker_process_environment(
        repository_root
    )
    return actual


def _relative_module_name(path: Path) -> tuple[str, bool]:
    initializer = path.name == "__init__.py"
    parts = list(path.with_suffix("").parts)
    if initializer:
        parts.pop()
    return ".".join(parts), initializer


def _module_files(module: str, repository_root: Path) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    discovered: list[Path] = []
    for index in range(1, len(parts)):
        initializer = Path(*parts[:index]) / "__init__.py"
        if (repository_root / initializer).is_file():
            discovered.append(initializer)
    module_file = Path(*parts).with_suffix(".py")
    package_file = Path(*parts) / "__init__.py"
    if (repository_root / module_file).is_file():
        discovered.append(module_file)
    elif (repository_root / package_file).is_file():
        discovered.append(package_file)
    return discovered


def _imported_local_paths(
    node: ast.Import | ast.ImportFrom,
    current_path: Path,
    repository_root: Path,
) -> set[Path]:
    modules: set[str] = set()
    current_module, current_is_initializer = _relative_module_name(current_path)
    current_package = (
        current_module
        if current_is_initializer
        else current_module.rpartition(".")[0]
    )
    if isinstance(node, ast.Import):
        modules.update(alias.name for alias in node.names)
    else:
        if node.level:
            package_parts = current_package.split(".") if current_package else []
            remove = node.level - 1
            if remove > len(package_parts):
                return set()
            prefix_parts = package_parts[: len(package_parts) - remove]
            if node.module:
                prefix_parts.extend(node.module.split("."))
            base = ".".join(prefix_parts)
        else:
            base = node.module or ""
        if base:
            modules.add(base)
        for alias in node.names:
            if alias.name != "*":
                modules.add(f"{base}.{alias.name}" if base else alias.name)
    paths: set[Path] = set()
    for module in modules:
        paths.update(_module_files(module, repository_root))
    return paths


def discover_import_closure(
    entry_paths: Iterable[str | os.PathLike[str]],
    repository_root: Path = REPOSITORY_ROOT,
) -> list[Path]:
    pending = {Path(path) for path in entry_paths}
    discovered: set[Path] = set()
    while pending:
        current = min(pending, key=lambda item: item.as_posix())
        pending.remove(current)
        if current in discovered:
            continue
        absolute = repository_path(current, repository_root)
        if _path_type(absolute) != "regular_file":
            raise ValueError(f"{current}: import-closure member is not a regular file")
        source = absolute.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=current.as_posix())
        discovered.add(current)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pending.update(
                    _imported_local_paths(node, current, repository_root) - discovered
                )
    return sorted(discovered, key=lambda item: item.as_posix())


def import_closure_inventory(
    entry_paths: Iterable[str | os.PathLike[str]],
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    inventory = []
    for path in discover_import_closure(entry_paths, repository_root):
        binding = _tracked_binding(path, repository_root=repository_root)
        binding["package_initializer"] = path.name == "__init__.py"
        inventory.append(binding)
    return inventory


def validate_import_closure(
    expected: Sequence[Mapping[str, Any]],
    entry_paths: Iterable[str | os.PathLike[str]],
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    actual = import_closure_inventory(entry_paths, repository_root)
    if actual != list(expected):
        raise ValueError("repository-local static import closure mismatch")
    return actual


def _load_json_metadata(path: Path, repository_root: Path) -> Any:
    if path == Path(
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json"
    ):
        raise ValueError("the pre-2025 anchor is hash-only and must not be parsed")
    with repository_path(path, repository_root).open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def validate_config_metadata(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    portfolio = _load_json_metadata(
        Path(DIRECT_AUTHORITY_BINDINGS[0][1]), repository_root
    )
    expected_weights = {sleeve["name"]: sleeve["configured_weight"] for sleeve in SLEEVES}
    if portfolio.get("weights") != expected_weights or portfolio.get(
        "gross_weight"
    ) != 9.0:
        raise ValueError("Gross9 portfolio weights do not match G9CB-1")

    base = _load_json_metadata(
        Path(DIRECT_AUTHORITY_BINDINGS[1][1]), repository_root
    )
    expected_base_sources = {
        "cand_rex_veto_7": "configs/live/rex_veto_7_candidate.json",
        "fresh_kimchi_fx": "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
        "frozen_annual_rank7": "configs/shadow/frozen_annual_rank7_2026-07-16.json",
        "markov_transition_long": (
            "configs/shadow/markov_transition_long_2026-07-16.json"
        ),
        "rex_taker_low_range_position": (
            "configs/shadow/rex_taker_low_range_position_2026-07-16.json"
        ),
    }
    observed_base_sources = {
        item["name"]: item["source"] for item in base.get("base_sleeves", [])
    }
    if observed_base_sources != expected_base_sources:
        raise ValueError("base portfolio sleeve-source bindings mismatch")

    checks = {
        "cand_rex_veto_7": {
            "path": Path("configs/live/rex_veto_7_candidate.json"),
            "side": "AUTO",
            "hold_bars": 144,
            "entry_delay_bars": 1,
        },
        "fresh_kimchi_fx": {
            "path": Path("configs/shadow/fresh_kimchi_fx_2026-07-16.json"),
            "side": "AUTO",
            "hold_bars": 288,
            "entry_delay_bars": 1,
            "take_bps": 400,
            "stop_bps": 250,
        },
        "frozen_annual_rank7": {
            "path": Path("configs/shadow/frozen_annual_rank7_2026-07-16.json"),
            "side": "LONG",
            "hold_bars": 576,
            "entry_delay_bars": 1,
            "bundle_manifest_hash": RANK7_BUNDLE_MANIFEST_HASH,
        },
        "markov_transition_long": {
            "path": Path("configs/shadow/markov_transition_long_2026-07-16.json"),
            "side": "LONG",
            "hold_bars": 576,
            "entry_delay_bars": 1,
        },
        "rex_taker_low_range_position": {
            "path": Path(
                "configs/shadow/rex_taker_low_range_position_2026-07-16.json"
            ),
            "side": "AUTO",
            "hold_bars": 144,
            "entry_delay_bars": 1,
        },
    }
    for name, contract in checks.items():
        metadata = _load_json_metadata(contract["path"], repository_root)
        for key, expected in contract.items():
            if key != "path" and metadata.get(key) != expected:
                raise ValueError(f"{name}: config metadata mismatch for {key}")
    return {
        "gross_weight": 9.0,
        "portfolio_weights": expected_weights,
        "base_sleeve_sources": expected_base_sources,
        "sleeve_contracts_authenticated": list(checks),
    }


def _rank7_declared_files(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[tuple[str, str]]:
    metadata = _load_json_metadata(RANK7_BUNDLE_MANIFEST_PATH, repository_root)
    if metadata.get("bundle_manifest_hash") != RANK7_BUNDLE_MANIFEST_HASH:
        raise ValueError("Rank7 internal manifest hash mismatch")
    declared = [
        (
            str(metadata.get("hourly_history", {}).get("path")),
            str(metadata.get("hourly_history", {}).get("sha256")),
        )
    ]
    declared.extend(
        (str(model.get("path")), str(model.get("sha256")))
        for model in metadata.get("models", [])
    )
    expected = [(path, digest) for path, digest, _blob in RANK7_FILE_BINDINGS]
    if declared != expected:
        raise ValueError("Rank7 manifest-declared file inventory mismatch")
    expected_exits = {
        "funding": {"hold_bars": 576, "stop_bps": 1_000_000, "take_bps": 400},
        "premium": {"hold_bars": 144, "stop_bps": 300, "take_bps": 1_000_000},
    }
    if metadata.get("exits_by_source") != expected_exits:
        raise ValueError("Rank7 source-routed exit metadata mismatch")
    return declared


def validate_rank7_bundle(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    _rank7_declared_files(repository_root)
    files = []
    for relative, digest, blob in RANK7_FILE_BINDINGS:
        path = RANK7_BUNDLE_ROOT / relative
        files.append(
            _tracked_binding(
                path,
                repository_root=repository_root,
                expected_sha256=digest,
                expected_blob=blob,
            )
        )
    return {
        "bundle_manifest_hash": RANK7_BUNDLE_MANIFEST_HASH,
        "declared_files": files,
    }


def _declared_sources(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[tuple[str, str, str]]:
    metadata = _load_json_metadata(SOURCE_MANIFEST_PATH, repository_root)
    if metadata.get("schema_version") != 1:
        raise ValueError("source manifest schema mismatch")
    declared = [
        (str(item.get("name")), str(item.get("path")), str(item.get("sha256")))
        for item in metadata.get("sources", [])
    ]
    if declared != list(SOURCE_BINDINGS):
        raise ValueError("source manifest ordered inventory mismatch")
    return declared


def _optional_git_metadata(
    path: str, repository_root: Path
) -> dict[str, str | None]:
    candidate = Path(path)
    if candidate.is_absolute():
        return {"git_blob": None, "git_mode": None}
    try:
        blob, mode = git_blob(candidate, repository_root)
    except (ValueError, subprocess.CalledProcessError):
        return {"git_blob": None, "git_mode": None}
    return {"git_blob": blob, "git_mode": mode}


def validate_sources(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    _declared_sources(repository_root)
    bindings = []
    for name, logical_path, digest in SOURCE_BINDINGS:
        absolute = repository_path(logical_path, repository_root)
        path_type = _path_type(absolute)
        if path_type not in {"regular_file", "symlink"}:
            raise ValueError(f"{logical_path}: invalid preregistration path type")
        resolved = absolute.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"{logical_path}: resolved source is not a regular file")
        actual = sha256_file(absolute)
        if actual != digest:
            raise ValueError(f"{logical_path}: source SHA-256 mismatch")
        binding = {
            "name": name,
            "logical_path": logical_path,
            "resolved_path": (
                str(resolved)
                if Path(logical_path).is_absolute()
                else Path(logical_path).as_posix()
            ),
            "path_type": path_type,
            "resolved_path_type": _path_type(resolved),
            "size_bytes": resolved.stat().st_size,
            "bytes_read_for_sha256_preclaim": resolved.stat().st_size,
            "sha256": actual,
        }
        binding.update(_optional_git_metadata(logical_path, repository_root))
        bindings.append(binding)
    return bindings


def source_preclaim_disclosures(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    frozen = repository_path(FROZEN_OPEN_INTEREST_GZIP_PATH, repository_root)
    frozen_type = _path_type(frozen)
    if frozen_type == "symlink":
        if frozen.resolve(strict=True) != FROZEN_OPEN_INTEREST_GZIP_RESOLVED_PATH:
            raise ValueError("frozen open-interest gzip resolved path mismatch")
    elif frozen_type != "regular_file":
        raise ValueError(
            "frozen open-interest gzip must be the disclosed symlink or "
            "an exact regular-file restoration"
        )
    if sha256_file(frozen) != FROZEN_OPEN_INTEREST_GZIP_SHA256:
        raise ValueError("frozen open-interest gzip SHA-256 mismatch")
    if frozen.resolve().stat().st_size != FROZEN_OPEN_INTEREST_GZIP_SIZE:
        raise ValueError("frozen open-interest gzip size mismatch")
    validate_file(
        OPEN_INTEREST_PATH,
        OPEN_INTEREST_SHA256,
        expected_size=OPEN_INTEREST_SIZE,
        repository_root=repository_root,
    )
    return {
        "frozen_open_interest_gzip_logical_path": str(
            FROZEN_OPEN_INTEREST_GZIP_PATH
        ),
        "frozen_open_interest_gzip_resolved_path": str(
            FROZEN_OPEN_INTEREST_GZIP_RESOLVED_PATH
        ),
        "frozen_open_interest_gzip_size_bytes": FROZEN_OPEN_INTEREST_GZIP_SIZE,
        "frozen_open_interest_gzip_sha256": FROZEN_OPEN_INTEREST_GZIP_SHA256,
        "frozen_open_interest_gzip_opaque_bytes_opened_preclaim": True,
        "frozen_open_interest_gzip_decompressed_preclaim": False,
        "frozen_open_interest_gzip_headers_decoded_preclaim": 0,
        "frozen_open_interest_gzip_rows_decoded_preclaim": 0,
        "frozen_open_interest_gzip_fields_or_values_opened_preclaim": 0,
        "open_interest_logical_path": str(OPEN_INTEREST_PATH),
        "open_interest_artifact_size_bytes": OPEN_INTEREST_SIZE,
        "open_interest_artifact_bytes_read_for_sha256_preclaim": (
            OPEN_INTEREST_SIZE
        ),
        "open_interest_sha256_preclaim": OPEN_INTEREST_SHA256,
        "open_interest_headers_decoded_preclaim": 0,
        "open_interest_rows_decoded_preclaim": 0,
        "open_interest_fields_or_values_opened_preclaim": 0,
    }


def _direct_authority_inventory(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    records = []
    for name, path, digest, blob in DIRECT_AUTHORITY_BINDINGS:
        record = _tracked_binding(
            path,
            repository_root=repository_root,
            expected_sha256=digest,
            expected_blob=blob,
        )
        record["name"] = name
        records.append(record)
    return records


def _protocol_inventory(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    return [
        _tracked_binding(path, repository_root=repository_root)
        for path in sorted(PROTOCOL_PATHS, key=lambda item: item.as_posix())
    ]


def _authority_decision_binding(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    binding = _tracked_binding(
        AUTHORITY_DECISION_PATH,
        repository_root=repository_root,
        expected_sha256=AUTHORITY_DECISION_SHA256,
        expected_blob=AUTHORITY_DECISION_GIT_BLOB,
    )
    binding["authority_commit"] = AUTHORITY_DECISION_COMMIT
    return binding


def _authority_amendment_binding(
    *,
    identity: str,
    path: Path,
    sha256: str,
    git_blob: str,
    authority_commit: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    binding = _tracked_binding(
        path,
        repository_root=repository_root,
        expected_sha256=sha256,
        expected_blob=git_blob,
    )
    binding = {"identity": identity, **binding}
    binding["authority_commit"] = authority_commit
    return binding


def _authority_amendment_bindings(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    return [
        _authority_amendment_binding(
            identity="G9CB-1A",
            path=RANK7_AUTHORITY_AMENDMENT_PATH,
            sha256=RANK7_AUTHORITY_AMENDMENT_SHA256,
            git_blob=RANK7_AUTHORITY_AMENDMENT_GIT_BLOB,
            authority_commit=RANK7_AUTHORITY_AMENDMENT_COMMIT,
            repository_root=repository_root,
        ),
        _authority_amendment_binding(
            identity="G9CB-1B",
            path=RUNTIME_ISOLATION_AMENDMENT_PATH,
            sha256=RUNTIME_ISOLATION_AMENDMENT_SHA256,
            git_blob=RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB,
            authority_commit=RUNTIME_ISOLATION_AMENDMENT_COMMIT,
            repository_root=repository_root,
        ),
    ]


def _manifest_without_hash(
    repository_root: Path, *, require_git_seal: bool
) -> dict[str, Any]:
    git_seal = (
        validate_git_seal(repository_root)
        if require_git_seal
        else {
            "expected_branch": EXPECTED_BRANCH,
            "expected_upstream": f"origin/{EXPECTED_BRANCH}",
            "required_head_equals_upstream": True,
            "required_worktree_and_index_clean": True,
            "observed_upstream_name": f"origin/{EXPECTED_BRANCH}",
        }
    )
    runtime_closure = import_closure_inventory(
        RUNTIME_IMPORT_ROOTS, repository_root
    )
    environment = validate_environment(repository_root)
    config_evidence = validate_config_metadata(repository_root)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "identity": IDENTITY,
        "authority_decision": _authority_decision_binding(repository_root),
        "direct_authority_verification_commit": (
            DIRECT_AUTHORITY_VERIFICATION_COMMIT
        ),
        "git_seal": git_seal,
        "candidate_independence": {
            "candidate_identity_present": False,
            "candidate_artifacts_opened": False,
            "comparator_clock_rows_opened": 0,
            "comparator_clocks_preseen_by_research_program": True,
        },
        "domain": {
            "start_inclusive": "2023-06-01T00:00:00Z",
            "end_exclusive": "2026-06-01T00:00:00Z",
            "bar_seconds": 300,
            "interval_semantics": "half_open",
        },
        "sleeves": list(SLEEVES),
        "configured_weight_sum": 9.0,
        "interval_geometry": {
            "timestamps": "YYYY-MM-DDTHH:MM:SSZ",
            "epoch_alignment_seconds": 300,
            "strict_entry_order_within_sleeve": True,
            "duplicate_entry_within_sleeve_forbidden": True,
            "per_sleeve_non_overlap": True,
            "touching_intervals_allowed": True,
            "cross_sleeve_overlap_allowed": True,
            "complete_intervals_only": True,
            "barrier_exit": (
                "first_5m_boundary_after_first_occupied_touching_bar"
            ),
        },
        "serialization": {
            "json": {
                "encoding": "UTF-8",
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": True,
                "allow_nan": False,
                "file_trailing_lf": True,
                "manifest_hash": (
                    "SHA256 canonical compact JSON excluding manifest_hash "
                    "without trailing LF"
                ),
            },
            "csv": {
                "columns": [
                    "identity",
                    "sleeve",
                    "sleeve_order",
                    "configured_weight",
                    "interval_index",
                    "entry_time_utc",
                    "exit_time_utc",
                    "side",
                ],
                "encoding": "UTF-8",
                "bom": False,
                "dialect": "RFC-4180",
                "delimiter": ",",
                "line_ending": "LF",
                "blank_lines": False,
                "final_lf": True,
                "row_order": ["sleeve_order", "interval_index"],
                "equivalent_sort": [
                    "sleeve_order",
                    "entry_time_utc",
                    "exit_time_utc",
                    "side",
                ],
                "allowed_sides": [1, -1],
            },
            "gzip": {
                "members": 1,
                "compression_level": 9,
                "original_filename": "",
                "mtime": 0,
                "comment": None,
                "extra_field": None,
                "xfl": 2,
                "os_byte": 255,
                "prefix_hex": "1f8b08000000000002ff",
            },
        },
        "bindings": {
            "protocol": _protocol_inventory(repository_root),
            "authority_amendments": _authority_amendment_bindings(
                repository_root
            ),
            "direct_authority": _direct_authority_inventory(repository_root),
            "config_metadata_evidence": config_evidence,
            "runtime_import_roots": [
                path.as_posix() for path in RUNTIME_IMPORT_ROOTS
            ],
            "runtime_import_closure": runtime_closure,
            "rank7_bundle": validate_rank7_bundle(repository_root),
            "source_manifest_ordered_inventory": validate_sources(repository_root),
            "environment": environment,
        },
        "pre2025_anchor_boundary": {
            "pre2025_anchor_bytes_hashed": True,
            "pre2025_anchor_git_blob_authenticated": True,
            "pre2025_anchor_json_parsed": False,
            "pre2025_anchor_value_rows_opened": 0,
        },
        "source_preclaim_disclosures": source_preclaim_disclosures(
            repository_root
        ),
        "creation_evidence_boundary": dict(CREATION_EVIDENCE_BOUNDARY),
        "output_paths": {
            "preregistration": PREREGISTRATION_PATH.as_posix(),
            "access_claim": ACCESS_CLAIM_PATH.as_posix(),
            "attempt_sentinel": ATTEMPT_SENTINEL_PATH.as_posix(),
            "worker_capability_consumption_ledgers": [
                path.as_posix()
                for path in WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS
            ],
            "canonical_csv_gzip": BUNDLE_PATH.as_posix(),
            "final_manifest": FINAL_MANIFEST_PATH.as_posix(),
        },
        "pre_access_claim_contract": {
            "parent_protocol_commit": "P",
            "claim_commit": "C",
            "claim_only_direct_child": True,
            "claim_hash_excludes_claim_hash": True,
            "zero_value_access": True,
            "retry_allowed": False,
        },
        "attempt_sentinel_contract": {
            "status": "attempt_consumed_before_runtime_or_value_access",
            "one_shot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "zero_runtime_imports": True,
            "zero_value_row_access": True,
            "canonical_mode_octal": "0444",
            "publish": "same_directory_fsync_hard_link_create_only",
            "next_operation": (
                "single_slot1_popen_with_anonymous_pipe_capability"
            ),
            "worker_capability_kind": "anonymous_pipe_v1",
            "worker_capability_slots": [1, 2],
        },
        "access_counter_names": {
            "file_access": [
                "bytes_read_by_logical_source",
                "source_files_opened",
                "model_files_opened",
                "runtime_modules_imported",
            ],
            "rows_decoded": list(SOURCE_COUNTER_NAMES),
            "rows_used": [
                "causal_feature_rows_by_source",
                "prediction_rows_scored",
                "outcome_dependent_ohlc_rows_examined",
                "rank7_training_trades_replayed",
                "rank7_net_labels_computed",
                "rank7_adverse_labels_computed",
                "rank7_price_factor_values_used",
                "rank7_funding_factor_values_used",
                "rank7_funding_debit_factor_values_used",
                "rank7_adverse_price_factor_values_used",
                "rank7_fee_factor_values_used",
                "rank7_bundle_activation_rows_scored",
                "rank7_bundle_parity_rows_compared",
            ],
            "per_sleeve": list(PER_SLEEVE_COUNTER_NAMES),
        },
        "permanent_prohibited_counters": dict(PERMANENT_PROHIBITED_COUNTERS),
        "two_pass_protocol": {
            "fresh_subprocesses": 2,
            "independent_runtime_imports": True,
            "independent_input_reads": True,
            "separate_same_filesystem_staging_directories": True,
            "cross_pass_state_forbidden": True,
            "compressed_csv_bytes_identical": True,
            "decompressed_csv_bytes_identical": True,
            "core_json_bytes_identical": True,
        },
        "publication_protocol": {
            "manifest_last": True,
            "canonical_mode_octal": "0444",
            "publish": "same_directory_fsync_hard_link_create_only",
            "csv_without_manifest_is_authority": False,
            "network_access": False,
        },
        "forbidden_computations": [
            "portfolio_return",
            "portfolio_pnl",
            "funding_cash",
            "cagr",
            "mdd",
            "economic_rank",
            "candidate_metric",
            "overlap_metric",
        ],
        "one_shot_policy": {
            "one_shot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "repair_allowed": False,
            "terminal_failure_action": (
                "TERMINAL_G9CB1_ATTEMPT_CONSUMED_NO_RETRY"
            ),
        },
    }


def build_manifest(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    require_git_seal: bool = True,
) -> dict[str, Any]:
    manifest = _manifest_without_hash(
        repository_root, require_git_seal=require_git_seal
    )
    manifest["manifest_hash"] = canonical_hash(manifest)
    validate_manifest(
        manifest,
        repository_root=repository_root,
        verify_files=False,
        verify_environment=False,
        verify_git_seal=False,
    )
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    verify_files: bool = True,
    verify_environment: bool = True,
    verify_git_seal: bool = True,
) -> None:
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("preregistration protocol version mismatch")
    if manifest.get("identity") != IDENTITY:
        raise ValueError("preregistration identity mismatch")
    if manifest.get("manifest_hash") != canonical_hash(manifest):
        raise ValueError("preregistration manifest_hash mismatch")
    if manifest.get("candidate_independence") != {
        "candidate_identity_present": False,
        "candidate_artifacts_opened": False,
        "comparator_clock_rows_opened": 0,
        "comparator_clocks_preseen_by_research_program": True,
    }:
        raise ValueError("candidate-independence boundary mismatch")
    if manifest.get("creation_evidence_boundary") != CREATION_EVIDENCE_BOUNDARY:
        raise ValueError("creation evidence boundary mismatch")
    amendments = manifest.get("bindings", {}).get("authority_amendments")
    expected_amendments = [
        {
            "identity": "G9CB-1A",
            "path": RANK7_AUTHORITY_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": RANK7_AUTHORITY_AMENDMENT_SHA256,
            "git_blob": RANK7_AUTHORITY_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": RANK7_AUTHORITY_AMENDMENT_COMMIT,
        },
        {
            "identity": "G9CB-1B",
            "path": RUNTIME_ISOLATION_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": RUNTIME_ISOLATION_AMENDMENT_SHA256,
            "git_blob": RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": RUNTIME_ISOLATION_AMENDMENT_COMMIT,
        },
    ]
    if amendments != expected_amendments:
        raise ValueError("authority amendment bindings mismatch")
    if manifest.get("source_preclaim_disclosures", {}).get(
        "frozen_open_interest_gzip_opaque_bytes_opened_preclaim"
    ) is not True:
        raise ValueError("accidental opaque gzip disclosure is missing")
    if manifest.get("source_preclaim_disclosures", {}).get(
        "frozen_open_interest_gzip_decompressed_preclaim"
    ) is not False:
        raise ValueError("gzip preclaim decompression disclosure mismatch")
    if verify_git_seal:
        validate_git_seal(repository_root)
    if verify_environment:
        actual_environment = validate_environment(repository_root)
        if manifest.get("bindings", {}).get("environment") != actual_environment:
            raise ValueError("manifest environment inventory mismatch")
    if verify_files:
        rebuilt = build_manifest(repository_root, require_git_seal=False)
        if rebuilt != dict(manifest):
            raise ValueError("preregistration does not match authenticated metadata")


def _validate_output_path(
    output: Path, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    canonical = repository_path(PREREGISTRATION_PATH, repository_root)
    resolved_output = repository_path(output, repository_root)
    if resolved_output != canonical:
        raise ValueError(f"only canonical preregistration path is allowed: {canonical}")
    if resolved_output.parent != repository_path("results", repository_root):
        raise ValueError("preregistration must be a singleton under results")
    return resolved_output


def write_once(
    manifest: Mapping[str, Any],
    output: Path = PREREGISTRATION_PATH,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> bool:
    validate_manifest(
        manifest,
        repository_root=repository_root,
        verify_files=True,
        verify_environment=True,
        verify_git_seal=True,
    )
    target = _validate_output_path(output, repository_root)
    expected_bytes = canonical_json_bytes(dict(manifest), trailing_lf=True)
    existing_type = _path_type(target)
    if existing_type != "missing":
        if existing_type != "regular_file":
            raise FileExistsError(
                "immutable preregistration path is not a regular file"
            )
        actual_bytes = target.read_bytes()
        if actual_bytes != expected_bytes:
            raise FileExistsError("immutable preregistration exists with other bytes")
        parsed = json.loads(actual_bytes)
        validate_manifest(parsed, repository_root=repository_root)
        return False

    target.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _path_type(target) != "regular_file":
                raise FileExistsError(
                    "immutable preregistration race produced a non-regular path"
                )
            actual_bytes = target.read_bytes()
            if actual_bytes == expected_bytes:
                return False
            raise FileExistsError(
                "immutable preregistration won a race with other bytes"
            )
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if target.read_bytes() != expected_bytes:
            raise RuntimeError("canonical preregistration byte verification failed")
        if stat.S_IMODE(target.stat().st_mode) != 0o444:
            raise RuntimeError("canonical preregistration mode verification failed")
        return True
    finally:
        temporary.unlink(missing_ok=True)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def load_existing_artifact(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    verify_files: bool = True,
) -> dict[str, Any]:
    target = _validate_output_path(PREREGISTRATION_PATH, repository_root)
    if _path_type(target) != "regular_file":
        raise ValueError("preregistration artifact must be a regular file")
    raw = target.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("preregistration must have exactly one trailing LF")
    manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest, trailing_lf=True):
        raise ValueError("preregistration bytes are not canonical JSON")
    validate_manifest(
        manifest,
        repository_root=repository_root,
        verify_files=verify_files,
        verify_environment=verify_files,
        verify_git_seal=verify_files,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify the existing canonical artifact without writing",
    )
    arguments = parser.parse_args(argv)
    if arguments.verify_existing:
        manifest = load_existing_artifact()
        print(manifest["manifest_hash"])
        return 0
    manifest = build_manifest()
    created = write_once(manifest)
    print(
        f"{'created' if created else 'verified'} {PREREGISTRATION_PATH} "
        f"{manifest['manifest_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
