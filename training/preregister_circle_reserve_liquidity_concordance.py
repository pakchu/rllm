"""Produce the frozen, source-unseen CRLC-336 preregistration artifact."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any, Iterable, Mapping, Sequence


POLICY_ID = "CRLC-336"
PROTOCOL_VERSION = "circle_reserve_liquidity_concordance_preregistration_v1"
EXPECTED_BRANCH = "codex/circle-reserve-liquidity-concordance-20260730"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DECISION_PATH = Path(
    "docs/circle-reserve-fund-nmfp-source-axis-decision-2026-07-30.md"
)
SOURCE_DECISION_SHA256 = (
    "e37218f2d5ba56b7c63938b9f5eca447562c9cd36af6acbf982032aaad0ee1a4"
)
SOURCE_DECISION_COMMIT = "0d99f67cef61a3ad9b33cd5b5783440322f93f93"
SOURCE_DECISION_GIT_BLOB = "b121dc5538585d38d227f396023e32a865ba0884"
MECHANISM_DECISION_PATH = Path(
    "docs/circle-reserve-liquidity-concordance-mechanism-decision-2026-07-30.md"
)
MECHANISM_DECISION_SHA256 = (
    "21aa1a242fb398e11201c1f587c41d86565bd114be8c37f30275eee60e27d08e"
)
MECHANISM_DECISION_COMMIT = "a061bcf30140c078e664fd91a8003733971bcff7"
MECHANISM_DECISION_GIT_BLOB = "803325b93b0399756c182fd220a4341911eb352f"
PRODUCER_PATH = Path(
    "training/preregister_circle_reserve_liquidity_concordance.py"
)
TEST_PATH = Path(
    "tests/test_preregister_circle_reserve_liquidity_concordance.py"
)
DEFAULT_OUTPUT = Path(
    "results/circle_reserve_liquidity_concordance_"
    "preregistration_2026-07-30.json"
)

ESDI_AUTHORITY_COMMIT = "f3de120a288f17e562e3f5cf7952ee77f6511fa7"
ESDI_PREREGISTRATION_PATH = Path(
    "results/ethereum_settlement_demand_impulse_"
    "preregistration_2026-07-30.json"
)
ESDI_PREREGISTRATION_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
ESDI_PREREGISTRATION_GIT_BLOB = "1b3d8b244426c0876d2995ce4a23159961d3cfa6"
ESDI_MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)
ESDI_NOVELTY_HELPER_PATH = Path(
    "training/preregister_ethereum_settlement_demand_impulse.py"
)
ESDI_NOVELTY_HELPER_SHA256 = (
    "1c7d7c822f16818ce0bc8fa0be99db0fe156882dbb76bf804ae19232f2a53b26"
)
ESDI_NOVELTY_HELPER_GIT_BLOB = "26c99dd77083cb0432160292c723da5ac2c019c3"
ESDI_ECONOMICS_HELPER_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_economics.py"
)
ESDI_ECONOMICS_HELPER_SHA256 = (
    "fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235"
)
ESDI_ECONOMICS_HELPER_GIT_BLOB = "ff305a740636bdf90a2bfd53ae13d93cd9f994b9"
ESDI_COMPARATOR_COUNT = 18
ESDI_COMPARATOR_SUBTREE_SHA256 = (
    "0d13c9de1e098446aaaa78b9a24c7d05c7ec375df05d79c9f8969792546bd4a3"
)
ESDI_GROSS9_SUBTREE_SHA256 = (
    "d79c79789ed48c7c2a94bac4474583798c2306bd320abb2617c354878c3578fe"
)
ESDI_GROSS9_AUTHORITY_SHA256 = (
    "b3490c484d3fda1d5b649498e0d84325e203cd2664086e68cebd76509a54957e"
)
ESDI_RUNTIME_CLOSURE_SHA256 = (
    "ffffb68c0900836ba06b573398c4825bd9d15161a9e36818aeb68fc33a86d84a"
)

FUTURE_PROTOCOL_PATHS = (
    Path("training/build_circle_reserve_fund_nmfp_source.py"),
    Path("tests/test_build_circle_reserve_fund_nmfp_source.py"),
    Path(
        "training/evaluate_circle_reserve_liquidity_concordance_"
        "source_support.py"
    ),
    Path(
        "tests/test_evaluate_circle_reserve_liquidity_concordance_"
        "source_support.py"
    ),
    Path(
        "training/evaluate_circle_reserve_liquidity_concordance_novelty.py"
    ),
    Path(
        "tests/test_evaluate_circle_reserve_liquidity_concordance_novelty.py"
    ),
    Path(
        "training/evaluate_circle_reserve_liquidity_concordance_economics.py"
    ),
    Path(
        "tests/test_evaluate_circle_reserve_liquidity_concordance_economics.py"
    ),
)

SOURCE_IDENTITY = {
    "registrant_name": "BlackRock Funds",
    "registrant_cik": "0000844779",
    "registrant_lei": "549300OZUEVJZHOBFP42",
    "series_name": "Circle Reserve Fund",
    "series_id": "S000077205",
    "series_lei": "549300X6KEJFVQHDAG85",
}
CONTEXT_ONLY_TICKER = "USDXX"
REPORT_MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year, month in (
        (2022 + (offset + 10) // 12, (offset + 10) % 12 + 1)
        for offset in range(42)
    )
)
DISCOVERY_MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year, month in (
        (2022 + (offset + 11) // 12, (offset + 11) % 12 + 1)
        for offset in range(42)
    )
)
DISCOVERY_DAY_COUNT = 630
ELIGIBLE_FORMS = ("N-MFP2", "N-MFP2/A", "N-MFP3", "N-MFP3/A")
ORIGINAL_FORMS = ("N-MFP2", "N-MFP3")
AMENDMENT_FORMS = ("N-MFP2/A", "N-MFP3/A")
CONTROL_NAMES = (
    "daily_path_only",
    "weekly_path_only",
    "wam_change_only",
    "wal_change_only",
    "path_pair",
    "maturity_pair",
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
    "one_bar_delayed_entry",
)
SOURCE_CONTROL_NAMES = CONTROL_NAMES[:6]
SAME_PARENT_CONTROL_NAMES = CONTROL_NAMES[6:]
CANDIDATE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
GROSS9_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}
PERIODS = {
    "2023H2": ("2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "2024": ("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "selection": ("2023-06-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "future25": ("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "future26": ("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    "combined_future": (
        "2025-01-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ),
    "full": ("2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
}
MINIMUM_SIGNALS = {
    "2023H2": 4,
    "2024": 8,
    "selection": 12,
    "future25": 8,
    "future26": 2,
    "full": 25,
}
NOVELTY_FUNCTIONS = (
    "entries_in_domain",
    "exact_entry_jaccard",
    "bidirectional_entry_containment",
    "fraction_at_most",
    "signed_exposure_5m",
    "occupied_bar_jaccard",
    "squared_signed_exposure_pearson",
)
EVIDENCE_BOUNDARIES = (
    "production_daily_indexes_opened",
    "production_feed_archives_opened",
    "source_incidence_opened",
    "liquidity_values_opened",
    "wam_wal_values_opened",
    "candidate_incidence_opened",
    "comparator_rows_opened",
    "gross9_rows_opened",
    "btc_market_rows_opened",
    "funding_rows_opened",
    "returns_opened",
    "pnl_opened",
    "cagr_opened",
    "strict_mdd_opened",
    "outcomes_opened",
)
TOP_LEVEL_TYPES = {
    "protocol_version": "str",
    "policy_id": "str",
    "status": "str",
    "singleton": "bool",
    "frozen_preregistration": "object",
    "source": "object",
    "feature_and_signal": "object",
    "execution": "object",
    "calendars": "object",
    "support_gates": "object",
    "controls": "object",
    "novelty": "object",
    "economic_contract": "object",
    "gross9": "object",
    "strict_sequence": "array[str]",
    "producer_effects": "object",
    **{name: "bool" for name in EVIDENCE_BOUNDARIES},
    "manifest_hash": "str",
}

_ACCESSION_RE = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or str(path).startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("CRLC-336 repository path is unsafe")
    return candidate


def _read_regular(path: str | Path) -> bytes:
    candidate = _safe_path(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(REPOSITORY_ROOT / candidate, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("CRLC-336 dependency is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("CRLC-336 Git identity validation failed") from error


def _git_blob(raw: bytes, object_id: str) -> str:
    if not _HEX40_RE.fullmatch(object_id):
        raise RuntimeError("CRLC-336 requires SHA-1 Git object format")
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"CRLC-336 {label} is not canonical JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"CRLC-336 {label} root must be an object")
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise RuntimeError(f"CRLC-336 {label} bytes are not canonical")
    return value


def _nested_file_authorities(value: Any) -> dict[Path, str]:
    found: dict[Path, str] = {}
    if isinstance(value, Mapping):
        if set(("path", "sha256")).issubset(value):
            path = value["path"]
            digest = value["sha256"]
            if isinstance(path, str) and isinstance(digest, str):
                candidate = _safe_path(path)
                if not _HEX64_RE.fullmatch(digest):
                    raise RuntimeError("CRLC-336 authority SHA-256 is invalid")
                previous = found.get(candidate)
                if previous is not None and previous != digest:
                    raise RuntimeError("CRLC-336 authority path hash conflicts")
                found[candidate] = digest
        for nested in value.values():
            for path, digest in _nested_file_authorities(nested).items():
                previous = found.get(path)
                if previous is not None and previous != digest:
                    raise RuntimeError("CRLC-336 authority path hash conflicts")
                found[path] = digest
    elif isinstance(value, list):
        for nested in value:
            for path, digest in _nested_file_authorities(nested).items():
                previous = found.get(path)
                if previous is not None and previous != digest:
                    raise RuntimeError("CRLC-336 authority path hash conflicts")
                found[path] = digest
    return found


def load_esdi_authority() -> dict[str, Any]:
    raw = _read_regular(ESDI_PREREGISTRATION_PATH)
    if hashlib.sha256(raw).hexdigest() != ESDI_PREREGISTRATION_SHA256:
        raise RuntimeError("CRLC-336 ESDI preregistration file SHA drift")
    payload = _canonical_json(raw, "ESDI preregistration")
    if payload.get("manifest_hash") != ESDI_MANIFEST_HASH:
        raise RuntimeError("CRLC-336 ESDI manifest hash drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != ESDI_MANIFEST_HASH:
        raise RuntimeError("CRLC-336 ESDI manifest is internally invalid")
    novelty = payload.get("novelty")
    gross9 = payload.get("gross9")
    if not isinstance(novelty, Mapping) or not isinstance(gross9, Mapping):
        raise RuntimeError("CRLC-336 ESDI authority is incomplete")
    comparators = novelty.get("frozen_comparator_artifacts")
    if not isinstance(comparators, Mapping) or len(comparators) != ESDI_COMPARATOR_COUNT:
        raise RuntimeError("CRLC-336 comparator registry count drift")
    if canonical_hash(comparators) != ESDI_COMPARATOR_SUBTREE_SHA256:
        raise RuntimeError("CRLC-336 comparator registry hash drift")
    if canonical_hash(gross9) != ESDI_GROSS9_SUBTREE_SHA256:
        raise RuntimeError("CRLC-336 Gross9 subtree hash drift")
    authority = gross9.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("CRLC-336 Gross9 authority is missing")
    if canonical_hash(authority) != ESDI_GROSS9_AUTHORITY_SHA256:
        raise RuntimeError("CRLC-336 Gross9 authority hash drift")
    closure = authority.get("runtime_code_closure")
    if not isinstance(closure, Mapping):
        raise RuntimeError("CRLC-336 runtime closure is missing")
    if canonical_hash(closure) != ESDI_RUNTIME_CLOSURE_SHA256:
        raise RuntimeError("CRLC-336 runtime closure hash drift")
    return {
        "frozen_comparator_artifacts": copy.deepcopy(dict(comparators)),
        "gross9": copy.deepcopy(dict(gross9)),
    }


def esdi_bound_path_hashes() -> dict[Path, str]:
    authority = load_esdi_authority()["gross9"]["authority"]
    found = _nested_file_authorities(authority)
    closure = authority["runtime_code_closure"]
    paths = closure.get("paths")
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise RuntimeError("CRLC-336 runtime closure paths are invalid")
    closure_sha = closure.get("sha256")
    if isinstance(closure_sha, Mapping):
        for raw_path in paths:
            digest = closure_sha.get(raw_path)
            if not isinstance(digest, str) or not _HEX64_RE.fullmatch(digest):
                raise RuntimeError("CRLC-336 closure path hash is missing")
            found[_safe_path(raw_path)] = digest
    else:
        # The ESDI artifact binds closure files through its repository identity.
        identity = load_esdi_preregistration_raw()["frozen_preregistration"][
            "repository_identity"
        ]
        sha = identity.get("sha256")
        if not isinstance(sha, Mapping):
            raise RuntimeError("CRLC-336 ESDI repository SHA map is missing")
        for raw_path in paths:
            digest = sha.get(raw_path)
            if not isinstance(digest, str) or not _HEX64_RE.fullmatch(digest):
                raise RuntimeError("CRLC-336 closure path SHA is missing")
            found[_safe_path(raw_path)] = digest
    return dict(sorted(found.items(), key=lambda item: str(item[0])))


def load_esdi_preregistration_raw() -> dict[str, Any]:
    return _canonical_json(
        _read_regular(ESDI_PREREGISTRATION_PATH),
        "ESDI preregistration",
    )


def validate_frozen_decision_commits() -> None:
    decision_authorities = (
        (
            SOURCE_DECISION_COMMIT,
            SOURCE_DECISION_PATH,
            SOURCE_DECISION_GIT_BLOB,
            "source-axis",
        ),
        (
            MECHANISM_DECISION_COMMIT,
            MECHANISM_DECISION_PATH,
            MECHANISM_DECISION_GIT_BLOB,
            "mechanism",
        ),
    )
    for commit, path, blob, label in decision_authorities:
        observed = _parse_tree(
            _git("ls-tree", "-z", commit, "--", str(path)),
            {str(path)},
        )
        if observed != {str(path): blob}:
            raise RuntimeError(f"CRLC-336 frozen {label} commit blob drift")


def validate_frozen_dependencies() -> None:
    expected = {
        SOURCE_DECISION_PATH: SOURCE_DECISION_SHA256,
        MECHANISM_DECISION_PATH: MECHANISM_DECISION_SHA256,
        ESDI_PREREGISTRATION_PATH: ESDI_PREREGISTRATION_SHA256,
        ESDI_NOVELTY_HELPER_PATH: ESDI_NOVELTY_HELPER_SHA256,
        ESDI_ECONOMICS_HELPER_PATH: ESDI_ECONOMICS_HELPER_SHA256,
    }
    expected.update(esdi_bound_path_hashes())
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(
                f"CRLC-336 frozen dependency changed: {path}: {actual} != {digest}"
            )
    authority_paths = {
        str(ESDI_PREREGISTRATION_PATH): ESDI_PREREGISTRATION_GIT_BLOB,
        str(ESDI_NOVELTY_HELPER_PATH): ESDI_NOVELTY_HELPER_GIT_BLOB,
        str(ESDI_ECONOMICS_HELPER_PATH): ESDI_ECONOMICS_HELPER_GIT_BLOB,
    }
    authority_blobs = _parse_tree(
        _git(
            "ls-tree",
            "-z",
            ESDI_AUTHORITY_COMMIT,
            "--",
            *sorted(authority_paths),
        ),
        set(authority_paths),
    )
    if authority_blobs != authority_paths:
        raise RuntimeError("CRLC-336 ESDI authority commit blob drift")
    validate_frozen_decision_commits()
    load_esdi_authority()


def committed_identity_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                SOURCE_DECISION_PATH,
                MECHANISM_DECISION_PATH,
                PRODUCER_PATH,
                TEST_PATH,
                ESDI_PREREGISTRATION_PATH,
                ESDI_NOVELTY_HELPER_PATH,
                ESDI_ECONOMICS_HELPER_PATH,
                *esdi_bound_path_hashes(),
            },
            key=str,
        )
    )


def _parse_tree(raw: bytes, expected: set[str]) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("CRLC-336 Git tree record is malformed") from error
        if mode != "100644" or object_type != "blob" or not _HEX40_RE.fullmatch(object_id):
            raise RuntimeError("CRLC-336 identity path is not a plain SHA-1 blob")
        blobs[path] = object_id
    if set(blobs) != expected:
        raise RuntimeError("CRLC-336 committed identity paths are incomplete")
    return blobs


def _branch_status_is_clean(raw: bytes, head: str, temporary: Path | None = None) -> bool:
    expected = [
        f"# branch.oid {head}".encode(),
        f"# branch.head {EXPECTED_BRANCH}".encode(),
        f"# branch.upstream origin/{EXPECTED_BRANCH}".encode(),
        b"# branch.ab +0 -0",
    ]
    lines = raw.splitlines()
    if temporary is None:
        return lines == expected
    return lines in (expected, [*expected, f"? {temporary}".encode()])


def validate_repository_identity(identity: Mapping[str, Any]) -> None:
    expected_keys = {
        "branch",
        "head_commit",
        "head_tree",
        "upstream",
        "upstream_commit",
        "git_blobs",
        "sha256",
        "whole_worktree_clean_required",
        "head_equals_upstream_required",
        "protocol_seal_hash",
    }
    if set(identity) != expected_keys:
        raise RuntimeError("CRLC-336 repository identity schema drift")
    head = identity.get("head_commit")
    tree = identity.get("head_tree")
    upstream_commit = identity.get("upstream_commit")
    blobs = identity.get("git_blobs")
    sha = identity.get("sha256")
    if (
        identity.get("branch") != EXPECTED_BRANCH
        or identity.get("upstream") != f"origin/{EXPECTED_BRANCH}"
        or not isinstance(head, str)
        or not _HEX40_RE.fullmatch(head)
        or not isinstance(tree, str)
        or not _HEX40_RE.fullmatch(tree)
        or upstream_commit != head
        or identity.get("whole_worktree_clean_required") is not True
        or identity.get("head_equals_upstream_required") is not True
        or not isinstance(blobs, Mapping)
        or not isinstance(sha, Mapping)
        or set(blobs) != set(sha)
    ):
        raise RuntimeError("CRLC-336 repository identity values drift")
    if any(
        not isinstance(path, str)
        or not isinstance(blob, str)
        or not _HEX40_RE.fullmatch(blob)
        or not isinstance(sha.get(path), str)
        or not _HEX64_RE.fullmatch(sha[path])
        for path, blob in blobs.items()
    ):
        raise RuntimeError("CRLC-336 repository identity digests drift")
    seal = canonical_hash({"git_blobs": dict(blobs), "sha256": dict(sha)})
    if identity.get("protocol_seal_hash") != seal:
        raise RuntimeError("CRLC-336 protocol seal hash drift")
    expected_paths = {str(path) for path in committed_identity_paths()}
    if set(blobs) != expected_paths:
        raise RuntimeError("CRLC-336 repository identity path set drift")
    fixed_blobs = {
        str(SOURCE_DECISION_PATH): SOURCE_DECISION_GIT_BLOB,
        str(MECHANISM_DECISION_PATH): MECHANISM_DECISION_GIT_BLOB,
        str(ESDI_PREREGISTRATION_PATH): ESDI_PREREGISTRATION_GIT_BLOB,
        str(ESDI_NOVELTY_HELPER_PATH): ESDI_NOVELTY_HELPER_GIT_BLOB,
        str(ESDI_ECONOMICS_HELPER_PATH): ESDI_ECONOMICS_HELPER_GIT_BLOB,
    }
    if any(blobs.get(path) != expected for path, expected in fixed_blobs.items()):
        raise RuntimeError("CRLC-336 fixed Git blob authority drift")
    fixed_sha256 = {
        str(SOURCE_DECISION_PATH): SOURCE_DECISION_SHA256,
        str(MECHANISM_DECISION_PATH): MECHANISM_DECISION_SHA256,
        str(ESDI_PREREGISTRATION_PATH): ESDI_PREREGISTRATION_SHA256,
        str(ESDI_NOVELTY_HELPER_PATH): ESDI_NOVELTY_HELPER_SHA256,
        str(ESDI_ECONOMICS_HELPER_PATH): ESDI_ECONOMICS_HELPER_SHA256,
    }
    if any(sha.get(path) != expected for path, expected in fixed_sha256.items()):
        raise RuntimeError("CRLC-336 fixed SHA-256 authority drift")


def frozen_repository_identity() -> dict[str, Any]:
    validate_frozen_dependencies()
    revision = _git("rev-parse", "--show-toplevel", "HEAD", "HEAD^{tree}", "@{upstream}").decode().splitlines()
    if len(revision) != 4:
        raise RuntimeError("CRLC-336 Git revision evidence is incomplete")
    root, head, tree, upstream_commit = revision
    if Path(root).resolve() != REPOSITORY_ROOT.resolve() or head != upstream_commit:
        raise RuntimeError("CRLC-336 requires pushed exact HEAD")
    status = _git(
        "status",
        "--porcelain=v2",
        "--branch",
        "--ahead-behind",
        "--untracked-files=all",
    )
    if not _branch_status_is_clean(status, head):
        raise RuntimeError("CRLC-336 creation requires a clean pushed branch")
    paths = [str(path) for path in committed_identity_paths()]
    blobs = _parse_tree(_git("ls-tree", "-z", head, "--", *paths), set(paths))
    sha: dict[str, str] = {}
    for path, blob in blobs.items():
        raw = _read_regular(path)
        if _git_blob(raw, blob) != blob:
            raise RuntimeError("CRLC-336 worktree differs from committed blob")
        sha[path] = hashlib.sha256(raw).hexdigest()
    identity: dict[str, Any] = {
        "branch": EXPECTED_BRANCH,
        "head_commit": head,
        "head_tree": tree,
        "upstream": f"origin/{EXPECTED_BRANCH}",
        "upstream_commit": upstream_commit,
        "git_blobs": {path: blobs[path] for path in sorted(blobs)},
        "sha256": {path: sha[path] for path in sorted(sha)},
        "whole_worktree_clean_required": True,
        "head_equals_upstream_required": True,
        "protocol_seal_hash": "",
    }
    identity["protocol_seal_hash"] = canonical_hash(
        {"git_blobs": identity["git_blobs"], "sha256": identity["sha256"]}
    )
    validate_repository_identity(identity)
    return identity


def validate_recorded_repository(identity: Mapping[str, Any]) -> None:
    validate_frozen_dependencies()
    validate_repository_identity(identity)
    head = str(identity["head_commit"])
    paths = sorted(identity["git_blobs"])
    records = _parse_tree(_git("ls-tree", "-z", head, "--", *paths), set(paths))
    if records != identity["git_blobs"]:
        raise RuntimeError("CRLC-336 recorded Git blobs are unavailable")
    for path, expected in identity["sha256"].items():
        raw = _git("show", f"{head}:{path}")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("CRLC-336 recorded path SHA is unavailable")
        if _git_blob(raw, records[path]) != records[path]:
            raise RuntimeError("CRLC-336 recorded path blob is invalid")


def parse_exact_decimal(value: str) -> Fraction:
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        raise ValueError("CRLC-336 decimal must be canonical plain nonnegative")
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("CRLC-336 decimal is invalid") from error
    if not decimal.is_finite() or decimal < 0:
        raise ValueError("CRLC-336 decimal is invalid")
    return Fraction(decimal)


def path_vote(values: Sequence[str | Fraction]) -> int:
    if not values:
        raise ValueError("CRLC-336 path cannot be empty")
    parsed = tuple(
        parse_exact_decimal(value) if isinstance(value, str) else value
        for value in values
    )
    if any(not isinstance(value, Fraction) or value < 0 for value in parsed):
        raise ValueError("CRLC-336 path values must be exact nonnegative Fractions")
    balance = 2 * parsed[-1] - min(parsed) - max(parsed)
    return 1 if balance > 0 else -1 if balance < 0 else 0


def maturity_vote(current: int, previous: int) -> int:
    if type(current) is not int or type(previous) is not int or min(current, previous) < 0:
        raise ValueError("CRLC-336 maturity values must be nonnegative integers")
    return 1 if current < previous else -1 if current > previous else 0


def primary_vote(component_votes: Sequence[int]) -> int:
    values = tuple(component_votes)
    if len(values) != 4 or any(type(value) is not int or value not in {-1, 0, 1} for value in values):
        raise ValueError("CRLC-336 primary requires four exact component votes")
    total = sum(values)
    return 1 if total > 0 else -1 if total < 0 else 0


def canonical_utc_timestamp(value: str | datetime) -> str:
    if isinstance(value, str):
        if not _TIMESTAMP_RE.fullmatch(value):
            raise ValueError("CRLC-336 timestamp grammar is invalid")
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError as error:
            raise ValueError("CRLC-336 timestamp is invalid") from error
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("CRLC-336 timestamp is noncanonical")
        return value
    if not isinstance(value, datetime) or value.tzinfo is None or value.microsecond:
        raise ValueError("CRLC-336 datetime must be aware whole-second")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_signed(value: int, *, nonzero: bool = True) -> str:
    if type(value) is not int or (nonzero and value == 0):
        raise ValueError("CRLC-336 identity integer is invalid")
    return str(value)


def primary_signal_id(accession: str, source_available_at: str, vote_sum: int) -> str:
    if not isinstance(accession, str) or not _ACCESSION_RE.fullmatch(accession):
        raise ValueError("CRLC-336 accession grammar is invalid")
    if type(vote_sum) is not int or vote_sum not in {
        -4,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        4,
    }:
        raise ValueError("CRLC-336 primary vote sum is outside the frozen domain")
    timestamp = canonical_utc_timestamp(source_available_at)
    raw = f"{POLICY_ID}|primary|{accession}|{timestamp}|{_canonical_signed(vote_sum)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_control_signal_id(
    control: str,
    accession: str,
    source_available_at: str,
    control_vote_sum: int,
) -> str:
    allowed = {
        "daily_path_only": {-1, 1},
        "weekly_path_only": {-1, 1},
        "wam_change_only": {-1, 1},
        "wal_change_only": {-1, 1},
        "path_pair": {-2, -1, 1, 2},
        "maturity_pair": {-2, -1, 1, 2},
    }
    if (
        control not in allowed
        or type(control_vote_sum) is not int
        or control_vote_sum not in allowed[control]
    ):
        raise ValueError("CRLC-336 source-control vote is invalid")
    if not isinstance(accession, str) or not _ACCESSION_RE.fullmatch(accession):
        raise ValueError("CRLC-336 accession grammar is invalid")
    timestamp = canonical_utc_timestamp(source_available_at)
    raw = (
        f"{POLICY_ID}|control|{control}|{accession}|{timestamp}|"
        f"{_canonical_signed(control_vote_sum)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deterministic_random_side(primary_id: str) -> int:
    if not isinstance(primary_id, str) or not _HEX64_RE.fullmatch(primary_id):
        raise ValueError("CRLC-336 primary signal ID is invalid")
    raw = f"{POLICY_ID}|{primary_id}|RANDOM_SIDE".encode("utf-8")
    return 1 if hashlib.sha256(raw).digest()[0] < 128 else -1


def same_parent_control_signal_id(
    control: str,
    primary_id: str,
    entry_time: str,
    exit_time: str,
    side: int,
) -> str:
    if control not in SAME_PARENT_CONTROL_NAMES:
        raise ValueError("CRLC-336 same-parent control name is invalid")
    if not isinstance(primary_id, str) or not _HEX64_RE.fullmatch(primary_id):
        raise ValueError("CRLC-336 primary signal ID is invalid")
    entry = canonical_utc_timestamp(entry_time)
    exit_ = canonical_utc_timestamp(exit_time)
    if exit_ <= entry or type(side) is not int or side not in {-1, 1}:
        raise ValueError("CRLC-336 same-parent control interval is invalid")
    if control == "constant_long" and side != 1:
        raise ValueError("CRLC-336 constant-long side is invalid")
    if control == "constant_short" and side != -1:
        raise ValueError("CRLC-336 constant-short side is invalid")
    if (
        control == "deterministic_random_side"
        and side != deterministic_random_side(primary_id)
    ):
        raise ValueError("CRLC-336 deterministic-random side is invalid")
    side_text = "LONG" if side == 1 else "SHORT"
    raw = f"{POLICY_ID}|control|{control}|{primary_id}|{entry}|{exit_}|{side_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def schedule(decision_time: str) -> tuple[str, str]:
    canonical = canonical_utc_timestamp(decision_time)
    parsed = datetime.strptime(canonical, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    entry = parsed + timedelta(minutes=5)
    exit_ = entry + timedelta(hours=336)
    return canonical_utc_timestamp(entry), canonical_utc_timestamp(exit_)


def authenticated_gross9_weights() -> dict[str, float]:
    gross9 = load_esdi_authority()["gross9"]
    weights = gross9.get("weights")
    if not isinstance(weights, Mapping) or dict(weights) != GROSS9_WEIGHTS:
        raise RuntimeError("CRLC-336 Gross9 weights differ from ESDI authority")
    baseline = gross9.get("baseline_gross")
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
        or float(baseline) != 9.0
    ):
        raise RuntimeError("CRLC-336 Gross9 configured gross authority drift")
    return dict(GROSS9_WEIGHTS)


def same_gross_weights(candidate_weight: float) -> dict[str, float]:
    if isinstance(candidate_weight, bool) or not isinstance(
        candidate_weight, (int, float)
    ):
        raise ValueError("CRLC-336 candidate weight must be a frozen numeric value")
    weight = float(candidate_weight)
    if weight not in CANDIDATE_WEIGHTS:
        raise ValueError("CRLC-336 candidate weight is outside the frozen grid")
    scale = (9.0 - weight) / 9.0
    treatment = {
        name: value * scale
        for name, value in authenticated_gross9_weights().items()
    }
    treatment["crlc"] = weight
    if not math.isclose(sum(treatment.values()), 9.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("CRLC-336 same-gross arithmetic drift")
    return treatment


def rank_same_gross_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observed: list[float] = []
    for raw in rows:
        value = raw.get("candidate_weight")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("CRLC-336 ranking weight type is invalid")
        observed.append(float(value))
    if sorted(observed) != list(CANDIDATE_WEIGHTS) or len(set(observed)) != 4:
        raise ValueError("CRLC-336 ranking requires the exact four-weight grid")
    normalized: list[dict[str, Any]] = []
    for raw, weight in zip(rows, observed):
        raw_score = raw.get("minimum_improvement")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("CRLC-336 ranking score type is invalid")
        score = float(raw_score)
        if not math.isfinite(score) or type(raw.get("passes")) is not bool:
            raise ValueError("CRLC-336 ranking row is invalid")
        row = dict(raw)
        row["candidate_weight"] = weight
        row["minimum_improvement"] = score
        normalized.append(row)
    ranked = sorted(
        normalized,
        key=lambda row: (
            -float(row["minimum_improvement"]),
            float(row["candidate_weight"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["frozen"] = bool(rank == 1 and row["passes"])
    if not ranked[0]["passes"]:
        raise RuntimeError("CRLC-336 raw rank one failed; no substitution")
    return ranked


def downstream_artifact_contract() -> dict[str, Any]:
    return {
        "paths": {
            "preregistration": str(DEFAULT_OUTPUT),
            "source_access_claim": (
                "results/circle_reserve_fund_nmfp_source_access_"
                "claim_2026-07-30.json"
            ),
            "source_csv": (
                "data/circle_reserve_fund_nmfp_source_2026-07-30.csv.gz"
            ),
            "source_manifest": (
                "results/circle_reserve_fund_nmfp_source_manifest_"
                "2026-07-30.json"
            ),
            "primary_clock": (
                "results/circle_reserve_liquidity_concordance_primary_"
                "clock_2026-07-30.csv.gz"
            ),
            "control_clocks": (
                "results/circle_reserve_liquidity_concordance_control_"
                "clocks_2026-07-30.csv.gz"
            ),
            "source_support": (
                "results/circle_reserve_liquidity_concordance_source_"
                "support_2026-07-30.json"
            ),
            "novelty_claim": (
                "results/circle_reserve_liquidity_concordance_novelty_"
                "claim_2026-07-30.json"
            ),
            "novelty": (
                "results/circle_reserve_liquidity_concordance_novelty_"
                "2026-07-30.json"
            ),
            "economics_directory": (
                "results/circle_reserve_liquidity_concordance_economics_"
                "2026-07-30"
            ),
        },
        "json": {
            "encoding": "UTF-8",
            "sort_keys": True,
            "indent": 2,
            "ensure_ascii": True,
            "allow_nan": False,
            "trailing_lf_count": 1,
            "manifest_hash": "SHA256 compact sorted JSON excluding manifest_hash",
        },
        "gzip_csv": {
            "encoding": "UTF-8",
            "line_ending": "LF",
            "compression_level": 9,
            "filename": "",
            "mtime": 0,
        },
        "future_protocol_paths_metadata_only": [str(path) for path in FUTURE_PROTOCOL_PATHS],
        "future_bytes_or_hashes_opened": False,
        "later_source_claim_binds_every_future_blob_and_sha256": True,
    }


def _core_manifest(repository_identity: Mapping[str, Any]) -> dict[str, Any]:
    validate_repository_identity(repository_identity)
    authority = load_esdi_authority()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "source_unseen_outcome_blind_write_once",
        "singleton": True,
        "frozen_preregistration": {
            "source_decision": {
                "path": str(SOURCE_DECISION_PATH),
                "sha256": SOURCE_DECISION_SHA256,
                "commit": SOURCE_DECISION_COMMIT,
                "git_blob": SOURCE_DECISION_GIT_BLOB,
            },
            "mechanism_decision": {
                "path": str(MECHANISM_DECISION_PATH),
                "sha256": MECHANISM_DECISION_SHA256,
                "commit": MECHANISM_DECISION_COMMIT,
                "git_blob": MECHANISM_DECISION_GIT_BLOB,
            },
            "producer": {
                "path": str(PRODUCER_PATH),
                "test_path": str(TEST_PATH),
                "producer_and_test_bound_in_repository_identity": True,
            },
            "esdi_authority": {
                "authority_commit": ESDI_AUTHORITY_COMMIT,
                "preregistration": {
                    "path": str(ESDI_PREREGISTRATION_PATH),
                    "sha256": ESDI_PREREGISTRATION_SHA256,
                    "git_blob": ESDI_PREREGISTRATION_GIT_BLOB,
                    "manifest_hash": ESDI_MANIFEST_HASH,
                },
                "novelty_helper": {
                    "path": str(ESDI_NOVELTY_HELPER_PATH),
                    "sha256": ESDI_NOVELTY_HELPER_SHA256,
                    "git_blob": ESDI_NOVELTY_HELPER_GIT_BLOB,
                    "functions": list(NOVELTY_FUNCTIONS),
                },
                "economics_helper": {
                    "path": str(ESDI_ECONOMICS_HELPER_PATH),
                    "sha256": ESDI_ECONOMICS_HELPER_SHA256,
                    "git_blob": ESDI_ECONOMICS_HELPER_GIT_BLOB,
                },
                "subtrees": {
                    "comparators": ESDI_COMPARATOR_SUBTREE_SHA256,
                    "gross9": ESDI_GROSS9_SUBTREE_SHA256,
                    "gross9_authority": ESDI_GROSS9_AUTHORITY_SHA256,
                    "runtime_code_closure": ESDI_RUNTIME_CLOSURE_SHA256,
                },
            },
            "repository_identity": copy.deepcopy(dict(repository_identity)),
            "serialization": downstream_artifact_contract()["json"],
            "downstream_artifacts": downstream_artifact_contract(),
        },
        "source": {
            "source_id": "CRF-NMFP",
            "authority": "SEC EDGAR first-dissemination static daily index and Feed archives",
            "identity": copy.deepcopy(SOURCE_IDENTITY),
            "context_only_ticker": CONTEXT_ONLY_TICKER,
            "context_only_ticker_is_never_identity": True,
            "expected_report_months": list(REPORT_MONTHS),
            "expected_original_count": 42,
            "warmup_report_months": ["2022-11", "2022-12"],
            "discovery_months": list(DISCOVERY_MONTHS),
            "discovery_days_each_month": list(range(1, 16)),
            "daily_index_path_count": DISCOVERY_DAY_COUNT,
            "forms": {
                "eligible": list(ELIGIBLE_FORMS),
                "original": list(ORIGINAL_FORMS),
                "amendment_audit_only": list(AMENDMENT_FORMS),
                "transition": {
                    "before_local_date": "2024-06-11 => N-MFP2",
                    "on_or_after_local_date": "2024-06-11 => N-MFP3",
                },
            },
            "transport": {
                "scheme": "https",
                "host": "www.sec.gov",
                "declared_contact_user_agent_required": True,
                "serialized_contact_forbidden": True,
                "accept_encoding": "identity",
                "minimum_request_interval_seconds": "0.20",
                "attempts_per_url": 1,
                "user_agent_environment_variable": "CRLC_SEC_USER_AGENT",
                "redirects": 0,
                "daily_index_404_only_expected_non_200": True,
                "allowed_path_templates": [
                    "/Archives/edgar/daily-index/<YYYY>/QTR<n>/master.<YYYYMMDD>.idx",
                    "/Archives/edgar/Feed/<YYYY>/QTR<n>/<YYYYMMDD>.nc.tar.gz",
                    "/Archives/edgar/data/844779/<accession_without_dashes>/index.json",
                    "/Archives/edgar/data/844779/<accession_without_dashes>/primary_doc.xml",
                    "/Archives/edgar/data/844779/<accession_without_dashes>/<accession_with_dashes>.txt",
                ],
                "forbidden": [
                    "redirects",
                    "authentication",
                    "cookies",
                    "JavaScript",
                    "search engines",
                    "mirrors",
                    "current full or quarterly indexes",
                    "SEC bulk N-MFP data",
                    "BlackRock files",
                    "Circle APIs",
                ],
            },
            "first_dissemination": {
                "daily_index_exact_cik_numeric": 844779,
                "feed_only_when_index_has_candidate": True,
                "feed_tar_no_path_extraction": True,
                "exactly_one_primary_doc_xml": True,
                "current_index_xml_submission_parity_required": True,
                "byte_mismatch_or_removal": "terminal",
                "amendment_values_parsed": False,
            },
            "xml": {
                "exact_official_n_mfp2_and_n_mfp3_namespaces": True,
                "required_identity_elements": [
                    "reportDate",
                    "registrantFullName",
                    "cik",
                    "registrantLEIId",
                    "nameOfSeries",
                    "leiOfSeries",
                    "seriesId",
                    "moneyMarketFundCategory",
                    "govMoneyMrktFundFlag",
                ],
                "required_source_elements": [
                    "averagePortfolioMaturity",
                    "averageLifeMaturity",
                    "liquidAssetsDetails",
                    "totalValueDailyLiquidAssets",
                    "totalValueWeeklyLiquidAssets",
                    "percentageDailyLiquidAssets",
                    "percentageWeeklyLiquidAssets",
                    "totalLiquidAssetsNearPercentDate",
                ],
                "retained": [
                    "accession",
                    "form",
                    "report_date",
                    "acceptance_datetime_et",
                    "source_available_at_utc",
                    *SOURCE_IDENTITY.keys(),
                    "wam_days",
                    "wal_days",
                    "ordered daily_pct/weekly_pct exact-decimal path",
                ],
                "forbidden_values": [
                    "dollar-valued liquidity",
                    "net assets",
                    "shares",
                    "shareholder flows",
                    "yields",
                    "NAV",
                    "class data",
                    "securities",
                    "holdings",
                    "issuers",
                    "counterparties",
                    "CUSIPs",
                    "maturity dates",
                    "explanatory text",
                ],
                "DTD_entity_XInclude_XSLT_network_resolution": "terminal",
            },
            "validation": {
                "category": "Government",
                "government_flag": "Y",
                "wam_integer_range": [0, 60],
                "wal_integer_range": [0, 120],
                "wam_at_most_wal": True,
                "liquidity_detail_count_range": [15, 31],
                "first_detail_day_at_most": 4,
                "maximum_adjacent_day_gap": 4,
                "last_detail_zero_to_three_days_before_report": True,
                "percentage_range": [0, 100],
                "daily_percentage_at_most_weekly": True,
                "all_42_months_consecutive_unique": True,
            },
            "availability": {
                "acceptance_zone": "America/New_York",
                "zone_less_ambiguous_or_nonexistent": "terminal",
                "historical_floor": "12:00:00Z on acceptance local date + 5 calendar days",
                "trading_latency_minutes": 5,
                "live_availability_is_later_than_durable_local_commit": True,
            },
            "source_csv_columns": [
                "accession",
                "form",
                "report_date",
                "acceptance_datetime_et",
                "source_available_at_utc",
                *SOURCE_IDENTITY.keys(),
                "wam_days",
                "wal_days",
                "liquidity_path_json",
            ],
        },
        "feature_and_signal": {
            "warmup": "first valid source report cannot emit",
            "path_balance": "2*end-min-max",
            "path_vote": {"positive": 1, "negative": -1, "zero": 0},
            "maturity_vote": {"lower": 1, "higher": -1, "tie": 0},
            "components": [
                "daily_path_vote",
                "weekly_path_vote",
                "wam_change_vote",
                "wal_change_vote",
            ],
            "vote_sum": "arithmetic sum of four equal integer votes",
            "side": {"positive": "LONG", "negative": "SHORT", "zero": "NONE"},
            "primary_identity": (
                "CRLC-336|primary|<accession>|<YYYY-MM-DDTHH:MM:SSZ>|"
                "<canonical_signed_vote_sum>"
            ),
            "signal_id": "lowercase SHA256 hex of exact UTF-8 identity",
            "no_fit_threshold_magnitude_or_regime": True,
        },
        "execution": {
            "decision_time": "source_available_at_utc",
            "entry_time": "decision_time + 5 elapsed minutes",
            "exit_time": "entry_time + 336 elapsed hours",
            "reserved_interval": "[entry_time,exit_time)",
            "sort": ["entry_time", "signal_id"],
            "overlap": "accept iff entry_time >= previous accepted exit_time",
            "overlap_queue_replacement_netting_release": False,
            "suppressed_overlap_required": 0,
            "standalone_leverage": 0.5,
            "tp_sl_dynamic_exit": False,
        },
        "calendars": {
            "timezone": "UTC",
            "interval_semantics": "half-open",
            "periods": {name: list(bounds) for name, bounds in PERIODS.items()},
            "period_membership": "decision entry and exit all fully contained",
            "crossing_trade": "excluded whole never clipped",
            "global_overlap_before_period_filter": True,
            "full_calendar_years": 3.0,
            "other_calendar_years": "seconds/(365.25*86400)",
        },
        "support_gates": {
            "source_integrity": [
                "all 630 daily-index receipts",
                "exact Feed membership and first-dissemination extraction",
                "exact current archive parity",
                "exactly 42 originals",
                "exact N-MFP2/N-MFP3 transition",
                "exact identity schema path maturity and clocks",
                "byte-identical deterministic rebuild",
                "zero forbidden or outcome access",
            ],
            "minimum_signals": copy.deepcopy(MINIMUM_SIGNALS),
            "side_support": {
                "selection_each_side": 3,
                "full_each_side": 6,
                "combined_future_each_side": 3,
            },
            "zero_duplicate_or_overlap": True,
            "vote_diversity_population": (
                "all valid post-warmup originals with hypothetical decision entry "
                "and exit fully contained in full before zero-vote and overlap"
            ),
            "vote_diversity": {
                "each_component_positive_min": 4,
                "each_component_negative_min": 4,
                "mixed_component_fraction_min": [1, 4],
                "primary_zero_vote_fraction_max": [3, 10],
                "single_control_disagree_or_abstain_min": 4,
                "pair_control_disagree_or_abstain_min": 2,
                "fraction_arithmetic": "exact integer cross multiplication",
            },
        },
        "controls": {
            "order": list(CONTROL_NAMES),
            "source_derived": {
                "single": list(SOURCE_CONTROL_NAMES[:4]),
                "pair": list(SOURCE_CONTROL_NAMES[4:]),
                "identity": (
                    "CRLC-336|control|<name>|<accession>|"
                    "<YYYY-MM-DDTHH:MM:SSZ>|<raw_nonzero_control_vote_sum>"
                ),
            },
            "same_parent": {
                "names": list(SAME_PARENT_CONTROL_NAMES),
                "random_digest": (
                    "SHA256(raw UTF-8 CRLC-336|<primary_id>|RANDOM_SIDE)."
                    "digest()[0] < 128 is LONG"
                ),
                "identity": (
                    "CRLC-336|control|<name>|<primary_id>|<entry>|<exit>|"
                    "<LONG_or_SHORT>"
                ),
            },
            "one_bar_delay_minutes": 5,
            "cannot_become_alternate_candidate": True,
        },
        "novelty": {
            "opens_after_source_support_pass_only": True,
            "opens_clock_rows_only": True,
            "authority": {
                "frozen_comparator_artifacts": authority[
                    "frozen_comparator_artifacts"
                ],
                "item_count": ESDI_COMPARATOR_COUNT,
                "subtree_sha256": ESDI_COMPARATOR_SUBTREE_SHA256,
                "functions": list(NOVELTY_FUNCTIONS),
            },
            "prior_source_family": {
                "minimum_common_domain_entries": 10,
                "exact_entry_jaccard_max": [1, 5],
                "bidirectional_containment_window_seconds": 24 * 3600,
                "bidirectional_containment_max": [1, 2],
                "absolute_signed_exposure_correlation_max": [2, 5],
            },
            "gross9_sleeves": {
                "all_positive_weight_required": True,
                "exact_entry_jaccard_max": [1, 10],
                "bidirectional_containment_window_seconds": 6 * 3600,
                "bidirectional_containment_max": [7, 20],
                "occupied_bar_jaccard_max": [1, 4],
                "absolute_signed_exposure_correlation_max": [7, 20],
            },
            "rational_inclusive_gates": True,
            "missing_malformed_hash_drift_empty_or_zero_variance": "terminal",
            "failure": "permanent retirement before economics",
        },
        "economic_contract": {
            "authority": {
                "path": str(ESDI_ECONOMICS_HELPER_PATH),
                "sha256": ESDI_ECONOMICS_HELPER_SHA256,
                "git_blob": ESDI_ECONOMICS_HELPER_GIT_BLOB,
            },
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "leverage": 0.5,
            "cost_rate_per_notional_side": {"base": 0.0006, "stress": 0.001},
            "funding": "exact realized causal funding cash flow",
            "strict_mdd": "global/pre-entry HWM plus intratrade path",
            "float_arithmetic": "authenticated ESDI NumPy/Python float64",
            "round_before_gate": False,
            "cagr": {
                "full_years": 3.0,
                "other_years": "seconds/(365.25*86400)",
                "formula": "expm1(log(final_equity)/years) with ESDI saturation",
            },
            "ratio": (
                "min(float64_max,CAGR/max(strict_MDD,1e-15)) if CAGR>0 else 0"
            ),
            "mean_gross_underlying_bp": (
                "mean(side*(exit_open/entry_open-1)*10000) over completed "
                "contained primary trades"
            ),
            "standalone_gates_each_cost": {
                "absolute_return": ">0",
                "cagr_to_strict_mdd": ">=3.0",
                "strict_mdd": "<=0.15",
                "mean_gross_underlying_bp": ">=20",
                "liquidation_safe": True,
            },
            "selection_signflip": {
                "p_value_max": 0.20,
                "cluster": "UTC entry calendar month",
                "exact_enumeration_nonzero_clusters_max": 20,
                "numpy_seed": 20260730,
                "monte_carlo_samples": 10000,
                "one_sided": True,
                "bootstrap": False,
            },
            "control_superiority": {
                "primary_ratio_gt_all_same_parent_direction_controls": True,
                "primary_ratio_gt_source_controls_min": 4,
                "primary_return_gt_flip_and_random": True,
                "same_authenticated_market_and_funding_arrays": True,
            },
            "stage_order": [
                "2023H2",
                "2024",
                "selection",
                "same_gross",
                "future25",
                "future26",
                "combined_future",
                "stitched_full",
            ],
            "stop_on_first_failure": True,
        },
        "gross9": {
            "authority": authority["gross9"],
            "baseline_weights": copy.deepcopy(GROSS9_WEIGHTS),
            "baseline_configured_gross": 9.0,
            "candidate_weights": list(CANDIDATE_WEIGHTS),
            "treatment": "scale every Gross9 sleeve by (9-c)/9 and add CRLC at c",
            "treatment_configured_gross": 9.0,
            "candidate_actual_notional_gross": "0.5*c",
            "selection_cells": [
                "2023H2/base",
                "2023H2/stress",
                "2024/base",
                "2024/stress",
            ],
            "cell_gates": {
                "ratio_improvement_min": 0.05,
                "return_retention_min": 0.97,
                "treatment_return_positive": True,
                "liquidation_safe": True,
            },
            "mdd_strict_reduction_in_at_least_one_cell": True,
            "ranking": (
                "all four rows sorted by raw (-minimum_improvement,weight); "
                "raw rank one must pass; no lower-rank substitution"
            ),
            "future_rerank_or_alternate_weight": False,
            "future_subperiod_gates": {
                "periods": ["future25", "future26"],
                "each_cost_ratio_improvement_min": 0.05,
                "each_cost_return_retention_min": 0.97,
                "each_cost_treatment_return_positive": True,
                "each_cost_liquidation_safe": True,
                "mdd_strict_reduction_in_at_least_one_cost_per_period": True,
            },
            "combined_future_gates": {
                "fresh_nonstitched_evaluation": True,
                "each_cost_ratio_improvement_min": 0.05,
                "each_cost_return_retention_min": 0.97,
                "each_cost_treatment_return_positive": True,
                "each_cost_liquidation_safe": True,
                "mdd_strict_reduction_in_at_least_one_cost": True,
                "candidate_completed_trades_min": 10,
                "candidate_active_utc_entry_months_min": 10,
                "candidate_signflip_p_value_max": 0.20,
            },
            "combined_future_fresh_period": list(PERIODS["combined_future"]),
            "stitched_full_is_confirmation_not_selection": True,
        },
        "strict_sequence": [
            "source and mechanism decisions plus preregistration producer and synthetic tests committed and pushed",
            "write-once preregistration artifact created committed and pushed",
            "source builder support novelty economics and tests committed and pushed with preregistration hash bound",
            "source-access claim committed and pushed sealing every future protocol blob before any production daily-index or Feed request",
            "complete 630-path source replay once and stop terminally on first failure",
            "write-once source artifacts committed without evaluator change",
            "source support run and stop on first failure",
            "novelty claim committed then novelty run and stop on first failure",
            "economic claim before each ordered stage and stop on first failure",
            "independent sealed-input byte reproduction tests review commit and push",
        ],
        "producer_effects": {
            "official_metadata_and_schema_spec_probes_previously_disclosed": True,
            "production_source_urls_requested": 0,
            "production_source_rows_opened": 0,
            "esdi_metadata_artifacts_opened": 1,
            "future_protocol_files_opened_or_hashed": 0,
            "artifact_write_once": True,
            "source_unseen": True,
            "outcome_blind": True,
        },
        **{name: False for name in EVIDENCE_BOUNDARIES},
    }


def build_manifest(repository_identity: Mapping[str, Any]) -> dict[str, Any]:
    core = _core_manifest(repository_identity)
    return {**core, "manifest_hash": canonical_hash(core)}


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "str":
        return type(value) is str
    if expected == "bool":
        return type(value) is bool
    if expected == "object":
        return type(value) is dict
    if expected == "array[str]":
        return type(value) is list and all(type(item) is str for item in value)
    raise RuntimeError("CRLC-336 unknown schema type")


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if set(payload) != set(TOP_LEVEL_TYPES) or any(
        not _matches_type(payload.get(key), expected)
        for key, expected in TOP_LEVEL_TYPES.items()
    ):
        raise RuntimeError("CRLC-336 preregistration top-level schema drift")
    identity = payload["frozen_preregistration"].get("repository_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("CRLC-336 repository identity is missing")
    expected = build_manifest(identity)
    if dict(payload) != expected:
        raise RuntimeError("CRLC-336 preregistration differs from frozen code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("CRLC-336 preregistration manifest hash drift")
    if any(payload[name] is not False for name in EVIDENCE_BOUNDARIES):
        raise RuntimeError("CRLC-336 preregistration evidence boundary opened")
    if REPORT_MONTHS[0] != "2022-11" or REPORT_MONTHS[-1] != "2026-04" or len(REPORT_MONTHS) != 42:
        raise RuntimeError("CRLC-336 report-month generation drift")
    if DISCOVERY_MONTHS[0] != "2022-12" or DISCOVERY_MONTHS[-1] != "2026-05" or len(DISCOVERY_MONTHS) != 42:
        raise RuntimeError("CRLC-336 discovery-month generation drift")
    if sum(authenticated_gross9_weights().values()) != 9.0:
        raise RuntimeError("CRLC-336 Gross9 baseline drift")
    if payload["gross9"].get("baseline_weights") != GROSS9_WEIGHTS:
        raise RuntimeError("CRLC-336 manifest Gross9 weights drift")


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_manifest(payload)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _output_path(path: str | Path) -> Path:
    candidate = _safe_path(path)
    if candidate != DEFAULT_OUTPUT:
        raise RuntimeError("CRLC-336 output must equal the frozen singleton path")
    return candidate


def _validate_creation_publish_state(identity: Mapping[str, Any], temporary: Path) -> None:
    status = _git(
        "status",
        "--porcelain=v2",
        "--branch",
        "--ahead-behind",
        "--untracked-files=all",
    )
    if not _branch_status_is_clean(status, str(identity["head_commit"]), temporary):
        raise RuntimeError("CRLC-336 worktree changed before artifact publish")


def write_once(
    output: str | Path = DEFAULT_OUTPUT,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    candidate = _output_path(output)
    full = REPOSITORY_ROOT / candidate
    if full.exists() or full.is_symlink():
        raw = _read_regular(candidate)
        stored = _canonical_json(raw, "existing preregistration")
        validate_manifest(stored)
        if raw != canonical_manifest_bytes(stored):
            raise RuntimeError("CRLC-336 existing preregistration bytes drift")
        if payload is not None and dict(payload) != stored:
            raise RuntimeError("CRLC-336 supplied preregistration payload drift")
        validate_recorded_repository(
            stored["frozen_preregistration"]["repository_identity"]
        )
        return "verified_existing", stored

    identity = frozen_repository_identity()
    expected = build_manifest(identity)
    if payload is not None and dict(payload) != expected:
        raise RuntimeError("CRLC-336 supplied preregistration payload drift")
    canonical = canonical_manifest_bytes(expected)
    temporary = candidate.parent / (
        f".{candidate.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    temporary_full = REPOSITORY_ROOT / temporary
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_full,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(canonical)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        _validate_creation_publish_state(identity, temporary)
        try:
            os.link(temporary_full, full, follow_symlinks=False)
        except FileExistsError:
            if _read_regular(candidate) != canonical:
                raise RuntimeError("CRLC-336 preregistration race drift")
            return "verified_existing", expected
        return "created", expected
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary_full.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    status, payload = write_once()
    print(
        json.dumps(
            {
                "status": status,
                "path": str(DEFAULT_OUTPUT),
                "manifest_hash": payload["manifest_hash"],
                **{name: payload[name] for name in EVIDENCE_BOUNDARIES},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
