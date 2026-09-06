"""Produce the frozen, outcome-blind TUSI-168 preregistration artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import types
from typing import Any, Mapping


_BOOTSTRAP_ESDI_HELPER_PATH = Path(
    "training/preregister_ethereum_settlement_demand_impulse.py"
)
_BOOTSTRAP_ESDI_HELPER_SHA256 = (
    "1c7d7c822f16818ce0bc8fa0be99db0fe156882dbb76bf804ae19232f2a53b26"
)


def _bootstrap_git(repository_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("TUSI-168 helper preflight Git failure") from error
    return completed.stdout


def _bootstrap_read_regular(
    repository_root: Path,
    relative_path: Path,
) -> bytes:
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.name in {"", ".", ".."}
    ):
        raise RuntimeError("TUSI-168 helper preflight path is unsafe")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(repository_root, directory_flags)
    try:
        for part in relative_path.parent.parts:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            relative_path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise RuntimeError(
                    "TUSI-168 helper preflight requires a regular file"
                )
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise RuntimeError("TUSI-168 helper preflight path is unsafe") from error
    finally:
        os.close(descriptor)


def _bootstrap_git_blob(raw: bytes, object_id: str) -> str:
    if len(object_id) == 40:
        digest = hashlib.sha1()
    elif len(object_id) == 64:
        digest = hashlib.sha256()
    else:
        raise RuntimeError("TUSI-168 helper Git object format is unsupported")
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _load_verified_esdi_helper() -> tuple[types.ModuleType, Path, str, str]:
    module_path = Path(__file__).resolve(strict=True)
    repository_root = module_path.parent.parent
    revision = _bootstrap_git(
        repository_root,
        "rev-parse",
        "--show-toplevel",
        "HEAD",
    ).decode("utf-8").splitlines()
    if len(revision) != 2:
        raise RuntimeError("TUSI-168 helper preflight revision is incomplete")
    git_root, head_commit = revision
    if Path(git_root).resolve(strict=True) != repository_root:
        raise RuntimeError("TUSI-168 helper preflight repository root drift")
    if not (
        len(head_commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in head_commit)
    ):
        raise RuntimeError("TUSI-168 helper preflight HEAD is invalid")

    record = _bootstrap_git(
        repository_root,
        "ls-tree",
        "-z",
        head_commit,
        "--",
        str(_BOOTSTRAP_ESDI_HELPER_PATH),
    )
    records = [item for item in record.split(b"\0") if item]
    if len(records) != 1:
        raise RuntimeError("TUSI-168 helper committed blob is missing")
    metadata, raw_path = records[0].split(b"\t", 1)
    mode, object_type, object_id = metadata.decode("ascii").split()
    if (
        mode != "100644"
        or object_type != "blob"
        or raw_path.decode("utf-8") != str(_BOOTSTRAP_ESDI_HELPER_PATH)
    ):
        raise RuntimeError("TUSI-168 helper is not a committed plain Git blob")

    raw = _bootstrap_read_regular(
        repository_root,
        _BOOTSTRAP_ESDI_HELPER_PATH,
    )
    if hashlib.sha256(raw).hexdigest() != _BOOTSTRAP_ESDI_HELPER_SHA256:
        raise RuntimeError("TUSI-168 helper preflight SHA-256 drift")
    if _bootstrap_git_blob(raw, object_id) != object_id:
        raise RuntimeError("TUSI-168 helper differs from its committed Git blob")

    helper_path = repository_root / _BOOTSTRAP_ESDI_HELPER_PATH
    helper = types.ModuleType("training._tusi_verified_esdi_authority")
    helper.__file__ = str(helper_path)
    helper.__package__ = "training"
    helper.__loader__ = None
    helper.__spec__ = None
    code = compile(
        raw,
        str(helper_path),
        "exec",
        dont_inherit=True,
    )
    exec(code, helper.__dict__)
    return helper, repository_root, head_commit, object_id


(
    esdi,
    _BOOTSTRAP_REPOSITORY_ROOT,
    _BOOTSTRAP_HEAD_COMMIT,
    _BOOTSTRAP_HELPER_GIT_BLOB,
) = _load_verified_esdi_helper()


POLICY_ID = "TUSI-168"
PROTOCOL_VERSION = "tron_usdt_supply_impulse_preregistration_v1"
EXPECTED_BRANCH = "codex/tron-usdt-supply-impulse-20260730"
REPOSITORY_ROOT = _BOOTSTRAP_REPOSITORY_ROOT
SOURCE_DECISION_PATH = Path(
    "docs/tron-usdt-supply-events-source-axis-decision-2026-07-30.md"
)
SOURCE_DECISION_SHA256 = (
    "ad742cc261b5dfa23bdc7cd730e3e6bd01c3d19687806fe9c65c995893ce3300"
)
MECHANISM_DECISION_PATH = Path(
    "docs/tron-usdt-supply-impulse-mechanism-decision-2026-07-30.md"
)
MECHANISM_DECISION_SHA256 = (
    "29fa76ca1fd1f86910d257e3d3bc05dcc76de0c5de2da88cd90a3930a5c65e98"
)
DEFAULT_OUTPUT = Path(
    "results/tron_usdt_supply_impulse_preregistration_2026-07-30.json"
)
PRODUCER_PATH = Path("training/preregister_tron_usdt_supply_impulse.py")
TEST_PATH = Path("tests/test_preregister_tron_usdt_supply_impulse.py")
ESDI_HELPER_PATH = Path(
    "training/preregister_ethereum_settlement_demand_impulse.py"
)
ESDI_HELPER_SHA256 = _BOOTSTRAP_ESDI_HELPER_SHA256
ESDI_PREREGISTRATION_PATH = Path(
    "results/ethereum_settlement_demand_impulse_"
    "preregistration_2026-07-30.json"
)
ESDI_PREREGISTRATION_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
ESDI_MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)
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
    Path("training/build_tron_usdt_supply_events.py"),
    Path("tests/test_build_tron_usdt_supply_events.py"),
    Path("training/evaluate_tron_usdt_supply_impulse_source_support.py"),
    Path("tests/test_evaluate_tron_usdt_supply_impulse_source_support.py"),
    Path("training/evaluate_tron_usdt_supply_impulse_novelty.py"),
    Path("tests/test_evaluate_tron_usdt_supply_impulse_novelty.py"),
    Path("training/evaluate_tron_usdt_supply_impulse_economics.py"),
    Path("tests/test_evaluate_tron_usdt_supply_impulse_economics.py"),
)
PREREGISTRATION_PATHS = (
    SOURCE_DECISION_PATH,
    MECHANISM_DECISION_PATH,
    PRODUCER_PATH,
    TEST_PATH,
    ESDI_HELPER_PATH,
    ESDI_PREREGISTRATION_PATH,
)

RUNTIME_CODE_ROOTS = tuple(esdi.RUNTIME_CODE_ROOTS)
RUNTIME_CODE_CLOSURE_PATHS = tuple(esdi.RUNTIME_CODE_CLOSURE_PATHS)
FROZEN_RUNTIME_ENVIRONMENT = copy.deepcopy(esdi.FROZEN_RUNTIME_ENVIRONMENT)
FROZEN_DISTRIBUTION_INVENTORY_COUNT = (
    esdi.FROZEN_DISTRIBUTION_INVENTORY_COUNT
)
FROZEN_DISTRIBUTION_INVENTORY_SHA256 = (
    esdi.FROZEN_DISTRIBUTION_INVENTORY_SHA256
)

CHAIN_ID = "0x2b6653dc"
BASE58_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
LOG_ADDRESS = "0xa614f803b6fd780986a42c78ec9c7f77e6ded13c"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
TOPICS = {
    "Issue(uint256)": (
        "0xcb8241adb0c3fdb35b70c24ce35c5eb0c"
        "17af7431c99f827d44a445ca624176a"
    ),
    "Redeem(uint256)": (
        "0x702d5967f45f6513a38ffc42d6ba9bf230"
        "bd40e8f53b16363c7eb4fd2deb9a44"
    ),
    "DestroyedBlackFunds(address,uint256)": (
        "0x61e6e66b0d6339b2980aecc6ccc0039736"
        "791f0ccde9ed512e789a7fbdd698c6"
    ),
    "Deprecate(address)": (
        "0xcc358699805e9a8b7f77b522628c7cb9abd"
        "07d9efb86b6fb616af1609036a99e"
    ),
    "Transfer(address,address,uint256)": (
        "0xddf252ad1be2c89b69c2b068fc378daa952"
        "ba7f163c4a11628f55a4df523b3ef"
    ),
}
BOUNDARIES = [
    {
        "utc": "2023-01-01T00:00:00Z",
        "first_block_at_or_after": 47_313_358,
        "block_hash": (
            "0x0000000002d1f1ce5e430281e5308004cf19dd6e31afd4402b670fc05da5b340"
        ),
    },
    {
        "utc": "2023-06-01T00:00:00Z",
        "first_block_at_or_after": 51_652_374,
        "block_hash": (
            "0x0000000003142716b7305d5d621414bc745837a849273ce4eab4b200c598af9d"
        ),
    },
    {
        "utc": "2024-01-01T00:00:00Z",
        "first_block_at_or_after": 57_811_194,
        "block_hash": (
            "0x00000000037220fa937d59050fab5c3740ef10f5f7715b0f45035353878cd98f"
        ),
    },
    {
        "utc": "2025-01-01T00:00:00Z",
        "first_block_at_or_after": 68_346_198,
        "block_hash": (
            "0x000000000412e156401b47b5e85900fecd1744a7dd70e0ec6d7c9db7b5b7b8fd"
        ),
    },
    {
        "utc": "2026-01-01T00:00:00Z",
        "first_block_at_or_after": 78_854_231,
        "block_hash": (
            "0x0000000004b338578474ba7a2a5fd3f2e19d303cb79f30d0b8e05ee361607b33"
        ),
    },
    {
        "utc": "2026-06-01T00:00:00Z",
        "first_block_at_or_after": 83_201_056,
        "block_hash": (
            "0x0000000004f58c20deab323895309dd25eecc6bbbe4cd6c940713da2d78ca67a"
        ),
    },
]
TRANSPORTS = [
    {
        "role": "primary",
        "scheme": "https",
        "hostname": "api.trongrid.io",
        "port": 443,
    },
    {
        "role": "verification",
        "scheme": "https",
        "hostname": "tron-mainnet.core.chainstack.com",
        "port": 443,
    },
]
TRANSPORT_BATCH_LIMITS = {"primary": 100, "verification": 30}
EVIDENCE_BOUNDARIES = (
    "precutoff_source_rows_opened",
    "source_incidence_opened",
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
INDEPENDENT_CONTROLS = (
    "issue_only",
    "redeem_only",
    "include_destroyed_black_funds",
    "count_net_side",
)
SAME_PRIMARY_PARENT_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
    "one_bar_delayed_entry",
)
FROZEN_CONTROL_ORDER = (
    *INDEPENDENT_CONTROLS,
    *SAME_PRIMARY_PARENT_CONTROLS,
)
SUPPORT_CLOCK_COLUMNS = (
    "policy_id",
    "control",
    "window",
    "constituent_identities_json",
    "source_identity",
    "constituent_count",
    "bucket_amount_raw",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
)
PREREGISTRATION_TOP_LEVEL_TYPES = {
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


# Reuse the already-frozen ESDI pure metric and environment authorities.
entries_in_domain = esdi.entries_in_domain
exact_entry_jaccard = esdi.exact_entry_jaccard
bidirectional_entry_containment = esdi.bidirectional_entry_containment
fraction_at_most = esdi.fraction_at_most
signed_exposure_5m = esdi.signed_exposure_5m
occupied_bar_jaccard = esdi.occupied_bar_jaccard
squared_signed_exposure_pearson = esdi.squared_signed_exposure_pearson
current_runtime_environment = esdi.current_runtime_environment
validate_runtime_environment = esdi.validate_runtime_environment
discover_runtime_code_closure = esdi.discover_runtime_code_closure
validate_runtime_code_closure = esdi.validate_runtime_code_closure


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def downstream_artifact_contract() -> dict[str, Any]:
    return {
        "paths": {
            "preregistration": str(DEFAULT_OUTPUT),
            "source_replay_claim": (
                "results/tron_usdt_supply_events_source_replay_"
                "claim_2026-07-30.json"
            ),
            "primary_support_clock": (
                "results/tron_usdt_supply_impulse_primary_"
                "clock_2026-07-30.csv.gz"
            ),
            "control_support_clocks": (
                "results/tron_usdt_supply_impulse_control_"
                "clocks_2026-07-30.csv.gz"
            ),
            "source_support_report": (
                "results/tron_usdt_supply_impulse_source_"
                "support_2026-07-30.json"
            ),
        },
        "json_serialization": (
            "sorted-key two-space-indented ASCII JSON plus one LF; internal "
            "manifest_hash is SHA-256 compact sorted-key JSON excluding that "
            "field"
        ),
        "gzip_csv_serialization": {
            "encoding": "UTF-8",
            "line_ending": "LF",
            "compression_level": 9,
            "filename": "",
            "mtime": 0,
        },
        "support_clock_csv": {
            "header_order": list(SUPPORT_CLOCK_COLUMNS),
            "field_contract": {
                "policy_id": "exact TUSI-168",
                "control": "primary or one frozen control string",
                "window": "selection, future25, or future26",
                "constituent_identities_json": (
                    "exact compact JSON used for source identity"
                ),
                "source_identity": "exactly 64 lowercase hex characters",
                "constituent_count": "canonical base-10 integer",
                "bucket_amount_raw": (
                    "canonical base-10 signed amount or count integer"
                ),
                "times": "whole-second YYYY-MM-DDTHH:MM:SSZ",
                "side": "LONG or SHORT",
                "null_or_extra_column_allowed": False,
            },
            "primary_file_control_values": ["primary"],
            "control_file_order": list(FROZEN_CONTROL_ORDER),
            "each_control_sort": (
                "entry_time,decision_time,source_identity,side"
            ),
            "future_append_views": {
                "same_semantic_fields": True,
                "additional_field": "accepted",
                "representation": "report objects, not a third CSV schema",
            },
        },
        "preregistration_json_schema": {
            "top_level_keys_and_types": copy.deepcopy(
                PREREGISTRATION_TOP_LEVEL_TYPES
            ),
            "unknown_or_missing_keys": "terminal failure",
            "ordered_arrays": {
                "boundaries": "chronological",
                "transports": ["primary", "verification"],
                "controls": list(FROZEN_CONTROL_ORDER),
                "strict_sequence": "displayed order",
                "closure_paths": "ESDI authority order",
            },
            "json_object_insertion_order_semantics": False,
        },
        "source_support_report_json_schema": {
            "top_level_keys_and_types": {
                "protocol_version": "str",
                "policy_id": "str",
                "status": "str",
                "terminal": "bool",
                "artifact_eligible": "bool",
                "support_passed": "bool",
                "decision": "str",
                "registration": "object",
                "source_contract": "object",
                "raw_candidate_counts": "object[int]",
                "accepted_clock_counts": "object[int]",
                "period_diagnostics": "object",
                "support_audit": "object",
                "support_checks": "object[bool]",
                "future_append_selection_invariance": "object",
                "control_overlap": "object",
                "clock_artifacts": "object[str]",
                "evidence_boundary": "object",
                "source_support_precedes_novelty": "bool",
                "novelty_comparator_market_or_outcome_artifacts_opened": "bool",
                "manifest_hash": "str",
            },
            "count_object_keys": ["primary", *FROZEN_CONTROL_ORDER],
            "period_diagnostic_keys": [
                "selection",
                "2023H2",
                "2024",
                "2024H1",
                "2024H2",
                "future25",
                "2025H1",
                "2025H2",
                "future26",
                "full",
            ],
            "clock_artifact_keys": ["primary_sha256", "controls_sha256"],
            "future_append": (
                "prefix rule, total difference count, then frozen-control-order "
                "raw/accepted row counts and SHA-256 pairs"
            ),
            "unknown_or_missing_nested_keys": "terminal failure",
        },
        "circularity_boundary": {
            "preregistration_binds": (
                "two decisions, producer/test, ESDI preregistration and metric "
                "authority, complete Gross9 closure and environment"
            ),
            "future_protocol_paths_are_metadata_only": True,
            "not_yet_final_evaluator_hashes_in_preregistration": False,
            "later_evaluators_hardcode_preregistration_artifact_file_sha256": True,
            "replay_claim_binds": (
                "preregistration artifact plus every source, support, novelty, "
                "economics, imported helper, and test blob"
            ),
        },
    }


def frozen_comparator_registry() -> dict[str, dict[str, Any]]:
    authority = load_esdi_preregistration_authority()
    return copy.deepcopy(authority["frozen_comparator_artifacts"])


def frozen_gross9_authority() -> dict[str, Any]:
    authority = load_esdi_preregistration_authority()
    return copy.deepcopy(authority["gross9"]["authority"])


def committed_identity_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                *PREREGISTRATION_PATHS,
                *RUNTIME_CODE_CLOSURE_PATHS,
            }
        )
    )


def _open_dependency(path: str | Path) -> int:
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("TUSI-168 dependency path must be repository-relative")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(REPOSITORY_ROOT, directory_flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            candidate.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            os.close(file_descriptor)
            raise RuntimeError("TUSI-168 dependency must be a regular non-symlink")
        return file_descriptor
    except OSError as error:
        raise RuntimeError(
            "TUSI-168 dependency path is missing or unsafe"
        ) from error
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    descriptor = _open_dependency(path)
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _dependency_bytes(path: str | Path) -> bytes:
    descriptor = _open_dependency(path)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_esdi_preregistration_authority() -> dict[str, Any]:
    """Validate the committed ESDI artifact before copying metadata subtrees."""

    raw = _dependency_bytes(ESDI_PREREGISTRATION_PATH)
    if hashlib.sha256(raw).hexdigest() != ESDI_PREREGISTRATION_SHA256:
        raise RuntimeError("TUSI-168 ESDI preregistration file hash drift")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "TUSI-168 ESDI preregistration is not canonical JSON"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("TUSI-168 ESDI preregistration root is invalid")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if (
        payload.get("manifest_hash") != ESDI_MANIFEST_HASH
        or canonical_hash(core) != ESDI_MANIFEST_HASH
    ):
        raise RuntimeError("TUSI-168 ESDI manifest hash drift")

    try:
        registry = payload["novelty"]["frozen_comparator_artifacts"]
        gross9 = payload["gross9"]
        gross9_authority = gross9["authority"]
        runtime_closure = gross9_authority["runtime_code_closure"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("TUSI-168 ESDI authority subtree is missing") from error
    if (
        not isinstance(registry, dict)
        or len(registry) != ESDI_COMPARATOR_COUNT
        or canonical_hash(registry) != ESDI_COMPARATOR_SUBTREE_SHA256
    ):
        raise RuntimeError("TUSI-168 ESDI comparator registry drift")
    if (
        not isinstance(gross9, dict)
        or canonical_hash(gross9) != ESDI_GROSS9_SUBTREE_SHA256
        or not isinstance(gross9_authority, dict)
        or canonical_hash(gross9_authority) != ESDI_GROSS9_AUTHORITY_SHA256
        or not isinstance(runtime_closure, dict)
        or canonical_hash(runtime_closure) != ESDI_RUNTIME_CLOSURE_SHA256
    ):
        raise RuntimeError("TUSI-168 ESDI Gross9 authority drift")

    validate_runtime_code_closure()
    validate_runtime_environment()
    current_environment = current_runtime_environment()
    if (
        runtime_closure.get("roots")
        != [str(path) for path in RUNTIME_CODE_ROOTS]
        or runtime_closure.get("paths")
        != [str(path) for path in RUNTIME_CODE_CLOSURE_PATHS]
        or runtime_closure.get("environment_lock_paths")
        != ["pyproject.toml", "uv.lock"]
        or runtime_closure.get("exact_runtime_environment")
        != current_environment
        or runtime_closure.get("required_runtime_abi_and_selected_packages")
        != FROZEN_RUNTIME_ENVIRONMENT
        or runtime_closure.get("all_distribution_inventory_count")
        != FROZEN_DISTRIBUTION_INVENTORY_COUNT
        or runtime_closure.get("all_distribution_inventory_sha256")
        != FROZEN_DISTRIBUTION_INVENTORY_SHA256
    ):
        raise RuntimeError("TUSI-168 ESDI runtime authority drift")
    return {
        "artifact_manifest_hash": ESDI_MANIFEST_HASH,
        "frozen_comparator_artifacts": copy.deepcopy(registry),
        "gross9": copy.deepcopy(gross9),
    }


def _committed_file_sha256(path: str | Path, expected_blob: str) -> str:
    descriptor = _open_dependency(path)
    try:
        size = os.fstat(descriptor).st_size
        if len(expected_blob) == 40:
            git_digest = hashlib.sha1()
        elif len(expected_blob) == 64:
            git_digest = hashlib.sha256()
        else:
            raise RuntimeError("TUSI-168 Git object format is unsupported")
        git_digest.update(f"blob {size}\0".encode("ascii"))
        file_digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            file_digest.update(chunk)
            git_digest.update(chunk)
        if git_digest.hexdigest() != expected_blob:
            raise RuntimeError(
                "TUSI-168 file bytes differ from committed Git blob"
            )
        return file_digest.hexdigest()
    finally:
        os.close(descriptor)


def validate_frozen_documents_and_helper() -> None:
    expected = {
        SOURCE_DECISION_PATH: SOURCE_DECISION_SHA256,
        MECHANISM_DECISION_PATH: MECHANISM_DECISION_SHA256,
        ESDI_HELPER_PATH: ESDI_HELPER_SHA256,
        ESDI_PREREGISTRATION_PATH: ESDI_PREREGISTRATION_SHA256,
    }
    for path, frozen_hash in expected.items():
        actual = sha256_file(path)
        if actual != frozen_hash:
            raise RuntimeError(
                f"TUSI-168 frozen dependency changed: {path}: "
                f"{actual} != {frozen_hash}"
            )


def _git(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("TUSI-168 Git identity validation failed") from error
    return completed.stdout


def _hex_digest(value: Any, lengths: set[int]) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _clean_branch_status(
    raw: bytes,
    expected_head: str,
    expected_temporary: Path | None = None,
) -> bool:
    expected_lines = [
        f"# branch.oid {expected_head}".encode("utf-8"),
        f"# branch.head {EXPECTED_BRANCH}".encode("utf-8"),
        (
            f"# branch.upstream origin/{EXPECTED_BRANCH}".encode("utf-8")
        ),
        b"# branch.ab +0 -0",
    ]
    if expected_temporary is None:
        return raw.splitlines() == expected_lines
    temporary_line = f"? {expected_temporary}".encode("utf-8")
    return raw.splitlines() in (
        expected_lines,
        [*expected_lines, temporary_line],
    )


def frozen_repository_identity() -> dict[str, Any]:
    """Bind a wholly clean worktree, protocol blobs, and pushed exact HEAD."""

    validate_frozen_documents_and_helper()
    validate_runtime_code_closure()
    validate_runtime_environment()
    paths = [str(path) for path in committed_identity_paths()]

    status = _git(
        "status",
        "--porcelain=v2",
        "--branch",
        "--ahead-behind",
        "--untracked-files=all",
    )
    if not _clean_branch_status(status, _BOOTSTRAP_HEAD_COMMIT):
        raise RuntimeError(
            "TUSI-168 initial creation requires a clean whole worktree"
        )
    rev_parse = _git(
        "rev-parse",
        "--show-toplevel",
        "HEAD^{tree}",
    ).decode("utf-8").splitlines()
    if len(rev_parse) != 2:
        raise RuntimeError("TUSI-168 Git HEAD/upstream evidence is incomplete")
    root, tree = rev_parse
    head = _BOOTSTRAP_HEAD_COMMIT
    upstream_commit = head
    if Path(root).resolve() != REPOSITORY_ROOT.resolve():
        raise RuntimeError("TUSI-168 ran outside the repository root")
    branch = EXPECTED_BRANCH
    expected_upstream = f"origin/{EXPECTED_BRANCH}"
    upstream = expected_upstream
    records = _git("ls-tree", "-z", head, "--", *paths).split(b"\0")
    blobs: dict[str, str] = {}
    for record in records:
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if mode != "100644" or object_type != "blob":
            raise RuntimeError("TUSI-168 protocol path is not a plain Git blob")
        blobs[path] = object_id
    if set(blobs) != set(paths):
        raise RuntimeError("TUSI-168 committed protocol blobs are incomplete")
    if blobs.get(str(ESDI_HELPER_PATH)) != _BOOTSTRAP_HELPER_GIT_BLOB:
        raise RuntimeError("TUSI-168 helper blob changed after preflight")

    identity = {
        "branch": branch,
        "head_commit": head,
        "head_tree": tree,
        "upstream": upstream,
        "upstream_ref": f"refs/remotes/{upstream}",
        "upstream_commit": upstream_commit,
        "head_equals_upstream_required": True,
        "git_blobs": {path: blobs[path] for path in sorted(blobs)},
        "sha256": {
            path: _committed_file_sha256(path, blobs[path])
            for path in sorted(paths)
        },
        "whole_worktree_clean_required": True,
        "bound_paths_clean_against_head_required": True,
        "protocol_seal_hash": "",
    }
    identity["protocol_seal_hash"] = canonical_hash(
        {
            "git_blobs": identity["git_blobs"],
            "sha256": identity["sha256"],
        }
    )
    validate_repository_identity(identity)
    return identity


def validate_creation_publish_state(
    identity: Mapping[str, Any],
    expected_temporary: Path,
) -> None:
    """Recheck clean pushed identity immediately before canonical publish."""

    validate_repository_identity(identity)
    status = _git(
        "status",
        "--porcelain=v2",
        "--branch",
        "--ahead-behind",
        "--untracked-files=all",
    )
    if not _clean_branch_status(
        status,
        identity["head_commit"],
        expected_temporary,
    ):
        raise RuntimeError(
            "TUSI-168 worktree changed before artifact publish"
        )


def _plain_git_blobs(
    raw: bytes,
    expected_paths: set[str],
    label: str,
) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(
                f"TUSI-168 {label} Git tree record is malformed"
            ) from error
        if mode != "100644" or object_type != "blob":
            raise RuntimeError(
                f"TUSI-168 {label} path is not a plain Git blob"
            )
        blobs[path] = object_id
    if set(blobs) != expected_paths:
        raise RuntimeError(f"TUSI-168 {label} Git tree paths are incomplete")
    return blobs


def validate_existing_artifact_repository(
    identity: Mapping[str, Any],
    artifact_bytes: bytes,
) -> None:
    """Validate the stored producer seal and its first artifact-only child."""

    validate_repository_identity(identity)
    bound_paths = sorted(identity["git_blobs"])
    producer_commit = identity["head_commit"]
    revision = _git(
        "rev-parse",
        "--show-toplevel",
        "HEAD",
        "@{upstream}",
        f"{producer_commit}^{{tree}}",
    ).decode("utf-8").splitlines()
    if len(revision) != 4:
        raise RuntimeError("TUSI-168 existing artifact revision is incomplete")
    root, current_head, current_upstream, producer_tree = revision
    if Path(root).resolve() != REPOSITORY_ROOT.resolve():
        raise RuntimeError("TUSI-168 existing artifact repository root drift")
    if producer_tree != identity["head_tree"]:
        raise RuntimeError("TUSI-168 stored producer Git tree drift")
    branch = _git("branch", "--show-current").decode("utf-8").strip()
    upstream = _git(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    ).decode("utf-8").strip()
    if (
        branch != EXPECTED_BRANCH
        or upstream != f"origin/{EXPECTED_BRANCH}"
        or current_head != current_upstream
    ):
        raise RuntimeError(
            "TUSI-168 existing artifact requires pushed canonical branch"
        )

    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *bound_paths,
        str(DEFAULT_OUTPUT),
    )
    if status:
        raise RuntimeError(
            "TUSI-168 existing artifact paths are not committed-clean"
        )
    producer_blobs = _plain_git_blobs(
        _git("ls-tree", "-z", producer_commit, "--", *bound_paths),
        set(bound_paths),
        "stored producer",
    )
    if producer_blobs != dict(identity["git_blobs"]):
        raise RuntimeError("TUSI-168 stored producer Git blobs drift")
    if producer_blobs.get(str(ESDI_HELPER_PATH)) != (
        _BOOTSTRAP_HELPER_GIT_BLOB
    ):
        raise RuntimeError("TUSI-168 stored helper blob differs from preflight")
    actual_sha256 = {
        path: _committed_file_sha256(path, producer_blobs[path])
        for path in bound_paths
    }
    if actual_sha256 != dict(identity["sha256"]):
        raise RuntimeError("TUSI-168 stored producer file SHA-256 drift")

    _git("merge-base", "--is-ancestor", producer_commit, current_head)
    added_commits = _git(
        "log",
        "--format=%H",
        "--reverse",
        "--diff-filter=A",
        "--no-renames",
        f"{producer_commit}..{current_head}",
        "--",
        str(DEFAULT_OUTPUT),
    ).decode("utf-8").splitlines()
    if len(added_commits) != 1:
        raise RuntimeError(
            "TUSI-168 pushed history lacks one artifact add commit"
        )
    artifact_commit = added_commits[0]
    parent_record = _git(
        "rev-list",
        "--parents",
        "-n",
        "1",
        artifact_commit,
    ).decode("utf-8").split()
    if parent_record != [artifact_commit, producer_commit]:
        raise RuntimeError(
            "TUSI-168 artifact add commit is not the producer direct child"
        )
    changed_paths = {
        item.decode("utf-8")
        for item in _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "--no-renames",
            producer_commit,
            artifact_commit,
        ).split(b"\0")
        if item
    }
    if changed_paths != {str(DEFAULT_OUTPUT)}:
        raise RuntimeError(
            "TUSI-168 artifact add commit changed non-artifact paths"
        )
    if _git(
        "ls-tree",
        "-z",
        producer_commit,
        "--",
        str(DEFAULT_OUTPUT),
    ):
        raise RuntimeError("TUSI-168 artifact existed in the producer commit")
    artifact_add_blobs = _plain_git_blobs(
        _git(
            "ls-tree",
            "-z",
            artifact_commit,
            "--",
            str(DEFAULT_OUTPUT),
        ),
        {str(DEFAULT_OUTPUT)},
        "artifact add",
    )
    _git("merge-base", "--is-ancestor", artifact_commit, current_head)
    later_artifact_commits = _git(
        "log",
        "--format=%H",
        "--full-history",
        "--no-renames",
        f"{artifact_commit}..{current_head}",
        "--",
        str(DEFAULT_OUTPUT),
    )
    if later_artifact_commits:
        raise RuntimeError(
            "TUSI-168 artifact changed after its write-once add commit"
        )

    later_bound_commits = _git(
        "log",
        "--format=%H",
        "--full-history",
        "--no-renames",
        f"{artifact_commit}..{current_head}",
        "--",
        *bound_paths,
    )
    if later_bound_commits:
        raise RuntimeError(
            "TUSI-168 bound path changed after artifact registration"
        )

    current_paths = [*bound_paths, str(DEFAULT_OUTPUT)]
    current_blobs = _plain_git_blobs(
        _git("ls-tree", "-z", current_head, "--", *current_paths),
        set(current_paths),
        "current artifact descendant",
    )
    if any(
        current_blobs[path] != producer_blobs[path] for path in bound_paths
    ):
        raise RuntimeError("TUSI-168 bound protocol changed after registration")
    artifact_blob = current_blobs[str(DEFAULT_OUTPUT)]
    if (
        artifact_blob != artifact_add_blobs[str(DEFAULT_OUTPUT)]
        or _bootstrap_git_blob(artifact_bytes, artifact_blob) != artifact_blob
    ):
        raise RuntimeError("TUSI-168 committed artifact bytes drift")

    final_revision = _git("rev-parse", "HEAD", "@{upstream}").decode(
        "utf-8"
    ).splitlines()
    if final_revision != [current_head, current_upstream]:
        raise RuntimeError(
            "TUSI-168 existing artifact Git identity changed during validation"
        )
    final_status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *bound_paths,
        str(DEFAULT_OUTPUT),
    )
    if final_status:
        raise RuntimeError(
            "TUSI-168 existing artifact paths changed during validation"
        )


def validate_repository_identity(identity: Mapping[str, Any]) -> None:
    expected_paths = sorted(str(path) for path in committed_identity_paths())
    if set(identity) != {
        "branch",
        "head_commit",
        "head_tree",
        "upstream",
        "upstream_ref",
        "upstream_commit",
        "head_equals_upstream_required",
        "git_blobs",
        "sha256",
        "whole_worktree_clean_required",
        "bound_paths_clean_against_head_required",
        "protocol_seal_hash",
    }:
        raise RuntimeError("TUSI-168 repository identity schema drift")
    if (
        identity.get("branch") != EXPECTED_BRANCH
        or identity.get("upstream") != f"origin/{EXPECTED_BRANCH}"
        or identity.get("upstream_ref")
        != f"refs/remotes/origin/{EXPECTED_BRANCH}"
        or identity.get("head_commit") != identity.get("upstream_commit")
        or identity.get("head_equals_upstream_required") is not True
        or identity.get("whole_worktree_clean_required") is not True
        or identity.get("bound_paths_clean_against_head_required") is not True
        or not _hex_digest(identity.get("head_commit"), {40, 64})
        or not _hex_digest(identity.get("head_tree"), {40, 64})
    ):
        raise RuntimeError("TUSI-168 repository identity values drift")
    if sorted(identity.get("git_blobs", {})) != expected_paths:
        raise RuntimeError("TUSI-168 repository identity Git paths differ")
    if sorted(identity.get("sha256", {})) != expected_paths:
        raise RuntimeError("TUSI-168 repository identity SHA-256 paths differ")
    if not all(
        _hex_digest(value, {40, 64})
        for value in identity["git_blobs"].values()
    ):
        raise RuntimeError("TUSI-168 repository Git blob is invalid")
    if not all(
        _hex_digest(value, {64}) for value in identity["sha256"].values()
    ):
        raise RuntimeError("TUSI-168 repository SHA-256 is invalid")
    if identity["sha256"].get(str(SOURCE_DECISION_PATH)) != (
        SOURCE_DECISION_SHA256
    ):
        raise RuntimeError("TUSI-168 source decision hash drift")
    if identity["sha256"].get(str(MECHANISM_DECISION_PATH)) != (
        MECHANISM_DECISION_SHA256
    ):
        raise RuntimeError("TUSI-168 mechanism decision hash drift")
    if identity["sha256"].get(str(ESDI_HELPER_PATH)) != ESDI_HELPER_SHA256:
        raise RuntimeError("TUSI-168 ESDI helper hash drift")
    if identity["sha256"].get(str(ESDI_PREREGISTRATION_PATH)) != (
        ESDI_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("TUSI-168 ESDI preregistration hash drift")
    seal = {
        "git_blobs": identity["git_blobs"],
        "sha256": identity["sha256"],
    }
    if identity.get("protocol_seal_hash") != canonical_hash(seal):
        raise RuntimeError("TUSI-168 protocol seal hash drift")


def _core_manifest(repository_identity: Mapping[str, Any]) -> dict[str, Any]:
    validate_repository_identity(repository_identity)
    esdi_authority = load_esdi_preregistration_authority()
    esdi_gross9 = esdi_authority["gross9"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "outcome_blind_write_once_preincidence",
        "singleton": True,
        "frozen_preregistration": {
            "source_decision": {
                "path": str(SOURCE_DECISION_PATH),
                "sha256": SOURCE_DECISION_SHA256,
            },
            "mechanism_decision": {
                "path": str(MECHANISM_DECISION_PATH),
                "sha256": MECHANISM_DECISION_SHA256,
            },
            "producer": {
                "path": str(PRODUCER_PATH),
                "test_path": str(TEST_PATH),
                "producer_and_test_hash_bound_in_repository_identity": True,
            },
            "expected_post_preregistration_protocol": {
                "paths_metadata_only": [
                    str(path) for path in FUTURE_PROTOCOL_PATHS
                ],
                "included_in_preregistration_repository_identity": False,
                "existence_required_at_preregistration_creation": False,
                "bytes_or_hashes_read_by_preregistration_producer": False,
                "required_later_binding": (
                    "each future module and test binds this TUSI preregistration "
                    "artifact file SHA-256 and manifest_hash as constants"
                ),
                "separate_source_replay_claim": (
                    "created and committed after all eight paths are committed; "
                    "seals their Git blobs and SHA-256 before replay"
                ),
            },
            "esdi_authority_helper": {
                "path": str(ESDI_HELPER_PATH),
                "sha256": ESDI_HELPER_SHA256,
                "git_blob_and_sha256_bound_in_repository_identity": True,
                "helper_drift_is_terminal": True,
            },
            "esdi_preregistration_authority": {
                "path": str(ESDI_PREREGISTRATION_PATH),
                "file_sha256": ESDI_PREREGISTRATION_SHA256,
                "manifest_hash": ESDI_MANIFEST_HASH,
                "git_blob_and_sha256_bound_in_repository_identity": True,
                "comparator_registry": {
                    "json_path": "novelty.frozen_comparator_artifacts",
                    "item_count": ESDI_COMPARATOR_COUNT,
                    "canonical_compact_sorted_sha256": (
                        ESDI_COMPARATOR_SUBTREE_SHA256
                    ),
                },
                "gross9_subtree": {
                    "json_path": "gross9",
                    "canonical_compact_sorted_sha256": (
                        ESDI_GROSS9_SUBTREE_SHA256
                    ),
                },
                "gross9_authority_subtree": {
                    "json_path": "gross9.authority",
                    "canonical_compact_sorted_sha256": (
                        ESDI_GROSS9_AUTHORITY_SHA256
                    ),
                },
                "runtime_closure_subtree": {
                    "json_path": "gross9.authority.runtime_code_closure",
                    "canonical_compact_sorted_sha256": (
                        ESDI_RUNTIME_CLOSURE_SHA256
                    ),
                },
                "all_hashes_validated_before_deep_copy": True,
            },
            "repository_identity": copy.deepcopy(dict(repository_identity)),
            "serialization": {
                "encoding": "UTF-8",
                "sort_keys": True,
                "indent": 2,
                "ensure_ascii": True,
                "allow_nan": False,
                "trailing_lf_count": 1,
                "manifest_hash": (
                    "SHA256 compact sorted-key JSON excluding manifest_hash"
                ),
            },
            "downstream_artifact_contracts": downstream_artifact_contract(),
        },
        "source": {
            "authority": "TRON mainnet",
            "hosted_rpc_role": "independent transports, not authorities",
            "chain_id": CHAIN_ID,
            "contract": {
                "base58": BASE58_CONTRACT,
                "log_address": LOG_ADDRESS,
                "zero_address": ZERO_ADDRESS,
                "decimals": 6,
            },
            "topics": copy.deepcopy(TOPICS),
            "exact_event_labels": [
                "Issue",
                "Redeem",
                "DestroyedBlackFunds",
                "Deprecate",
            ],
            "semantic_supply_direction": {
                "Issue": 1,
                "Redeem": -1,
                "DestroyedBlackFunds": -1,
                "Deprecate": 0,
            },
            "primary_event_types": ["Issue", "Redeem"],
            "destroyed_black_funds_primary": False,
            "deprecate_terminates_source_v1": True,
            "boundaries": copy.deepcopy(BOUNDARIES),
            "first_source_block": 47_313_358,
            "exclusive_end_block": 83_201_056,
            "confirmation_blocks": 64,
            "availability": "canonical timestamp of event block N+64",
            "last_admissible_event_block": 83_200_991,
            "last_confirmation_block": 83_201_055,
            "raw_log_envelope": {
                "inclusive": [47_313_358, 83_200_991],
                "inclusive_block_count": 35_887_634,
                "confirmation_only_not_event_queried": [
                    83_200_992,
                    83_201_055,
                ],
                "exclusive_boundary_header": 83_201_056,
            },
            "chunks": {
                "count": 7_178,
                "maximum_blocks": 5_000,
                "full_size_chunk_count": 7_177,
                "full_size_chunk_blocks": 5_000,
                "final_chunk": {
                    "inclusive": [83_198_358, 83_200_991],
                    "block_count": 2_634,
                },
                "inclusive_contiguous_fixed": True,
                "gap_overlap_dynamic_split_or_partial_final_chunk": False,
            },
            "transports": copy.deepcopy(TRANSPORTS),
            "maximum_json_rpc_batch_by_role": copy.deepcopy(
                TRANSPORT_BATCH_LIMITS
            ),
            "transport_runtime_configuration": {
                "environment_variables": [
                    "TRON_PRIMARY_RPC_URL",
                    "TRON_VERIFY_RPC_URL",
                ],
                "provider_path_nonempty_and_runtime_memory_only": True,
                "userinfo_query_and_fragment_forbidden": True,
                "url_path_query_userinfo_or_credential_serialized": False,
                "independent_roles_must_resolve_to_distinct_hostnames": True,
            },
            "filters_per_chunk": [
                "topic0 in {Issue,Redeem,DestroyedBlackFunds,Deprecate}",
                "Transfer with indexed from == zero",
                "Transfer with indexed to == zero",
            ],
            "canonical_raw_log_fields": [
                "address",
                "block_number",
                "block_hash",
                "transaction_hash",
                "transaction_index",
                "log_index",
                "topics",
                "data",
                "removed",
            ],
            "raw_shape_validation": {
                "address": "exact frozen 20-byte log address",
                "Issue": {
                    "topic_count": 1,
                    "data_bytes": 32,
                    "data_words": ["positive_uint256_amount_raw"],
                },
                "Redeem": {
                    "topic_count": 1,
                    "data_bytes": 32,
                    "data_words": ["positive_uint256_amount_raw"],
                },
                "DestroyedBlackFunds": {
                    "topic_count": 1,
                    "data_bytes": 64,
                    "data_words": [
                        "exact_left_zero_padded_20_byte_actor_address",
                        "positive_uint256_amount_raw",
                    ],
                },
                "Deprecate": {
                    "topic_count": 1,
                    "data_bytes": 32,
                    "data_words": [
                        "exact_left_zero_padded_20_byte_replacement_address"
                    ],
                },
                "Transfer": {
                    "topic_count": 3,
                    "data_bytes": 32,
                    "indexed_address_words": [
                        "exact_left_zero_padded_20_byte_from",
                        "exact_left_zero_padded_20_byte_to",
                    ],
                    "data_words": ["positive_uint256_amount_raw"],
                },
                "quantities_topics_and_data_canonical": True,
                "extra_topics_short_or_long_data_nonzero_address_padding_"
                "or_zero_amount": (
                    "terminal"
                ),
                "removed_must_be_false": True,
            },
            "pairing": {
                "Issue": (
                    "exactly one same-transaction same-block same-amount "
                    "Transfer(zero,actor_address,amount)"
                ),
                "Redeem": (
                    "exactly one same-transaction same-block same-amount "
                    "Transfer(actor_address,zero,amount)"
                ),
                "bidirectional_exact": True,
                "orphan_semantic_events": 0,
                "orphan_zero_address_transfers": 0,
                "actor_and_amount_equality": "exact",
                "one_to_many_or_many_to_one": "terminal",
                "transaction_receipt_status_success_required": True,
                "successful_receipt_status": "0x1",
                "companion_transfer_is_second_economic_event": False,
            },
            "normalized_fields_only": [
                "event_type",
                "supply_direction",
                "actor_address",
                "amount_raw",
                "block_number",
                "block_hash",
                "transaction_hash",
                "transaction_index",
                "log_index",
                "paired_transfer_log_index",
                "event_timestamp_utc",
                "confirmation_block",
                "confirmation_block_hash",
                "available_at_utc",
            ],
            "replay": {
                "dual_exact_per_chunk_filter_and_global_agreement": True,
                "receipt_event_and_confirmation_headers_dual_exact": True,
                "common_finalized_head_covers_all_confirmation_blocks": True,
                "attempts_per_request": 1,
                "inter_batch_throttle": {
                    "elapsed_seconds": 0.25,
                    "applies_per_transport_after_first_batch": True,
                    "cli_parameter": False,
                    "bound_in_replay_claim_and_source_manifest": True,
                },
                "retry_backoff_provider_substitution_checkpoint_resume": False,
                "partial_publication": False,
            },
            "pre_replay_evidence_boundary": {
                "prior_precutoff_rpc": (
                    "twelve metadata-only boundary and immediately preceding "
                    "block-header calls"
                ),
                "prior_contract_logs_or_receipts_opened": 0,
                "replay_claim_required_before": [
                    "first precutoff eth_getLogs request",
                    "first precutoff eth_getTransactionReceipt request",
                    "any event or source incidence",
                ],
            },
        },
        "feature_and_signal": {
            "eligible_event_types": ["Issue", "Redeem"],
            "candidate_entry_open": (
                "ceil_to_5m(available_at_utc)+5 elapsed minutes; aligned "
                "availability still waits 5 minutes"
            ),
            "group_key": "exact candidate_entry_open",
            "net_supply_raw": "sum(Issue.amount_raw)-sum(Redeem.amount_raw)",
            "amount_arithmetic": "exact six-decimal integers",
            "side": {
                "positive": "LONG",
                "negative": "SHORT",
                "zero": "ABSTAIN",
            },
            "source_identity": {
                "sort_key": [
                    "block_number",
                    "transaction_index",
                    "log_index",
                    "transaction_hash",
                    "event_type",
                    "amount_raw",
                ],
                "row_shape": "six-element JSON array",
                "number_encoding": "JSON integers",
                "transaction_hash_encoding": "lowercase 0x",
                "event_type_encoding": "exact event string",
                "serialization": (
                    "UTF-8 compact JSON; sorted object keys; separators "
                    "(',',':'); ASCII escaping; finite values; no newline"
                ),
                "digest": "SHA-256 of canonical constituent bytes",
                "tie_break_or_outcome_encoding": False,
            },
            "decision_time": "maximum constituent available_at_utc",
            "rank_quantile_threshold_clipping_winsorization_or_amount_floor": False,
            "lookback_event_onset_side_or_hold_search": False,
        },
        "execution": {
            "entry_time": "candidate_entry_open",
            "exit_time": "entry_time + 168 elapsed hours",
            "hold_hours": 168,
            "hold_bars_5m": 2_016,
            "standalone_leverage": 0.5,
            "candidate_order": [
                "entry_time",
                "decision_time",
                "source_identity",
                "side",
            ],
            "reservation": {
                "scope": "one global position",
                "interval": "[entry_time,exit_time)",
                "accept": "entry_time >= previous accepted exit_time",
                "suppressed_candidates_queued": False,
            },
            "scheduler": {
                "runs_per_independent_construction": 1,
                "steps": [
                    "build every raw candidate from complete eligible source",
                    (
                        "assign to exactly one disjoint main window in order "
                        "selection,future25,future26 only when both entry and "
                        "exit are contained; discard main-boundary crossers"
                    ),
                    (
                        "sort union by frozen raw-candidate order and run one "
                        "global nonoverlap reservation"
                    ),
                    "define full accepted clock as accepted union",
                    (
                        "derive diagnostics and economic periods only by "
                        "entry-and-exit-contained projection"
                    ),
                ],
                "diagnostic_or_economic_period_nonoverlap_rerun": False,
                "projected_periods": [
                    "2023H2",
                    "2024",
                    "2024H1",
                    "2024H2",
                    "2025H1",
                    "2025H2",
                ],
                "diagnostic_boundary_crossing_selection_trade": (
                    "retained in selection reservation clock; skipped from "
                    "diagnostic; never truncated or replaced"
                ),
            },
            "pyramiding_stop_take_profit_trailing_or_early_close": False,
            "research_window_crossing": "skip; never truncate",
        },
        "calendars": {
            "containment": "half-open; both entry and exit contained",
            "full": ["2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
            "selection": [
                "2023-06-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "future25": [
                "2025-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
            "future26": [
                "2026-01-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "selection_reports": {
                "2023H2": [
                    "2023-06-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
                "2024H1": [
                    "2024-01-01T00:00:00Z",
                    "2024-07-01T00:00:00Z",
                ],
                "2024H2": [
                    "2024-07-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                ],
            },
            "future25_reports": {
                "2025H1": [
                    "2025-01-01T00:00:00Z",
                    "2025-07-01T00:00:00Z",
                ],
                "2025H2": [
                    "2025-07-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ],
            },
            "full_cagr_wall_clock_years": 3,
        },
        "support_gates": {
            "source_exact_zero_differences": {
                "dual_raw_log_replay": 0,
                "chunk_gaps_overlaps_missing_response_ids": 0,
                "receipt_header": 0,
                "issue_mint_transfer_pair": 0,
                "redeem_burn_transfer_pair": 0,
                "deprecate_events": 0,
                "future_append_selection": 0,
            },
            "future_append_invariance": {
                "fresh_constructions": {
                    "prefix": (
                        "source rows with available_at_utc < "
                        "2025-01-01T00:00:00Z"
                    ),
                    "full": "complete source artifact",
                },
                "independent_constructions": [
                    "primary",
                    "issue_only",
                    "redeem_only",
                    "include_destroyed_black_funds",
                    "count_net_side",
                ],
                "compared_views": [
                    "every raw candidate assigned to selection",
                    "every accepted selection clock row",
                ],
                "compared_fields": [
                    "control",
                    "sorted_constituent_identities",
                    "source_identity",
                    "constituent_count",
                    "exact_signed_bucket_amount_or_count",
                    "decision_time",
                    "entry",
                    "exit",
                    "side",
                ],
                "row_serialization": "canonical compact JSON",
                "same_primary_parent_controls": (
                    "regenerated and compared from prefix/full accepted primary "
                    "views"
                ),
                "row_order_field_or_sha256_differences_allowed": 0,
            },
            "accepted_trade_minimums": {
                "selection": 8,
                "2023H2": 2,
                "2024H1": 2,
                "2024H2": 2,
                "future25": 4,
                "2025H1": 1,
                "2025H2": 1,
                "future26": 2,
            },
            "utc_entry_month_share": {
                "periods": ["selection", "future25", "future26"],
                "numerator": (
                    "maximum count sharing the same UTC entry_time calendar month"
                ),
                "denominator": "all accepted primary entries in that period",
                "maximum_exact_inclusive": {
                    "numerator": 1,
                    "denominator": 2,
                },
                "comparison": "integer cross multiplication",
                "zero_denominator": "terminal failure",
            },
            "full_accepted_entry_gap": {
                "entries": "accepted primary entries contained in full",
                "sort": "strictly increasing entry_time integer UTC seconds",
                "differences": "consecutive entry timestamps only",
                "boundary_to_first_and_last_to_boundary_included": False,
                "minimum_entries": 2,
                "fewer_than_minimum": "terminal failure",
                "maximum_seconds_exact_inclusive": 240 * 86_400,
            },
            "long_short_counts_reported_but_not_floored": True,
            "identity_bucket_amount_side_entry_exit_hash_unique_reproducible": True,
            "all_support_before_comparator_or_outcome_rows": True,
            "failure_action": "retire TUSI-168 unchanged before outcomes",
        },
        "controls": {
            "independent_own_bucket_and_scheduler": [
                "issue_only",
                "redeem_only",
                "include_destroyed_black_funds",
                "count_net_side",
            ],
            "same_primary_parent_no_regroup_or_nonoverlap_rerun": [
                "exact_direction_flip",
                "deterministic_random_side",
                "constant_long",
                "constant_short",
                "one_bar_delayed_entry",
            ],
            "independent_rebuild_rule": (
                "construct own exact causal buckets, sort own raw candidates, "
                "and run own global nonoverlap scheduler"
            ),
            "definitions": {
                "issue_only": (
                    "independent rebuild using only Issue; always LONG"
                ),
                "redeem_only": (
                    "independent rebuild using only Redeem; always SHORT"
                ),
                "include_destroyed_black_funds": (
                    "independent rebuild adding DestroyedBlackFunds.amount_raw "
                    "as negative supply contamination diagnostic"
                ),
                "count_net_side": (
                    "independent rebuild using sign(Issue row count minus "
                    "Redeem row count) in each exact bucket"
                ),
                "exact_direction_flip": (
                    "same accepted primary parents with every side reversed"
                ),
                "deterministic_random_side": (
                    "SHA256 UTF-8(source_identity|TUSI-168|RANDOM_SIDE); first "
                    "byte <128 LONG else SHORT"
                ),
                "constant_long": "same accepted primary parents fixed LONG",
                "constant_short": "same accepted primary parents fixed SHORT",
                "one_bar_delayed_entry": (
                    "same accepted primary parents; entry_time and exit_time "
                    "both shifted exactly +300 seconds"
                ),
            },
            "one_bar_delayed_projection": {
                "all_shifted_parent_rows_retained_in_support_artifact": True,
                "parent_main_window_label_role": "provenance only",
                "each_diagnostic_and_economic_projection": (
                    "recheck both shifted timestamps against its own half-open "
                    "period"
                ),
                "shifted_boundary_crosser_action": (
                    "drop before any market or funding lookup"
                ),
                "dropped_report_fields": ["count", "source_identity"],
                "may_suppress_replace_or_reschedule_another_parent": False,
            },
            "support_reports_incidence_and_exact_overlap_without_price": True,
            "controls_cannot_replace_or_repair_primary": True,
        },
        "novelty": {
            "opens_only_after_complete_source_support": True,
            "frozen_comparator_artifacts": copy.deepcopy(
                esdi_authority["frozen_comparator_artifacts"]
            ),
            "registry_authority": {
                "path": str(ESDI_PREREGISTRATION_PATH),
                "json_path": "novelty.frozen_comparator_artifacts",
                "item_count": ESDI_COMPARATOR_COUNT,
                "canonical_compact_sorted_sha256": (
                    ESDI_COMPARATOR_SUBTREE_SHA256
                ),
                "import_without_deletion_or_reinterpretation": True,
                "groups_marked_separate_are_evaluated_separately": True,
            },
            "comparator_min_common_domain_entries_to_gate": 10,
            "minimum_applied_after_common_domain_filtering": True,
            "below_minimum": "reported not gated; never silently discarded",
            "prior_source_family_thresholds_exact_inclusive": {
                "exact_entry_jaccard": {"numerator": 1, "denominator": 5},
                "candidate_24h_containment": {
                    "numerator": 1,
                    "denominator": 2,
                },
                "squared_signed_exposure_pearson": {
                    "numerator": 4,
                    "denominator": 25,
                },
            },
            "gross9_each_positive_weight_sleeve_thresholds_exact_inclusive": {
                "exact_entry_jaccard": {"numerator": 1, "denominator": 10},
                "candidate_6h_containment": {
                    "numerator": 7,
                    "denominator": 20,
                },
                "occupied_bar_jaccard": {
                    "numerator": 1,
                    "denominator": 4,
                },
                "squared_signed_exposure_pearson": {
                    "numerator": 49,
                    "denominator": 400,
                },
            },
            "pearson_gate": (
                "absolute signed-exposure Pearson, implemented as exact "
                "squared Pearson with inclusive rational threshold"
            ),
            "metric_contract": {
                "entries": (
                    "unique strictly increasing integer UTC seconds after both "
                    "clocks use the identical half-open common domain"
                ),
                "exact_entry_jaccard": "|A intersect B| / |A union B|",
                "bidirectional_containment": (
                    "maximum of both directional fractions with inclusive "
                    "distance at most the frozen window"
                ),
                "signed_exposure": (
                    "exact {-1,0,1} on each 300-second bar open from sorted "
                    "contained nonoverlapping [entry,exit) intervals"
                ),
                "occupied_bar_jaccard": (
                    "Jaccard of indexes having nonzero signed exposure"
                ),
                "squared_signed_exposure_pearson": (
                    "covariance_numerator^2/(left_variance_numerator*"
                    "right_variance_numerator)"
                ),
                "inclusive_threshold_comparison": (
                    "exact integer cross multiplication"
                ),
                "terminal_undefined_inputs": [
                    "duplicate_or_unsorted_timestamps",
                    "empty_metric_denominator",
                    "unequal_exposure_vectors",
                    "zero_pearson_variance",
                ],
                "undefined_values_zero_filled": False,
            },
            "executable_metric_authority": {
                "module": str(ESDI_HELPER_PATH),
                "functions": [
                    "entries_in_domain",
                    "exact_entry_jaccard",
                    "bidirectional_entry_containment",
                    "fraction_at_most",
                    "signed_exposure_5m",
                    "occupied_bar_jaccard",
                    "squared_signed_exposure_pearson",
                ],
                "helper_git_blob_and_sha256_bound": True,
            },
            "all_frozen_comparators_and_positive_weight_gross9_sleeves_required": True,
            "missing_malformed_hash_drift_or_capability_mismatch": (
                "terminal failure"
            ),
            "comparator_removal_after_overlap_seen": False,
        },
        "economic_contract": {
            "evaluator_committed_tested_and_hash_bound_before_rows_open": True,
            "accounting_code_authority": {
                "path": (
                    "training/evaluate_ethereum_settlement_demand_"
                    "impulse_economics.py"
                ),
                "sha256": (
                    "fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235"
                ),
                "tusi_imports_strict_pure_helpers": True,
                "bound_by_later_source_replay_claim": True,
                "included_in_preregistration_repository_identity": False,
            },
            "standalone_accounting": {
                "leverage": 0.5,
                "base_cost_bp_per_notional_side": 6,
                "stress_cost_bp_per_notional_side": 10,
                "funding_interval": "entry_time <= funding_time < exit_time",
                "entry_price": "next BTCUSDT perpetual 5m open",
                "exit_price": "exact scheduled BTCUSDT perpetual 5m open",
                "full_calendar_cagr": True,
                "position_formulas": {
                    "allocated_equity": "E * w",
                    "quantity": "allocated_equity * 0.5 / O",
                    "entry_cost": "abs(quantity * O) * cost_rate",
                    "funding_cash": (
                        "-s * quantity * funding_rate * settlement_mark"
                    ),
                    "exit_price_cash": "s * quantity * (exit_open - O)",
                    "exit_cost": "abs(quantity * exit_open) * cost_rate",
                    "marked_equity_at_bar_open": (
                        "cash + sum(s * quantity * (open - entry_open))"
                    ),
                    "entry_and_exit_cost_charged_exactly_once": True,
                },
                "strict_mdd_hwm_initial_equity": 1.0,
                "strict_mdd_hwm_scope": (
                    "global full-calendar including idle pre-entry time and "
                    "every earlier sleeve or trade; no per-trade reset"
                ),
                "strict_mdd_bar_order": [
                    (
                        "mark pre-cost equity at bar open against the global HWM"
                    ),
                    (
                        "execute scheduled exits at that open and charge exit "
                        "cost; then execute entries and charge entry cost"
                    ),
                    (
                        "apply realized funding credits and debits whose "
                        "timestamp equals the bar"
                    ),
                    (
                        "update HWM with max(pre-cost open equity, post-entry-"
                        "cost equity plus funding credits plus favorable "
                        "long-high or short-low excursion)"
                    ),
                    (
                        "update trough from pre-cost open-equity baseline, "
                        "subtracting aggregate entry and exit costs once, with "
                        "funding credits/debits, adverse excursion, and "
                        "hypothetical liquidation cost at adverse mark"
                    ),
                ],
                "strict_mdd_formulas": {
                    "upper_t": "max(E_pre, E_pre - C_in + F_plus + A_plus)",
                    "hwm_t": "max(HWM_previous, upper_t)",
                    "lower_t": (
                        "min(E_pre, E_pre - C_in - C_out + F_plus + F_minus "
                        "+ A_minus - Q * P_bad * cost_rate)"
                    ),
                    "mdd": "max_t(1 - lower_t / HWM_t)",
                    "funding_credit_constraint": "F_plus >= 0",
                    "funding_debit_constraint": "F_minus <= 0",
                    "favorable_excursion_constraint": "A_plus >= 0",
                    "adverse_excursion_constraint": "A_minus <= 0",
                    "excursion_quantity": (
                        "net signed quantity after that bar exits and entries"
                    ),
                    "liquidation_quantity": "gross quantity",
                    "exit_open_already_marked_in_E_pre": True,
                    "C_out_is_single_exit_cost_event_not_second_charge": True,
                },
                "close_only_truncated_or_favorable_adverse_reordering": False,
                "nonpositive_liquidation_envelope_equity": "terminal failure",
                "period_metrics": {
                    "calendar_years_full": 3.0,
                    "calendar_years_other": (
                        "(end-start).total_seconds()/(365.25*86400)"
                    ),
                    "absolute_return": "final_equity - 1",
                    "cagr": "exp(log(final_equity)/years) - 1",
                    "positive_cagr_to_mdd": "CAGR / max(MDD, 1e-15)",
                    "nonpositive_cagr_to_mdd": 0,
                    "mean_gross_underlying_bp": (
                        "arithmetic mean of side*(exit_open/entry_open-1)*10000 "
                        "over contained trades"
                    ),
                },
            },
            "calendar_month_clustered_sign_flip": {
                "helper": "calendar_month_clustered_signflip",
                "trade_record_order": [
                    "entry_time",
                    "exit_time",
                    "source_identity",
                ],
                "cluster": "UTC entry month",
                "cluster_value": (
                    "accumulate Python float net return on allocated equity"
                ),
                "ordered_vector": (
                    "NumPy float64 in ascending YYYY-MM order"
                ),
                "discard_only_when_absolute_sum_at_most": 1e-15,
                "observed": "left-to-right ordered-vector sum",
                "observed_total_nonpositive_p": 1,
                "exact_enumeration_when_nonzero_months_at_most": 20,
                "exact_sign_stream": (
                    "itertools.product((-1.0,1.0),repeat=m) in native order"
                ),
                "one_sided_exceedance": (
                    "np.dot(np.asarray(signs),ordered) >= observed - 1e-15"
                ),
                "monte_carlo_when_nonzero_months_above": 20,
                "rng": "numpy.default_rng(20260730)",
                "sign_vectors": 10_000,
                "batch_rows": [4_096, 4_096, 1_808],
                "random_draw": (
                    "rng.integers(0,2,size=(batch,m),dtype=np.int8)"
                ),
                "sign_conversion": (
                    "signs.astype(np.float64)*2.0-1.0"
                ),
                "matrix_evaluation": "row-major signs @ ordered",
                "monte_carlo_p": "(exceed+1)/10001",
            },
            "standalone_gate_base_and_stress_each_opened_period": {
                "absolute_return": ">0",
                "full_calendar_cagr_to_strict_mdd": ">=3.0",
                "strict_mdd": "<=0.15",
                "mean_gross_underlying_bp": ">=20",
                "calendar_month_clustered_signflip_p": "<=0.10",
            },
            "independent_control_superiority": {
                "metric": "full_calendar_CAGR / strict_MDD",
                "comparison": "primary strictly greater",
                "period_scope": "every opened standalone period",
                "cost_scope": ["base", "stress"],
                "controls": [
                    "issue_only",
                    "redeem_only",
                    "include_destroyed_black_funds",
                    "count_net_side",
                ],
                "gate_when_contained_accepted_trades_at_least": 1,
                "zero_trade_control": "reported and not superiority-gated",
                "undefined_nonzero_support_metric": "terminal failure",
                "primary_absolute_gates_still_required": True,
            },
            "same_primary_parent_complete_qualification": {
                "controls": [
                    "exact_direction_flip",
                    "deterministic_random_side",
                    "constant_long",
                    "constant_short",
                ],
                "definition": (
                    "passes all five standalone gates under both base and "
                    "stress in every standalone period opened before stop"
                ),
                "any_control_may_completely_qualify": False,
                "one_bar_delayed_entry": (
                    "timing sensitivity diagnostic only; cannot replace or "
                    "select primary"
                ),
            },
            "period_open_order": [
                "2023H2",
                "2024",
                "selection",
                "same_gross_gross9_selection_and_weight",
                "future25",
                "future26",
                "stitched_full",
            ],
            "stop_permanently_at_first_failure": True,
            "later_periods_veto_only": True,
        },
        "gross9": {
            "authority": copy.deepcopy(esdi_gross9["authority"]),
            "esdi_artifact_binding": {
                "path": str(ESDI_PREREGISTRATION_PATH),
                "file_sha256": ESDI_PREREGISTRATION_SHA256,
                "manifest_hash": ESDI_MANIFEST_HASH,
                "gross9_subtree_sha256": ESDI_GROSS9_SUBTREE_SHA256,
                "authority_subtree_sha256": ESDI_GROSS9_AUTHORITY_SHA256,
                "runtime_closure_subtree_sha256": (
                    ESDI_RUNTIME_CLOSURE_SHA256
                ),
                "complete_authority_closure_and_environment_deep_copied": True,
                "current_closure_and_environment_match_required": True,
            },
            "weights": copy.deepcopy(esdi_gross9["weights"]),
            "baseline_gross": 9.0,
            "candidate_weights": [0.25, 0.50, 0.75, 1.00],
            "treatment": "scale every Gross9 sleeve by (9-w)/9; add TUSI at w",
            "same_configured_gross": 9.0,
            "selection_periods": ["2023H2", "2024"],
            "requirements": {
                "applies_to_every_candidate_weight": True,
                "periods": ["2023H2", "2024"],
                "cost_settings": ["base", "stress"],
                "base_and_stress_cagr_mdd_improvement_min": 0.05,
                "unscaled_absolute_return_retention_min": 0.97,
                "base_and_stress_absolute_return_positive": True,
                "strict_mdd_reduced_in_at_least_one_of_four_period_cost_cells": True,
            },
            "ranking": (
                "maximum minimum base/stress improvement across both periods; "
                "tie lower weight"
            ),
            "freeze_rank": 1,
            "future_uses_only_frozen_weight": True,
            "future_contract": {
                "periods_evaluated_independently": ["future25", "future26"],
                "cost_settings": ["base", "stress"],
                "cagr_mdd_improvement_min_each_cost": 0.05,
                "baseline_absolute_return_retention_min_each_cost": 0.97,
                "treatment_return_positive_each_cost": True,
                "liquidation_safety_each_cost": True,
                "strict_mdd_lower_than_baseline_in_at_least_one_cost_per_period": True,
            },
            "future_weight_grid_opened": False,
            "future_rerank_or_alternate_weight": False,
        },
        "strict_sequence": [
            (
                "source_axis_and_mechanism_decisions plus preregistration "
                "producer and tests committed and pushed"
            ),
            "write_once_preregistration_artifact created committed and pushed",
            (
                "future builder evaluators and tests implemented committed and "
                "pushed with preregistration artifact SHA-256 and manifest_hash "
                "bound as constants"
            ),
            (
                "separate source-replay claim created committed and pushed, "
                "sealing all future protocol Git blobs and SHA-256 before the "
                "first precutoff event/log/receipt replay RPC and before source "
                "incidence; prior metadata-only boundary headers acknowledged"
            ),
            "complete_source_replay_once",
            "write_once_source_csv_and_manifest_committed_without_evaluator_change",
            "source_support_then_novelty_then_economics_stop_on_first_failure",
        ],
        "producer_effects": {
            "network_calls": 0,
            "non_git_subprocess_calls": 0,
            "helper_import_preflight_git_read_only_subprocess_calls": 2,
            "artifact_creation_git_read_only_subprocess_calls": 4,
            "git_read_only_subprocess_calls": 6,
            "source_csv_or_source_rows_opened": 0,
            "esdi_preregistration_metadata_artifacts_opened": 1,
            "comparator_data_rows_opened": 0,
            "gross9_data_rows_opened": 0,
            "market_rows_opened": 0,
            "funding_rows_opened": 0,
            "future_protocol_files_opened_or_hashed": 0,
            "bound_committed_paths_hashed": len(committed_identity_paths()),
        },
        **{name: False for name in EVIDENCE_BOUNDARIES},
    }


def build_manifest(repository_identity: Mapping[str, Any]) -> dict[str, Any]:
    core = _core_manifest(repository_identity)
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_sanitized_transports(transports: Any) -> None:
    if transports != TRANSPORTS or any(
        set(transport) != {"role", "scheme", "hostname", "port"}
        for transport in transports
    ):
        raise RuntimeError("TUSI-168 transport serialization is not sanitized")
    transport_bytes = json.dumps(
        transports,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if any(token in transport_bytes for token in ("://", "/", "@", "?", "#")):
        raise RuntimeError("TUSI-168 transport secret surface serialized")


def _matches_schema_type(value: Any, schema_type: str) -> bool:
    if schema_type == "str":
        return type(value) is str
    if schema_type == "bool":
        return type(value) is bool
    if schema_type == "object":
        return type(value) is dict
    if schema_type == "array[str]":
        return type(value) is list and all(type(item) is str for item in value)
    raise RuntimeError("TUSI-168 unknown preregistration schema type")


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if set(payload) != set(PREREGISTRATION_TOP_LEVEL_TYPES) or any(
        not _matches_schema_type(payload.get(key), schema_type)
        for key, schema_type in PREREGISTRATION_TOP_LEVEL_TYPES.items()
    ):
        raise RuntimeError("TUSI-168 preregistration top-level schema drift")
    identity = payload.get("frozen_preregistration", {}).get(
        "repository_identity"
    )
    if not isinstance(identity, Mapping):
        raise RuntimeError("TUSI-168 repository identity is missing")
    expected = build_manifest(identity)
    if dict(payload) != expected:
        raise RuntimeError("TUSI-168 preregistration differs from frozen code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("TUSI-168 internal manifest hash mismatch")
    if any(payload.get(name) is not False for name in EVIDENCE_BOUNDARIES):
        raise RuntimeError("TUSI-168 evidence boundary opened")
    if sum(payload["gross9"]["weights"].values()) != 9.0:
        raise RuntimeError("TUSI-168 Gross9 weights do not sum to 9.0")
    validate_sanitized_transports(payload["source"]["transports"])


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


def _output_relative(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("TUSI-168 output must be repository-relative")
    if candidate != DEFAULT_OUTPUT:
        raise RuntimeError("TUSI-168 output must equal the frozen singleton path")
    return candidate


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise RuntimeError("TUSI-168 output parent is unsafe") from error


def _open_regular(parent_fd: int, filename: str) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_fd)
    except OSError as error:
        if isinstance(error, FileNotFoundError):
            raise
        raise RuntimeError("TUSI-168 output path is unsafe") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("TUSI-168 output is not a regular file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_open_regular(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_regular(parent_fd: int, filename: str) -> bytes:
    descriptor = _open_regular(parent_fd, filename)
    try:
        return _read_open_regular(descriptor)
    finally:
        os.close(descriptor)


def _artifact_file_identity(file_status: os.stat_result) -> tuple[int, ...]:
    return (
        file_status.st_dev,
        file_status.st_ino,
        file_status.st_mode,
        file_status.st_size,
        file_status.st_mtime_ns,
        file_status.st_ctime_ns,
    )


def write_once(
    output: str | Path = DEFAULT_OUTPUT,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Atomically create the artifact or verify an identical regular file."""

    candidate = _output_relative(output)
    parent_fd = _open_parent(candidate)
    try:
        existing_fd = _open_regular(parent_fd, candidate.name)
    except FileNotFoundError:
        existing_fd = None
    if existing_fd is not None:
        try:
            initial_status = os.fstat(existing_fd)
            existing = _read_open_regular(existing_fd)
            try:
                stored = json.loads(existing.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "TUSI-168 existing preregistration drift: "
                    "not canonical JSON"
                ) from error
            if not isinstance(stored, dict):
                raise RuntimeError(
                    "TUSI-168 existing preregistration root drift"
                )
            canonical_existing = canonical_manifest_bytes(stored)
            if existing != canonical_existing:
                raise RuntimeError("TUSI-168 existing preregistration drift")
            if payload is not None and dict(payload) != stored:
                raise RuntimeError(
                    "TUSI-168 supplied preregistration payload drift"
                )
            identity = stored["frozen_preregistration"]["repository_identity"]
            validate_existing_artifact_repository(identity, existing)
            final_status = os.fstat(existing_fd)
            try:
                path_status = os.stat(
                    candidate.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise RuntimeError(
                    "TUSI-168 existing preregistration race drift"
                ) from error
            if (
                _read_open_regular(existing_fd) != existing
                or _artifact_file_identity(final_status)
                != _artifact_file_identity(initial_status)
                or not stat.S_ISREG(path_status.st_mode)
                or (path_status.st_dev, path_status.st_ino)
                != (final_status.st_dev, final_status.st_ino)
            ):
                raise RuntimeError(
                    "TUSI-168 existing preregistration race drift"
                )
            return "verified_existing", stored
        finally:
            os.close(existing_fd)
            os.close(parent_fd)

    os.close(parent_fd)
    validate_frozen_documents_and_helper()
    identity = frozen_repository_identity()
    expected = build_manifest(identity)
    if payload is not None and dict(payload) != expected:
        raise RuntimeError("TUSI-168 supplied preregistration payload drift")
    canonical = canonical_manifest_bytes(expected)
    parent_fd = _open_parent(candidate)
    temporary = f".{candidate.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    temporary_created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        validate_creation_publish_state(
            identity,
            candidate.parent / temporary,
        )
        try:
            os.link(
                temporary,
                candidate.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular(parent_fd, candidate.name) != canonical:
                raise RuntimeError("TUSI-168 preregistration race drift")
            return "verified_existing", expected
        os.fsync(parent_fd)
        return "created", expected
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.fsync(parent_fd)
        os.close(parent_fd)


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
