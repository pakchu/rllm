"""Strict, staged economics evaluator for the frozen TUSI-168 policy.

The numerical accounting implementation is not copied here.  It is loaded
from the committed, hash-bound ESDI authority and its pure helpers are reused
directly.  Production entry points authenticate source support and novelty
before any callback capable of opening market, funding, return, or PnL rows is
invoked.  Direct frames and callbacks are synthetic-test interfaces only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import csv
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import types
from typing import Any, NamedTuple, cast

import pandas as pd


POLICY_ID = "TUSI-168"
PROTOCOL_VERSION = "tron_usdt_supply_impulse_economics_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE_PATH = Path(
    "training/evaluate_tron_usdt_supply_impulse_economics.py"
)
EVALUATOR_TEST_PATH = Path(
    "tests/test_evaluate_tron_usdt_supply_impulse_economics.py"
)
PREREGISTRATION_ARTIFACT = Path(
    "results/tron_usdt_supply_impulse_preregistration_2026-07-30.json"
)
PREREGISTRATION_ARTIFACT_SHA256 = (
    "54817044b8df76dc347ed64b6fe5f6f2dfdddcdb211bded4ba2b1af133d49067"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d67cd1b67632ae92e9458395e729627a6f4c3b4b75ce97187653eac3a09e40c1"
)
ESDI_ECONOMICS_AUTHORITY_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_economics.py"
)
ESDI_ECONOMICS_AUTHORITY_SHA256 = (
    "fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235"
)
SOURCE_SUPPORT_ARTIFACT = Path(
    "results/tron_usdt_supply_impulse_source_support_2026-07-30.json"
)
PRIMARY_CLOCK_ARTIFACT = Path(
    "results/tron_usdt_supply_impulse_primary_clock_2026-07-30.csv.gz"
)
CONTROL_CLOCK_ARTIFACT = Path(
    "results/tron_usdt_supply_impulse_control_clocks_2026-07-30.csv.gz"
)
NOVELTY_ARTIFACT = Path(
    "results/tron_usdt_supply_impulse_novelty_2026-07-30.json"
)
SOURCE_SUPPORT_PROTOCOL_VERSION = "tron_usdt_supply_impulse_source_support_v1"
NOVELTY_PROTOCOL_VERSION = "tron_usdt_supply_impulse_novelty_v1"
ECONOMIC_RECEIPT_PROTOCOL = "tron_usdt_supply_impulse_economic_stage_receipt_v1"
ECONOMIC_ATTEMPT_PROTOCOL = (
    "tron_usdt_supply_impulse_economic_attempt_claim_v1"
)

INDEPENDENT_CONTROLS = (
    "issue_only",
    "redeem_only",
    "include_destroyed_black_funds",
    "count_net_side",
)
SAME_PARENT_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
)
TIMING_CONTROL = "one_bar_delayed_entry"
CONTROL_NAMES = INDEPENDENT_CONTROLS + SAME_PARENT_CONTROLS + (TIMING_CONTROL,)
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
PERIODS = {
    "2023H2": (
        pd.Timestamp("2023-06-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
    "2024": (
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "selection": (
        pd.Timestamp("2023-06-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "future25": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "future26": (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-06-01T00:00:00Z"),
    ),
    "full": (
        pd.Timestamp("2023-06-01T00:00:00Z"),
        pd.Timestamp("2026-06-01T00:00:00Z"),
    ),
}
SELECTION_PERIODS = {
    "2023H2": PERIODS["2023H2"],
    "2024": PERIODS["2024"],
}
ECONOMIC_STAGE_ORDER = (
    "2023H2",
    "2024",
    "selection",
    "same_gross",
    "future25",
    "future26",
    "full",
)
STAGE_CUTOFFS = {
    "2023H2": PERIODS["2023H2"][1],
    "2024": PERIODS["2024"][1],
    "selection": PERIODS["selection"][1],
    "same_gross": PERIODS["selection"][1],
    "future25": PERIODS["future25"][1],
    "future26": PERIODS["future26"][1],
    "full": PERIODS["full"][1],
}
STAGE_RECEIPT_NAMES = {
    stage: f"tron_usdt_supply_impulse_economics_{stage}_2026-07-30.json"
    for stage in ECONOMIC_STAGE_ORDER
}
STAGE_ATTEMPT_NAMES = {
    stage: (
        f"tron_usdt_supply_impulse_economics_{stage}_"
        "attempt_claim_2026-07-30.json"
    )
    for stage in ECONOMIC_STAGE_ORDER
}
CANONICAL_RESULTS_ROOT = REPOSITORY_ROOT / "results"
PRODUCTION_OUTPUT_NAMES = frozenset(
    (*STAGE_RECEIPT_NAMES.values(), *STAGE_ATTEMPT_NAMES.values())
)


class _Publication(NamedTuple):
    status: str
    sha256: str


class _LoadedReceipt(NamedTuple):
    payload: dict[str, Any]
    raw: bytes
    sha256: str


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("TUSI-168 Git provenance check failed") from error


def _canonical_absolute_path(
    value: str | Path,
    *,
    label: str,
) -> Path:
    raw = os.fspath(value)
    if not raw or "\x00" in raw:
        raise RuntimeError(f"TUSI-168 {label} path is unsafe")
    pieces = raw.split(os.sep)
    if raw.startswith(os.sep):
        pieces = pieces[1:]
    if (
        not raw.startswith(os.sep)
        or raw.endswith(os.sep)
        or any(piece in {"", ".", ".."} for piece in pieces)
    ):
        raise RuntimeError(f"TUSI-168 {label} path is not canonical absolute")
    return Path(raw)


def _prepare_output_target(
    path: str | Path,
    *,
    root: str | Path,
    production: bool,
) -> tuple[Path, Path]:
    root_path = _canonical_absolute_path(root, label="output root")
    raw_target = os.fspath(path)
    if Path(raw_target).is_absolute():
        target = _canonical_absolute_path(raw_target, label="output")
    else:
        if (
            not raw_target
            or raw_target.endswith(os.sep)
            or any(
                piece in {"", ".", ".."}
                for piece in raw_target.split(os.sep)
            )
        ):
            raise RuntimeError("TUSI-168 output path is not canonical relative")
        target = root_path / raw_target
    try:
        relative = target.relative_to(root_path)
    except ValueError as error:
        raise RuntimeError("TUSI-168 output escapes its explicit root") from error
    if not relative.parts or target.name in {"", ".", ".."}:
        raise RuntimeError("TUSI-168 output leaf is unsafe")
    if production:
        if (
            root_path != CANONICAL_RESULTS_ROOT
            or len(relative.parts) != 1
            or target.name not in PRODUCTION_OUTPUT_NAMES
        ):
            raise RuntimeError(
                "TUSI-168 production output is not an exact results path"
            )
    else:
        resolved_root = Path(os.path.realpath(root_path))
        if resolved_root == CANONICAL_RESULTS_ROOT or (
            resolved_root.is_relative_to(CANONICAL_RESULTS_ROOT)
        ):
            raise RuntimeError(
                "TUSI-168 synthetic output root cannot resolve under results"
            )
    return root_path, target


def _open_directory_component(
    parent_fd: int,
    component: str,
    *,
    create: bool,
) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(component, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(component, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        return os.open(component, flags, dir_fd=parent_fd)


def _open_absolute_directory(path: Path) -> int:
    descriptor = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            next_descriptor = _open_directory_component(
                descriptor,
                component,
                create=False,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_output_parent(
    path: str | Path,
    *,
    root: str | Path,
    production: bool,
    create: bool,
) -> tuple[int, str, Path]:
    root_path, target = _prepare_output_target(
        path,
        root=root,
        production=production,
    )
    try:
        descriptor = _open_absolute_directory(root_path)
    except OSError as error:
        raise RuntimeError("TUSI-168 output root has an unsafe ancestor") from error
    relative = target.relative_to(root_path)
    try:
        for component in relative.parts[:-1]:
            next_descriptor = _open_directory_component(
                descriptor,
                component,
                create=create,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise RuntimeError("TUSI-168 output parent is unsafe") from error
    return descriptor, relative.parts[-1], target


def _read_fd_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _write_fd_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeError("TUSI-168 output write made no progress")
        offset += written


def _read_output_bytes(
    path: str | Path,
    *,
    root: str | Path,
    production: bool,
) -> bytes:
    parent_fd, leaf, _ = _open_output_parent(
        path,
        root=root,
        production=production,
        create=False,
    )
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise RuntimeError("TUSI-168 output leaf is unsafe or missing") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError("TUSI-168 output leaf is not a regular file")
            return _read_fd_all(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _output_leaf_exists(
    path: str | Path,
    *,
    root: str | Path,
    production: bool,
) -> bool:
    parent_fd, leaf, _ = _open_output_parent(
        path,
        root=root,
        production=production,
        create=False,
    )
    try:
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent_fd)


def _read_regular(relative_path: Path) -> bytes:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("TUSI-168 authority path is unsafe")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(REPOSITORY_ROOT, directory_flags)
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
                raise RuntimeError("TUSI-168 authority is not a regular file")
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise RuntimeError("TUSI-168 authority path is unsafe") from error
    finally:
        os.close(descriptor)


def _git_blob(raw: bytes, object_id: str) -> str:
    digest = hashlib.sha1() if len(object_id) == 40 else hashlib.sha256()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _load_esdi_authority() -> tuple[Any, str]:
    head = _git("rev-parse", "HEAD").decode("ascii").strip()
    record = _git(
        "ls-tree", "-z", head, "--", ESDI_ECONOMICS_AUTHORITY_PATH.as_posix()
    )
    rows = [row for row in record.split(b"\0") if row]
    if len(rows) != 1:
        raise RuntimeError("TUSI-168 ESDI authority committed blob is missing")
    metadata, raw_path = rows[0].split(b"\t", 1)
    mode, object_type, object_id = metadata.decode("ascii").split()
    if (
        mode != "100644"
        or object_type != "blob"
        or raw_path.decode("utf-8") != ESDI_ECONOMICS_AUTHORITY_PATH.as_posix()
    ):
        raise RuntimeError("TUSI-168 ESDI authority Git identity drift")
    raw = _read_regular(ESDI_ECONOMICS_AUTHORITY_PATH)
    if hashlib.sha256(raw).hexdigest() != ESDI_ECONOMICS_AUTHORITY_SHA256:
        raise RuntimeError("TUSI-168 ESDI authority SHA-256 drift")
    if _git_blob(raw, object_id) != object_id:
        raise RuntimeError("TUSI-168 ESDI authority differs from committed blob")
    module_name = (
        "training._tusi_verified_esdi_economics_"
        f"{ESDI_ECONOMICS_AUTHORITY_SHA256}"
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(REPOSITORY_ROOT / ESDI_ECONOMICS_AUTHORITY_PATH)
    module.__package__ = "training"
    module.__loader__ = None
    module.__spec__ = None
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        code = compile(
            raw,
            module.__file__,
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module, object_id


esdi, ESDI_ECONOMICS_AUTHORITY_GIT_BLOB = _load_esdi_authority()

# Exact strict accounting authority bindings.  These identities are deliberate:
# TUSI must not fork CAGR, MDD, funding/cost, sign-flip, or Gross9 arithmetic.
LEVERAGE = esdi.LEVERAGE
BASE_COST_RATE = esdi.BASE_COST_RATE
STRESS_COST_RATE = esdi.STRESS_COST_RATE
BASELINE_GROSS = esdi.BASELINE_GROSS
GROSS9_WEIGHTS = esdi.GROSS9_WEIGHTS
GROSS9_SLEEVES = esdi.GROSS9_SLEEVES
CANDIDATE_WEIGHTS = esdi.CANDIDATE_WEIGHTS
MARKET_COLUMNS = esdi.MARKET_COLUMNS
FUNDING_COLUMNS = esdi.FUNDING_COLUMNS
CLOCK_COLUMNS = esdi.CLOCK_COLUMNS
validate_clock = esdi.validate_clock
validate_market = esdi.validate_market
validate_funding = esdi.validate_funding
calendar_month_clustered_signflip = esdi.calendar_month_clustered_signflip
standalone_gate_checks = esdi.standalone_gate_checks
simulate_portfolio = esdi.simulate_portfolio
_simulate_portfolio = esdi._simulate_portfolio
_full_calendar_cagr = esdi._full_calendar_cagr
_calendar_years = esdi._calendar_years


def _json_ready(value: Any) -> Any:
    return cast(Any, esdi._json_ready(value))


def canonical_hash(payload: Any) -> str:
    return cast(str, esdi.canonical_hash(_json_ready(payload)))


def canonical_json_bytes(payload: Any) -> bytes:
    return cast(bytes, esdi.canonical_json_bytes(_json_ready(payload)))


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        canonical = _canonical_absolute_path(candidate, label="hash target")
        try:
            parent_fd = _open_absolute_directory(canonical.parent)
        except OSError as error:
            raise RuntimeError(
                f"TUSI-168 hash target is unsafe: {path}"
            ) from error
        try:
            try:
                descriptor = os.open(
                    canonical.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as error:
                raise RuntimeError(
                    f"TUSI-168 hash target is unsafe: {path}"
                ) from error
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RuntimeError(
                        f"TUSI-168 hash target is not a regular file: {path}"
                    )
                raw = _read_fd_all(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
    else:
        raw = _read_regular(candidate)
    digest = hashlib.sha256()
    digest.update(raw)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_bound_preregistration(
    path: str | Path = PREREGISTRATION_ARTIFACT,
) -> dict[str, Any]:
    if Path(path) != PREREGISTRATION_ARTIFACT:
        raise RuntimeError("TUSI-168 production preregistration path is canonical")
    target = REPOSITORY_ROOT / PREREGISTRATION_ARTIFACT
    raw = _read_regular(PREREGISTRATION_ARTIFACT)
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("TUSI-168 preregistration artifact SHA-256 drift")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("TUSI-168 preregistration JSON is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("TUSI-168 preregistration is not an object")
    expected_serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected_serialized:
        raise RuntimeError("TUSI-168 preregistration serialization drift")
    validate_frozen_contract(payload)
    if target.resolve() != (REPOSITORY_ROOT / path).resolve():
        raise RuntimeError("TUSI-168 preregistration path drift")
    return payload


def validate_frozen_contract(registration: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        key: value for key, value in registration.items() if key != "manifest_hash"
    }
    if (
        registration.get("protocol_version")
        != "tron_usdt_supply_impulse_preregistration_v1"
        or registration.get("policy_id") != POLICY_ID
        or registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH
    ):
        raise RuntimeError("TUSI-168 preregistration identity or manifest drift")
    accounting = (
        registration.get("economic_contract", {})
        .get("accounting_code_authority", {})
    )
    if accounting != {
        "path": ESDI_ECONOMICS_AUTHORITY_PATH.as_posix(),
        "sha256": ESDI_ECONOMICS_AUTHORITY_SHA256,
        "tusi_imports_strict_pure_helpers": True,
        "bound_by_later_source_replay_claim": True,
        "included_in_preregistration_repository_identity": False,
    }:
        raise RuntimeError("TUSI-168 strict accounting authority drift")
    gross9 = registration.get("gross9")
    if not isinstance(gross9, Mapping):
        raise RuntimeError("TUSI-168 Gross9 contract is absent")
    if (
        gross9.get("weights") != dict(GROSS9_WEIGHTS)
        or gross9.get("candidate_weights") != list(CANDIDATE_WEIGHTS)
        or float(cast(float, gross9.get("baseline_gross", -1.0)))
        != BASELINE_GROSS
        or float(cast(float, gross9.get("same_configured_gross", -1.0)))
        != BASELINE_GROSS
        or gross9.get("selection_periods") != list(SELECTION_PERIODS)
        or gross9.get("future_uses_only_frozen_weight") is not True
        or gross9.get("future_weight_grid_opened") is not False
        or gross9.get("future_rerank_or_alternate_weight") is not False
    ):
        raise RuntimeError("TUSI-168 frozen Gross9 contract drift")
    binding = gross9.get("esdi_artifact_binding")
    if not isinstance(binding, Mapping) or (
        binding.get("file_sha256")
        != "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
        or binding.get("manifest_hash")
        != "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
        or binding.get("gross9_subtree_sha256")
        != "d79c79789ed48c7c2a94bac4474583798c2306bd320abb2617c354878c3578fe"
        or binding.get("authority_subtree_sha256")
        != "b3490c484d3fda1d5b649498e0d84325e203cd2664086e68cebd76509a54957e"
        or binding.get("runtime_closure_subtree_sha256")
        != "ffffb68c0900836ba06b573398c4825bd9d15161a9e36818aeb68fc33a86d84a"
    ):
        raise RuntimeError("TUSI-168 ESDI Gross9 binding drift")
    return {
        "validated": True,
        "preregistration_sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "esdi_economics_sha256": ESDI_ECONOMICS_AUTHORITY_SHA256,
        "esdi_economics_git_blob": ESDI_ECONOMICS_AUTHORITY_GIT_BLOB,
    }


def _validate_manifest(payload: Mapping[str, Any], label: str) -> None:
    manifest = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if not _is_sha256(manifest) or manifest != canonical_hash(core):
        raise RuntimeError(f"TUSI-168 {label} manifest drift")


def validate_passed_source_support(
    payload: Mapping[str, Any],
    *,
    exact: bool = True,
) -> dict[str, Any]:
    _validate_manifest(payload, "source-support")
    required = {
        "protocol_version",
        "policy_id",
        "status",
        "terminal",
        "artifact_eligible",
        "support_passed",
        "decision",
        "registration",
        "source_contract",
        "raw_candidate_counts",
        "accepted_clock_counts",
        "period_diagnostics",
        "support_audit",
        "support_checks",
        "future_append_selection_invariance",
        "control_overlap",
        "clock_artifacts",
        "evidence_boundary",
        "source_support_precedes_novelty",
        "novelty_comparator_market_or_outcome_artifacts_opened",
        "manifest_hash",
    }
    if exact and set(payload) != required:
        raise RuntimeError("TUSI-168 source-support exact schema drift")
    checks = payload.get("support_checks")
    artifacts = payload.get("clock_artifacts")
    control_order = ("primary",) + CONTROL_NAMES
    period_order = (
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
    )
    raw_counts = payload.get("raw_candidate_counts")
    accepted_counts = payload.get("accepted_clock_counts")
    diagnostics = payload.get("period_diagnostics")
    registration = payload.get("registration")
    if (
        payload.get("protocol_version") != SOURCE_SUPPORT_PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
        or payload.get("status") != "source_support_passed"
        or payload.get("terminal") is not True
        or payload.get("artifact_eligible") is not True
        or payload.get("support_passed") is not True
        or payload.get("decision") != "SOURCE_SUPPORT_PASS"
        or payload.get("source_support_precedes_novelty") is not True
        or payload.get(
            "novelty_comparator_market_or_outcome_artifacts_opened"
        )
        is not False
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(raw_counts, Mapping)
        or tuple(raw_counts) != control_order
        or any(type(value) is not int for value in raw_counts.values())
        or not isinstance(accepted_counts, Mapping)
        or tuple(accepted_counts) != control_order
        or any(type(value) is not int for value in accepted_counts.values())
        or not isinstance(diagnostics, Mapping)
        or set(diagnostics) != set(period_order)
        or not isinstance(registration, Mapping)
        or registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or registration.get("mode") != "artifact"
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"primary_sha256", "controls_sha256"}
        or not all(_is_sha256(value) for value in artifacts.values())
    ):
        raise RuntimeError("TUSI-168 source support did not pass exactly")
    return dict(payload)


def validate_passed_novelty(
    payload: Mapping[str, Any],
    *,
    exact: bool = True,
) -> dict[str, Any]:
    _validate_manifest(payload, "novelty")
    if (
        payload.get("protocol_version") != NOVELTY_PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
        or payload.get("status") not in (None, "novelty_passed")
        or payload.get("decision")
        not in (None, "NOVELTY_PASS_OPEN_STRICT_ECONOMICS")
        or payload.get("terminal") not in (None, False)
    ):
        raise RuntimeError("TUSI-168 novelty identity drift")
    novelty = payload.get("novelty")
    if not isinstance(novelty, Mapping):
        raise RuntimeError("TUSI-168 novelty result is absent")
    if novelty.get("passed") is not True or novelty.get("terminal") is not False:
        raise RuntimeError("TUSI-168 novelty did not pass")
    failed_checks = novelty.get("failed_checks")
    if failed_checks not in (None, []):
        raise RuntimeError("TUSI-168 novelty contains failed checks")
    evidence = payload.get("evidence_boundary")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("TUSI-168 novelty evidence boundary is absent")
    forbidden_true = (
        "candidate_market_rows_opened",
        "candidate_funding_rows_opened",
        "candidate_outcome_rows_opened",
        "candidate_returns_or_pnl_computed",
        "portfolio_return_or_pnl_metrics_computed",
    )
    if any(evidence.get(key) is not False for key in forbidden_true):
        raise RuntimeError("TUSI-168 novelty leaked economic or future rows")
    source_binding = payload.get("source_support")
    if not isinstance(source_binding, Mapping) or not all(
        _is_sha256(source_binding.get(key)) for key in ("sha256", "manifest_hash")
    ):
        raise RuntimeError("TUSI-168 novelty source-support binding drift")
    if exact:
        required = {
            "protocol_version",
            "policy_id",
            "status",
            "terminal",
            "decision",
            "preregistration",
            "attempt_claim",
            "source_support",
            "candidate_clock",
            "gross9_clock_artifact",
            "registry",
            "novelty",
            "evidence_boundary",
            "manifest_hash",
        }
        expected_evidence = {
            "source_support_report_bytes_opened": True,
            "source_support_primary_clock_bytes_opened": True,
            "comparator_clock_artifact_bytes_opened": True,
            "gross9_clock_artifact_bytes_opened": True,
            "candidate_market_rows_opened": False,
            "candidate_funding_rows_opened": False,
            "candidate_outcome_rows_opened": False,
            "candidate_returns_or_pnl_computed": False,
            "portfolio_return_or_pnl_metrics_computed": False,
            "network_calls": 0,
        }
        preregistration = payload.get("preregistration")
        registry = payload.get("registry")
        gross9_rows = novelty.get("gross9_sleeves")
        prior_rows = novelty.get("prior_source_comparators")
        if (
            set(payload) != required
            or dict(evidence) != expected_evidence
            or preregistration
            != {
                "path": PREREGISTRATION_ARTIFACT.as_posix(),
                "sha256": PREREGISTRATION_ARTIFACT_SHA256,
                "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            }
            or not isinstance(registry, Mapping)
            or registry.get("artifacts") != 18
            or registry.get("canonical_compact_sorted_sha256")
            != "0d13c9de1e098446aaaa78b9a24c7d05c7ec375df05d79c9f8969792546bd4a3"
            or not isinstance(prior_rows, list)
            or len(prior_rows) != registry.get("comparator_groups")
            or any(
                not isinstance(row, Mapping) or row.get("passed") is not True
                for row in prior_rows
            )
            or not isinstance(gross9_rows, list)
            or [row.get("sleeve") for row in gross9_rows if isinstance(row, Mapping)]
            != list(GROSS9_SLEEVES)
            or any(
                not isinstance(row, Mapping)
                or row.get("passed") is not True
                or float(cast(float, row.get("weight", -1.0)))
                != float(GROSS9_WEIGHTS[str(row.get("sleeve"))])
                for row in gross9_rows
            )
        ):
            raise RuntimeError("TUSI-168 novelty exact schema drift")
    return dict(payload)


def _assert_committed_clean(relative_path: Path) -> None:
    _git("ls-files", "--error-unmatch", "--", relative_path.as_posix())
    if _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative_path.as_posix(),
    ):
        raise RuntimeError(f"TUSI-168 artifact is not committed-clean: {relative_path}")
    committed = _git("show", f"HEAD:{relative_path.as_posix()}")
    if committed != _read_regular(relative_path):
        raise RuntimeError(f"TUSI-168 committed artifact byte drift: {relative_path}")


def _load_json_artifact(path: Path, label: str) -> dict[str, Any]:
    raw = _read_regular(path)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"TUSI-168 {label} JSON is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"TUSI-168 {label} is not an object")
    expected = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected:
        raise RuntimeError(f"TUSI-168 {label} serialization drift")
    return payload


def _is_exact_repository_artifact_path(value: Any, relative_path: Path) -> bool:
    if not isinstance(value, str):
        return False
    return value in {
        relative_path.as_posix(),
        (REPOSITORY_ROOT / relative_path).as_posix(),
    }


def _validate_novelty_attempt_claim(
    novelty: Mapping[str, Any],
    *,
    novelty_module: Any,
) -> None:
    attempt_path = Path(novelty_module.DEFAULT_ATTEMPT_CLAIM_PATH)
    _assert_committed_clean(attempt_path)
    raw = _read_regular(attempt_path)
    try:
        claim = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("TUSI-168 novelty attempt claim is invalid") from error
    if not isinstance(claim, dict) or raw != canonical_json_bytes(claim):
        raise RuntimeError("TUSI-168 novelty attempt claim is noncanonical")
    claim_keys = {
        "protocol_version",
        "policy_id",
        "status",
        "one_shot",
        "retry_or_repair_after_failure",
        "preregistration",
        "source_support",
        "candidate_clock",
        "gross9_clock_artifact",
        "canonical_output",
        "claim_hash",
    }
    _exact_keys(claim, claim_keys, "novelty attempt claim")
    core = {key: value for key, value in claim.items() if key != "claim_hash"}
    report_binding = novelty.get("attempt_claim")
    report_support = novelty.get("source_support")
    report_candidate = novelty.get("candidate_clock")
    report_gross9 = novelty.get("gross9_clock_artifact")
    if (
        claim.get("protocol_version")
        != novelty_module.ATTEMPT_CLAIM_PROTOCOL_VERSION
        or claim.get("policy_id") != POLICY_ID
        or claim.get("status") != "claimed_before_comparator_access"
        or claim.get("one_shot") is not True
        or claim.get("retry_or_repair_after_failure") is not False
        or claim.get("preregistration") != novelty.get("preregistration")
        or not isinstance(report_binding, Mapping)
        or report_binding
        != {
            "path": attempt_path.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "claim_hash": claim.get("claim_hash"),
        }
        or claim.get("claim_hash") != canonical_hash(core)
        or not isinstance(report_support, Mapping)
        or not _is_exact_repository_artifact_path(
            report_support.get("path"),
            SOURCE_SUPPORT_ARTIFACT,
        )
        or claim.get("source_support")
        != {
            key: report_support.get(key)
            for key in ("path", "sha256", "manifest_hash")
        }
        or not isinstance(report_candidate, Mapping)
        or not _is_exact_repository_artifact_path(
            report_candidate.get("path"),
            Path(novelty_module.DEFAULT_PRIMARY_CLOCK_PATH),
        )
        or claim.get("candidate_clock")
        != {
            "path": novelty_module.DEFAULT_PRIMARY_CLOCK_PATH.as_posix(),
            "sha256": report_candidate.get("sha256"),
        }
        or not isinstance(report_gross9, Mapping)
        or not _is_exact_repository_artifact_path(
            report_gross9.get("path"),
            Path(novelty_module.DEFAULT_GROSS9_CLOCKS_PATH),
        )
        or claim.get("gross9_clock_artifact") != dict(report_gross9)
        or claim.get("canonical_output")
        != novelty_module.DEFAULT_OUTPUT_PATH.as_posix()
    ):
        raise RuntimeError("TUSI-168 novelty attempt/report binding drift")


def load_production_phase_gates() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate support first, then novelty; opens no economic rows."""

    load_bound_preregistration()
    _assert_committed_clean(SOURCE_SUPPORT_ARTIFACT)
    novelty_module = importlib.import_module(
        "training.evaluate_tron_usdt_supply_impulse_novelty"
    )
    try:
        verified_support = novelty_module.load_passed_source_support(
            SOURCE_SUPPORT_ARTIFACT,
            production=True,
        )
    except Exception as error:
        raise RuntimeError(
            "TUSI-168 committed source support did not authenticate"
        ) from error
    support = validate_passed_source_support(
        cast(Mapping[str, Any], verified_support.payload),
        exact=True,
    )
    _assert_committed_clean(NOVELTY_ARTIFACT)
    novelty = _load_json_artifact(NOVELTY_ARTIFACT, "novelty")
    try:
        canonical = novelty_module.canonical_report_bytes(novelty)
    except Exception as error:
        raise RuntimeError(
            "TUSI-168 committed novelty report did not authenticate"
        ) from error
    if canonical != _read_regular(NOVELTY_ARTIFACT):
        raise RuntimeError("TUSI-168 novelty report byte drift")
    checked = validate_passed_novelty(novelty, exact=True)
    _validate_novelty_attempt_claim(
        checked,
        novelty_module=novelty_module,
    )
    checked_support = checked.get("source_support")
    if not isinstance(checked_support, Mapping):
        raise RuntimeError("TUSI-168 novelty source-support binding is absent")
    if (
        not _is_exact_repository_artifact_path(
            checked_support.get("path"),
            SOURCE_SUPPORT_ARTIFACT,
        )
        or checked_support.get("sha256") != sha256_file(SOURCE_SUPPORT_ARTIFACT)
        or checked_support.get("manifest_hash") != support["manifest_hash"]
        or checked_support.get("passed") is not True
    ):
        raise RuntimeError("TUSI-168 novelty is not bound to exact source support")
    return support, checked


def _clock_frame(frame: pd.DataFrame, *, control: str) -> pd.DataFrame:
    if tuple(frame.columns) == CLOCK_COLUMNS:
        return cast(pd.DataFrame, validate_clock(frame, sleeve=control))
    if tuple(frame.columns) != SUPPORT_CLOCK_COLUMNS:
        raise RuntimeError(f"TUSI-168 {control} support clock schema drift")
    selected = frame.loc[frame["control"].eq(control)].copy()
    if not selected["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError(f"TUSI-168 {control} policy drift")
    sides = selected["side"].map({"LONG": 1, "SHORT": -1})
    if sides.isna().any():
        raise RuntimeError(f"TUSI-168 {control} side drift")
    economics = pd.DataFrame(
        {
            "entry_time": selected["entry_time_utc"],
            "exit_time": selected["exit_time_utc"],
            "side": sides,
        },
        columns=CLOCK_COLUMNS,
    )
    return cast(pd.DataFrame, validate_clock(economics, sleeve=control))


def _read_support_clock(path: Path, expected_sha256: str) -> pd.DataFrame:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("TUSI-168 support clock SHA-256 drift")
    raw = _read_regular(path)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            decoded = handle.read().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError("TUSI-168 support clock gzip is invalid") from error
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if tuple(reader.fieldnames or ()) != SUPPORT_CLOCK_COLUMNS:
        raise RuntimeError("TUSI-168 support clock header drift")
    rows = list(reader)
    return pd.DataFrame.from_records(rows, columns=SUPPORT_CLOCK_COLUMNS)


def load_production_clocks(
    support: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    artifacts = support.get("clock_artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("TUSI-168 support clock bindings are absent")
    _assert_committed_clean(PRIMARY_CLOCK_ARTIFACT)
    _assert_committed_clean(CONTROL_CLOCK_ARTIFACT)
    primary_rows = _read_support_clock(
        PRIMARY_CLOCK_ARTIFACT, str(artifacts.get("primary_sha256", ""))
    )
    control_rows = _read_support_clock(
        CONTROL_CLOCK_ARTIFACT, str(artifacts.get("controls_sha256", ""))
    )
    if set(primary_rows["control"]) != {"primary"}:
        raise RuntimeError("TUSI-168 primary support clock contains controls")
    if tuple(dict.fromkeys(control_rows["control"])) != CONTROL_NAMES:
        raise RuntimeError("TUSI-168 control support clock order drift")
    return (
        _clock_frame(primary_rows, control="primary"),
        {name: _clock_frame(control_rows, control=name) for name in CONTROL_NAMES},
    )


def evaluate_standalone_period(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clock: pd.DataFrame,
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("TUSI-168 direct standalone frames are synthetic-only")
    return cast(
        dict[str, Any],
        esdi._evaluate_standalone_period(
            market,
            funding,
            _clock_frame(clock, control="primary"),
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        ),
    )


def evaluate_primary_superiority(
    primary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(controls) != CONTROL_NAMES:
        raise RuntimeError("TUSI-168 controls are missing, extra, or reordered")
    checks: dict[str, bool] = {}
    for cost in ("base", "stress"):
        primary_ratio = float(primary[cost]["metrics"]["cagr_to_strict_mdd"])
        for name in INDEPENDENT_CONTROLS:
            metrics = controls[name][cost]["metrics"]
            trades = int(metrics["trades"])
            if trades == 0:
                checks[f"{cost}_{name}_zero_trade_not_gated"] = True
            else:
                ratio = float(metrics["cagr_to_strict_mdd"])
                if not math.isfinite(ratio):
                    raise RuntimeError(
                        f"TUSI-168 nonzero {name} metric is undefined"
                    )
                checks[f"{cost}_strictly_exceeds_{name}"] = primary_ratio > ratio
    for name in SAME_PARENT_CONTROLS:
        checks[f"{name}_cannot_completely_qualify"] = not bool(
            controls[name]["passes"]
        )
    return {"checks": checks, "passes": all(checks.values())}


def evaluate_standalone_period_with_controls(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    primary_clock: pd.DataFrame,
    control_clocks: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("TUSI-168 direct control frames are synthetic-only")
    if tuple(control_clocks) != CONTROL_NAMES:
        raise RuntimeError("TUSI-168 controls are missing, extra, or reordered")
    primary = evaluate_standalone_period(
        market,
        funding,
        primary_clock,
        start=start,
        end=end,
        synthetic=True,
    )
    controls = {
        name: evaluate_standalone_period(
            market,
            funding,
            control_clocks[name],
            start=start,
            end=end,
            synthetic=True,
        )
        for name in CONTROL_NAMES
    }
    superiority = evaluate_primary_superiority(primary, controls)
    return {
        "primary": primary,
        "controls": controls,
        "primary_superiority": superiority,
        "one_bar_delayed_entry_is_diagnostic_only": True,
        "passes": bool(primary["passes"] and superiority["passes"]),
    }


def _rename_candidate(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {
            (new if key == old else key): _rename_candidate(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item, old, new) for item in value]
    if value == old:
        return new
    return value


def same_gross_weights(candidate_weight: float) -> dict[str, float]:
    weights = cast(dict[str, float], esdi.same_gross_weights(candidate_weight))
    weights["tusi"] = weights.pop("esdi")
    return weights


def evaluate_same_gross_weight(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    gross9_clocks: Mapping[str, pd.DataFrame],
    tusi_clock: pd.DataFrame,
    candidate_weight: float,
    *,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("TUSI-168 direct same-gross frames are synthetic-only")
    result = esdi.evaluate_same_gross_weight(
        market,
        funding,
        gross9_clocks,
        _clock_frame(tusi_clock, control="primary"),
        candidate_weight,
        periods=periods,
        synthetic=True,
    )
    return cast(dict[str, Any], _rename_candidate(result, "esdi", "tusi"))


def _to_esdi_same_gross_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _rename_candidate(dict(row), "tusi", "esdi"))


def rank_same_gross_treatments(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = esdi.rank_same_gross_treatments(
        [_to_esdi_same_gross_row(row) for row in rows]
    )
    return cast(list[dict[str, Any]], _rename_candidate(ranked, "esdi", "tusi"))


def future_veto(
    frozen_selection: Mapping[str, Any],
    future_rows: Mapping[str, Mapping[str, Any]],
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("TUSI-168 injected future rows are synthetic-only")
    result = esdi.future_veto(
        _to_esdi_same_gross_row(frozen_selection),
        {
            name: _to_esdi_same_gross_row(row)
            for name, row in future_rows.items()
        },
        synthetic=True,
    )
    return cast(dict[str, Any], _rename_candidate(result, "esdi", "tusi"))


def _validate_synthetic_phase_gates(
    support: Mapping[str, Any],
    novelty: Mapping[str, Any],
) -> None:
    validate_passed_source_support(support, exact=False)
    checked = validate_passed_novelty(novelty, exact=False)
    binding = checked["source_support"]
    if binding.get("manifest_hash") != support["manifest_hash"]:
        raise RuntimeError("TUSI-168 novelty source-support manifest mismatch")


def _exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(payload) != expected:
        raise RuntimeError(f"TUSI-168 {label} exact schema drift")


def _exact_boolean_mapping(
    observed: Any,
    expected: Mapping[str, bool],
    label: str,
) -> None:
    if (
        not isinstance(observed, Mapping)
        or set(observed) != set(expected)
        or any(
            type(observed[key]) is not bool
            or observed[key] is not expected[key]
            for key in expected
        )
    ):
        raise RuntimeError(f"TUSI-168 {label} checks were forged")


def _validate_same_gross_cost_row(
    value: Any,
    *,
    label: str,
    includes_improvement: bool,
) -> tuple[bool, bool, float]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"TUSI-168 {label} cost row is invalid")
    expected_keys = {"treatment", "unscaled_gross9", "checks"}
    if includes_improvement:
        expected_keys.add("improvement")
    _exact_keys(value, expected_keys, f"{label} cost row")
    treatment = value["treatment"]
    baseline = value["unscaled_gross9"]
    if not isinstance(treatment, Mapping) or not isinstance(baseline, Mapping):
        raise RuntimeError(f"TUSI-168 {label} metrics are invalid")
    try:
        expected_checks = cast(
            dict[str, bool],
            esdi._same_gross_period_checks(treatment, baseline),
        )
        improvement = float(
            cast(float, treatment["cagr_to_strict_mdd"])
        ) - float(
            cast(float, baseline["cagr_to_strict_mdd"])
        )
        mdd_reduced = float(
            cast(float, treatment["strict_mdd"])
        ) < float(
            cast(float, baseline["strict_mdd"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"TUSI-168 {label} metrics did not reproduce"
        ) from error
    if not math.isfinite(improvement):
        raise RuntimeError(f"TUSI-168 {label} improvement is not finite")
    _exact_boolean_mapping(value["checks"], expected_checks, label)
    if includes_improvement and (
        type(value["improvement"]) is not float
        or not math.isclose(
            value["improvement"],
            improvement,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError(f"TUSI-168 {label} improvement was forged")
    return all(expected_checks.values()), mdd_reduced, improvement


def _validate_same_gross_ranking(
    ranking: Any,
) -> float | None:
    if not isinstance(ranking, list) or len(ranking) != len(CANDIDATE_WEIGHTS):
        raise RuntimeError("TUSI-168 same-gross ranking grid is incomplete")
    row_keys = {
        "candidate_weight",
        "treatment_weights",
        "baseline_weights",
        "periods",
        "period_order",
        "fresh_evaluation",
        "strict_mdd_reduced_in_at_least_one_period",
        "minimum_improvement",
        "passes",
        "rank",
        "frozen",
    }
    normalized: list[tuple[float, float, bool, int, bool]] = []
    for row in ranking:
        if not isinstance(row, Mapping):
            raise RuntimeError("TUSI-168 same-gross ranking row is invalid")
        _exact_keys(row, row_keys, "same-gross ranking row")
        if type(row["candidate_weight"]) is not float:
            raise RuntimeError("TUSI-168 same-gross weight type drift")
        weight = float(cast(float, row["candidate_weight"]))
        try:
            minimum, passes = esdi._derive_same_gross_summary(
                _to_esdi_same_gross_row(row)
            )
        except Exception as error:
            raise RuntimeError(
                "TUSI-168 same-gross ranking row did not reproduce"
            ) from error
        periods = cast(Mapping[str, Any], row["periods"])
        cost_summaries = [
            _validate_same_gross_cost_row(
                cast(Mapping[str, Any], periods[period])[cost],
                label=f"same-gross {period} {cost}",
                includes_improvement=True,
            )
            for period in SELECTION_PERIODS
            for cost in ("base", "stress")
        ]
        mdd_reduced = any(summary[1] for summary in cost_summaries)
        if (
            not math.isfinite(weight)
            or weight not in CANDIDATE_WEIGHTS
            or type(row["minimum_improvement"]) is not float
            or not math.isclose(
                float(cast(float, row["minimum_improvement"])),
                float(minimum),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or row["passes"] is not passes
            or row["strict_mdd_reduced_in_at_least_one_period"]
            is not mdd_reduced
            or type(row["rank"]) is not int
            or type(row["frozen"]) is not bool
        ):
            raise RuntimeError("TUSI-168 same-gross ranking row was forged")
        normalized.append(
            (
                weight,
                float(minimum),
                bool(passes),
                int(row["rank"]),
                bool(row["frozen"]),
            )
        )
    if (
        sorted(weight for weight, *_ in normalized) != list(CANDIDATE_WEIGHTS)
        or len({weight for weight, *_ in normalized}) != len(CANDIDATE_WEIGHTS)
    ):
        raise RuntimeError("TUSI-168 same-gross weights are not the exact grid")
    expected = sorted(
        normalized,
        key=lambda item: (-item[1], item[0]),
    )
    if normalized != expected:
        raise RuntimeError("TUSI-168 same-gross ranking order drift")
    for expected_rank, row in enumerate(normalized, start=1):
        _, _, passes, rank, frozen = row
        if (
            rank != expected_rank
            or frozen is not (expected_rank == 1 and passes)
        ):
            raise RuntimeError("TUSI-168 same-gross rank/freeze drift")
    return normalized[0][0] if normalized[0][2] else None


def _validate_standalone_period_result(value: Any, label: str) -> bool:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"TUSI-168 {label} period result is invalid")
    _exact_keys(value, {"base", "stress", "passes"}, f"{label} period result")
    cost_passes: list[bool] = []
    for cost in ("base", "stress"):
        row = value[cost]
        if not isinstance(row, Mapping):
            raise RuntimeError(f"TUSI-168 {label} {cost} result is invalid")
        _exact_keys(
            row,
            {"metrics", "checks", "passes"},
            f"{label} {cost} result",
        )
        metrics = row["metrics"]
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"TUSI-168 {label} {cost} metrics are invalid")
        try:
            expected_checks = cast(
                dict[str, bool],
                standalone_gate_checks(metrics),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"TUSI-168 {label} {cost} metrics did not reproduce"
            ) from error
        _exact_boolean_mapping(
            row["checks"],
            expected_checks,
            f"{label} {cost}",
        )
        expected_pass = all(expected_checks.values())
        if type(row["passes"]) is not bool or row["passes"] is not expected_pass:
            raise RuntimeError(f"TUSI-168 {label} {cost} pass was forged")
        cost_passes.append(expected_pass)
    expected_period_pass = all(cost_passes)
    if (
        type(value["passes"]) is not bool
        or value["passes"] is not expected_period_pass
    ):
        raise RuntimeError(f"TUSI-168 {label} period pass was forged")
    return expected_period_pass


def _validate_standalone_summary(value: Any, label: str) -> bool:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"TUSI-168 {label} standalone summary is invalid")
    _exact_keys(
        value,
        {
            "primary",
            "controls",
            "primary_superiority",
            "one_bar_delayed_entry_is_diagnostic_only",
            "passes",
        },
        f"{label} standalone summary",
    )
    controls = value["controls"]
    if (
        not isinstance(value["primary"], Mapping)
        or not isinstance(controls, Mapping)
        or set(controls) != set(CONTROL_NAMES)
        or not isinstance(value["primary_superiority"], Mapping)
        or value["one_bar_delayed_entry_is_diagnostic_only"] is not True
        or type(value["passes"]) is not bool
    ):
        raise RuntimeError(f"TUSI-168 {label} standalone summary drift")
    primary_passed = _validate_standalone_period_result(
        value["primary"],
        f"{label} primary",
    )
    ordered_controls = {
        name: cast(Mapping[str, Any], controls[name])
        for name in CONTROL_NAMES
    }
    for name, control in ordered_controls.items():
        _validate_standalone_period_result(
            control,
            f"{label} {name}",
        )
    superiority = value["primary_superiority"]
    _exact_keys(
        cast(Mapping[str, Any], superiority),
        {"checks", "passes"},
        f"{label} primary superiority",
    )
    try:
        expected_superiority = evaluate_primary_superiority(
            cast(Mapping[str, Any], value["primary"]),
            ordered_controls,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"TUSI-168 {label} superiority did not reproduce"
        ) from error
    _exact_boolean_mapping(
        cast(Mapping[str, Any], superiority)["checks"],
        cast(Mapping[str, bool], expected_superiority["checks"]),
        f"{label} primary superiority",
    )
    expected_superiority_pass = bool(expected_superiority["passes"])
    if (
        type(cast(Mapping[str, Any], superiority)["passes"]) is not bool
        or cast(Mapping[str, Any], superiority)["passes"]
        is not expected_superiority_pass
    ):
        raise RuntimeError(f"TUSI-168 {label} superiority pass was forged")
    expected_pass = primary_passed and expected_superiority_pass
    if value["passes"] is not expected_pass:
        raise RuntimeError(f"TUSI-168 {label} standalone pass was forged")
    return expected_pass


def _validate_stage_result(
    stage: str,
    result: Mapping[str, Any],
    *,
    expected_frozen_weight: float | None,
) -> float | None:
    if type(result.get("passed")) is not bool:
        raise RuntimeError(f"TUSI-168 {stage} result pass flag is invalid")
    if stage in {"2023H2", "2024", "selection"}:
        _exact_keys(result, {"passed", "standalone"}, f"{stage} result")
        standalone_passed = _validate_standalone_summary(
            result["standalone"],
            stage,
        )
        if result["passed"] is not standalone_passed:
            raise RuntimeError(f"TUSI-168 {stage} standalone result drift")
        return None
    if stage == "same_gross":
        _exact_keys(result, {"passed", "ranking"}, "same-gross result")
        frozen = _validate_same_gross_ranking(result["ranking"])
        if result["passed"] is not (frozen is not None):
            raise RuntimeError("TUSI-168 same-gross pass/freeze drift")
        return frozen
    if stage in {"future25", "future26", "full"}:
        _exact_keys(
            result,
            {
                "passed",
                "frozen_weight",
                "reranked",
                "standalone",
                "same_gross",
            },
            f"{stage} result",
        )
        if expected_frozen_weight not in CANDIDATE_WEIGHTS:
            raise RuntimeError(f"TUSI-168 {stage} lacks prior frozen weight")
        standalone = result["standalone"]
        same_gross = result["same_gross"]
        standalone_passed = _validate_standalone_summary(standalone, stage)
        if (
            type(result["frozen_weight"]) is not float
            or not isinstance(same_gross, Mapping)
        ):
            raise RuntimeError(
                f"TUSI-168 {stage} changed weight, reranked, or forged pass"
            )
        _exact_keys(
            same_gross,
            {
                "candidate_weight",
                "costs",
                "strict_mdd_reduced",
                "passes",
            },
            f"{stage} same-gross result",
        )
        costs = same_gross["costs"]
        if (
            type(same_gross["candidate_weight"]) is not float
            or type(same_gross["passes"]) is not bool
            or type(same_gross["strict_mdd_reduced"]) is not bool
            or not isinstance(costs, Mapping)
            or tuple(costs) != ("base", "stress")
        ):
            raise RuntimeError(
                f"TUSI-168 {stage} changed weight, reranked, or forged pass"
            )
        cost_results = [
            _validate_same_gross_cost_row(
                costs[cost],
                label=f"{stage} {cost}",
                includes_improvement=False,
            )
            for cost in ("base", "stress")
        ]
        derived_same_gross_pass = bool(
            all(summary[0] for summary in cost_results)
            and any(summary[1] for summary in cost_results)
        )
        if (
            same_gross["strict_mdd_reduced"]
            is not any(summary[1] for summary in cost_results)
            or same_gross["passes"] is not derived_same_gross_pass
            or float(cast(float, result["frozen_weight"]))
            != expected_frozen_weight
            or float(cast(float, same_gross.get("candidate_weight", -1.0)))
            != expected_frozen_weight
            or result["reranked"] is not False
            or result["passed"]
            is not bool(standalone_passed and same_gross["passes"])
        ):
            raise RuntimeError(
                f"TUSI-168 {stage} changed weight, reranked, or forged pass"
            )
        return expected_frozen_weight
    raise RuntimeError(f"TUSI-168 unknown economics stage: {stage}")


def _receipt(
    stage: str,
    result: Mapping[str, Any],
    *,
    attempt_binding: Mapping[str, Any],
    prior_sha256: str | None,
    novelty_manifest_hash: str,
    expected_frozen_weight: float | None,
    synthetic: bool,
) -> dict[str, Any]:
    authenticated_attempt = _validate_attempt_binding(
        attempt_binding,
        stage=stage,
        novelty_manifest_hash=novelty_manifest_hash,
        prior_sha256=prior_sha256,
        frozen_weight=expected_frozen_weight,
    )
    frozen = _validate_stage_result(
        stage,
        result,
        expected_frozen_weight=expected_frozen_weight,
    )
    core: dict[str, Any] = {
        "protocol_version": ECONOMIC_RECEIPT_PROTOCOL,
        "policy_id": POLICY_ID,
        "execution_mode": "synthetic_only" if synthetic else "production",
        "attempt_claim": authenticated_attempt,
        "stage": stage,
        "cutoff_exclusive": cast(pd.Timestamp, STAGE_CUTOFFS[stage]).isoformat(),
        "passed": bool(result["passed"]),
        "novelty_manifest_hash": novelty_manifest_hash,
        "prior_receipt_sha256": prior_sha256,
        "result": _json_ready(result),
    }
    if stage == "same_gross":
        core["frozen_weight"] = frozen
    elif stage in {"future25", "future26", "full"}:
        core["frozen_weight"] = expected_frozen_weight
        core["reranked"] = False
    return {**core, "manifest_hash": canonical_hash(core)}


def _attempt_claim(
    stage: str,
    *,
    novelty_manifest_hash: str,
    prior_sha256: str | None,
    frozen_weight: float | None,
) -> dict[str, Any]:
    if stage not in ECONOMIC_STAGE_ORDER:
        raise RuntimeError(f"TUSI-168 unknown economics stage: {stage}")
    if not _is_sha256(novelty_manifest_hash):
        raise RuntimeError(f"TUSI-168 {stage} novelty hash is invalid")
    if (stage == ECONOMIC_STAGE_ORDER[0]) is not (prior_sha256 is None):
        raise RuntimeError(f"TUSI-168 {stage} prior receipt hash is invalid")
    if prior_sha256 is not None and not _is_sha256(prior_sha256):
        raise RuntimeError(f"TUSI-168 {stage} prior receipt hash is invalid")
    core: dict[str, Any] = {
        "protocol_version": ECONOMIC_ATTEMPT_PROTOCOL,
        "policy_id": POLICY_ID,
        "status": "claimed_before_stage_rows",
        "stage": stage,
        "cutoff_exclusive": cast(
            pd.Timestamp, STAGE_CUTOFFS[stage]
        ).isoformat(),
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "novelty_manifest_hash": novelty_manifest_hash,
        "prior_receipt_sha256": prior_sha256,
        "canonical_receipt": STAGE_RECEIPT_NAMES[stage],
    }
    if stage in {"future25", "future26", "full"}:
        if (
            type(frozen_weight) is not float
            or frozen_weight not in CANDIDATE_WEIGHTS
        ):
            raise RuntimeError(f"TUSI-168 {stage} attempt lacks frozen weight")
        core["frozen_weight"] = frozen_weight
    elif frozen_weight is not None:
        raise RuntimeError(f"TUSI-168 {stage} attempt opened a future weight")
    return {**core, "manifest_hash": canonical_hash(core)}


def _validate_attempt_binding(
    binding: Mapping[str, Any],
    *,
    stage: str,
    novelty_manifest_hash: str,
    prior_sha256: str | None,
    frozen_weight: float | None,
) -> dict[str, Any]:
    _exact_keys(
        binding,
        {"path", "sha256", "manifest_hash", "content"},
        f"{stage} attempt binding",
    )
    expected_content = _attempt_claim(
        stage,
        novelty_manifest_hash=novelty_manifest_hash,
        prior_sha256=prior_sha256,
        frozen_weight=frozen_weight,
    )
    expected_raw = canonical_json_bytes(expected_content)
    content = binding["content"]
    try:
        content_raw = (
            canonical_json_bytes(content)
            if isinstance(content, Mapping)
            else b""
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"TUSI-168 {stage} attempt binding content is invalid"
        ) from error
    expected = {
        "path": STAGE_ATTEMPT_NAMES[stage],
        "sha256": hashlib.sha256(expected_raw).hexdigest(),
        "manifest_hash": expected_content["manifest_hash"],
        "content": expected_content,
    }
    if content_raw != expected_raw or dict(binding) != expected:
        raise RuntimeError(f"TUSI-168 {stage} attempt binding drift")
    return expected


def _load_attempt_claim(
    path: str | Path,
    *,
    root: str | Path,
    stage: str,
    novelty_manifest_hash: str,
    prior_sha256: str | None,
    frozen_weight: float | None,
    production: bool,
) -> dict[str, Any]:
    raw = _read_output_bytes(path, root=root, production=production)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"TUSI-168 {stage} attempt claim is invalid") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"TUSI-168 {stage} attempt bytes are noncanonical")
    expected = _attempt_claim(
        stage,
        novelty_manifest_hash=novelty_manifest_hash,
        prior_sha256=prior_sha256,
        frozen_weight=frozen_weight,
    )
    if raw != canonical_json_bytes(expected):
        raise RuntimeError(f"TUSI-168 {stage} attempt content drift")
    binding = {
        "path": STAGE_ATTEMPT_NAMES[stage],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_hash": payload["manifest_hash"],
        "content": payload,
    }
    return _validate_attempt_binding(
        binding,
        stage=stage,
        novelty_manifest_hash=novelty_manifest_hash,
        prior_sha256=prior_sha256,
        frozen_weight=frozen_weight,
    )


def _load_completed_receipt(
    path: str | Path,
    *,
    root: str | Path,
    stage: str,
    attempt_binding: Mapping[str, Any],
    novelty_manifest_hash: str,
    prior_sha256: str | None,
    expected_frozen_weight: float | None,
    synthetic: bool,
) -> _LoadedReceipt:
    authenticated_attempt = _validate_attempt_binding(
        attempt_binding,
        stage=stage,
        novelty_manifest_hash=novelty_manifest_hash,
        prior_sha256=prior_sha256,
        frozen_weight=expected_frozen_weight,
    )
    raw = _read_output_bytes(
        path,
        root=root,
        production=not synthetic,
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"TUSI-168 {stage} receipt is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"TUSI-168 {stage} receipt is not an object")
    _validate_manifest(payload, f"{stage} receipt")
    expected_keys = {
        "protocol_version",
        "policy_id",
        "execution_mode",
        "attempt_claim",
        "stage",
        "cutoff_exclusive",
        "passed",
        "novelty_manifest_hash",
        "prior_receipt_sha256",
        "result",
        "manifest_hash",
    }
    if stage == "same_gross":
        expected_keys.add("frozen_weight")
    elif stage in {"future25", "future26", "full"}:
        expected_keys.update(("frozen_weight", "reranked"))
    _exact_keys(payload, expected_keys, f"{stage} receipt")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError(f"TUSI-168 {stage} receipt result is invalid")
    frozen = _validate_stage_result(
        stage,
        result,
        expected_frozen_weight=expected_frozen_weight,
    )
    if (
        raw != canonical_json_bytes(payload)
        or payload.get("protocol_version") != ECONOMIC_RECEIPT_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("attempt_claim") != authenticated_attempt
        or payload.get("stage") != stage
        or payload.get("cutoff_exclusive")
        != cast(pd.Timestamp, STAGE_CUTOFFS[stage]).isoformat()
        or payload.get("execution_mode")
        != ("synthetic_only" if synthetic else "production")
        or payload.get("novelty_manifest_hash") != novelty_manifest_hash
        or payload.get("prior_receipt_sha256") != prior_sha256
        or payload.get("passed") is not result["passed"]
        or (
            stage == "same_gross"
            and payload.get("frozen_weight") != frozen
        )
        or (
            stage in {"future25", "future26", "full"}
            and (
                payload.get("frozen_weight") != expected_frozen_weight
                or payload.get("reranked") is not False
            )
        )
    ):
        raise RuntimeError(f"TUSI-168 {stage} receipt binding drift")
    return _LoadedReceipt(
        payload=payload,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _gross9_frames(verified: Any) -> dict[str, pd.DataFrame]:
    clocks = getattr(verified, "clocks", None)
    if not isinstance(clocks, Mapping) or tuple(clocks) != GROSS9_SLEEVES:
        raise RuntimeError("TUSI-168 Gross9 verified clocks are incomplete")
    output: dict[str, pd.DataFrame] = {}
    for sleeve in GROSS9_SLEEVES:
        intervals = cast(Sequence[Any], clocks[sleeve])
        rows = [
            {
                "entry_time": pd.Timestamp(int(row.entry), unit="s", tz="UTC"),
                "exit_time": pd.Timestamp(int(row.exit), unit="s", tz="UTC"),
                "side": int(row.side),
            }
            for row in intervals
        ]
        frame = pd.DataFrame.from_records(rows, columns=CLOCK_COLUMNS)
        output[sleeve] = cast(
            pd.DataFrame, validate_clock(frame, sleeve=sleeve)
        )
    return output


def _production_prerequisites(
    support: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], str, str]:
    _assert_committed_clean(EVALUATOR_SOURCE_PATH)
    _assert_committed_clean(EVALUATOR_TEST_PATH)
    _assert_committed_clean(ESDI_ECONOMICS_AUTHORITY_PATH)
    registration = load_bound_preregistration()
    esdi_registration = esdi.load_bound_preregistration()
    esdi.validate_frozen_contract(esdi_registration)
    novelty_module = importlib.import_module(
        "training.evaluate_tron_usdt_supply_impulse_novelty"
    )
    verified_support = novelty_module.load_passed_source_support(
        SOURCE_SUPPORT_ARTIFACT,
        production=True,
    )
    verified_gross9 = novelty_module.load_gross9_clock_artifact(
        registration=registration,
        source_support=verified_support,
        production=True,
    )
    gross9 = _gross9_frames(verified_gross9)
    market_path, funding_path = esdi._source_paths_from_authority(
        esdi_registration
    )
    if support["manifest_hash"] != verified_support.manifest_hash:
        raise RuntimeError("TUSI-168 production source-support binding drift")
    return gross9, str(market_path), str(funding_path)


def _production_stage_evaluator(
    stage: str,
    inputs: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    market = cast(pd.DataFrame, inputs["market"])
    funding = cast(pd.DataFrame, inputs["funding"])
    primary = cast(pd.DataFrame, inputs["primary_clock"])
    controls = cast(Mapping[str, pd.DataFrame], inputs["control_clocks"])
    gross9 = cast(Mapping[str, pd.DataFrame], inputs["gross9_clocks"])
    if stage in {"2023H2", "2024", "selection"}:
        raw_start, raw_end = PERIODS[stage]
        start = cast(pd.Timestamp, raw_start)
        end = cast(pd.Timestamp, raw_end)
        standalone = evaluate_standalone_period_with_controls(
            market,
            funding,
            primary,
            controls,
            start=start,
            end=end,
            synthetic=True,
        )
        return {"passed": bool(standalone["passes"]), "standalone": standalone}
    if stage == "same_gross":
        rows = [
            evaluate_same_gross_weight(
                market,
                funding,
                gross9,
                primary,
                weight,
                periods=cast(
                    Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
                    SELECTION_PERIODS,
                ),
                synthetic=True,
            )
            for weight in CANDIDATE_WEIGHTS
        ]
        ranking = cast(
            list[dict[str, Any]],
            _rename_candidate(
                esdi._rank_same_gross_treatments(
                    [_to_esdi_same_gross_row(row) for row in rows],
                    require_passing_freeze=False,
                ),
                "esdi",
                "tusi",
            ),
        )
        frozen_weight = _validate_same_gross_ranking(ranking)
        return {
            "passed": frozen_weight is not None,
            "ranking": ranking,
        }
    if stage in {"future25", "future26", "full"}:
        weight = float(state.get("frozen_weight", -1.0))
        if weight not in CANDIDATE_WEIGHTS:
            raise RuntimeError("TUSI-168 future stage lacks frozen weight")
        raw_start, raw_end = PERIODS[stage]
        start = cast(pd.Timestamp, raw_start)
        end = cast(pd.Timestamp, raw_end)
        standalone = evaluate_standalone_period_with_controls(
            market,
            funding,
            primary,
            controls,
            start=start,
            end=end,
            synthetic=True,
        )
        same_gross = esdi._evaluate_same_gross_future_period(
            market,
            funding,
            gross9,
            primary,
            weight,
            start=start,
            end=end,
        )
        same_gross = _rename_candidate(same_gross, "esdi", "tusi")
        return {
            "passed": bool(
                standalone["passes"] and same_gross["passes"]
            ),
            "frozen_weight": weight,
            "reranked": False,
            "standalone": standalone,
            "same_gross": same_gross,
        }
    raise RuntimeError(f"TUSI-168 unknown economics stage: {stage}")


def run_staged_economics(
    *,
    synthetic: bool = False,
    source_support: Mapping[str, Any] | None = None,
    novelty: Mapping[str, Any] | None = None,
    stage_loader: Callable[[str, pd.Timestamp], Mapping[str, Any]] | None = None,
    stage_evaluator: Callable[
        [str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ]
    | None = None,
    receipt_root: str | Path = REPOSITORY_ROOT / "results",
) -> dict[str, Any]:
    injected = any(
        value is not None
        for value in (source_support, novelty, stage_loader, stage_evaluator)
    )
    if injected and not synthetic:
        raise RuntimeError("TUSI-168 injected phase inputs are synthetic-only")
    root = _canonical_absolute_path(receipt_root, label="receipt root")
    if synthetic:
        if (
            source_support is None
            or novelty is None
            or stage_loader is None
            or stage_evaluator is None
        ):
            raise RuntimeError(
                "TUSI-168 synthetic staged runner requires all inputs"
            )
        if root == CANONICAL_RESULTS_ROOT or root.is_relative_to(
            CANONICAL_RESULTS_ROOT
        ):
            raise RuntimeError("TUSI-168 synthetic receipts cannot use results/")
        _validate_synthetic_phase_gates(source_support, novelty)
        active_support = source_support
        active_novelty = novelty
        loader = stage_loader
        evaluator = stage_evaluator
    else:
        if root != CANONICAL_RESULTS_ROOT:
            raise RuntimeError("TUSI-168 production receipts use results/")
        active_support, active_novelty = load_production_phase_gates()
        gross9, market_path, funding_path = _production_prerequisites(
            active_support
        )
        primary, controls = load_production_clocks(active_support)

        def production_loader(
            stage: str, cutoff: pd.Timestamp
        ) -> Mapping[str, Any]:
            if cutoff != STAGE_CUTOFFS[stage]:
                raise RuntimeError("TUSI-168 stage cutoff drift")
            return {
                "market": esdi._load_market_prefix(market_path, cutoff),
                "funding": esdi._load_funding_prefix(funding_path, cutoff),
                "primary_clock": primary,
                "control_clocks": controls,
                "gross9_clocks": gross9,
            }

        loader = production_loader
        evaluator = _production_stage_evaluator

    state: dict[str, Any] = {}
    completed: list[str] = []
    prior_sha: str | None = None
    for stage in ECONOMIC_STAGE_ORDER:
        receipt_path = STAGE_RECEIPT_NAMES[stage]
        attempt_path = STAGE_ATTEMPT_NAMES[stage]
        production_output = not synthetic
        receipt_exists = _output_leaf_exists(
            receipt_path,
            root=root,
            production=production_output,
        )
        attempt_exists = _output_leaf_exists(
            attempt_path,
            root=root,
            production=production_output,
        )
        expected_frozen = cast(float | None, state.get("frozen_weight"))
        if receipt_exists:
            if not attempt_exists:
                raise RuntimeError(
                    f"TUSI-168 {stage} completion lacks attempt claim"
                )
            attempt_binding = _load_attempt_claim(
                attempt_path,
                root=root,
                stage=stage,
                novelty_manifest_hash=str(active_novelty["manifest_hash"]),
                prior_sha256=prior_sha,
                frozen_weight=expected_frozen,
                production=production_output,
            )
            loaded_receipt = _load_completed_receipt(
                receipt_path,
                root=root,
                stage=stage,
                attempt_binding=attempt_binding,
                novelty_manifest_hash=str(active_novelty["manifest_hash"]),
                prior_sha256=prior_sha,
                expected_frozen_weight=expected_frozen,
                synthetic=synthetic,
            )
            receipt = loaded_receipt.payload
            result = cast(Mapping[str, Any], receipt["result"])
            if stage == "same_gross" and receipt.get("passed") is True:
                frozen = _validate_stage_result(
                    stage,
                    result,
                    expected_frozen_weight=None,
                )
                if frozen is None:
                    raise RuntimeError(
                        "TUSI-168 resumed freeze lacks passed rank one"
                    )
                state["frozen_weight"] = frozen
            prior_sha = loaded_receipt.sha256
            completed.append(stage)
            if receipt.get("passed") is not True:
                return {
                    "passed": False,
                    "terminal": True,
                    "stopped_at": stage,
                    "completed_stages": completed,
                    **state,
                }
            continue
        if attempt_exists:
            raise RuntimeError(
                f"TUSI-168 {stage} was claimed without completion; "
                "retry or repair is forbidden"
            )
        attempt_publication = write_once_result(
            attempt_path,
            _attempt_claim(
                stage,
                novelty_manifest_hash=str(
                    active_novelty["manifest_hash"]
                ),
                prior_sha256=prior_sha,
                frozen_weight=expected_frozen,
            ),
            root=root,
            production=production_output,
        )
        attempt_binding = _load_attempt_claim(
            attempt_path,
            root=root,
            stage=stage,
            novelty_manifest_hash=str(active_novelty["manifest_hash"]),
            prior_sha256=prior_sha,
            frozen_weight=expected_frozen,
            production=production_output,
        )
        if attempt_binding["sha256"] != attempt_publication.sha256:
            raise RuntimeError(
                f"TUSI-168 {stage} attempt changed after publication"
            )
        inputs = loader(stage, cast(pd.Timestamp, STAGE_CUTOFFS[stage]))
        result = dict(evaluator(stage, inputs, dict(state)))
        frozen = _validate_stage_result(
            stage,
            result,
            expected_frozen_weight=expected_frozen,
        )
        if stage == "same_gross":
            if frozen is not None:
                state["frozen_weight"] = frozen
        receipt = _receipt(
            stage,
            result,
            attempt_binding=attempt_binding,
            prior_sha256=prior_sha,
            novelty_manifest_hash=str(active_novelty["manifest_hash"]),
            expected_frozen_weight=expected_frozen,
            synthetic=synthetic,
        )
        receipt_publication = write_once_result(
            receipt_path,
            receipt,
            root=root,
            production=production_output,
        )
        prior_sha = receipt_publication.sha256
        completed.append(stage)
        if result.get("passed") is not True:
            return {
                "passed": False,
                "terminal": True,
                "stopped_at": stage,
                "completed_stages": completed,
                **state,
            }
    return {
        "passed": True,
        "terminal": False,
        "stopped_at": None,
        "completed_stages": completed,
        **state,
    }


def _open_existing_leaf(parent_fd: int, leaf: str) -> bytes:
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise RuntimeError("TUSI-168 result leaf is unsafe") from error
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError("TUSI-168 result leaf is not a regular file")
        return _read_fd_all(descriptor)
    finally:
        os.close(descriptor)


def _publish_result(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path,
    production: bool,
) -> _Publication:
    raw = canonical_json_bytes(payload)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    parent_fd, leaf, _ = _open_output_parent(
        path,
        root=root,
        production=production,
        create=True,
    )
    temporary = f".{leaf}.{secrets.token_hex(16)}.tmp"
    temporary_created = False
    try:
        try:
            existing = _open_existing_leaf(parent_fd, leaf)
        except RuntimeError as error:
            try:
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise error
        else:
            if existing != raw:
                raise RuntimeError("TUSI-168 write-once result already differs")
            return _Publication("verified_existing", raw_sha256)

        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW,
                0o444,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise RuntimeError("TUSI-168 temporary output creation failed") from error
        temporary_created = True
        try:
            _write_fd_all(descriptor, raw)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            temporary_stat = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            winner = _open_existing_leaf(parent_fd, leaf)
            if winner != raw:
                raise RuntimeError("TUSI-168 atomic publication raced")
            return _Publication("verified_existing", raw_sha256)
        linked = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            linked.st_dev != temporary_stat.st_dev
            or linked.st_ino != temporary_stat.st_ino
            or not stat.S_ISREG(linked.st_mode)
        ):
            raise RuntimeError("TUSI-168 published hardlink identity drift")
        os.fsync(parent_fd)
        return _Publication("created", raw_sha256)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def write_once_result(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path,
    production: bool = False,
) -> _Publication:
    """Create deterministic bytes atomically under an explicit safe root."""

    return _publish_result(
        path,
        payload,
        root=root,
        production=production,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("economics",))
    parser.parse_args(argv)
    result = run_staged_economics()
    print(json.dumps(_json_ready(result), sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "BASELINE_GROSS",
    "BASE_COST_RATE",
    "CANDIDATE_WEIGHTS",
    "CLOCK_COLUMNS",
    "CONTROL_NAMES",
    "ECONOMIC_STAGE_ORDER",
    "ESDI_ECONOMICS_AUTHORITY_GIT_BLOB",
    "ESDI_ECONOMICS_AUTHORITY_SHA256",
    "FUNDING_COLUMNS",
    "GROSS9_SLEEVES",
    "GROSS9_WEIGHTS",
    "INDEPENDENT_CONTROLS",
    "LEVERAGE",
    "MARKET_COLUMNS",
    "PERIODS",
    "POLICY_ID",
    "PREREGISTRATION_ARTIFACT_SHA256",
    "PREREGISTRATION_MANIFEST_HASH",
    "PROTOCOL_VERSION",
    "SAME_PARENT_CONTROLS",
    "SELECTION_PERIODS",
    "SOURCE_SUPPORT_ARTIFACT",
    "STAGE_CUTOFFS",
    "STAGE_RECEIPT_NAMES",
    "STRESS_COST_RATE",
    "calendar_month_clustered_signflip",
    "canonical_hash",
    "canonical_json_bytes",
    "evaluate_primary_superiority",
    "evaluate_same_gross_weight",
    "evaluate_standalone_period",
    "evaluate_standalone_period_with_controls",
    "future_veto",
    "load_bound_preregistration",
    "load_production_phase_gates",
    "rank_same_gross_treatments",
    "run_staged_economics",
    "same_gross_weights",
    "sha256_file",
    "simulate_portfolio",
    "standalone_gate_checks",
    "validate_clock",
    "validate_frozen_contract",
    "validate_passed_novelty",
    "validate_passed_source_support",
    "write_once_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
