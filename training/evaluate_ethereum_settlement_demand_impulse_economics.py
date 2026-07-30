"""Strict, hash-bound ESDI-288 economics and Gross9 evaluation.

Production entry points validate the complete frozen trust chain and use only
canonical, staged loaders.  Direct frames, callbacks, and reconstructed clocks
are accepted solely through explicit ``synthetic=True`` test interfaces.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from training import build_ethereum_settlement_demand_impulse_source as source_builder
from training import preregister_ethereum_settlement_demand_impulse as prereg


POLICY_ID = "ESDI-288"
PROTOCOL_VERSION = "ethereum_settlement_demand_impulse_economics_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_economics.py"
)
EVALUATOR_TEST_PATH = Path(
    "tests/test_evaluate_ethereum_settlement_demand_impulse_economics.py"
)
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)
SOURCE_SUPPORT_ARTIFACT = Path(
    "results/ethereum_settlement_demand_impulse_source_support_2026-07-30.json"
)
NOVELTY_ARTIFACT = Path(
    "results/ethereum_settlement_demand_impulse_novelty_2026-07-30.json"
)
NOVELTY_PROTOCOL_VERSION = "ethereum_settlement_demand_impulse_novelty_v1"
GROSS9_CLOCK_PROTOCOL = "ethereum_settlement_demand_impulse_gross9_clocks_v1"
GROSS9_CLOCK_ARTIFACT = Path(
    "results/ethereum_settlement_demand_impulse_gross9_clocks_2026-07-30.json"
)
GROSS9_ATTEMPT_CLAIM = Path(
    "results/"
    "ethereum_settlement_demand_impulse_gross9_attempt_claim_2026-07-30.json"
)
GROSS9_ATTEMPT_CLAIM_PROTOCOL = (
    "ethereum_settlement_demand_impulse_gross9_attempt_claim_v1"
)
GROSS9_CLOCK_EVIDENCE_BOUNDARY = {
    "gross9_runtime_market_and_feature_rows_opened": True,
    "gross9_outcome_dependent_path_rows_opened": True,
    "gross9_full_domain_clock_paths_reproduced": True,
    "full_domain_paths_used_for_preregistered_structural_novelty_veto": True,
    "future_rows_used_for_economic_weight_ranking": False,
    "future_rows_used_for_structural_candidate_veto": True,
    "gross9_portfolio_return_or_pnl_metrics_computed": False,
}
NOVELTY_EVIDENCE_BOUNDARY = {
    "candidate_market_rows_opened": False,
    "candidate_outcome_rows_opened": False,
    "gross9_clock_artifact_bytes_opened": True,
    "gross9_outcome_dependent_clock_paths_certified": True,
    "gross9_full_domain_paths_used_for_preregistered_structural_novelty_veto": True,
    "future_rows_used_for_economic_weight_ranking": False,
    "future_rows_used_for_structural_candidate_veto": True,
    "portfolio_return_or_pnl_metrics_computed": False,
}
ECONOMIC_RECEIPT_PROTOCOL = (
    "ethereum_settlement_demand_impulse_economic_stage_receipt_v1"
)

LEVERAGE = 0.5
BASE_COST_RATE = 6.0 / 10_000.0
STRESS_COST_RATE = 10.0 / 10_000.0
BASELINE_GROSS = 9.0
GROSS9_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}
GROSS9_SLEEVES = tuple(GROSS9_WEIGHTS)
CANDIDATE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
CONTROL_NAMES = (
    "base_fee_one_epoch_stale",
    "gas_utilization_only",
    "base_fee_no_tail",
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
    "one_bar_delayed_entry",
)
SIGNFLIP_SEED = 20_260_730
SIGNFLIP_SAMPLES = 10_000
FIVE_MINUTES = pd.Timedelta(minutes=5)

MARKET_COLUMNS = ("timestamp", "open", "high", "low")
FUNDING_COLUMNS = ("funding_time", "funding_rate", "settlement_mark")
CLOCK_COLUMNS = ("entry_time", "exit_time", "side")
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
    stage: (
        "ethereum_settlement_demand_impulse_economics_"
        f"{stage.lower()}_2026-07-30.json"
    )
    for stage in ECONOMIC_STAGE_ORDER
}
ECONOMIC_STAGE_ATTEMPT_CLAIM_PROTOCOL = (
    "ethereum_settlement_demand_impulse_economic_stage_attempt_claim_v1"
)
STAGE_ATTEMPT_CLAIM_NAMES = {
    stage: (
        "ethereum_settlement_demand_impulse_economics_"
        f"{stage.lower()}_attempt_claim_2026-07-30.json"
    )
    for stage in ECONOMIC_STAGE_ORDER
}


@dataclass(frozen=True)
class ValidationContext:
    """Injectable observations used by synthetic tests and offline validators."""

    runtime_environment: Mapping[str, Any] | None = None
    file_hashes: Mapping[str, str] | None = None
    discovered_runtime_closure: Sequence[str] | None = None
    source_manifest: Mapping[str, Any] | None = None
    source_hashes: Mapping[str, str] | None = None
    rank7_bundle_manifest: Mapping[str, Any] | None = None
    rank7_bundle_hashes: Mapping[str, str] | None = None
    repository_sha256: Mapping[str, str] | None = None
    repository_git_blobs: Mapping[str, str] | None = None


@dataclass
class _Position:
    sleeve: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    quantity: float
    allocated_equity: float
    entry_cost: float
    funding_cash: float = 0.0


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"ESDI hash target is not a regular file: {path}")
    with candidate.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(_json_ready(payload))


def _claim_binding(
    path: Path,
    raw: bytes,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "claim_hash": str(payload["claim_hash"]),
    }


def _create_exact_attempt_claim(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    target = _repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuntimeError(f"ESDI {label} attempt-claim parent is unsafe")
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"ESDI {label} attempt is already claimed")
    raw = canonical_json_bytes(payload)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"ESDI {label} attempt-claim write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _claim_binding(path, raw, payload)


def _load_exact_attempt_claim(
    path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    target = _repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"ESDI {label} attempt claim is invalid")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ESDI {label} attempt claim is invalid") from error
    if (
        not isinstance(payload, Mapping)
        or dict(payload) != dict(expected)
        or raw != canonical_json_bytes(payload)
    ):
        raise RuntimeError(f"ESDI {label} attempt claim drift")
    return _claim_binding(path, raw, payload)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ESDI result JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def load_bound_preregistration(
    path: str | Path = PREREGISTRATION_ARTIFACT,
) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("ESDI preregistration artifact is missing or symlinked")
    target = candidate.resolve()
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("ESDI preregistration artifact SHA drift")
    payload = json.loads(raw)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ESDI preregistration manifest drift")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("ESDI preregistration identity drift")
    if any(
        payload.get(name) is not False
        for name in (
            "btc_market_rows_opened",
            "funding_rows_opened",
            "gross9_rows_opened",
            "outcomes_opened",
        )
    ):
        raise RuntimeError("ESDI preregistration is not outcome blind")
    return payload


def _declared_dependency_hashes(authority: Mapping[str, Any]) -> dict[str, str]:
    records: dict[str, str] = {}

    def add(binding: Mapping[str, Any], label: str) -> None:
        path = str(binding.get("path", ""))
        digest = str(binding.get("sha256", ""))
        if not path or not _is_sha256(digest):
            raise RuntimeError(f"ESDI malformed Gross9 binding: {label}")
        if path in records and records[path] != digest:
            raise RuntimeError(f"ESDI conflicting Gross9 binding: {path}")
        records[path] = digest

    add(authority["portfolio"], "portfolio")
    add(authority["base_portfolio"], "base_portfolio")
    add(authority["transitive_source_manifest"], "transitive_source_manifest")
    add(authority["pre2025_anchor"], "pre2025_anchor")
    for name, binding in authority["runtime"].items():
        add(binding, f"runtime.{name}")
    for name, sleeve in authority["sleeves"].items():
        add(sleeve["config"], f"sleeves.{name}.config")
        if "bundle_manifest" in sleeve:
            add(sleeve["bundle_manifest"], f"sleeves.{name}.bundle_manifest")
    return records


def _observed_hashes(
    paths: Iterable[str], injected: Mapping[str, str] | None
) -> dict[str, str]:
    expected = tuple(str(path) for path in paths)
    if injected is not None:
        observed = {str(path): str(digest) for path, digest in injected.items()}
        if set(observed) != set(expected):
            raise RuntimeError("ESDI injected hash inventory drift")
        return observed
    return {path: sha256_file(path) for path in expected}


def _discover_gross9_adapter_closure() -> tuple[Path, ...]:
    discovered: set[Path] = set()
    pending = [
        Path("training/audit_gross9_pullback_premium_overheat_marginal.py")
    ]
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        discovered.add(path)
        try:
            tree = ast.parse(
                prereg._dependency_bytes(path),
                filename=str(path),
            )
        except (SyntaxError, UnicodeDecodeError) as error:
            raise RuntimeError(
                f"ESDI cannot parse Gross9 adapter dependency: {path}"
            ) from error
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                modules.append(node.module)
            for module in modules:
                for imported_path in prereg._local_import_paths(module):
                    if imported_path not in discovered:
                        pending.append(imported_path)
    return tuple(sorted(discovered))


def _load_source_manifest(
    authority: Mapping[str, Any],
    context: ValidationContext,
) -> Mapping[str, Any]:
    if context.source_manifest is not None:
        return context.source_manifest
    path = _repository_path(authority["transitive_source_manifest"]["path"])
    return json.loads(path.read_text(encoding="utf-8"))


def _rank7_bundle_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("bundle_manifest_hash", None)
    return canonical_hash(payload)


def _load_rank7_bundle_manifest(
    authority: Mapping[str, Any],
    context: ValidationContext,
    *,
    synthetic: bool,
) -> Mapping[str, Any]:
    if context.rank7_bundle_manifest is not None:
        return context.rank7_bundle_manifest
    if synthetic:
        raise RuntimeError(
            "ESDI synthetic validation requires the Rank7 bundle manifest"
        )
    binding = authority["sleeves"]["frozen_annual_rank7"]["bundle_manifest"]
    path = _repository_path(binding["path"])
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("ESDI Rank7 bundle manifest is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("ESDI Rank7 bundle manifest is not an object")
    return manifest


def _rank7_transitive_bindings(
    authority: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    if manifest.get("bundle_manifest_hash") != _rank7_bundle_manifest_hash(
        manifest
    ):
        raise RuntimeError("ESDI Rank7 bundle internal manifest hash drift")
    model_rows = manifest.get("models")
    history_row = manifest.get("hourly_history")
    if (
        not isinstance(model_rows, list)
        or len(model_rows) != 3
        or not isinstance(history_row, Mapping)
    ):
        raise RuntimeError("ESDI Rank7 transitive inventory is malformed")

    manifest_path = Path(
        str(
            authority["sleeves"]["frozen_annual_rank7"]["bundle_manifest"][
                "path"
            ]
        )
    )
    bundle_root = _repository_path(manifest_path).parent
    expected: dict[str, str] = {}

    def add(row: Mapping[str, Any], label: str) -> None:
        raw_path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if not raw_path or Path(raw_path).is_absolute() or not _is_sha256(
            digest
        ):
            raise RuntimeError(f"ESDI Rank7 transitive binding is malformed: {label}")
        target = (bundle_root / raw_path).resolve()
        if not target.is_relative_to(bundle_root) or not target.is_relative_to(
            REPOSITORY_ROOT
        ):
            raise RuntimeError(
                f"ESDI Rank7 transitive path escapes bundle: {label}"
            )
        repository_path = target.relative_to(REPOSITORY_ROOT).as_posix()
        if repository_path in expected:
            raise RuntimeError("ESDI Rank7 transitive path is duplicated")
        expected[repository_path] = digest

    for index, row in enumerate(model_rows):
        if not isinstance(row, Mapping):
            raise RuntimeError("ESDI Rank7 model binding is malformed")
        add(row, f"models[{index}]")
    add(history_row, "hourly_history")
    return expected


def _validate_rank7_bundle_transitives(
    authority: Mapping[str, Any],
    context: ValidationContext,
    *,
    synthetic: bool,
) -> dict[str, str]:
    manifest = _load_rank7_bundle_manifest(
        authority, context, synthetic=synthetic
    )
    expected = _rank7_transitive_bindings(authority, manifest)
    if synthetic and context.rank7_bundle_hashes is None:
        raise RuntimeError(
            "ESDI synthetic validation requires all Rank7 transitive hashes"
        )
    if context.rank7_bundle_hashes is not None and set(
        str(path) for path in context.rank7_bundle_hashes
    ) != set(expected):
        raise RuntimeError("ESDI Rank7 transitive hash inventory drift")
    observed = _observed_hashes(expected, context.rank7_bundle_hashes)
    for path, digest in expected.items():
        if observed[path] != digest:
            raise RuntimeError(f"ESDI Rank7 transitive SHA drift: {path}")
    return dict(sorted(observed.items()))


def _production_git_blobs(paths: Sequence[str]) -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["git", "ls-tree", "-z", "HEAD", "--", *paths],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("ESDI repository Git blob validation failed") from error
    observed: dict[str, str] = {}
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if mode != "100644" or object_type != "blob":
            raise RuntimeError("ESDI repository identity contains a non-plain blob")
        observed[path] = object_id
    return observed


def _validate_repository_identity(
    registration: Mapping[str, Any],
    context: ValidationContext,
    *,
    synthetic: bool,
) -> dict[str, Any]:
    identity = registration.get("frozen_preregistration", {}).get(
        "repository_identity"
    )
    if not isinstance(identity, Mapping):
        raise RuntimeError("ESDI preregistration repository identity is missing")
    expected_sha = identity.get("sha256")
    expected_blobs = identity.get("git_blobs")
    if (
        not isinstance(expected_sha, Mapping)
        or not isinstance(expected_blobs, Mapping)
        or len(expected_sha) != 77
        or len(expected_blobs) != 77
        or tuple(expected_sha) != tuple(expected_blobs)
    ):
        raise RuntimeError("ESDI preregistration must bind exactly 77 repository files")
    paths = tuple(str(path) for path in expected_sha)
    if synthetic:
        observed_sha = context.repository_sha256
        observed_blobs = context.repository_git_blobs
        if (
            not isinstance(observed_sha, Mapping)
            or not isinstance(observed_blobs, Mapping)
            or len(observed_sha) != 77
            or len(observed_blobs) != 77
        ):
            raise RuntimeError("ESDI synthetic validation requires all 77 repository bindings")
        observed_sha = {str(key): str(value) for key, value in observed_sha.items()}
        observed_blobs = {
            str(key): str(value) for key, value in observed_blobs.items()
        }
    else:
        observed_sha = {path: sha256_file(path) for path in paths}
        observed_blobs = _production_git_blobs(paths)
    if observed_sha != {
        str(key): str(value) for key, value in expected_sha.items()
    }:
        raise RuntimeError("ESDI repository identity SHA-256 drift")
    if observed_blobs != {
        str(key): str(value) for key, value in expected_blobs.items()
    }:
        raise RuntimeError("ESDI repository identity Git blob drift")
    return {
        "paths": list(paths),
        "sha256": dict(observed_sha),
        "git_blobs": dict(observed_blobs),
    }


def _run_evaluator_synthetic_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(EVALUATOR_TEST_PATH),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "ESDI evaluator synthetic test suite could not start"
        ) from error
    if completed.returncode != 0:
        raise RuntimeError("ESDI evaluator synthetic test suite failed")
    return {
        "command": [
            "python",
            "-m",
            "pytest",
            "-q",
            str(EVALUATOR_TEST_PATH),
        ],
        "passed": True,
    }


def _validate_evaluator_source_identity() -> dict[str, Any]:
    paths = (str(EVALUATOR_SOURCE_PATH), str(EVALUATOR_TEST_PATH))
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *paths,
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("ESDI evaluator source identity validation failed") from error
    if status.stdout:
        raise RuntimeError("ESDI evaluator and tests must be committed and clean")
    blobs = _production_git_blobs(paths)
    if set(blobs) != set(paths):
        raise RuntimeError("ESDI evaluator Git blob identity is incomplete")
    sha256 = {path: sha256_file(path) for path in paths}
    core = {
        "paths": list(paths),
        "sha256": sha256,
        "git_blobs": blobs,
        "synthetic_tests": _run_evaluator_synthetic_tests(),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_frozen_contract(
    registration: Mapping[str, Any],
    *,
    context: ValidationContext | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Validate every frozen prerequisite before clocks or outcomes may be used."""

    if context is not None and not synthetic:
        raise RuntimeError("ESDI injected ValidationContext is synthetic-only")
    if synthetic and context is None:
        raise RuntimeError("ESDI synthetic validation requires ValidationContext")
    ctx = context or ValidationContext()
    if registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ESDI preregistration manifest drift")
    if registration.get("policy_id") != POLICY_ID:
        raise RuntimeError("ESDI preregistration identity drift")
    execution = registration.get("execution", {})
    expected_execution = {
        "leverage": LEVERAGE,
        "base_cost_bp_per_notional_side": 6,
        "stress_cost_bp_per_notional_side": 10,
        "aligned_availability_still_waits_seconds": 300,
        "entry": "ceil_to_5m(available_at)+300 elapsed seconds",
        "entry_price": "BTCUSDT perpetual 5m open at entry",
        "exit_price": "BTCUSDT perpetual 5m open at exact scheduled exit",
        "hold_bars_5m": 288,
        "hold_seconds": 86_400,
        "pyramiding_stop_take_profit_trailing_or_early_close": False,
        "candidate_order": [
            "entry_time",
            "available_at",
            "epoch_id",
            "side",
        ],
        "reservation": {
            "accept": "entry_time >= previous accepted exit_time",
            "interval": "[entry_time,exit_time)",
            "scope": "one global position",
            "suppressed_candidates_queued": False,
        },
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise RuntimeError(f"ESDI execution contract drift: {key}")
    if execution.get("funding") != {
        "interval": "entry_time <= funding_time < exit_time",
        "cash": "-side_sign*quantity*funding_rate*settlement_mark_price",
        "realized_only": True,
    } or execution.get("split_crossing_action") != "skip; never truncate":
        raise RuntimeError("ESDI execution funding contract drift")
    authority = registration.get("gross9", {}).get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("ESDI Gross9 authority is absent")
    if set(authority) != {
        "portfolio",
        "base_portfolio",
        "pre2025_anchor",
        "runtime",
        "runtime_code_closure",
        "sleeves",
        "transitive_source_manifest",
        "clock_reconstruction",
    }:
        raise RuntimeError("ESDI Gross9 authority schema drift")
    if (
        tuple(authority["runtime"])
        != ("portfolio_live.py", "rank7_runtime.py", "rex_llm_live.py")
        or tuple(authority["sleeves"]) != GROSS9_SLEEVES
        or authority["pre2025_anchor"].get("metadata_only_until_economic_stage")
        is not True
        or {
            sleeve: authority["sleeves"][sleeve].get("side")
            for sleeve in GROSS9_SLEEVES
        }
        != {
            "cand_rex_veto_7": "AUTO from exact REX decision",
            "fresh_kimchi_fx": "AUTO from exact exclusive long/short gates",
            "frozen_annual_rank7": "LONG",
            "markov_transition_long": "LONG",
            "rex_taker_low_range_position": "AUTO from exact REX decision",
        }
    ):
        raise RuntimeError("ESDI Gross9 authority inventory drift")
    expected_reconstruction = {
        "stage": "after ESDI source-support pass and before ESDI economics",
        "five_signed_sleeves_required": True,
        "exact_runtime_config_and_transitive_hash_validation_required": True,
        "failure_or_missing_dependency_is_terminal": True,
    }
    if authority["clock_reconstruction"] != expected_reconstruction:
        raise RuntimeError("ESDI Gross9 reconstruction authority drift")
    if registration["gross9"].get("weights") != GROSS9_WEIGHTS:
        raise RuntimeError("ESDI Gross9 weights drift")
    gross9_contract = registration["gross9"]
    if (
        float(gross9_contract.get("baseline_gross", -1.0)) != BASELINE_GROSS
        or tuple(float(value) for value in gross9_contract.get("candidate_weights", ()))
        != CANDIDATE_WEIGHTS
        or gross9_contract.get("treatment")
        != "scale every sleeve by (9-w)/9 and add ESDI at w"
        or gross9_contract.get("configured_treatment_gross") != BASELINE_GROSS
        or gross9_contract.get("future_uses_only_frozen_weight") is not True
        or gross9_contract.get("future_rerank_or_alternate_weight") is not False
        or gross9_contract.get("matching_execution_costs_exact_funding_and_strict_mdd")
        is not True
        or gross9_contract.get("selection_periods") != list(SELECTION_PERIODS)
        or gross9_contract.get("comparison")
        != "unscaled authoritative Gross9 baseline at gross 9.0"
        or gross9_contract.get("ranking")
        != (
            "maximum minimum base/stress 2023H2/2024 improvement; "
            "tie lower weight"
        )
        or gross9_contract.get("freeze_rank") != 1
        or gross9_contract.get("requirements")
        != {
            "base_and_stress_cagr_mdd_improvement_min": 0.05,
            "unscaled_absolute_return_retention_min": 0.97,
            "base_and_stress_absolute_return_positive": True,
            "strict_mdd_reduced_in_at_least_one_selection_period": True,
        }
    ):
        raise RuntimeError("ESDI same-gross contract drift")
    economic = registration.get("economic_contract")
    if not isinstance(economic, Mapping) or economic != {
        "evaluator_committed_tested_and_hash_bound_before_rows_open": True,
        "standalone_gate_base_and_stress_each_period": {
            "absolute_return": ">0",
            "full_calendar_cagr_to_strict_mdd": ">=3.0",
            "strict_mdd": "<=0.15",
            "mean_gross_underlying_bp": ">=20",
            "calendar_month_clustered_signflip_p": "<=0.10",
        },
        "strict_mdd": (
            "global/pre-entry HWM; favorable OHLC and funding credits before "
            "adverse OHLC, funding debits, liquidation envelope and exit cost"
        ),
        "primary_strictly_exceeds": [
            "gas_utilization_only",
            "base_fee_one_epoch_stale",
        ],
        "cannot_completely_qualify": [
            "exact_direction_flip",
            "deterministic_random_side",
            "constant_long",
            "constant_short",
        ],
    }:
        raise RuntimeError("ESDI strict MDD contract drift")
    calendars = registration.get("calendars")
    expected_calendars = {
        name: [
            PERIODS[name][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            PERIODS[name][1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        ]
        for name in ("selection", "future25", "future26", "full")
    }
    if not isinstance(calendars, Mapping) or any(
        calendars.get(name) != bounds
        for name, bounds in expected_calendars.items()
    ) or calendars.get("full_cagr_wall_clock_years") != 3:
        raise RuntimeError("ESDI economic calendar drift")
    registration_core = {
        key: value
        for key, value in registration.items()
        if key != "manifest_hash"
    }
    if canonical_hash(registration_core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ESDI preregistration canonical manifest drift")

    closure = authority["runtime_code_closure"]
    exact_environment = closure["exact_runtime_environment"]
    if (
        closure.get("roots")
        != [
            "execution/portfolio_live.py",
            "execution/rank7_runtime.py",
            "execution/rex_llm_live.py",
        ]
        or closure.get("environment_lock_paths") != ["pyproject.toml", "uv.lock"]
        or closure.get("ast_import_closure_must_match_before_artifact_creation")
        is not True
        or closure.get("bound_by_git_blob_and_sha256_in_repository_identity")
        is not True
        or closure.get("runtime_environment_must_match_before_artifact_creation")
        is not True
        or closure.get("required_runtime_abi_and_selected_packages")
        != {
            key: exact_environment[key]
            for key in ("python", "platform", "packages")
        }
    ):
        raise RuntimeError("ESDI Gross9 runtime closure contract drift")
    observed_environment = (
        ctx.runtime_environment
        if ctx.runtime_environment is not None
        else prereg.current_runtime_environment()
    )
    if observed_environment != exact_environment:
        raise RuntimeError("ESDI Gross9 runtime environment changed")
    if (
        observed_environment.get("all_distributions_count")
        != closure["all_distribution_inventory_count"]
        or observed_environment.get("all_distributions_sha256")
        != closure["all_distribution_inventory_sha256"]
    ):
        raise RuntimeError("ESDI Gross9 distribution inventory changed")

    expected_paths = tuple(str(path) for path in closure["paths"])
    environment_lock_paths = tuple(
        str(path) for path in closure["environment_lock_paths"]
    )
    expected_runtime_code_paths = tuple(
        path for path in expected_paths if path not in environment_lock_paths
    )
    discovered = (
        tuple(str(path) for path in ctx.discovered_runtime_closure)
        if ctx.discovered_runtime_closure is not None
        else tuple(
            (
                path.relative_to(REPOSITORY_ROOT).as_posix()
                if path.is_absolute()
                else path.as_posix()
            )
            for path in prereg.discover_runtime_code_closure()
        )
    )
    if discovered != expected_runtime_code_paths:
        raise RuntimeError("ESDI Gross9 runtime import closure changed")

    repository_identity = _validate_repository_identity(
        registration, ctx, synthetic=synthetic
    )
    adapter_discovered = tuple(
        path.as_posix() for path in _discover_gross9_adapter_closure()
    )
    expected_adapter_extras = tuple(
        path.as_posix()
        for path in source_builder.GROSS9_ADAPTER_EXTRA_CLOSURE_PATHS
    )
    if (
        tuple(
            path.as_posix()
            for path in source_builder.PROTOCOL_PATHS
            if path in source_builder.GROSS9_ADAPTER_EXTRA_CLOSURE_PATHS
        )
        != expected_adapter_extras
    ):
        raise RuntimeError("ESDI Gross9 adapter protocol seal inventory drift")
    runtime_code_set = set(expected_runtime_code_paths)
    adapter_extra_set = set(adapter_discovered) - runtime_code_set
    if (
        tuple(sorted(set(adapter_discovered))) != adapter_discovered
        or adapter_extra_set != set(expected_adapter_extras)
        or (
            set(discovered) | set(adapter_discovered)
            != runtime_code_set | set(expected_adapter_extras)
        )
    ):
        raise RuntimeError("ESDI Gross9 adapter import closure changed")
    adapter_extra_hashes = _observed_hashes(expected_adapter_extras, None)

    declared = _declared_dependency_hashes(authority)
    observed = _observed_hashes(declared, ctx.file_hashes)
    if set(observed) != set(declared):
        missing = sorted(set(declared) - set(observed))
        raise RuntimeError(f"ESDI Gross9 dependency hash missing: {missing}")
    for path, expected in declared.items():
        if observed[path] != expected:
            raise RuntimeError(f"ESDI Gross9 dependency SHA drift: {path}")

    rank7_bundle_hashes = _validate_rank7_bundle_transitives(
        authority, ctx, synthetic=synthetic
    )

    source_manifest = _load_source_manifest(authority, ctx)
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("ESDI Gross9 source manifest is malformed")
    if not synthetic and (
        set(source_manifest) != {"schema_version", "as_of", "sources"}
        or source_manifest.get("schema_version") != 1
        or source_manifest.get("as_of") != "2026-07-16"
        or [row.get("name") for row in sources if isinstance(row, Mapping)]
        != [
            "market_5m",
            "funding",
            "premium",
            "open_interest",
            "rex_taker_train",
            "rex_taker_test",
            "rex_taker_eval",
            "rex_veto_source",
        ]
    ):
        raise RuntimeError("ESDI Gross9 source manifest inventory drift")
    source_expected: dict[str, str] = {}
    for row in sources:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "path",
            "sha256",
        }:
            raise RuntimeError("ESDI Gross9 source manifest row is malformed")
        path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if not path or not _is_sha256(digest) or path in source_expected:
            raise RuntimeError("ESDI Gross9 source manifest binding is malformed")
        source_expected[path] = digest
    source_observed = _observed_hashes(source_expected, ctx.source_hashes)
    if set(source_observed) != set(source_expected):
        missing = sorted(set(source_expected) - set(source_observed))
        raise RuntimeError(f"ESDI Gross9 source hash missing: {missing}")
    for path, expected in source_expected.items():
        if source_observed[path] != expected:
            raise RuntimeError(f"ESDI Gross9 source SHA drift: {path}")

    return {
        "validated": True,
        "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "dependency_hashes": dict(sorted(observed.items())),
        "runtime_closure_paths": list(expected_runtime_code_paths),
        "environment_lock_paths": list(environment_lock_paths),
        "gross9_adapter_closure_paths": list(adapter_discovered),
        "gross9_adapter_extra_hashes": dict(
            sorted(adapter_extra_hashes.items())
        ),
        "source_hashes": dict(sorted(source_observed.items())),
        "rank7_bundle_hashes": rank7_bundle_hashes,
        "runtime_environment": _json_ready(observed_environment),
        "repository_identity": repository_identity,
    }


def _utc_series(values: Any, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    series = pd.Series(parsed).reset_index(drop=True)
    if series.isna().any():
        raise RuntimeError(f"ESDI {label} contains missing timestamps")
    return series


def validate_clock(
    clock: pd.DataFrame,
    *,
    sleeve: str,
    allow_overlap: bool = False,
) -> pd.DataFrame:
    if tuple(clock.columns) != CLOCK_COLUMNS:
        raise RuntimeError(f"ESDI {sleeve} clock schema drift")
    out = clock.copy()
    out["entry_time"] = _utc_series(out["entry_time"], f"{sleeve} entry")
    out["exit_time"] = _utc_series(out["exit_time"], f"{sleeve} exit")
    numeric_side = pd.to_numeric(out["side"], errors="raise")
    if not numeric_side.isin((-1, 1)).all():
        raise RuntimeError(f"ESDI {sleeve} side is not signed")
    out["side"] = numeric_side.astype(np.int8)
    if out["entry_time"].duplicated().any() or not out["entry_time"].is_monotonic_increasing:
        raise RuntimeError(f"ESDI {sleeve} entries are duplicate or unsorted")
    if not (out["exit_time"] > out["entry_time"]).all():
        raise RuntimeError(f"ESDI {sleeve} clock has nonpositive holds")
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    for column in ("entry_time", "exit_time"):
        elapsed = (out[column] - epoch).dt.total_seconds().to_numpy()
        if np.any(np.mod(elapsed, 300.0) != 0.0):
            raise RuntimeError(f"ESDI {sleeve} clock is not 5m aligned")
    if not allow_overlap and len(out) > 1:
        previous_exit = out["exit_time"].iloc[:-1].reset_index(drop=True)
        next_entry = out["entry_time"].iloc[1:].reset_index(drop=True)
        if (next_entry < previous_exit).any():
            raise RuntimeError(f"ESDI {sleeve} clock overlaps")
    return out


def reconstruct_gross9_sleeve_clocks(
    registration: Mapping[str, Any],
    *,
    context: ValidationContext,
    reconstructors: Mapping[
        str, Callable[[Mapping[str, Any], Mapping[str, Any]], pd.DataFrame]
    ]
    | None = None,
    injected_clocks: Mapping[str, pd.DataFrame] | None = None,
    synthetic: bool = False,
) -> dict[str, pd.DataFrame]:
    """Reconstruct all five signed clocks or fail without a partial result."""

    if (reconstructors is not None or injected_clocks is not None) and not synthetic:
        raise RuntimeError(
            "ESDI injected clocks and reconstructors are synthetic-only"
        )
    validation = validate_frozen_contract(
        registration, context=context, synthetic=synthetic
    )
    if injected_clocks is not None:
        if not synthetic:
            raise RuntimeError("ESDI injected Gross9 clocks are synthetic-only")
        raw = injected_clocks
    else:
        if reconstructors is None or tuple(reconstructors) != GROSS9_SLEEVES:
            raise RuntimeError("ESDI requires exact five-sleeve reconstructors")
        authority = registration["gross9"]["authority"]
        built: dict[str, pd.DataFrame] = {}
        for sleeve in GROSS9_SLEEVES:
            try:
                built[sleeve] = reconstructors[sleeve](
                    authority["sleeves"][sleeve], validation
                )
            except Exception as error:
                raise RuntimeError(
                    f"ESDI Gross9 clock reconstruction failed: {sleeve}"
                ) from error
        raw = built
    if tuple(raw) != GROSS9_SLEEVES:
        raise RuntimeError("ESDI Gross9 clocks are missing, extra, or reordered")
    return {
        sleeve: validate_clock(raw[sleeve], sleeve=sleeve)
        for sleeve in GROSS9_SLEEVES
    }


def validate_market(market: pd.DataFrame) -> pd.DataFrame:
    if tuple(market.columns) != MARKET_COLUMNS:
        raise RuntimeError("ESDI market schema drift")
    out = market.copy()
    out["timestamp"] = _utc_series(out["timestamp"], "market")
    for column in ("open", "high", "low"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    if (
        out["timestamp"].duplicated().any()
        or not out["timestamp"].is_monotonic_increasing
        or (out["timestamp"].diff().dropna() != FIVE_MINUTES).any()
    ):
        raise RuntimeError("ESDI market grid is not exact contiguous 5m")
    values = out.loc[:, ("open", "high", "low")].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("ESDI market prices are not finite positive")
    if (out["high"] < out["open"]).any() or (out["low"] > out["open"]).any():
        raise RuntimeError("ESDI market OHLC envelope is invalid")
    if (out["high"] < out["low"]).any():
        raise RuntimeError("ESDI market high is below low")
    return out


def validate_funding(funding: pd.DataFrame) -> pd.DataFrame:
    if tuple(funding.columns) != FUNDING_COLUMNS:
        raise RuntimeError("ESDI funding schema drift")
    out = funding.copy()
    out["funding_time"] = _utc_series(out["funding_time"], "funding")
    for column in ("funding_rate", "settlement_mark"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    if (
        out["funding_time"].duplicated().any()
        or not out["funding_time"].is_monotonic_increasing
    ):
        raise RuntimeError("ESDI funding times are duplicate or unsorted")
    if not np.isfinite(out["funding_rate"]).all():
        raise RuntimeError("ESDI funding rates are not finite")
    if (
        not np.isfinite(out["settlement_mark"]).all()
        or (out["settlement_mark"] <= 0.0).any()
    ):
        raise RuntimeError("ESDI funding settlement marks are invalid")
    return out


def _period_clock(
    clock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, sleeve: str
) -> tuple[pd.DataFrame, int]:
    checked = validate_clock(clock, sleeve=sleeve)
    contained = (checked["entry_time"] >= start) & (checked["exit_time"] <= end)
    crossing = (
        (checked["entry_time"] < end)
        & (checked["exit_time"] > start)
        & ~contained
    )
    return checked.loc[contained].reset_index(drop=True), int(crossing.sum())


def _full_calendar_cagr(final_equity: float, years: float) -> float:
    """Compute CAGR in log space, saturating only beyond float64 range."""

    annual_log_return = math.log(final_equity) / years
    maximum = float(np.finfo(np.float64).max)
    if annual_log_return >= math.log(maximum):
        return maximum
    return math.expm1(annual_log_return)


def _calendar_years(lower: pd.Timestamp, upper: pd.Timestamp) -> float:
    if (lower, upper) == PERIODS["full"]:
        return 3.0
    return (upper - lower).total_seconds() / (365.25 * 86_400.0)


def _simulate_portfolio(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    cost_rate: float,
) -> dict[str, Any]:
    """Run exact-open fixed-quantity accounting with a strict MDD envelope."""

    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    if lower.tzinfo is None:
        lower = lower.tz_localize("UTC")
    else:
        lower = lower.tz_convert("UTC")
    if upper.tzinfo is None:
        upper = upper.tz_localize("UTC")
    else:
        upper = upper.tz_convert("UTC")
    if upper <= lower:
        raise ValueError("ESDI period must have positive calendar length")
    if not math.isfinite(cost_rate) or cost_rate < 0.0:
        raise ValueError("ESDI cost rate must be finite and nonnegative")

    bars = validate_market(market)
    funds = validate_funding(funding)
    funding_by_bar: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    period_funding = funds[
        (funds["funding_time"] >= lower)
        & (funds["funding_time"] < upper)
    ]
    for row in period_funding.itertuples(index=False):
        elapsed = pd.Timestamp(row.funding_time) - lower
        bar_number = int(elapsed // FIVE_MINUTES)
        bar_time = lower + bar_number * FIVE_MINUTES
        funding_by_bar.setdefault(bar_time, []).append(
            (float(row.funding_rate), float(row.settlement_mark))
        )
    period_bars = bars[
        (bars["timestamp"] >= lower) & (bars["timestamp"] <= upper)
    ].reset_index(drop=True)
    expected = int((upper - lower) / FIVE_MINUTES) + 1
    if (
        len(period_bars) != expected
        or period_bars["timestamp"].iloc[0] != lower
        or period_bars["timestamp"].iloc[-1] != upper
    ):
        raise RuntimeError("ESDI market does not cover the full period calendar")

    unknown = set(weights) - set(clocks)
    if unknown:
        raise RuntimeError(f"ESDI weights have no clocks: {sorted(unknown)}")
    checked_clocks: dict[str, pd.DataFrame] = {}
    skipped_crossers: dict[str, int] = {}
    for sleeve, clock in clocks.items():
        checked, skipped = _period_clock(clock, lower, upper, sleeve)
        checked_clocks[sleeve] = checked
        skipped_crossers[sleeve] = skipped
    entries: dict[pd.Timestamp, list[tuple[str, pd.Series]]] = {}
    exits: dict[pd.Timestamp, list[tuple[str, pd.Series]]] = {}
    for sleeve, clock in checked_clocks.items():
        weight = float(weights.get(sleeve, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise RuntimeError(f"ESDI invalid sleeve weight: {sleeve}")
        if weight == 0.0:
            continue
        for _, row in clock.iterrows():
            entries.setdefault(row["entry_time"], []).append((sleeve, row))
            exits.setdefault(row["exit_time"], []).append((sleeve, row))

    cash = 1.0
    hwm = 1.0
    maximum_drawdown = 0.0
    positions: list[_Position] = []
    trade_records: list[dict[str, Any]] = []
    path: list[dict[str, Any]] = []

    for _, bar in period_bars.iterrows():
        timestamp = bar["timestamp"]
        open_price = float(bar["open"])
        pre_cost_equity = cash + sum(
            position.side
            * position.quantity
            * (open_price - position.entry_price)
            for position in positions
        )
        if pre_cost_equity <= 0.0 or not math.isfinite(pre_cost_equity):
            raise RuntimeError("ESDI pre-cost equity is not liquidation safe")
        entry_cost_event = 0.0
        exit_cost_event = 0.0

        for sleeve, row in exits.get(timestamp, []):
            matches = [
                position
                for position in positions
                if position.sleeve == sleeve
                and position.entry_time == row["entry_time"]
                and position.exit_time == timestamp
            ]
            if len(matches) != 1:
                raise RuntimeError("ESDI exit has no unique open position")
            position = matches[0]
            price_cash = (
                position.side
                * position.quantity
                * (open_price - position.entry_price)
            )
            exit_cost = abs(position.quantity * open_price) * cost_rate
            cash += price_cash - exit_cost
            exit_cost_event += exit_cost
            positions.remove(position)
            net_cash = (
                price_cash
                + position.funding_cash
                - position.entry_cost
                - exit_cost
            )
            trade_records.append(
                {
                    "sleeve": sleeve,
                    "entry_time": position.entry_time,
                    "exit_time": timestamp,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": open_price,
                    "entry_cost": position.entry_cost,
                    "exit_cost": exit_cost,
                    "funding_cash": position.funding_cash,
                    "net_return_on_allocated_equity": (
                        net_cash / position.allocated_equity
                    ),
                    "gross_underlying_bp": (
                        position.side
                        * (open_price / position.entry_price - 1.0)
                        * 10_000.0
                    ),
                }
            )

        post_exit_equity = cash + sum(
            position.side
            * position.quantity
            * (open_price - position.entry_price)
            for position in positions
        )
        if post_exit_equity <= 0.0 or not math.isfinite(post_exit_equity):
            raise RuntimeError("ESDI post-exit equity is not liquidation safe")
        for sleeve, row in entries.get(timestamp, []):
            weight = float(weights[sleeve])
            allocated = post_exit_equity * weight
            quantity = allocated * LEVERAGE / open_price
            entry_cost = abs(quantity * open_price) * cost_rate
            position = _Position(
                sleeve=sleeve,
                entry_time=timestamp,
                exit_time=row["exit_time"],
                side=int(row["side"]),
                entry_price=open_price,
                quantity=quantity,
                allocated_equity=allocated,
                entry_cost=entry_cost,
            )
            cash -= entry_cost
            entry_cost_event += entry_cost
            positions.append(position)

        funding_credit = 0.0
        funding_debit = 0.0
        for rate, settlement_mark in funding_by_bar.get(timestamp, []):
            for position in positions:
                funding_cash = (
                    -position.side
                    * position.quantity
                    * rate
                    * settlement_mark
                )
                position.funding_cash += funding_cash
                cash += funding_cash
                if funding_cash > 0.0:
                    funding_credit += funding_cash
                elif funding_cash < 0.0:
                    funding_debit += funding_cash

        favorable = 0.0
        adverse = 0.0
        adverse_mark = open_price
        gross_quantity = sum(abs(position.quantity) for position in positions)
        net_signed_quantity = sum(
            position.side * position.quantity for position in positions
        )
        if abs(net_signed_quantity) <= 1e-15 * max(1.0, gross_quantity):
            net_signed_quantity = 0.0
        if timestamp < upper:
            if net_signed_quantity > 0.0:
                favorable = net_signed_quantity * (
                    float(bar["high"]) - open_price
                )
                adverse = net_signed_quantity * (
                    float(bar["low"]) - open_price
                )
                adverse_mark = float(bar["low"])
            elif net_signed_quantity < 0.0:
                favorable = net_signed_quantity * (
                    float(bar["low"]) - open_price
                )
                adverse = net_signed_quantity * (
                    float(bar["high"]) - open_price
                )
                adverse_mark = float(bar["high"])
        favorable = max(0.0, favorable)
        adverse = min(0.0, adverse)
        liquidation_cost = (
            gross_quantity * adverse_mark * cost_rate
        )

        # Frozen order: global/pre-entry HWM, then favorable OHLC and funding
        # credits, then adverse OHLC, funding debits, liquidation envelope,
        # and side costs.
        upper_equity = max(
            pre_cost_equity,
            pre_cost_equity
            - entry_cost_event
            + funding_credit
            + favorable,
        )
        hwm = max(hwm, upper_equity)
        lower_equity = min(
            pre_cost_equity,
            pre_cost_equity
            - entry_cost_event
            + funding_credit
            + funding_debit
            + adverse
            - liquidation_cost
            - exit_cost_event,
        )
        if (
            lower_equity <= 0.0
            or not math.isfinite(lower_equity)
            or not math.isfinite(upper_equity)
        ):
            raise RuntimeError("ESDI portfolio is not liquidation safe")
        maximum_drawdown = max(maximum_drawdown, 1.0 - lower_equity / hwm)
        marked_equity = cash + sum(
            position.side
            * position.quantity
            * (open_price - position.entry_price)
            for position in positions
        )
        if marked_equity <= 0.0 or not math.isfinite(marked_equity):
            raise RuntimeError("ESDI portfolio equity is not liquidation safe")
        path.append(
            {
                "timestamp": timestamp,
                "marked_equity": marked_equity,
                "hwm": hwm,
                "strict_lower_equity": lower_equity,
                "net_signed_btc_quantity": net_signed_quantity,
                "hypothetical_liquidation_cost": liquidation_cost,
            }
        )

    if positions:
        raise RuntimeError("ESDI period ended with open positions")
    final_equity = cash
    if final_equity <= 0.0 or not math.isfinite(final_equity):
        raise RuntimeError("ESDI final equity is invalid")
    years = _calendar_years(lower, upper)
    cagr = _full_calendar_cagr(final_equity, years)
    ratio = (
        min(
            float(np.finfo(np.float64).max),
            cagr / max(maximum_drawdown, 1e-15),
        )
        if cagr > 0.0
        else 0.0
    )
    gross_moves = [float(row["gross_underlying_bp"]) for row in trade_records]
    signflip = calendar_month_clustered_signflip(
        trade_records,
        seed=SIGNFLIP_SEED,
        samples=SIGNFLIP_SAMPLES,
    )
    return {
        "absolute_return": final_equity - 1.0,
        "cagr": cagr,
        "strict_mdd": maximum_drawdown,
        "cagr_to_strict_mdd": ratio,
        "mean_gross_underlying_bp": (
            float(np.mean(gross_moves)) if gross_moves else 0.0
        ),
        "calendar_month_clustered_signflip": signflip,
        "trades": len(trade_records),
        "liquidation_safe": True,
        "final_equity": final_equity,
        "trade_records": trade_records,
        "path": path,
        "calendar_start": lower,
        "calendar_end": upper,
        "cost_rate": cost_rate,
        "skipped_split_crossers": skipped_crossers,
        "split_crossers_truncated": 0,
    }


def simulate_portfolio(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    cost_rate: float,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("ESDI direct frame accounting is synthetic-only")
    return _simulate_portfolio(
        market,
        funding,
        clocks,
        weights,
        start=start,
        end=end,
        cost_rate=cost_rate,
    )


def calendar_month_clustered_signflip(
    trades: Sequence[Mapping[str, Any]],
    *,
    seed: int = SIGNFLIP_SEED,
    samples: int = SIGNFLIP_SAMPLES,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("ESDI sign-flip samples must be positive")
    clusters: dict[str, float] = {}
    for trade in trades:
        timestamp = pd.Timestamp(trade["entry_time"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        key = timestamp.strftime("%Y-%m")
        value = float(trade["net_return_on_allocated_equity"])
        if not math.isfinite(value):
            raise RuntimeError("ESDI sign-flip trade return is not finite")
        clusters[key] = clusters.get(key, 0.0) + value
    ordered = np.asarray(
        [
            clusters[key]
            for key in sorted(clusters)
            if abs(clusters[key]) > 1e-15
        ],
        dtype=float,
    )
    if len(ordered) == 0:
        return {
            "seed": int(seed),
            "samples": int(samples),
            "cluster_count": 0,
            "observed": 0.0,
            "p_value_one_sided": 1.0,
            "method": "empty",
        }
    observed = float(ordered.sum())
    method: str
    if observed <= 0.0:
        p_value = 1.0
        method = "nonpositive_observed"
    elif len(ordered) <= 20:
        exceed = 0
        total = 2 ** len(ordered)
        for signs in itertools.product((-1.0, 1.0), repeat=len(ordered)):
            if float(np.dot(np.asarray(signs), ordered)) >= observed - 1e-15:
                exceed += 1
        p_value = exceed / total
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        exceed = 0
        remaining = samples
        while remaining:
            batch = min(remaining, 4096)
            signs = rng.integers(0, 2, size=(batch, len(ordered)), dtype=np.int8)
            signs = signs.astype(np.float64) * 2.0 - 1.0
            exceed += int(np.count_nonzero(signs @ ordered >= observed - 1e-15))
            remaining -= batch
        p_value = (exceed + 1.0) / (samples + 1.0)
        method = "monte_carlo"
    return {
        "seed": int(seed),
        "samples": int(samples),
        "cluster_count": int(len(ordered)),
        "observed": observed,
        "p_value_one_sided": float(p_value),
        "method": method,
    }


def standalone_gate_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "absolute_return_positive": float(metrics["absolute_return"]) > 0.0,
        "cagr_to_strict_mdd_at_least_3": (
            float(metrics["cagr_to_strict_mdd"]) >= 3.0
        ),
        "strict_mdd_at_most_0p15": float(metrics["strict_mdd"]) <= 0.15,
        "mean_gross_underlying_bp_at_least_20": (
            float(metrics["mean_gross_underlying_bp"]) >= 20.0
        ),
        "calendar_month_clustered_signflip_p_at_most_0p10": (
            float(
                metrics["calendar_month_clustered_signflip"][
                    "p_value_one_sided"
                ]
            )
            <= 0.10
        ),
        "liquidation_safe": bool(metrics["liquidation_safe"]),
    }


def _evaluate_standalone_period(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clock: pd.DataFrame,
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, cost in (("base", BASE_COST_RATE), ("stress", STRESS_COST_RATE)):
        metrics = _simulate_portfolio(
            market,
            funding,
            {"esdi": clock},
            {"esdi": 1.0},
            start=start,
            end=end,
            cost_rate=cost,
        )
        checks = standalone_gate_checks(metrics)
        rows[name] = {
            "metrics": metrics,
            "checks": checks,
            "passes": all(checks.values()),
        }
    rows["passes"] = bool(rows["base"]["passes"] and rows["stress"]["passes"])
    return rows


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
        raise RuntimeError("ESDI direct standalone frames are synthetic-only")
    return _evaluate_standalone_period(
        market, funding, clock, start=start, end=end
    )


def evaluate_primary_superiority(
    primary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = ("gas_utilization_only", "base_fee_one_epoch_stale")
    disqualified = (
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
    )
    if tuple(controls) != CONTROL_NAMES:
        raise RuntimeError("ESDI standalone controls are missing, extra, or reordered")
    checks: dict[str, bool] = {}
    for cost in ("base", "stress"):
        primary_ratio = float(primary[cost]["metrics"]["cagr_to_strict_mdd"])
        for name in required:
            checks[f"{cost}_strictly_exceeds_{name}"] = primary_ratio > float(
                controls[name][cost]["metrics"]["cagr_to_strict_mdd"]
            )
    for name in disqualified:
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
    """Evaluate every frozen standalone gate and strict primary control."""

    if not synthetic:
        raise RuntimeError("ESDI direct standalone frames are synthetic-only")
    if tuple(control_clocks) != CONTROL_NAMES:
        raise RuntimeError("ESDI standalone controls are missing, extra, or reordered")
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
        "passes": bool(primary["passes"] and superiority["passes"]),
    }


def _evaluate_standalone_period_with_controls(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    primary_clock: pd.DataFrame,
    control_clocks: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    if tuple(control_clocks) != CONTROL_NAMES:
        raise RuntimeError("ESDI standalone controls are missing, extra, or reordered")
    primary = _evaluate_standalone_period(
        market, funding, primary_clock, start=start, end=end
    )
    controls = {
        name: _evaluate_standalone_period(
            market, funding, control_clocks[name], start=start, end=end
        )
        for name in CONTROL_NAMES
    }
    superiority = evaluate_primary_superiority(primary, controls)
    return {
        "primary": primary,
        "controls": controls,
        "primary_superiority": superiority,
        "passes": bool(primary["passes"] and superiority["passes"]),
    }


def same_gross_weights(candidate_weight: float) -> dict[str, float]:
    weight = float(candidate_weight)
    if weight not in CANDIDATE_WEIGHTS:
        raise ValueError("ESDI candidate weight is outside the frozen grid")
    scale = (BASELINE_GROSS - weight) / BASELINE_GROSS
    treatment = {
        sleeve: value * scale for sleeve, value in GROSS9_WEIGHTS.items()
    }
    treatment["esdi"] = weight
    if not math.isclose(
        sum(treatment.values()), BASELINE_GROSS, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("ESDI same-gross arithmetic did not preserve gross 9")
    return treatment


def _same_gross_period_checks(
    treatment: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, bool]:
    baseline_return = float(baseline["absolute_return"])
    retention = (
        float(treatment["absolute_return"]) / baseline_return
        if baseline_return > 0.0
        else -math.inf
    )
    return {
        "cagr_mdd_improvement_at_least_0p05": (
            float(treatment["cagr_to_strict_mdd"])
            - float(baseline["cagr_to_strict_mdd"])
            >= 0.05
        ),
        "unscaled_absolute_return_retention_at_least_0p97": retention >= 0.97,
        "absolute_return_positive": float(treatment["absolute_return"]) > 0.0,
        "liquidation_safe": bool(treatment["liquidation_safe"]),
    }


def _evaluate_same_gross_weight(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    gross9_clocks: Mapping[str, pd.DataFrame],
    esdi_clock: pd.DataFrame,
    candidate_weight: float,
    *,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    normalized_periods = {
        str(name): (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        for name, bounds in periods.items()
    }
    if tuple(normalized_periods) != tuple(SELECTION_PERIODS) or any(
        normalized_periods[name] != SELECTION_PERIODS[name]
        for name in SELECTION_PERIODS
    ):
        raise RuntimeError(
            "ESDI same-gross requires exact ordered 2023H2 and 2024 periods"
        )
    if tuple(gross9_clocks) != GROSS9_SLEEVES:
        raise RuntimeError("ESDI same-gross evaluation requires five Gross9 clocks")
    treatment_weights = same_gross_weights(candidate_weight)
    clocks = dict(gross9_clocks)
    clocks["esdi"] = esdi_clock
    output: dict[str, Any] = {
        "candidate_weight": float(candidate_weight),
        "treatment_weights": treatment_weights,
        "baseline_weights": dict(GROSS9_WEIGHTS),
        "periods": {},
        "period_order": list(SELECTION_PERIODS),
        "fresh_evaluation": True,
    }
    improvements: list[float] = []
    any_mdd_reduction = False
    all_checks = True
    for period, (start, end) in normalized_periods.items():
        period_row: dict[str, Any] = {}
        for cost_name, cost_rate in (
            ("base", BASE_COST_RATE),
            ("stress", STRESS_COST_RATE),
        ):
            baseline = _simulate_portfolio(
                market,
                funding,
                gross9_clocks,
                GROSS9_WEIGHTS,
                start=start,
                end=end,
                cost_rate=cost_rate,
            )
            treatment = _simulate_portfolio(
                market,
                funding,
                clocks,
                treatment_weights,
                start=start,
                end=end,
                cost_rate=cost_rate,
            )
            checks = _same_gross_period_checks(treatment, baseline)
            improvement = (
                float(treatment["cagr_to_strict_mdd"])
                - float(baseline["cagr_to_strict_mdd"])
            )
            improvements.append(improvement)
            any_mdd_reduction = any_mdd_reduction or (
                float(treatment["strict_mdd"]) < float(baseline["strict_mdd"])
            )
            all_checks = all_checks and all(checks.values())
            period_row[cost_name] = {
                "treatment": treatment,
                "unscaled_gross9": baseline,
                "checks": checks,
                "improvement": improvement,
            }
        output["periods"][period] = period_row
    output["strict_mdd_reduced_in_at_least_one_period"] = any_mdd_reduction
    output["minimum_improvement"] = min(improvements)
    output["passes"] = bool(all_checks and any_mdd_reduction)
    return output


def evaluate_same_gross_weight(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    gross9_clocks: Mapping[str, pd.DataFrame],
    esdi_clock: pd.DataFrame,
    candidate_weight: float,
    *,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("ESDI direct same-gross frames are synthetic-only")
    return _evaluate_same_gross_weight(
        market,
        funding,
        gross9_clocks,
        esdi_clock,
        candidate_weight,
        periods=periods,
    )


def _derive_same_gross_summary(row: Mapping[str, Any]) -> tuple[float, bool]:
    weight = float(row.get("candidate_weight", -1.0))
    if (
        weight not in CANDIDATE_WEIGHTS
        or row.get("treatment_weights") != same_gross_weights(weight)
        or row.get("baseline_weights") != GROSS9_WEIGHTS
    ):
        raise RuntimeError("ESDI same-gross row weight binding drifted")
    if row.get("fresh_evaluation") is not True or tuple(
        row.get("period_order", ())
    ) != tuple(SELECTION_PERIODS):
        raise RuntimeError("ESDI ranking requires fresh exact selection rows")
    periods = row.get("periods")
    if not isinstance(periods, Mapping) or tuple(periods) != tuple(
        SELECTION_PERIODS
    ):
        raise RuntimeError("ESDI ranking selection periods drifted")
    improvements: list[float] = []
    checks_pass = True
    mdd_reduced = False
    for period in SELECTION_PERIODS:
        period_row = periods[period]
        if not isinstance(period_row, Mapping) or tuple(period_row) != (
            "base",
            "stress",
        ):
            raise RuntimeError("ESDI ranking cost rows drifted")
        for cost in ("base", "stress"):
            cost_row = period_row[cost]
            treatment = cost_row["treatment"]
            baseline = cost_row["unscaled_gross9"]
            improvement = (
                float(treatment["cagr_to_strict_mdd"])
                - float(baseline["cagr_to_strict_mdd"])
            )
            derived_checks = _same_gross_period_checks(treatment, baseline)
            if not math.isfinite(improvement):
                raise RuntimeError("ESDI same-gross improvement is not finite")
            if (
                "improvement" in cost_row
                and not math.isclose(
                    float(cost_row["improvement"]),
                    improvement,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ) or (
                "checks" in cost_row
                and dict(cost_row["checks"]) != derived_checks
            ):
                raise RuntimeError("ESDI same-gross cost fields were forged")
            improvements.append(improvement)
            checks_pass = checks_pass and all(derived_checks.values())
            mdd_reduced = mdd_reduced or (
                float(treatment["strict_mdd"])
                < float(baseline["strict_mdd"])
            )
    return min(improvements), bool(checks_pass and mdd_reduced)


def _rank_same_gross_treatments(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_passing_freeze: bool,
) -> list[dict[str, Any]]:
    observed = [float(row["candidate_weight"]) for row in rows]
    if sorted(observed) != list(CANDIDATE_WEIGHTS) or len(set(observed)) != 4:
        raise RuntimeError("ESDI ranking requires the exact four-weight grid")
    derived: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        minimum, passes = _derive_same_gross_summary(row)
        if (
            "minimum_improvement" in row
            and not math.isclose(
                float(row["minimum_improvement"]),
                minimum,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ) or ("passes" in row and bool(row["passes"]) != passes):
            raise RuntimeError("ESDI same-gross derived fields were forged")
        row["minimum_improvement"] = minimum
        row["passes"] = passes
        derived.append(row)
    ranked = sorted(
        derived,
        key=lambda row: (
            -float(row["minimum_improvement"]),
            float(row["candidate_weight"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    if not ranked[0]["passes"] and require_passing_freeze:
        raise RuntimeError("ESDI has no passing same-gross treatment")
    for row in ranked:
        row["frozen"] = bool(row["rank"] == 1 and row["passes"])
    return ranked


def rank_same_gross_treatments(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _rank_same_gross_treatments(
        rows,
        require_passing_freeze=True,
    )


def future_veto(
    frozen_selection: Mapping[str, Any],
    future_rows: Mapping[str, Mapping[str, Any]],
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("ESDI injected future rows are synthetic-only")
    if int(frozen_selection.get("rank", 0)) != 1 or not bool(
        frozen_selection.get("frozen")
    ) or not bool(frozen_selection.get("passes")):
        raise RuntimeError("ESDI future evaluation requires frozen rank one")
    weight = float(frozen_selection["candidate_weight"])
    if weight not in CANDIDATE_WEIGHTS:
        raise RuntimeError("ESDI frozen weight is invalid")
    required = ("future25", "future26")
    if tuple(future_rows) != required:
        raise RuntimeError("ESDI future veto periods are missing or reordered")
    checks: dict[str, bool] = {}
    for period in required:
        row = future_rows[period]
        if float(row["candidate_weight"]) != weight:
            raise RuntimeError("ESDI future attempted to rerank or change weight")
        checks[period] = bool(row["passes"])
    return {
        "frozen_weight": weight,
        "checks": checks,
        "passes": all(checks.values()),
        "reranked": False,
    }


def _validate_receipt_manifest(receipt: Mapping[str, Any]) -> None:
    manifest_hash = receipt.get("manifest_hash")
    core = {key: value for key, value in receipt.items() if key != "manifest_hash"}
    if not isinstance(manifest_hash, str) or manifest_hash != canonical_hash(core):
        raise RuntimeError("ESDI economic receipt manifest drift")


def authorize_future_period(
    frozen_selection: Mapping[str, Any],
    period: str,
    future_row: Mapping[str, Any],
    *,
    future25_receipt: Mapping[str, Any] | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("ESDI injected future rows are synthetic-only")
    if (
        int(frozen_selection.get("rank", 0)) != 1
        or frozen_selection.get("frozen") is not True
        or frozen_selection.get("passes") is not True
    ):
        raise RuntimeError("ESDI future requires passed frozen selection rank one")
    if period not in {"future25", "future26"}:
        raise RuntimeError("ESDI future period is not frozen")
    weight = float(frozen_selection["candidate_weight"])
    if float(future_row.get("candidate_weight", -1.0)) != weight:
        raise RuntimeError("ESDI future attempted to rerank or change weight")
    if period == "future26":
        if not isinstance(future25_receipt, Mapping):
            raise RuntimeError("ESDI future26 requires hash-bound future25 pass")
        _validate_receipt_manifest(future25_receipt)
        expected_selection = frozen_selection.get("selection_receipt_sha256")
        if (
            future25_receipt.get("protocol_version")
            != ECONOMIC_RECEIPT_PROTOCOL
            or future25_receipt.get("stage") != "future25"
            or future25_receipt.get("passed") is not True
            or float(future25_receipt.get("frozen_weight", -1.0)) != weight
            or future25_receipt.get("selection_receipt_sha256")
            != expected_selection
        ):
            raise RuntimeError("ESDI future26 future25 receipt binding drift")
    return {
        "period": period,
        "frozen_weight": weight,
        "passed": bool(future_row.get("passes")),
        "reranked": False,
    }


def _clock_intervals(
    clock: pd.DataFrame,
    *,
    sleeve: str,
) -> list[dict[str, Any]]:
    checked = validate_clock(clock, sleeve=sleeve)
    domain_start, domain_end = PERIODS["full"]
    contained = (
        (checked["entry_time"] >= domain_start)
        & (checked["exit_time"] <= domain_end)
    )
    rows = checked.loc[contained].reset_index(drop=True)
    return [
        {
            "entry": pd.Timestamp(row.entry_time).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exit": pd.Timestamp(row.exit_time).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "side": "LONG" if int(row.side) > 0 else "SHORT",
        }
        for row in rows.itertuples(index=False)
    ]


def _build_gross9_clock_artifact(
    clocks: Mapping[str, pd.DataFrame],
    *,
    source_support_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(clocks) != GROSS9_SLEEVES:
        raise RuntimeError("ESDI Gross9 artifact requires exactly five ordered clocks")
    support = {
        "path": str(source_support_binding.get("path", "")),
        "sha256": str(source_support_binding.get("sha256", "")),
        "manifest_hash": str(source_support_binding.get("manifest_hash", "")),
    }
    if (
        not support["path"]
        or not _is_sha256(support["sha256"])
        or not _is_sha256(support["manifest_hash"])
    ):
        raise RuntimeError("ESDI Gross9 source-support binding is malformed")
    reconstruction = authority.get("clock_reconstruction")
    expected_reconstruction = {
        "stage": "after ESDI source-support pass and before ESDI economics",
        "five_signed_sleeves_required": True,
        "exact_runtime_config_and_transitive_hash_validation_required": True,
        "failure_or_missing_dependency_is_terminal": True,
    }
    if reconstruction != expected_reconstruction:
        raise RuntimeError("ESDI Gross9 clock reconstruction authority drift")
    clocks_payload: dict[str, Any] = {}
    for sleeve in GROSS9_SLEEVES:
        intervals = _clock_intervals(clocks[sleeve], sleeve=sleeve)
        clocks_payload[sleeve] = {
            "intervals": intervals,
            "sha256": canonical_hash({"intervals": intervals}),
        }
    core = {
        "protocol_version": GROSS9_CLOCK_PROTOCOL,
        "policy_id": POLICY_ID,
        "preregistration": {
            "path": str(PREREGISTRATION_ARTIFACT),
            "sha256": PREREGISTRATION_ARTIFACT_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "source_support": support,
        "authority_hash": canonical_hash(authority),
        "clocks": clocks_payload,
        "frozen_contract_validation": {
            **expected_reconstruction,
            "source_support_passed_before_reconstruction": True,
            "exact_runtime_config_and_transitive_hash_validation_passed": True,
            "five_signed_sleeves_validated": True,
            "gross9_common_domain": [
                "2023-06-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "evidence_boundary": dict(GROSS9_CLOCK_EVIDENCE_BOUNDARY),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _utc_market_dates(market: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"], utc=True))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise RuntimeError("ESDI Gross9 runtime market clock is invalid")
    return dates


def _rows_from_trades(
    trades: Iterable[Any],
    dates: pd.DatetimeIndex,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    rows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for trade in trades:
        entry_position = int(trade.entry_position)
        exit_position = int(trade.exit_position)
        if (
            entry_position < 0
            or exit_position >= len(dates)
            or entry_position >= exit_position
            or int(trade.side) not in {-1, 1}
        ):
            raise RuntimeError("ESDI Gross9 runtime trade geometry drift")
        if hasattr(trade, "entry_date") and pd.Timestamp(
            pd.to_datetime(trade.entry_date, utc=True, errors="raise")
        ) != dates[entry_position]:
            raise RuntimeError("ESDI Gross9 runtime trade timestamp drift")
        rows.append(
            (
                dates[entry_position],
                dates[exit_position],
                int(trade.side),
            )
        )
    return rows


def _verify_exact_trade_replays(
    trades: Sequence[Any],
    replay: Callable[[Any], Any | None],
    *,
    sleeve: str,
) -> None:
    fields = (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
    )
    for trade in trades:
        try:
            expected = replay(trade)
            observed_geometry = tuple(int(getattr(trade, field)) for field in fields)
            expected_geometry = (
                None
                if expected is None
                else tuple(int(getattr(expected, field)) for field in fields)
            )
        except Exception as error:
            raise RuntimeError(
                f"ESDI {sleeve} exact trade replay failed"
            ) from error
        if expected_geometry != observed_geometry:
            raise RuntimeError(f"ESDI {sleeve} side/exit replay drift")


def _frame_from_clock_rows(
    rows: Iterable[tuple[pd.Timestamp, pd.Timestamp, int]],
    sleeve: str,
) -> pd.DataFrame:
    start, end = PERIODS["full"]
    selected = [
        row for row in rows if row[0] >= start and row[1] <= end
    ]
    selected.sort(key=lambda row: (row[0], row[1], row[2]))
    frame = pd.DataFrame.from_records(selected, columns=CLOCK_COLUMNS)
    if frame.empty:
        frame = pd.DataFrame(columns=CLOCK_COLUMNS)
    return validate_clock(frame, sleeve=sleeve)


def _verify_fixed_clock_geometry(
    rows: Sequence[tuple[pd.Timestamp, pd.Timestamp, int]],
    dates: pd.DatetimeIndex,
    *,
    sleeve: str,
    hold_bars: int,
    allowed_sides: set[int],
) -> None:
    for entry, exit_, side in rows:
        entry_position = int(dates.get_indexer([entry])[0])
        exit_position = int(dates.get_indexer([exit_])[0])
        if (
            entry_position < 0
            or exit_position - entry_position != hold_bars
            or side not in allowed_sides
        ):
            raise RuntimeError(f"ESDI {sleeve} side/exit geometry drift")


def _verify_event_entry_positions(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[tuple[pd.Timestamp, pd.Timestamp, int]],
    dates: pd.DatetimeIndex,
    masks: Mapping[str, np.ndarray],
    *,
    sleeve: str,
) -> None:
    actual = {str(split): [] for split in masks}
    for entry, _exit, _side in rows:
        entry_position = int(dates.get_indexer([entry])[0])
        signal_position = entry_position - 1
        matching = [
            str(split)
            for split, mask in masks.items()
            if 0 <= signal_position < len(mask) and bool(mask[signal_position])
        ]
        if len(matching) != 1:
            raise RuntimeError(f"ESDI {sleeve} split mapping drift")
        actual[matching[0]].append(entry_position)
    expected: dict[str, list[int]] = {}
    for split in masks:
        matches = [
            event
            for event in events
            if event.get("split") == split
            and event.get("sleeve") == sleeve
        ]
        if len(matches) > 1:
            raise RuntimeError(f"ESDI {sleeve} aggregate event drift")
        entries = [] if not matches else matches[0].get("entry_positions")
        if not isinstance(entries, list):
            raise RuntimeError(f"ESDI {sleeve} event entries are missing")
        expected[str(split)] = [int(position) for position in entries]
    if actual != expected:
        raise RuntimeError(f"ESDI {sleeve} reconstructed entries drift")


def _reconstruct_mask_long_clock(
    runtime: Any,
    market: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    active: np.ndarray,
    *,
    hold: int,
    stride: int,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, int]], dict[str, int]]:
    dates = _utc_market_dates(market)
    positions = np.arange(
        143, max(0, len(market) - hold - 2), stride, dtype=np.int64
    )
    rows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    counts: dict[str, int] = {}
    for split, split_mask in masks.items():
        next_allowed = 0
        count = 0
        for raw_position in positions[split_mask[positions] & active[positions]]:
            position = int(raw_position)
            if position < next_allowed:
                continue
            path = runtime.portfolio.new_alpha._event_path(
                market,
                position,
                side="long",
                hold=hold,
                cost_rate=BASE_COST_RATE,
                entry_delay=1,
                leverage=LEVERAGE,
            )
            if path is None:
                continue
            event_return = path[0]
            nonzero = np.flatnonzero(np.abs(event_return) > 1e-15)
            exit_position = (
                int(nonzero[-1]) if len(nonzero) else position + hold + 1
            )
            if exit_position >= len(split_mask) or not split_mask[exit_position]:
                continue
            rows.append((dates[position + 1], dates[exit_position], 1))
            next_allowed = exit_position + 1
            count += 1
        counts[str(split)] = count
    return rows, counts


def _reconstruct_rex_taker_clock(
    runtime: Any,
    market: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, int]], dict[str, int]]:
    paths = (
        "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
        "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
        "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl",
    )
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        for line in runtime.portfolio.resolve_existing(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                unique[(int(row["signal_pos"]), str(row["date"]))] = row
    source_rows = sorted(unique.values(), key=lambda row: int(row["signal_pos"]))
    dates = _utc_market_dates(market)
    rows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    counts: dict[str, int] = {}
    for split, split_mask in masks.items():
        next_allowed = 0
        count = 0
        for row in source_rows:
            position = int(row["signal_pos"])
            if (
                position < next_allowed
                or position >= len(split_mask)
                or not split_mask[position]
            ):
                continue
            if pd.Timestamp(
                pd.to_datetime(row["date"], utc=True, errors="raise")
            ) != dates[position]:
                raise RuntimeError("ESDI REX-taker source mapping drift")
            if not runtime.portfolio.rex_gate_match(
                row, list(runtime.portfolio.REX_GATES)
            ):
                continue
            side_text = str((row.get("action") or {}).get("side", "")).lower()
            if side_text not in {"long", "short"}:
                continue
            path = runtime.portfolio.new_alpha._event_path(
                market,
                position,
                side=side_text,
                hold=144,
                cost_rate=BASE_COST_RATE,
                entry_delay=1,
                leverage=LEVERAGE,
            )
            if path is None:
                continue
            exit_position = position + 145
            if exit_position >= len(split_mask) or not split_mask[exit_position]:
                continue
            nonzero = np.flatnonzero(np.abs(path[0]) > 1e-15)
            if not len(nonzero) or int(nonzero[-1]) != exit_position:
                raise RuntimeError("ESDI REX-taker exit replay drift")
            side = 1 if side_text == "long" else -1
            rows.append((dates[position + 1], dates[exit_position], side))
            next_allowed = exit_position + 1
            count += 1
        counts[str(split)] = count
    return rows, counts


def _reconstruct_rex_veto_clock(
    runtime: Any,
    market: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, int]], dict[str, int]]:
    report = runtime.legacy_all.load_json(
        runtime.legacy_all.SCAN_FILES["rex_veto"]
    )
    candidates = runtime._unique_rex_rows(report, 50)
    if runtime.FROZEN_REX_ROW_INDEX >= len(candidates):
        raise RuntimeError("ESDI frozen REX-veto row is missing")
    gate_row = candidates[runtime.FROZEN_REX_ROW_INDEX]
    source_path = _repository_path(
        "data/rex_event_reasoning_policy_sft_20260712.jsonl"
    )
    source = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features = runtime.legacy_all._build_light_rex_features(market)
    dates = _utc_market_dates(market)
    rows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    counts: dict[str, int] = {}
    for split, split_mask in masks.items():
        next_allowed = 0
        count = 0
        for source_row in source:
            position = int(source_row.get("signal_pos", -1))
            if (
                position < 0
                or position >= len(split_mask)
                or not split_mask[position]
                or position < next_allowed
            ):
                continue
            if pd.Timestamp(
                pd.to_datetime(source_row["date"], utc=True, errors="raise")
            ) != dates[position]:
                raise RuntimeError("ESDI REX-veto source mapping drift")
            side_text = str(
                (source_row.get("base_event") or {}).get("base_side", "")
            ).lower()
            if side_text not in {"long", "short"} or not (
                runtime.legacy_all._rex_row_matches(
                    gate_row.get("gates", []), features, source_row
                )
            ):
                continue
            exit_position = position + 145
            if exit_position >= len(split_mask) or not split_mask[exit_position]:
                continue
            path = runtime.portfolio.new_alpha._event_path(
                market,
                position,
                side=side_text,
                hold=144,
                cost_rate=BASE_COST_RATE,
                entry_delay=1,
                leverage=LEVERAGE,
            )
            if path is None:
                continue
            nonzero = np.flatnonzero(np.abs(path[0]) > 1e-15)
            if not len(nonzero) or int(nonzero[-1]) != exit_position:
                raise RuntimeError("ESDI REX-veto exit replay drift")
            side = 1 if side_text == "long" else -1
            rows.append((dates[position + 1], dates[exit_position], side))
            next_allowed = exit_position + 1
            count += 1
        counts[str(split)] = count
    return rows, counts


def _reconstruct_gross9_runtime_clocks() -> dict[str, pd.DataFrame]:
    """Concrete frozen-runtime adapter.  It is called only after support passes."""

    import training.audit_gross9_pullback_premium_overheat_marginal as runtime

    cfg = runtime.Config()
    market, masks, events, source_meta = runtime.build_full_context(cfg)
    dates = _utc_market_dates(market)

    features = runtime.portfolio.feature_frame(market)
    markov_active = runtime.portfolio.markov_active(market, features)
    markov_rows, markov_counts = _reconstruct_mask_long_clock(
        runtime,
        market,
        masks,
        markov_active,
        hold=576,
        stride=12,
    )
    if markov_counts != {
        str(key): int(value)
        for key, value in source_meta["markov_counts"].items()
    }:
        raise RuntimeError("ESDI Markov Gross9 clock count drift")
    _verify_fixed_clock_geometry(
        markov_rows,
        dates,
        sleeve="markov_transition_long",
        hold_bars=576,
        allowed_sides={1},
    )
    _verify_event_entry_positions(
        events,
        markov_rows,
        dates,
        masks,
        sleeve="markov_transition_long",
    )

    taker_rows, taker_counts = _reconstruct_rex_taker_clock(
        runtime, market, masks
    )
    if taker_counts != {
        str(key): int(value)
        for key, value in source_meta["rex_counts"].items()
    }:
        raise RuntimeError("ESDI REX-taker Gross9 clock count drift")
    _verify_fixed_clock_geometry(
        taker_rows,
        dates,
        sleeve="rex_taker_low_range_position",
        hold_bars=144,
        allowed_sides={-1, 1},
    )
    _verify_event_entry_positions(
        events,
        taker_rows,
        dates,
        masks,
        sleeve="rex_taker_low_range_position",
    )

    veto_rows, veto_counts = _reconstruct_rex_veto_clock(
        runtime, market, masks
    )
    event_veto_counts = {
        str(split): sum(
            int(event.get("trade_count", 0))
            for event in events
            if event.get("split") == split
            and event.get("sleeve") == "cand_rex_veto_7"
        )
        for split in masks
    }
    if veto_counts != event_veto_counts:
        raise RuntimeError("ESDI REX-veto Gross9 clock count drift")
    _verify_fixed_clock_geometry(
        veto_rows,
        dates,
        sleeve="cand_rex_veto_7",
        hold_bars=144,
        allowed_sides={-1, 1},
    )
    _verify_event_entry_positions(
        events,
        veto_rows,
        dates,
        masks,
        sleeve="cand_rex_veto_7",
    )

    audit_cfg = runtime.portfolio.FreshAuditConfig(
        input_csv=str(runtime.portfolio.resolve_existing(cfg.market_csv)),
        funding_csv=str(runtime.portfolio.resolve_existing(cfg.funding_csv)),
        premium_csv=str(runtime.portfolio.resolve_existing(cfg.premium_csv)),
        output="/tmp/no_write_esdi_gross9.json",
        docs_output="",
        exclude_from=runtime.portfolio.FULL_CUTOFF,
    )
    fresh = runtime.portfolio.build_candidate_context(audit_cfg)
    rank7 = runtime.portfolio.build_rank7_context(audit_cfg)
    fresh_dates = _utc_market_dates(fresh["market"])
    rank7_dates = _utc_market_dates(rank7["base"]["context"]["market"])
    if not dates.equals(fresh_dates) or not dates.equals(rank7_dates):
        raise RuntimeError("ESDI Rank7/Fresh Gross9 market grid drift")
    fresh_rows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    rank7_rows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    fresh_counts: dict[str, int] = {}
    rank7_counts: dict[str, int] = {}
    fresh_spec = runtime.portfolio.CANDIDATE_SPEC
    rank7_funding_leg = np.asarray(
        rank7["base"]["context"]["funding_leg"], dtype=bool
    )

    def replay_fresh(trade: Any) -> Any | None:
        signal = int(trade.signal_position)
        long_active = bool(fresh["long_active"][signal])
        short_active = bool(fresh["short_active"][signal])
        if long_active == short_active:
            raise RuntimeError("ESDI Fresh side source is not exclusive")
        side = 1 if long_active else -1
        return fresh["engine"].trade_at(
            signal,
            side,
            int(fresh_spec["hold_bars"]),
            int(fresh_spec["take_bps"]),
            int(fresh_spec["stop_bps"]),
        )

    def replay_rank7(trade: Any) -> Any | None:
        signal = int(trade.signal_position)
        hold, take, stop = runtime.portfolio.rank7_action_spec(
            bool(rank7_funding_leg[signal])
        )
        return rank7["base"]["engine"].trade_at(
            signal, 1, int(hold), int(take), int(stop)
        )

    for split, (start, end) in runtime.portfolio.SPLIT_BOUNDS.items():
        fresh_trades = runtime.portfolio.candidate_schedule(
            fresh, start=start, end=end
        )
        rank7_trades = runtime.portfolio.rank7_schedule(
            rank7, start=start, end=end
        )
        _verify_exact_trade_replays(
            fresh_trades,
            replay_fresh,
            sleeve="fresh_kimchi_fx",
        )
        _verify_exact_trade_replays(
            rank7_trades,
            replay_rank7,
            sleeve="frozen_annual_rank7",
        )
        fresh_rows.extend(_rows_from_trades(fresh_trades, dates))
        rank7_rows.extend(_rows_from_trades(rank7_trades, dates))
        fresh_counts[split] = len(fresh_trades)
        rank7_counts[split] = len(rank7_trades)
    expected_path_counts = source_meta["path_counts"]
    if fresh_counts != {
        str(key): int(value)
        for key, value in expected_path_counts["fresh_kimchi_fx"].items()
    } or rank7_counts != {
        str(key): int(value)
        for key, value in expected_path_counts["frozen_annual_rank7"].items()
    }:
        raise RuntimeError("ESDI Rank7/Fresh Gross9 clock count drift")
    _verify_event_entry_positions(
        events,
        fresh_rows,
        dates,
        masks,
        sleeve="fresh_kimchi_fx",
    )
    _verify_event_entry_positions(
        events,
        rank7_rows,
        dates,
        masks,
        sleeve="frozen_annual_rank7",
    )

    return {
        "cand_rex_veto_7": _frame_from_clock_rows(
            veto_rows, "cand_rex_veto_7"
        ),
        "fresh_kimchi_fx": _frame_from_clock_rows(
            fresh_rows, "fresh_kimchi_fx"
        ),
        "frozen_annual_rank7": _frame_from_clock_rows(
            rank7_rows, "frozen_annual_rank7"
        ),
        "markov_transition_long": _frame_from_clock_rows(
            markov_rows, "markov_transition_long"
        ),
        "rex_taker_low_range_position": _frame_from_clock_rows(
            taker_rows, "rex_taker_low_range_position"
        ),
    }


def _gross9_attempt_claim_payload(
    *,
    source_support_binding: Mapping[str, Any],
    evaluator_source: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": GROSS9_ATTEMPT_CLAIM_PROTOCOL,
        "policy_id": POLICY_ID,
        "status": "claimed_before_gross9_dependency_and_path_rows",
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "preregistration": {
            "path": str(PREREGISTRATION_ARTIFACT),
            "sha256": PREREGISTRATION_ARTIFACT_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "source_support": dict(source_support_binding),
        "evaluator_source_manifest_hash": str(
            evaluator_source["manifest_hash"]
        ),
        "canonical_output": str(GROSS9_CLOCK_ARTIFACT),
    }
    return {**core, "claim_hash": canonical_hash(core)}


def reconstruct_production_gross9_clocks(
    *,
    source_support_path: str | Path = SOURCE_SUPPORT_ARTIFACT,
    output: str | Path = GROSS9_CLOCK_ARTIFACT,
) -> dict[str, Any]:
    """Validate support first, then reconstruct and seal all five frozen clocks."""

    if (
        Path(source_support_path) != SOURCE_SUPPORT_ARTIFACT
        or Path(output) != GROSS9_CLOCK_ARTIFACT
    ):
        raise RuntimeError("ESDI production Gross9 reconstruction requires canonical paths")
    support = _load_passed_source_support(source_support_path)
    support_binding = _artifact_binding(source_support_path, support)
    evaluator_source = _validate_evaluator_source_identity()
    attempt_payload = _gross9_attempt_claim_payload(
        source_support_binding=support_binding,
        evaluator_source=evaluator_source,
    )
    claim_target = _repository_path(GROSS9_ATTEMPT_CLAIM)
    output_target = _repository_path(GROSS9_CLOCK_ARTIFACT)
    if claim_target.exists() or claim_target.is_symlink():
        _load_exact_attempt_claim(
            GROSS9_ATTEMPT_CLAIM,
            attempt_payload,
            label="Gross9 reconstruction",
        )
        if output_target.is_symlink() or not output_target.is_file():
            raise RuntimeError(
                "ESDI claimed Gross9 reconstruction lacks its completion artifact"
            )
        return load_gross9_clock_artifact(GROSS9_CLOCK_ARTIFACT)
    if output_target.exists() or output_target.is_symlink():
        raise RuntimeError(
            "ESDI Gross9 artifact exists without its attempt claim"
        )
    _create_exact_attempt_claim(
        GROSS9_ATTEMPT_CLAIM,
        attempt_payload,
        label="Gross9 reconstruction",
    )
    registration = load_bound_preregistration()
    validation = validate_frozen_contract(registration)
    clocks = _reconstruct_gross9_runtime_clocks()
    if tuple(clocks) != GROSS9_SLEEVES or any(
        clocks[sleeve].empty for sleeve in GROSS9_SLEEVES
    ):
        raise RuntimeError("ESDI production Gross9 reconstruction is incomplete")
    payload = _build_gross9_clock_artifact(
        clocks,
        source_support_binding=support_binding,
        authority=registration["gross9"]["authority"],
    )
    _validate_gross9_clock_artifact(
        payload,
        source_support_binding=support_binding,
        authority=registration["gross9"]["authority"],
    )
    if not validation["validated"]:
        raise RuntimeError("ESDI Gross9 frozen validation did not complete")
    write_once_result(_repository_path(output), payload)
    return payload


def build_gross9_clock_artifact(
    clocks: Mapping[str, pd.DataFrame],
    *,
    source_support_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic:
        raise RuntimeError("ESDI injected Gross9 clocks are synthetic-only")
    return _build_gross9_clock_artifact(
        clocks,
        source_support_binding=source_support_binding,
        authority=authority,
    )


def _validate_gross9_clock_artifact(
    payload: Mapping[str, Any],
    *,
    source_support_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    required_keys = {
        "protocol_version",
        "policy_id",
        "preregistration",
        "source_support",
        "authority_hash",
        "clocks",
        "frozen_contract_validation",
        "manifest_hash",
    }
    if set(payload) != required_keys:
        raise RuntimeError("ESDI Gross9 clock artifact schema drift")
    if payload.get("protocol_version") != GROSS9_CLOCK_PROTOCOL:
        raise RuntimeError("ESDI Gross9 clock protocol drift")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("ESDI Gross9 clock policy drift")
    _validate_receipt_manifest(payload)
    if payload.get("preregistration") != {
        "path": str(PREREGISTRATION_ARTIFACT),
        "sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
    }:
        raise RuntimeError("ESDI Gross9 preregistration binding drift")
    if payload.get("source_support") != {
        "path": str(source_support_binding.get("path", "")),
        "sha256": str(source_support_binding.get("sha256", "")),
        "manifest_hash": str(source_support_binding.get("manifest_hash", "")),
    }:
        raise RuntimeError("ESDI Gross9 source-support binding drift")
    if payload.get("authority_hash") != canonical_hash(authority):
        raise RuntimeError("ESDI Gross9 authority hash drift")
    expected_frozen = {
        **dict(authority["clock_reconstruction"]),
        "source_support_passed_before_reconstruction": True,
        "exact_runtime_config_and_transitive_hash_validation_passed": True,
        "five_signed_sleeves_validated": True,
        "gross9_common_domain": [
            "2023-06-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ],
        "evidence_boundary": dict(GROSS9_CLOCK_EVIDENCE_BOUNDARY),
    }
    if payload.get("frozen_contract_validation") != expected_frozen:
        raise RuntimeError("ESDI Gross9 frozen validation drift")
    raw_sleeves = payload.get("clocks")
    if not isinstance(raw_sleeves, Mapping) or tuple(raw_sleeves) != GROSS9_SLEEVES:
        raise RuntimeError("ESDI Gross9 artifact sleeves drifted")
    clocks: dict[str, pd.DataFrame] = {}
    for sleeve in GROSS9_SLEEVES:
        record = raw_sleeves[sleeve]
        if not isinstance(record, Mapping) or set(record) != {
            "intervals",
            "sha256",
        }:
            raise RuntimeError(f"ESDI Gross9 sleeve record invalid: {sleeve}")
        intervals = record.get("intervals")
        if not isinstance(intervals, list):
            raise RuntimeError(f"ESDI Gross9 sleeve intervals invalid: {sleeve}")
        if record.get("sha256") != canonical_hash({"intervals": intervals}):
            raise RuntimeError(f"ESDI Gross9 sleeve hash drift: {sleeve}")
        normalized: list[dict[str, Any]] = []
        for row in intervals:
            if not isinstance(row, Mapping) or set(row) != {
                "entry",
                "exit",
                "side",
            }:
                raise RuntimeError(f"ESDI Gross9 interval schema drift: {sleeve}")
            side = {"LONG": 1, "SHORT": -1}.get(str(row["side"]))
            if side is None:
                raise RuntimeError(f"ESDI Gross9 interval side drift: {sleeve}")
            normalized.append(
                {
                    "entry_time": row["entry"],
                    "exit_time": row["exit"],
                    "side": side,
                }
            )
        frame = pd.DataFrame.from_records(normalized)
        if frame.empty:
            frame = pd.DataFrame(columns=CLOCK_COLUMNS)
        else:
            frame = frame.loc[:, CLOCK_COLUMNS]
        checked = validate_clock(frame, sleeve=sleeve)
        if _clock_intervals(checked, sleeve=sleeve) != intervals:
            raise RuntimeError(f"ESDI Gross9 sleeve domain drift: {sleeve}")
        clocks[sleeve] = checked
    return clocks


def load_gross9_clock_artifact(
    path: str | Path,
    *,
    source_support_binding: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    if (source_support_binding is not None or authority is not None) and not synthetic:
        raise RuntimeError("ESDI injected Gross9 artifact bindings are synthetic-only")
    if synthetic and (
        source_support_binding is None or authority is None
    ):
        raise RuntimeError("ESDI synthetic Gross9 artifact bindings are required")
    if not synthetic:
        if Path(path) != GROSS9_CLOCK_ARTIFACT:
            raise RuntimeError("ESDI production Gross9 artifact path is canonical")
        support_payload = _load_passed_source_support(SOURCE_SUPPORT_ARTIFACT)
        registration = load_bound_preregistration()
        validate_frozen_contract(registration)
        authority = registration["gross9"]["authority"]
        source_support_binding = _artifact_binding(
            SOURCE_SUPPORT_ARTIFACT, support_payload
        )
    target = Path(path)
    if not synthetic and not target.is_absolute():
        target = REPOSITORY_ROOT / target
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("ESDI Gross9 clock artifact is missing or symlinked")
    target = target.resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("ESDI Gross9 clock artifact JSON is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("ESDI Gross9 clock artifact is not an object")
    _validate_gross9_clock_artifact(
        payload,
        source_support_binding=source_support_binding,
        authority=authority,
    )
    return payload


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"ESDI {label} is missing or symlinked")
    target = candidate.resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ESDI {label} JSON is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"ESDI {label} is not an object")
    return payload


def _artifact_binding(path: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    manifest_hash = payload.get("manifest_hash")
    if not _is_sha256(manifest_hash):
        raise RuntimeError("ESDI artifact manifest binding is missing")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_hash": manifest_hash,
    }


def _load_passed_source_support(
    path: str | Path = SOURCE_SUPPORT_ARTIFACT,
) -> dict[str, Any]:
    if Path(path) != SOURCE_SUPPORT_ARTIFACT:
        raise RuntimeError("ESDI production source-support path is canonical")
    from training import (
        evaluate_ethereum_settlement_demand_impulse_novelty as novelty_module,
    )

    try:
        verified = novelty_module.load_passed_source_support(
            path,
            production=True,
        )
    except Exception as error:
        raise RuntimeError(
            "ESDI exact canonical source support did not validate"
        ) from error
    return json.loads(verified.raw_bytes)


def _validate_passed_novelty(
    payload: Mapping[str, Any],
    *,
    exact: bool = False,
) -> None:
    manifest_hash = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if manifest_hash != canonical_hash(core):
        raise RuntimeError("ESDI novelty artifact manifest drift")
    if (
        payload.get("protocol_version") != NOVELTY_PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
    ):
        raise RuntimeError("ESDI novelty policy drift")
    novelty = payload.get("novelty")
    passed = (
        novelty.get("passed")
        if isinstance(novelty, Mapping)
        else payload.get("passed")
    )
    if passed is not True:
        raise RuntimeError("ESDI novelty did not pass before economic rows")
    evidence_boundary = payload.get("evidence_boundary")
    if isinstance(evidence_boundary, Mapping) and (
        evidence_boundary.get("candidate_market_rows_opened") is not False
        or evidence_boundary.get("candidate_outcome_rows_opened") is not False
        or evidence_boundary.get("future_rows_used_for_economic_weight_ranking")
        is not False
        or evidence_boundary.get(
            "future_rows_used_for_structural_candidate_veto"
        )
        is not True
        or evidence_boundary.get("portfolio_return_or_pnl_metrics_computed")
        is not False
    ):
        raise RuntimeError("ESDI novelty evidence boundary drift")
    if exact:
        required = {
            "protocol_version",
            "policy_id",
            "preregistration",
            "attempt_claim",
            "source_support",
            "gross9_clock_artifact",
            "registry_artifacts",
            "registry_comparator_groups",
            "novelty",
            "evidence_boundary",
            "manifest_hash",
        }
        if set(payload) != required:
            raise RuntimeError("ESDI novelty exact schema drift")
        if (
            evidence_boundary != NOVELTY_EVIDENCE_BOUNDARY
            or not isinstance(payload.get("attempt_claim"), Mapping)
            or set(payload["attempt_claim"])
            != {"path", "sha256", "claim_hash"}
            or payload["attempt_claim"].get("path")
            != (
                "results/"
                "ethereum_settlement_demand_impulse_"
                "novelty_attempt_claim_2026-07-30.json"
            )
            or not _is_sha256(payload["attempt_claim"].get("sha256"))
            or not _is_sha256(payload["attempt_claim"].get("claim_hash"))
            or type(payload.get("registry_artifacts")) is not int
            or type(payload.get("registry_comparator_groups")) is not int
        ):
            raise RuntimeError("ESDI novelty evidence boundary drift")
        if set(novelty) != {
            "prior_source_comparators",
            "gross9_sleeves",
            "passed",
            "terminal",
            "failed_checks",
        }:
            raise RuntimeError("ESDI novelty result schema drift")
        prior_rows = novelty["prior_source_comparators"]
        gross9_rows = novelty["gross9_sleeves"]
        if (
            novelty.get("terminal") is not False
            or novelty.get("failed_checks") != []
            or not isinstance(prior_rows, list)
            or len(prior_rows) != payload["registry_comparator_groups"]
            or not isinstance(gross9_rows, list)
            or any(not isinstance(row, Mapping) for row in gross9_rows)
            or [row.get("sleeve") for row in gross9_rows]
            != list(GROSS9_SLEEVES)
        ):
            raise RuntimeError("ESDI novelty passing result drift")
        for row in prior_rows:
            if (
                not isinstance(row, Mapping)
                or row.get("passed") is not True
                or (
                    row.get("gating") is True
                    and (
                        not isinstance(row.get("checks"), Mapping)
                        or any(value is not True for value in row["checks"].values())
                    )
                )
            ):
                raise RuntimeError("ESDI novelty prior-comparator pass drift")
        for sleeve, row in zip(GROSS9_SLEEVES, gross9_rows):
            if (
                not isinstance(row, Mapping)
                or row.get("passed") is not True
                or float(row.get("weight", -1.0)) != GROSS9_WEIGHTS[sleeve]
                or not isinstance(row.get("checks"), Mapping)
                or any(value is not True for value in row["checks"].values())
            ):
                raise RuntimeError("ESDI novelty Gross9 pass drift")


def _load_passed_novelty(
    path: str | Path = NOVELTY_ARTIFACT,
) -> dict[str, Any]:
    if Path(path) != NOVELTY_ARTIFACT:
        raise RuntimeError("ESDI production novelty path is canonical")
    from training import (
        evaluate_ethereum_settlement_demand_impulse_novelty as novelty_module,
    )
    try:
        payload = novelty_module.load_reproduced_novelty_for_economics(path)
    except Exception as error:
        raise RuntimeError(
            "ESDI committed novelty report did not authenticate and reproduce"
        ) from error
    _validate_passed_novelty(payload, exact=True)
    return payload


def _validate_pre2025_anchor(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    binding = registration["gross9"]["authority"].get("pre2025_anchor")
    if not isinstance(binding, Mapping):
        raise RuntimeError("ESDI pre-2025 Gross9 anchor binding is missing")
    path = str(binding.get("path", ""))
    expected_sha = str(binding.get("sha256", ""))
    if not path or sha256_file(path) != expected_sha:
        raise RuntimeError("ESDI pre-2025 Gross9 anchor SHA drift")
    anchor = _read_json_object(path, "pre-2025 Gross9 anchor")
    exact_values = {
        "name": "gross9_pre2025_authoritative_anchor",
        "future_metrics_present": False,
        "accounting_version": "same_btc_low_high_v1",
        "selection_mode": "frozen_pre2025_allocation_rank_future_veto_only",
        "future_used_for_allocation_ranking": False,
        "future_can_only_veto_frozen_rank1": True,
    }
    if any(anchor.get(key) != value for key, value in exact_values.items()):
        raise RuntimeError("ESDI pre-2025 Gross9 anchor contract drift")
    raw_weights = anchor.get("weights")
    if not isinstance(raw_weights, Mapping):
        raise RuntimeError("ESDI pre-2025 Gross9 anchor weights are missing")
    weights = {
        str(name): float(value) for name, value in raw_weights.items()
    }
    if weights != GROSS9_WEIGHTS or not math.isclose(
        float(anchor.get("gross", -1.0)),
        BASELINE_GROSS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("ESDI pre-2025 Gross9 anchor weight drift")
    selection_stats = anchor.get("selection_stats")
    if not isinstance(selection_stats, Mapping) or set(selection_stats) != {
        "train",
        "test2024",
    }:
        raise RuntimeError("ESDI pre-2025 Gross9 anchor selection drift")
    metric_names = {
        "absolute_return_pct",
        "cagr_pct",
        "strict_mdd_pct",
        "cagr_to_strict_mdd",
    }
    for split, metrics in selection_stats.items():
        if (
            not isinstance(metrics, Mapping)
            or not metric_names.issubset(metrics)
            or type(metrics.get("trades")) is not int
            or metrics["trades"] < 0
            or not all(
                math.isfinite(float(metrics[name])) for name in metric_names
            )
        ):
            raise RuntimeError(
                f"ESDI pre-2025 Gross9 anchor metrics drift: {split}"
            )
    return {
        "path": path,
        "sha256": expected_sha,
        "payload_hash": canonical_hash(anchor),
        "selection_windows": ["train", "test2024"],
        "future_metrics_present": False,
    }


def _validate_production_economic_prerequisites() -> dict[str, Any]:
    """Revalidate the current frozen closure before any economic row loader."""

    registration = load_bound_preregistration()
    evaluator_source = _validate_evaluator_source_identity()
    frozen = validate_frozen_contract(registration)
    return {
        "frozen_contract_validation_hash": canonical_hash(frozen),
        "evaluator_source": evaluator_source,
    }


def _open_text(path: str | Path) -> Any:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"ESDI staged source is not a regular file: {path}")
    target = candidate.resolve()
    if target.suffix == ".gz":
        return gzip.open(target, "rt", encoding="utf-8", newline="")
    return target.open("rt", encoding="utf-8", newline="")


def _stream_csv_prefix(
    path: str | Path,
    *,
    time_column: str,
    cutoff: pd.Timestamp,
    include_cutoff: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    with _open_text(path) as handle:
        header_line = handle.readline()
        try:
            header = next(csv.reader([header_line]))
        except (StopIteration, csv.Error) as error:
            raise RuntimeError("ESDI staged CSV header drift") from error
        if (
            not header
            or len(header) != len(set(header))
            or header[0] != time_column
        ):
            raise RuntimeError("ESDI staged CSV header drift")
        previous: pd.Timestamp | None = None
        while True:
            timestamp_characters: list[str] = []
            while True:
                character = handle.read(1)
                if character == "":
                    if timestamp_characters:
                        raise RuntimeError("ESDI staged CSV row framing drift")
                    return header, rows
                if character == ",":
                    break
                if character in "\r\n" or character == '"':
                    raise RuntimeError(
                        "ESDI staged CSV first timestamp field is not canonical"
                    )
                timestamp_characters.append(character)
            timestamp_text = "".join(timestamp_characters)
            timestamp = pd.Timestamp(
                pd.to_datetime(timestamp_text, utc=True, errors="raise")
            )
            if previous is not None and timestamp <= previous:
                raise RuntimeError("ESDI staged CSV timestamps are not increasing")
            beyond = timestamp > cutoff if include_cutoff else timestamp >= cutoff
            if beyond:
                return header, rows
            remainder = handle.readline()
            if not remainder:
                raise RuntimeError("ESDI staged CSV row framing drift")
            try:
                values = next(
                    csv.reader([f"{timestamp_text},{remainder}"])
                )
            except (StopIteration, csv.Error) as error:
                raise RuntimeError("ESDI staged CSV row drift") from error
            if len(values) != len(header):
                raise RuntimeError("ESDI staged CSV row width drift")
            row = dict(zip(header, values))
            rows.append({str(key): str(value) for key, value in row.items()})
            previous = timestamp
            if include_cutoff and timestamp == cutoff:
                return header, rows


def _load_market_prefix(path: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    header, rows = _stream_csv_prefix(
        path,
        time_column="date",
        cutoff=cutoff,
        include_cutoff=True,
    )
    required = {"date", "open", "high", "low"}
    if not required.issubset(header):
        raise RuntimeError("ESDI market prefix columns drift")
    if not rows:
        raise RuntimeError("ESDI market prefix is empty")
    frame = pd.DataFrame.from_records(rows)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["date"], utc=True, errors="raise"),
            "open": pd.to_numeric(frame["open"], errors="raise"),
            "high": pd.to_numeric(frame["high"], errors="raise"),
            "low": pd.to_numeric(frame["low"], errors="raise"),
        },
        columns=MARKET_COLUMNS,
    )


def _load_funding_prefix(path: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    header, rows = _stream_csv_prefix(
        path,
        time_column="date",
        cutoff=cutoff,
        include_cutoff=False,
    )
    mark_column = next(
        (
            name
            for name in ("settlement_mark_price", "settlement_mark", "mark_price")
            if name in header
        ),
        None,
    )
    if "funding_rate" not in header or mark_column is None:
        raise RuntimeError("ESDI funding prefix lost settlement mark")
    if not rows:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    frame = pd.DataFrame.from_records(rows)
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                frame["date"], utc=True, errors="raise"
            ),
            "funding_rate": pd.to_numeric(
                frame["funding_rate"], errors="raise"
            ),
            "settlement_mark": pd.to_numeric(
                frame[mark_column], errors="raise"
            ),
        },
        columns=FUNDING_COLUMNS,
    )


def _source_paths_from_authority(
    registration: Mapping[str, Any],
) -> tuple[str, str]:
    authority = registration["gross9"]["authority"]
    manifest = json.loads(
        _repository_path(
            authority["transitive_source_manifest"]["path"]
        ).read_text(encoding="utf-8")
    )
    by_name = {
        str(row["name"]): str(row["path"])
        for row in manifest.get("sources", [])
        if isinstance(row, Mapping)
    }
    if "market_5m" not in by_name or "funding" not in by_name:
        raise RuntimeError("ESDI Gross9 source manifest lacks market/funding")
    return by_name["market_5m"], by_name["funding"]


def _clock_csv_rows(
    path: str | Path,
    expected_hash: str,
) -> list[dict[str, str]]:
    target = _repository_path(path)
    if sha256_file(target) != expected_hash:
        raise RuntimeError("ESDI source-support clock hash drift")
    with _open_text(target) as handle:
        reader = csv.DictReader(handle)
        required = {
            "policy_id",
            "control",
            "entry_time_utc",
            "exit_time_utc",
            "side",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("ESDI source-support clock columns drift")
        rows = [dict(row) for row in reader]
    return rows


def _clock_frame_from_csv_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    control: str,
) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("policy_id") != POLICY_ID or row.get("control") != control:
            continue
        side = {"LONG": 1, "SHORT": -1}.get(str(row.get("side")))
        if side is None:
            raise RuntimeError("ESDI source-support clock side drift")
        selected.append(
            {
                "entry_time": row["entry_time_utc"],
                "exit_time": row["exit_time_utc"],
                "side": side,
            }
        )
    return validate_clock(
        pd.DataFrame.from_records(selected, columns=CLOCK_COLUMNS),
        sleeve=control,
    )


def _load_esdi_clocks_from_support(
    support: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    artifacts = support.get("clock_artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("ESDI source-support clock bindings are missing")
    from training import (
        evaluate_ethereum_settlement_demand_impulse_source_support as support_module,
    )

    primary_rows = _clock_csv_rows(
        support_module.DEFAULT_PRIMARY_CLOCK_OUTPUT,
        str(artifacts.get("primary_sha256", "")),
    )
    control_rows = _clock_csv_rows(
        support_module.DEFAULT_CONTROL_CLOCK_OUTPUT,
        str(artifacts.get("controls_sha256", "")),
    )
    primary = _clock_frame_from_csv_rows(primary_rows, control="primary")
    controls = {
        name: _clock_frame_from_csv_rows(control_rows, control=name)
        for name in CONTROL_NAMES
    }
    return primary, controls


def _production_stage_loader(
    stage: str,
    cutoff: pd.Timestamp,
    *,
    novelty: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if stage not in ECONOMIC_STAGE_ORDER or cutoff != STAGE_CUTOFFS[stage]:
        raise RuntimeError("ESDI production stage cutoff drift")
    if novelty is not None:
        support_binding = novelty.get("source_support")
        gross9_binding = novelty.get("gross9_clock_artifact")
        if (
            not isinstance(support_binding, Mapping)
            or support_binding.get("sha256")
            != sha256_file(SOURCE_SUPPORT_ARTIFACT)
            or not isinstance(gross9_binding, Mapping)
            or gross9_binding.get("sha256") != sha256_file(GROSS9_CLOCK_ARTIFACT)
        ):
            raise RuntimeError("ESDI staged artifact hash changed after novelty")
    registration = load_bound_preregistration()
    support = _load_passed_source_support(SOURCE_SUPPORT_ARTIFACT)
    primary, controls = _load_esdi_clocks_from_support(support)
    support_binding = _artifact_binding(SOURCE_SUPPORT_ARTIFACT, support)
    clock_payload = _read_json_object(
        GROSS9_CLOCK_ARTIFACT, "Gross9 clock artifact"
    )
    gross9 = _validate_gross9_clock_artifact(
        clock_payload,
        source_support_binding=support_binding,
        authority=registration["gross9"]["authority"],
    )
    market_path, funding_path = _source_paths_from_authority(registration)
    return {
        "stage": stage,
        "market": _load_market_prefix(market_path, cutoff),
        "funding": _load_funding_prefix(funding_path, cutoff),
        "primary_clock": primary,
        "control_clocks": controls,
        "gross9_clocks": gross9,
    }


def _evaluate_same_gross_future_period(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    gross9_clocks: Mapping[str, pd.DataFrame],
    esdi_clock: pd.DataFrame,
    weight: float,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    treatment_weights = same_gross_weights(weight)
    clocks = {**gross9_clocks, "esdi": esdi_clock}
    costs: dict[str, Any] = {}
    passes = True
    mdd_reduced = False
    for name, rate in (("base", BASE_COST_RATE), ("stress", STRESS_COST_RATE)):
        baseline = _simulate_portfolio(
            market,
            funding,
            gross9_clocks,
            GROSS9_WEIGHTS,
            start=start,
            end=end,
            cost_rate=rate,
        )
        treatment = _simulate_portfolio(
            market,
            funding,
            clocks,
            treatment_weights,
            start=start,
            end=end,
            cost_rate=rate,
        )
        checks = _same_gross_period_checks(treatment, baseline)
        passes = passes and all(checks.values())
        mdd_reduced = mdd_reduced or (
            float(treatment["strict_mdd"]) < float(baseline["strict_mdd"])
        )
        costs[name] = {
            "treatment": treatment,
            "unscaled_gross9": baseline,
            "checks": checks,
        }
    return {
        "candidate_weight": weight,
        "costs": costs,
        "strict_mdd_reduced": mdd_reduced,
        "passes": bool(passes and mdd_reduced),
    }


def _production_stage_evaluator(
    stage: str,
    inputs: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    market = inputs["market"]
    funding = inputs["funding"]
    primary = inputs["primary_clock"]
    controls = inputs["control_clocks"]
    gross9 = inputs["gross9_clocks"]
    if stage in {"2023H2", "2024", "selection"}:
        start, end = PERIODS[stage]
        standalone = _evaluate_standalone_period_with_controls(
            market,
            funding,
            primary,
            controls,
            start=start,
            end=end,
        )
        return {
            "passed": bool(standalone["passes"]),
            "standalone": standalone,
        }
    if stage == "same_gross":
        rows = [
            _evaluate_same_gross_weight(
                market,
                funding,
                gross9,
                primary,
                weight,
                periods=SELECTION_PERIODS,
            )
            for weight in CANDIDATE_WEIGHTS
        ]
        ranked = _rank_same_gross_treatments(
            rows,
            require_passing_freeze=False,
        )
        frozen = ranked[0]
        if frozen.get("passes") is not True:
            return {
                "passed": False,
                "frozen_weight": None,
                "frozen_rank": None,
                "ranking": ranked,
            }
        return {
            "passed": True,
            "frozen_weight": float(frozen["candidate_weight"]),
            "frozen_rank": 1,
            "ranking": ranked,
        }
    if stage in {"future25", "future26"}:
        weight = float(state.get("frozen_weight", -1.0))
        if weight not in CANDIDATE_WEIGHTS:
            raise RuntimeError("ESDI future stage lacks frozen selection weight")
        start, end = PERIODS[stage]
        standalone = _evaluate_standalone_period_with_controls(
            market,
            funding,
            primary,
            controls,
            start=start,
            end=end,
        )
        same_gross = _evaluate_same_gross_future_period(
            market,
            funding,
            gross9,
            primary,
            weight,
            start=start,
            end=end,
        )
        return {
            "passed": bool(standalone["passes"] and same_gross["passes"]),
            "frozen_weight": weight,
            "standalone": standalone,
            "same_gross": same_gross,
        }
    if stage == "full":
        weight = float(state.get("frozen_weight", -1.0))
        if weight not in CANDIDATE_WEIGHTS:
            raise RuntimeError("ESDI stitched full stage lacks frozen weight")
        start, end = PERIODS["full"]
        standalone = _evaluate_standalone_period_with_controls(
            market,
            funding,
            primary,
            controls,
            start=start,
            end=end,
        )
        same_gross = _evaluate_same_gross_future_period(
            market,
            funding,
            gross9,
            primary,
            weight,
            start=start,
            end=end,
        )
        return {
            "passed": bool(standalone["passes"] and same_gross["passes"]),
            "frozen_weight": weight,
            "standalone": standalone,
            "same_gross": same_gross,
        }
    raise RuntimeError(f"ESDI unknown economic stage: {stage}")


def _validated_frozen_selection_weight(result: Mapping[str, Any]) -> float:
    weight = float(result.get("frozen_weight", -1.0))
    ranking = result.get("ranking")
    if (
        result.get("passed") is not True
        or int(result.get("frozen_rank", 0)) != 1
        or weight not in CANDIDATE_WEIGHTS
        or not isinstance(ranking, list)
        or len(ranking) != len(CANDIDATE_WEIGHTS)
    ):
        raise RuntimeError("ESDI same-gross did not produce passed frozen rank one")
    observed_weights: list[float] = []
    for expected_rank, row in enumerate(ranking, start=1):
        if not isinstance(row, Mapping):
            raise RuntimeError(
                "ESDI same-gross did not produce passed frozen rank one"
            )
        observed_weights.append(float(row.get("candidate_weight", -1.0)))
        if (
            int(row.get("rank", 0)) != expected_rank
            or bool(row.get("frozen")) != (expected_rank == 1)
        ):
            raise RuntimeError(
                "ESDI same-gross did not produce passed frozen rank one"
            )
    frozen = ranking[0]
    if (
        frozen.get("passes") is not True
        or float(frozen.get("candidate_weight", -1.0)) != weight
        or sorted(observed_weights) != list(CANDIDATE_WEIGHTS)
    ):
        raise RuntimeError("ESDI same-gross did not produce passed frozen rank one")
    return weight


def _stage_attempt_claim_payload(
    *,
    stage: str,
    novelty_manifest_hash: str,
    prior_receipt_sha256: str | None,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": ECONOMIC_STAGE_ATTEMPT_CLAIM_PROTOCOL,
        "policy_id": POLICY_ID,
        "status": "claimed_before_stage_rows",
        "stage": stage,
        "cutoff_exclusive": STAGE_CUTOFFS[stage].isoformat(),
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "novelty_manifest_hash": novelty_manifest_hash,
        "prior_receipt_sha256": prior_receipt_sha256,
        "canonical_receipt": STAGE_RECEIPT_NAMES[stage],
    }
    for name in (
        "frozen_weight",
        "selection_receipt_sha256",
        "frozen_contract_validation_hash",
        "evaluator_source_manifest_hash",
        "pre2025_anchor_sha256",
    ):
        if name in state:
            core[name] = _json_ready(state[name])
    return {**core, "claim_hash": canonical_hash(core)}


def _load_completed_stage_receipt(
    path: Path,
    *,
    stage: str,
    attempt_claim: Mapping[str, Any],
    novelty_manifest_hash: str,
    prior_receipt_sha256: str | None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"ESDI claimed {stage} stage lacks its completion receipt"
        )
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ESDI {stage} receipt JSON is invalid") from error
    if (
        not isinstance(receipt, Mapping)
        or raw != canonical_json_bytes(receipt)
    ):
        raise RuntimeError(f"ESDI {stage} receipt serialization drift")
    _validate_receipt_manifest(receipt)
    if (
        receipt.get("protocol_version") != ECONOMIC_RECEIPT_PROTOCOL
        or receipt.get("policy_id") != POLICY_ID
        or receipt.get("execution_mode") != "production"
        or receipt.get("stage") != stage
        or receipt.get("cutoff_exclusive")
        != STAGE_CUTOFFS[stage].isoformat()
        or receipt.get("novelty_manifest_hash") != novelty_manifest_hash
        or receipt.get("prior_receipt_sha256") != prior_receipt_sha256
        or receipt.get("attempt_claim") != dict(attempt_claim)
        or not isinstance(receipt.get("result"), Mapping)
        or receipt.get("passed")
        is not bool(receipt["result"].get("passed"))
    ):
        raise RuntimeError(f"ESDI {stage} completion receipt binding drift")
    return dict(receipt)


def _stage_receipt(
    *,
    stage: str,
    result: Mapping[str, Any],
    novelty_manifest_hash: str,
    prior_receipt_sha256: str | None,
    state: Mapping[str, Any],
    synthetic: bool,
    attempt_claim: Mapping[str, Any],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": ECONOMIC_RECEIPT_PROTOCOL,
        "policy_id": POLICY_ID,
        "execution_mode": "synthetic_only" if synthetic else "production",
        "attempt_claim": dict(attempt_claim),
        "stage": stage,
        "cutoff_exclusive": STAGE_CUTOFFS[stage].isoformat(),
        "passed": bool(result.get("passed")),
        "novelty_manifest_hash": novelty_manifest_hash,
        "prior_receipt_sha256": prior_receipt_sha256,
        "result": _json_ready(result),
    }
    if "frozen_weight" in state:
        core["frozen_weight"] = float(state["frozen_weight"])
    if "selection_receipt_sha256" in state:
        core["selection_receipt_sha256"] = str(
            state["selection_receipt_sha256"]
        )
    if "frozen_contract_validation_hash" in state:
        core["frozen_contract_validation_hash"] = str(
            state["frozen_contract_validation_hash"]
        )
    if "pre2025_anchor_sha256" in state:
        core["pre2025_anchor_sha256"] = str(state["pre2025_anchor_sha256"])
    if "evaluator_source_manifest_hash" in state:
        core["evaluator_source_manifest_hash"] = str(
            state["evaluator_source_manifest_hash"]
        )
    return {**core, "manifest_hash": canonical_hash(core)}


def run_staged_economics(
    *,
    synthetic: bool = False,
    novelty: Mapping[str, Any] | None = None,
    stage_loader: Callable[[str, pd.Timestamp], Mapping[str, Any]] | None = None,
    stage_evaluator: Callable[
        [str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ]
    | None = None,
    receipt_root: str | Path = REPOSITORY_ROOT / "results",
) -> dict[str, Any]:
    """Run the immutable economic sequence, loading no stage before authorization."""

    injected = novelty is not None or stage_loader is not None or stage_evaluator is not None
    if injected and not synthetic:
        raise RuntimeError("ESDI injected novelty/loaders/evaluators are synthetic-only")
    if synthetic and (
        novelty is None or stage_loader is None or stage_evaluator is None
    ):
        raise RuntimeError("ESDI synthetic staged runner requires all injections")
    if not synthetic:
        supplied_receipt_root = Path(receipt_root)
        if supplied_receipt_root.is_symlink() or supplied_receipt_root.resolve() != (
            REPOSITORY_ROOT / "results"
        ).resolve():
            raise RuntimeError(
                "ESDI production economic receipts require canonical paths"
            )
    else:
        canonical_results = (REPOSITORY_ROOT / "results").resolve()
        supplied_receipt_root = Path(receipt_root).resolve()
        if supplied_receipt_root == canonical_results or (
            supplied_receipt_root.is_relative_to(canonical_results)
        ):
            raise RuntimeError(
                "ESDI synthetic receipts cannot use canonical results paths"
            )
    active_novelty = (
        dict(novelty)
        if novelty is not None
        else _load_passed_novelty(NOVELTY_ARTIFACT)
    )
    # This check intentionally precedes the first loader call.
    _validate_passed_novelty(active_novelty)
    prerequisites = (
        {}
        if synthetic
        else _validate_production_economic_prerequisites()
    )
    if stage_loader is not None:
        loader = stage_loader
    elif synthetic:
        raise RuntimeError("ESDI synthetic staged loader is missing")
    else:
        loader = lambda stage, cutoff: _production_stage_loader(
            stage,
            cutoff,
            novelty=active_novelty,
        )
    evaluator = stage_evaluator or _production_stage_evaluator
    root = Path(receipt_root)
    state: dict[str, Any] = {}
    if prerequisites:
        state["frozen_contract_validation_hash"] = prerequisites[
            "frozen_contract_validation_hash"
        ]
        state["evaluator_source_manifest_hash"] = prerequisites[
            "evaluator_source"
        ]["manifest_hash"]
    receipts: dict[str, Any] = {}
    previous_sha: str | None = None

    for stage in ECONOMIC_STAGE_ORDER:
        if stage in {"future25", "future26", "full"}:
            prerequisite_stage = {
                "future25": "same_gross",
                "future26": "future25",
                "full": "future26",
            }[stage]
            prerequisite = receipts.get(prerequisite_stage)
            if (
                not isinstance(prerequisite, Mapping)
                or prerequisite.get("passed") is not True
            ):
                raise RuntimeError(
                    f"ESDI {stage} cannot load before hash-bound "
                    f"{prerequisite_stage} pass"
                )
            _validate_receipt_manifest(prerequisite)
            prerequisite_path = root / STAGE_RECEIPT_NAMES[prerequisite_stage]
            expected_sha = (
                state.get("selection_receipt_sha256")
                if stage == "future25"
                else previous_sha
            )
            prerequisite_bytes = (
                prerequisite_path.read_bytes()
                if prerequisite_path.is_file()
                and not prerequisite_path.is_symlink()
                else None
            )
            if (
                prerequisite_bytes is None
                or prerequisite_bytes != canonical_json_bytes(prerequisite)
                or hashlib.sha256(prerequisite_bytes).hexdigest() != expected_sha
            ):
                raise RuntimeError(
                    f"ESDI {stage} requires hash-bound "
                    f"{prerequisite_stage} receipt bytes"
                )
        attempt_payload = _stage_attempt_claim_payload(
            stage=stage,
            novelty_manifest_hash=str(active_novelty["manifest_hash"]),
            prior_receipt_sha256=previous_sha,
            state=state,
        )
        if synthetic:
            attempt_claim: Mapping[str, Any] = {
                "mode": "synthetic_only",
                "stage": stage,
            }
        else:
            claim_relative = Path("results") / STAGE_ATTEMPT_CLAIM_NAMES[stage]
            claim_path = root / STAGE_ATTEMPT_CLAIM_NAMES[stage]
            receipt_path = root / STAGE_RECEIPT_NAMES[stage]
            claim_exists = claim_path.exists() or claim_path.is_symlink()
            receipt_exists = (
                receipt_path.exists() or receipt_path.is_symlink()
            )
            if claim_exists != receipt_exists:
                raise RuntimeError(
                    f"ESDI {stage} attempt claim/completion is incomplete"
                )
            if claim_exists:
                attempt_claim = _load_exact_attempt_claim(
                    claim_relative,
                    attempt_payload,
                    label=f"{stage} economic stage",
                )
                receipt = _load_completed_stage_receipt(
                    receipt_path,
                    stage=stage,
                    attempt_claim=attempt_claim,
                    novelty_manifest_hash=str(
                        active_novelty["manifest_hash"]
                    ),
                    prior_receipt_sha256=previous_sha,
                )
                result = dict(receipt["result"])
                if stage == "same_gross" and receipt.get("passed") is True:
                    state["frozen_weight"] = (
                        _validated_frozen_selection_weight(result)
                    )
                for name in (
                    "pre2025_anchor_sha256",
                    "frozen_weight",
                    "selection_receipt_sha256",
                ):
                    if name in receipt:
                        state[name] = receipt[name]
                receipt_sha = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()
                receipts[stage] = receipt
                previous_sha = receipt_sha
                if stage == "same_gross":
                    state["selection_receipt_sha256"] = receipt_sha
                if not receipt["passed"]:
                    return {
                        "passed": False,
                        "stopped_at": stage,
                        "completed_stages": list(receipts),
                        **state,
                    }
                continue
            attempt_claim = _create_exact_attempt_claim(
                claim_relative,
                attempt_payload,
                label=f"{stage} economic stage",
            )
        if stage == "same_gross" and not synthetic:
            anchor = _validate_pre2025_anchor(load_bound_preregistration())
            state["pre2025_anchor_sha256"] = anchor["sha256"]
        inputs = loader(stage, STAGE_CUTOFFS[stage])
        result = dict(evaluator(stage, inputs, dict(state)))
        if stage == "same_gross" and result.get("passed") is True:
            weight = _validated_frozen_selection_weight(result)
            state["frozen_weight"] = weight
        if stage in {"future25", "future26", "full"}:
            if "frozen_weight" not in state or float(
                result.get("frozen_weight", -1.0)
            ) != float(state["frozen_weight"]):
                raise RuntimeError("ESDI future stage changed frozen weight")
        receipt = _stage_receipt(
            stage=stage,
            result=result,
            novelty_manifest_hash=str(active_novelty["manifest_hash"]),
            prior_receipt_sha256=previous_sha,
            state=state,
            synthetic=synthetic,
            attempt_claim=attempt_claim,
        )
        path = root / STAGE_RECEIPT_NAMES[stage]
        write_once_result(path, receipt)
        receipt_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        receipts[stage] = receipt
        previous_sha = receipt_sha
        if stage == "same_gross":
            state["selection_receipt_sha256"] = receipt_sha
        if not receipt["passed"]:
            return {
                "passed": False,
                "stopped_at": stage,
                "completed_stages": list(receipts),
                **state,
            }
    return {
        "passed": True,
        "stopped_at": None,
        "completed_stages": list(receipts),
        **state,
    }


def write_once_result(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    raw = canonical_json_bytes(payload)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise RuntimeError("ESDI result target is not a regular file")
        if target.read_bytes() != raw:
            raise RuntimeError("ESDI write-once result already differs")
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuntimeError("ESDI result parent is not a real directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise RuntimeError("ESDI write-once result raced with different bytes")
            return "verified_existing"
        directory_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("gross9-clocks", "economics"),
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "gross9-clocks":
        payload = reconstruct_production_gross9_clocks()
        result = {
            "status": "gross9_clocks_reconstructed",
            "output": str(GROSS9_CLOCK_ARTIFACT),
            "manifest_hash": payload["manifest_hash"],
        }
    else:
        result = run_staged_economics()
    print(json.dumps(_json_ready(result), sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "BASELINE_GROSS",
    "BASE_COST_RATE",
    "CANDIDATE_WEIGHTS",
    "CLOCK_COLUMNS",
    "CONTROL_NAMES",
    "ECONOMIC_RECEIPT_PROTOCOL",
    "ECONOMIC_STAGE_ORDER",
    "FUNDING_COLUMNS",
    "GROSS9_CLOCK_ARTIFACT",
    "GROSS9_CLOCK_PROTOCOL",
    "GROSS9_SLEEVES",
    "GROSS9_WEIGHTS",
    "LEVERAGE",
    "MARKET_COLUMNS",
    "NOVELTY_ARTIFACT",
    "PERIODS",
    "POLICY_ID",
    "PREREGISTRATION_ARTIFACT_SHA256",
    "PREREGISTRATION_MANIFEST_HASH",
    "PROTOCOL_VERSION",
    "SELECTION_PERIODS",
    "SOURCE_SUPPORT_ARTIFACT",
    "STAGE_CUTOFFS",
    "STAGE_RECEIPT_NAMES",
    "STRESS_COST_RATE",
    "ValidationContext",
    "authorize_future_period",
    "build_gross9_clock_artifact",
    "calendar_month_clustered_signflip",
    "canonical_hash",
    "canonical_json_bytes",
    "evaluate_primary_superiority",
    "evaluate_same_gross_weight",
    "evaluate_standalone_period",
    "evaluate_standalone_period_with_controls",
    "future_veto",
    "load_bound_preregistration",
    "load_gross9_clock_artifact",
    "rank_same_gross_treatments",
    "reconstruct_gross9_sleeve_clocks",
    "reconstruct_production_gross9_clocks",
    "run_staged_economics",
    "same_gross_weights",
    "sha256_file",
    "simulate_portfolio",
    "standalone_gate_checks",
    "validate_clock",
    "validate_frozen_contract",
    "write_once_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
