#!/usr/bin/env python3
"""Fit repaired PSIM S6R1 action-mean-residual Ridge FQI and gate schedules.

This runner reads the frozen 2020 transition ledger, source-only Gemma
features, and frozen 2020 PCA.  It never reads a raw market/funding payload and
never computes a 2021 reward or economic metric.  The write-once terminal
result authorizes a separate report-only 2021 transfer preregistration only
when the sealed schedule passes every outcome-blind readiness gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (
    preregister_psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi as prereg,
)
from training import psim_action_mean_residual_fqi as residual
from training import psim_semantic_feature_matrix as features

PROTOCOL_VERSION = (
    "psim_d8_rllm2_s6r1_train2020_action_mean_residual_ridge_fqi_v1"
)
ATTEMPT_PROTOCOL_VERSION = (
    "psim_d8_rllm2_s6r1_train2020_action_mean_residual_ridge_fqi_attempt_v1"
)
SCHEDULE_PROTOCOL_VERSION = (
    "psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi_schedule_gate_v1"
)
PREREGISTRATION_SHA256 = (
    "a12db8dd9899a9b0e81f2e0ff80f1acaf7695019717cdfb4d14c7755d06576dc"
)
PREREGISTRATION_MANIFEST_HASH = (
    "f527487ec53eefe7ee0ddcc931512e8f845c510d3566f4a17104082454c88aff"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else prereg.REPO_ROOT / candidate


def _canonical_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + (b"\n" if pretty else b"")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=prereg.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_committed_clean_runner() -> tuple[str, str]:
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    status = _git("status", "--porcelain", "--untracked-files=no")
    runner_sha = _sha256_file(prereg.RUNNER_PATH)
    committed = subprocess.run(
        ["git", "show", f"{head}:{prereg.RUNNER_PATH.as_posix()}"],
        cwd=prereg.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if (
        head != origin_main
        or status
        or hashlib.sha256(committed).hexdigest() != runner_sha
    ):
        raise RuntimeError(
            "PSIM S6R1 runner requires clean HEAD equal to origin/main"
        )
    return head, runner_sha


def validate_preregistration() -> dict[str, Any]:
    target = repository_path(prereg.DEFAULT_OUTPUT)
    if (
        target.is_symlink()
        or not target.is_file()
        or _sha256_file(target) != PREREGISTRATION_SHA256
    ):
        raise RuntimeError("PSIM S6R1 preregistration file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    if (
        payload.get("manifest_hash") != _canonical_hash(core)
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload != prereg.build_preregistration()
        or payload.get("candidate", {}).get("id") != prereg.STAGE_ID
        or payload.get("fitted_q_contract", {}).get("primary_policy_id")
        != prereg.PRIMARY_POLICY_ID
        or payload.get("control_family", {}).get("new_policy_family_ids")
        != list(prereg.POLICY_FAMILY_IDS)
        or payload.get("action_mean_residual_reward_contract", {}).get(
            "transform"
        )
        != "R_star[t,p,a] = R[t,p,a] - mu[p,a] + bar_mu[p]"
        or payload.get("action_mean_residual_reward_contract", {}).get(
            "s5_delta_reuse"
        )
        is not False
        or payload.get("repair_contract", {}).get("authorized_change")
        != (
            "replace residual.reconstruct_reward_tensor with "
            "residual.s5_core.reconstruct_reward_tensor"
        )
        or payload.get("repair_contract", {}).get(
            "authorized_change_count"
        )
        != 1
        or payload.get("scientific_contract_identity", {}).get(
            "unchanged_contract_hash"
        )
        != "2471fa5e5272ed33e970ba03c0b8268df4eaeb6a856eaa28e7fa8125b1e0b70d"
        or payload.get("pre2021_schedule_readiness_gate", {}).get(
            "must_run_before_any_2021_market_or_funding_read"
        )
        is not True
        or payload.get("execution_contract", {}).get(
            "no_raw_market_or_funding_payload_read"
        )
        is not True
        or payload.get("execution_contract", {}).get(
            "full_execute_smoke_test_before_commit_required"
        )
        is not True
    ):
        raise RuntimeError("PSIM S6R1 preregistration contract changed")
    return payload


def _write_once(path: str | Path, raw: bytes) -> None:
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError(f"PSIM S6R1 write-once artifact drift: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_once(path: str | Path, payload: Mapping[str, Any]) -> None:
    _write_once(path, _canonical_bytes(dict(payload), pretty=True))


def _deterministic_csv_gzip_bytes(
    frame: pd.DataFrame,
) -> tuple[bytes, str]:
    text = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        fileobj=output,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(text)
    return output.getvalue(), hashlib.sha256(text).hexdigest()


def _artifact_record(
    path: str | Path,
    raw: bytes,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "path": Path(path).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    if extra:
        record.update(dict(extra))
    return record


def _attempt_payload(
    execution_commit: str,
    runner_sha256: str,
) -> dict[str, Any]:
    core = {
        "protocol_version": ATTEMPT_PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha256,
        "preregistration": {
            "path": prereg.DEFAULT_OUTPUT.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "authorization": {
            "parse_frozen_2020_transition_ledger": True,
            "read_raw_market_or_funding_payload": False,
            "open_2021_policy_specific_outcomes": False,
            "open_2022_or_later_outcomes": False,
        },
        "access_boundary_at_attempt": {
            "raw_market_or_funding_paths_read": [],
            "2020_transition_ledger_rows_parsed": 0,
            "2020_residual_reward_values_created": 0,
            "2020_economic_metrics_computed": 0,
            "2021_market_or_funding_paths_read": [],
            "2021_reward_rows_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_policy_specific_outcomes_opened": False,
            "2022_or_later_outcomes_opened": False,
            "model_loaded": False,
            "model_forwards_started": 0,
        },
    }
    return {**core, "attempt_hash": _canonical_hash(core)}


def _write_or_validate_attempt(
    execution_commit: str,
    runner_sha256: str,
) -> dict[str, Any]:
    payload = _attempt_payload(execution_commit, runner_sha256)
    _write_json_once(prereg.ATTEMPT_PATH, payload)
    return payload


def _load_frozen_pca(train_indices: np.ndarray) -> features.FrozenPCA:
    target = repository_path(prereg.s4.PCA_PATH)
    with np.load(target, allow_pickle=False) as payload:
        required = {
            "components",
            "explained_variance",
            "fit_row_index",
            "mean",
            "singular_values",
        }
        if set(payload.files) != required:
            raise RuntimeError("PSIM S6R1 frozen PCA keys changed")
        components = np.asarray(payload["components"], dtype=np.float64)
        explained = np.asarray(
            payload["explained_variance"],
            dtype=np.float64,
        )
        fit_rows = np.asarray(payload["fit_row_index"], dtype=np.int64)
        mean = np.asarray(payload["mean"], dtype=np.float64)
        singular = np.asarray(
            payload["singular_values"],
            dtype=np.float64,
        )
    if (
        components.shape != (32, 2_560)
        or explained.shape != (32,)
        or mean.shape != (2_560,)
        or singular.shape != (32,)
        or not np.array_equal(fit_rows, train_indices)
        or not all(
            np.isfinite(array).all()
            for array in (components, explained, mean, singular)
        )
    ):
        raise RuntimeError("PSIM S6R1 frozen PCA content changed")
    return features.FrozenPCA(
        mean=mean,
        components=components,
        singular_values=singular,
        explained_variance=explained,
        fit_row_count=len(fit_rows),
    )


def _verify_source_artifacts() -> None:
    for path, expected in (
        (
            prereg.s4.s3.SOURCE_ROWS_PATH,
            prereg.s4.S3_SOURCE_ROWS_SHA256,
        ),
        (
            prereg.s4.s3.EMBEDDINGS_PATH,
            prereg.s4.S3_EMBEDDINGS_SHA256,
        ),
        (
            prereg.s4.s3.RELATION_LOGITS_PATH,
            prereg.s4.S3_RELATION_LOGITS_SHA256,
        ),
        (
            prereg.s4.s3.RELATION_ROWS_PATH,
            prereg.s4.S3_RELATION_ROWS_SHA256,
        ),
        (
            prereg.s4.TRANSITION_LEDGER_PATH,
            prereg.s5.S4_TRANSITION_LEDGER_SHA256,
        ),
        (prereg.s4.PCA_PATH, prereg.s5.S4_PCA_SHA256),
        (
            prereg.s4.SCHEDULE_PATH,
            prereg.s5.S4_BASE_SCHEDULES_SHA256,
        ),
        (
            prereg.s5.SCHEDULE_PATH,
            prereg.S5_BASE_SCHEDULES_SHA256,
        ),
    ):
        if _sha256_file(path) != expected:
            raise RuntimeError(f"PSIM S6R1 frozen input changed: {path}")


def _schedule_manifest(
    *,
    attempt_hash: str,
    residual_ledger_artifact: Mapping[str, Any],
    base_artifact: Mapping[str, Any],
    delayed_artifact: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": SCHEDULE_PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "fit_stage": "2020_frozen_transition_ledger",
        "target_stage": "2021_source_schedule_only",
        "attempt_hash": attempt_hash,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "input_bindings": {
            "source_rows_sha256": prereg.s4.S3_SOURCE_ROWS_SHA256,
            "embeddings_sha256": prereg.s4.S3_EMBEDDINGS_SHA256,
            "relation_logits_sha256": prereg.s4.S3_RELATION_LOGITS_SHA256,
            "transition_ledger_2020_sha256": (
                prereg.s5.S4_TRANSITION_LEDGER_SHA256
            ),
            "pca32_2020_sha256": prereg.s5.S4_PCA_SHA256,
            "s4_base_schedules_2021_sha256": (
                prereg.s5.S4_BASE_SCHEDULES_SHA256
            ),
            "s5_base_schedules_2021_sha256": (
                prereg.S5_BASE_SCHEDULES_SHA256
            ),
        },
        "primary_policy_id": prereg.PRIMARY_POLICY_ID,
        "policy_family_ids": list(prereg.POLICY_FAMILY_IDS),
        "residual_reward_ledger_2020": dict(residual_ledger_artifact),
        "base_schedules_2021": dict(base_artifact),
        "delayed_primary_schedule_2021": dict(delayed_artifact),
        "schedule_readiness": dict(readiness),
        "decision": "pass" if readiness["passed"] else "reject",
        "outcome_boundary": {
            "raw_market_or_funding_payload_opened": False,
            "2021_market_or_funding_payload_opened": False,
            "2021_reward_rows_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_policy_specific_outcomes_opened": False,
            "2022_or_later_outcomes_opened": False,
        },
        "global_2021_pristine_claim": False,
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def _expected_artifact_paths() -> dict[str, Path]:
    return {
        "residual_reward_ledger_2020": prereg.RESIDUAL_LEDGER_PATH,
        "base_schedules_2021": prereg.SCHEDULE_PATH,
        "delayed_primary_schedule_2021": prereg.DELAYED_SCHEDULE_PATH,
        "schedule_gate_manifest_2021": prereg.SCHEDULE_MANIFEST_PATH,
    }


def _expected_access_boundary(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    registration_paths = registration.get("access_boundary", {}).get(
        "source_files_read"
    )
    if not isinstance(registration_paths, list) or not all(
        isinstance(path, str) for path in registration_paths
    ):
        raise RuntimeError("PSIM S6R1 source access contract changed")
    return {
        "source_paths_read": sorted(
            {
                *registration_paths,
                prereg.s4.s3.SOURCE_ROWS_PATH.as_posix(),
                prereg.s4.s3.EMBEDDINGS_PATH.as_posix(),
                prereg.s4.s3.RELATION_LOGITS_PATH.as_posix(),
                prereg.s4.s3.RELATION_ROWS_PATH.as_posix(),
                prereg.s4.PCA_PATH.as_posix(),
                prereg.s4.TRANSITION_LEDGER_PATH.as_posix(),
                prereg.s4.SCHEDULE_PATH.as_posix(),
                prereg.s5.SCHEDULE_PATH.as_posix(),
                prereg.DEFAULT_OUTPUT.as_posix(),
            }
        ),
        "raw_market_or_funding_paths_read": [],
        "2020_transition_ledger_rows_parsed": 3_288,
        "2020_original_reward_values_read": 3_288,
        "2020_residual_reward_values_created": 3_288,
        "2020_economic_metrics_computed": 0,
        "2021_market_or_funding_paths_read": [],
        "2021_market_rows_parsed": 0,
        "2021_funding_rows_parsed": 0,
        "2021_reward_rows_created": 0,
        "2021_economic_metrics_computed": 0,
        "2021_policy_specific_outcomes_opened": False,
        "2022_or_later_outcomes_opened": False,
        "model_loaded": False,
        "model_forwards_started": 0,
    }


def _verify_existing_result(
    attempt: Mapping[str, Any],
) -> dict[str, Any] | None:
    target = repository_path(prereg.RESULT_PATH)
    if not target.exists():
        return None
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("PSIM S6R1 result path is unsafe")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    artifacts = payload.get("artifacts")
    expected_paths = _expected_artifact_paths()
    passed = payload.get("decision") == "pass"
    readiness = payload.get("schedule_readiness")
    readiness_gates = (
        readiness.get("gates")
        if isinstance(readiness, dict)
        else None
    )
    registration = prereg.build_preregistration()
    expected_action = (
        registration["pre2021_schedule_readiness_gate"][
            "pass_action" if passed else "failure_action"
        ]
    )
    expected_boundary = _expected_access_boundary(registration)
    if (
        payload.get("result_hash") != _canonical_hash(core)
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage_id") != prereg.STAGE_ID
        or payload.get("execution_commit") != attempt["execution_commit"]
        or payload.get("runner_sha256") != attempt["runner_sha256"]
        or payload.get("attempt_hash") != attempt["attempt_hash"]
        or payload.get("preregistration_manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
        or payload.get("decision") not in {"pass", "reject"}
        or payload.get("terminal_action") != expected_action
        or payload.get(
            "authorize_separate_2021_transfer_preregistration"
        )
        is not passed
        or payload.get("authorize_2022_or_later_outcomes") is not False
        or payload.get("qlora_authorized") is not False
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(expected_paths)
        or not isinstance(readiness, dict)
        or readiness.get("passed") is not passed
        or not isinstance(readiness_gates, dict)
        or not readiness_gates
        or readiness.get("passed")
        is not all(bool(value) for value in readiness_gates.values())
    ):
        raise RuntimeError("PSIM S6R1 existing result changed")
    for name, expected_path in expected_paths.items():
        artifact = artifacts[name]
        if (
            artifact.get("path") != expected_path.as_posix()
            or not isinstance(artifact.get("sha256"), str)
            or _sha256_file(expected_path) != artifact["sha256"]
        ):
            raise RuntimeError("PSIM S6R1 existing artifact changed")
    manifest = json.loads(
        repository_path(prereg.SCHEDULE_MANIFEST_PATH).read_text(
            encoding="utf-8"
        )
    )
    manifest_core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    boundary = payload.get("access_boundary")
    if (
        manifest.get("manifest_hash") != _canonical_hash(manifest_core)
        or manifest.get("protocol_version") != SCHEDULE_PROTOCOL_VERSION
        or manifest.get("attempt_hash") != attempt["attempt_hash"]
        or manifest.get("decision") != payload["decision"]
        or manifest.get("schedule_readiness")
        != payload.get("schedule_readiness")
        or manifest.get("residual_reward_ledger_2020")
        != artifacts["residual_reward_ledger_2020"]
        or manifest.get("base_schedules_2021")
        != artifacts["base_schedules_2021"]
        or manifest.get("delayed_primary_schedule_2021")
        != artifacts["delayed_primary_schedule_2021"]
        or boundary != expected_boundary
    ):
        raise RuntimeError("PSIM S6R1 existing schedule gate changed")
    return payload


def _validate_only() -> dict[str, Any]:
    payload = validate_preregistration()
    execution_commit, runner_sha = assert_committed_clean_runner()
    existing_outputs = [
        path.as_posix()
        for path in (
            prereg.ATTEMPT_PATH,
            prereg.RESULT_PATH,
            prereg.RESIDUAL_LEDGER_PATH,
            prereg.SCHEDULE_PATH,
            prereg.DELAYED_SCHEDULE_PATH,
            prereg.SCHEDULE_MANIFEST_PATH,
        )
        if repository_path(path).exists()
    ]
    return {
        "decision": "validated",
        "stage_id": prereg.STAGE_ID,
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha,
        "preregistration_manifest_hash": payload["manifest_hash"],
        "existing_outputs": existing_outputs,
        "raw_market_or_funding_paths_read": [],
        "2020_transition_ledger_rows_parsed": 0,
        "2021_policy_specific_outcomes_opened": False,
    }


def execute() -> dict[str, Any]:
    registration = validate_preregistration()
    execution_commit, runner_sha = assert_committed_clean_runner()
    attempt = _write_or_validate_attempt(execution_commit, runner_sha)
    existing = _verify_existing_result(attempt)
    if existing is not None:
        return existing

    _verify_source_artifacts()
    bundle = features.load_source_bundle(
        repository_path(prereg.s4.s3.SOURCE_ROWS_PATH),
        repository_path(prereg.s4.s3.EMBEDDINGS_PATH),
        repository_path(prereg.s4.s3.RELATION_LOGITS_PATH),
        repository_path(prereg.s4.s3.RELATION_ROWS_PATH),
    )
    train_indices = features.year_indices(bundle.rows, 2020)
    target_indices = features.year_indices(bundle.rows, 2021)
    if len(train_indices) != 366 or len(target_indices) != 365:
        raise RuntimeError("PSIM S6R1 annual source roster changed")
    train_rows = [bundle.rows[index] for index in train_indices]
    ledger = pd.read_csv(
        repository_path(prereg.s4.TRANSITION_LEDGER_PATH)
    )
    reconstructed = residual.s5_core.reconstruct_reward_tensor(
        ledger,
        train_rows,
    )
    transformed = residual.action_mean_residualize_rewards(
        reconstructed.reward_tensor,
        reconstructed.reachable,
    )
    residual_ledger = residual.build_residual_ledger(
        ledger,
        transformed.action_means,
        transformed.position_grand_means,
    )
    pca = _load_frozen_pca(train_indices)
    feature_family = features.build_feature_family(bundle, pca)
    family = residual.fit_policy_family(
        feature_family,
        train_indices,
        transformed.reward_tensor,
        reconstructed.terminal,
        reconstructed.reachable,
        train_rows,
    )
    target_schedules = residual.build_schedule_family(
        family,
        feature_family,
        target_indices,
        bundle.rows,
    )
    delayed_schedule = residual.build_delayed_primary_schedule(
        target_schedules
    )
    s4_schedules = pd.read_csv(
        repository_path(prereg.s4.SCHEDULE_PATH)
    )
    s5_schedules = pd.read_csv(
        repository_path(prereg.s5.SCHEDULE_PATH)
    )
    readiness = residual.evaluate_schedule_readiness(
        target_schedules,
        delayed_schedule,
        s4_schedules,
        s5_schedules,
    )

    residual_raw, residual_decoded_sha = _deterministic_csv_gzip_bytes(
        residual_ledger
    )
    base_raw, base_decoded_sha = _deterministic_csv_gzip_bytes(
        target_schedules
    )
    delayed_raw, delayed_decoded_sha = _deterministic_csv_gzip_bytes(
        delayed_schedule
    )
    residual_artifact = _artifact_record(
        prereg.RESIDUAL_LEDGER_PATH,
        residual_raw,
        extra={
            "decoded_sha256": residual_decoded_sha,
            "rows": len(residual_ledger),
            "residual_reward_values": int(
                np.isfinite(transformed.reward_tensor).sum()
            ),
        },
    )
    base_artifact = _artifact_record(
        prereg.SCHEDULE_PATH,
        base_raw,
        extra={
            "decoded_sha256": base_decoded_sha,
            "rows": len(target_schedules),
            "policy_count": len(prereg.POLICY_FAMILY_IDS),
        },
    )
    delayed_artifact = _artifact_record(
        prereg.DELAYED_SCHEDULE_PATH,
        delayed_raw,
        extra={
            "decoded_sha256": delayed_decoded_sha,
            "rows": len(delayed_schedule),
            "policy_count": 1,
        },
    )
    manifest = _schedule_manifest(
        attempt_hash=attempt["attempt_hash"],
        residual_ledger_artifact=residual_artifact,
        base_artifact=base_artifact,
        delayed_artifact=delayed_artifact,
        readiness=readiness,
    )
    manifest_raw = _canonical_bytes(manifest, pretty=True)
    manifest_artifact = _artifact_record(
        prereg.SCHEDULE_MANIFEST_PATH,
        manifest_raw,
        extra={"manifest_hash": manifest["manifest_hash"]},
    )

    _write_once(prereg.RESIDUAL_LEDGER_PATH, residual_raw)
    _write_once(prereg.SCHEDULE_PATH, base_raw)
    _write_once(prereg.DELAYED_SCHEDULE_PATH, delayed_raw)
    _write_once(prereg.SCHEDULE_MANIFEST_PATH, manifest_raw)
    artifacts = {
        "residual_reward_ledger_2020": residual_artifact,
        "base_schedules_2021": base_artifact,
        "delayed_primary_schedule_2021": delayed_artifact,
        "schedule_gate_manifest_2021": manifest_artifact,
    }
    for artifact in artifacts.values():
        if _sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("PSIM S6R1 published artifact hash changed")

    passed = bool(readiness["passed"])
    gate_contract = registration["pre2021_schedule_readiness_gate"]
    terminal_action = gate_contract[
        "pass_action" if passed else "failure_action"
    ]
    access_boundary = _expected_access_boundary(registration)
    observed_boundary_counts = {
        "2020_transition_ledger_rows_parsed": len(ledger),
        "2020_original_reward_values_read": int(
            np.isfinite(reconstructed.reward_tensor).sum()
        ),
        "2020_residual_reward_values_created": int(
            np.isfinite(transformed.reward_tensor).sum()
        ),
    }
    if any(
        access_boundary[key] != value
        for key, value in observed_boundary_counts.items()
    ):
        raise RuntimeError("PSIM S6R1 observed access boundary changed")
    result_core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "decision": "pass" if passed else "reject",
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha,
        "attempt_hash": attempt["attempt_hash"],
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "fit": {
            "fit_stage": "2020_frozen_transition_ledger",
            "fit_source_rows": len(train_indices),
            "reachable_state_position_rows": int(
                reconstructed.reachable.sum()
            ),
            "original_reward_values": int(
                np.isfinite(reconstructed.reward_tensor).sum()
            ),
            "residual_reward_values": int(
                np.isfinite(transformed.reward_tensor).sum()
            ),
            "fitted_q_count": family.fitted_q_count,
            "ridge_fitted_q_count": family.fitted_q_count,
            "extra_trees_fitted_q_count": 0,
            "primary_policy_id": prereg.PRIMARY_POLICY_ID,
            "primary_selected_from_2020_metrics": False,
            "action_mean_baseline_by_current_position": {
                position: {
                    action: float(
                        transformed.action_means[
                            position_index,
                            action_index,
                        ]
                    )
                    for action_index, action in enumerate(
                        (
                            "TARGET_FLAT",
                            "TARGET_SHORT",
                            "TARGET_LONG",
                        )
                    )
                }
                for position_index, position in enumerate(
                    (
                        "POSITION_SHORT",
                        "POSITION_FLAT",
                        "POSITION_LONG",
                    )
                )
            },
            "position_grand_mean_by_current_position": {
                position: float(
                    transformed.position_grand_means[position_index]
                )
                for position_index, position in enumerate(
                    (
                        "POSITION_SHORT",
                        "POSITION_FLAT",
                        "POSITION_LONG",
                    )
                )
            },
        },
        "schedule_readiness": readiness,
        "target_schedule_seal": {
            "target_stage": "2021_source_schedule_only",
            "source_rows": len(target_indices),
            "primary_policy_id": prereg.PRIMARY_POLICY_ID,
            "policy_family_ids": list(prereg.POLICY_FAMILY_IDS),
            "manifest_hash": manifest["manifest_hash"],
        },
        "artifacts": artifacts,
        "global_2021_outcome_disclosure": {
            "globally_pristine_2021_claim": False,
            "prior_unrelated_2021_transfer_result_exists": True,
            "s6r1_policy_specific_2021_metrics_exist": False,
            "allowed_future_role": (
                "protocol-isolated policy-specific report-only transfer"
            ),
        },
        "access_boundary": access_boundary,
        "authorize_separate_2021_transfer_preregistration": passed,
        "authorize_2022_or_later_outcomes": False,
        "qlora_authorized": False,
        "terminal_action": terminal_action,
    }
    result = {**result_core, "result_hash": _canonical_hash(result_core)}
    _write_json_once(prereg.RESULT_PATH, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _validate_only() if args.validate_only else execute()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
