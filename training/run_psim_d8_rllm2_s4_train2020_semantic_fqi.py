#!/usr/bin/env python3
"""Fit PSIM semantic FQI on 2020 and seal 2021 schedules.

The authorizing result is written only after the exact 2020 transition ledger,
2020-only PCA, complete base schedule family, delayed primary schedules, and
the outcome-blind 2021 schedule manifest are present and hash verified.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import bctp_family_evaluator as evaluator  # noqa: E402
from training import bctp_stage_sources as stage_sources  # noqa: E402
from training import bctp_transition_labels as bctp_labels  # noqa: E402
from training import preregister_psim_d8_rllm2_s4_semantic_fqi as prereg  # noqa: E402
from training import psim_semantic_feature_matrix as feature_matrix  # noqa: E402
from training import psim_semantic_fqi_policies as fqi  # noqa: E402
from training import psim_semantic_transition_labels as transitions  # noqa: E402


PROTOCOL_VERSION = "psim_d8_rllm2_s4_train2020_semantic_fqi_seal_v1"
ATTEMPT_PROTOCOL_VERSION = (
    "psim_d8_rllm2_s4_train2020_semantic_fqi_attempt_v1"
)
SCHEDULE_PROTOCOL_VERSION = (
    "psim_d8_rllm2_s4_semantic_fqi_schedule_seal_v1"
)
PREREGISTRATION_SHA256 = (
    "2c3be7d92f61248dd26d428a9793f1ec1e703e794e6454970c8a6c2804fc2a34"
)
PREREGISTRATION_MANIFEST_HASH = (
    "81eaa2c6ea80e792a20d4abf6e341355d66ea3b0bed7cd359af040139ce91845"
)
SCHEDULE_COLUMNS = (
    "policy_id",
    "sequence_id",
    "entry_time",
    "target",
)
FIVE_MINUTES = pd.Timedelta(minutes=5)
BASE_COST_RATE = 0.0006
STRESS_COST_RATE = 0.0010

FEATURE_NAME_BY_POLICY = {
    "semantic_ridge_fqi": "semantic",
    "semantic_extra_trees_fqi": "semantic",
    "metadata_frontmatter_only": "metadata_frontmatter_only",
    "path_section_diff_size_only": "path_section_diff_size_only",
    "cadence_revision_topology_only": "cadence_revision_topology_only",
    "shuffled_eip_bip_daily_relation": "shuffled_eip_bip_daily_relation",
    "shuffled_old_new_pairing": "shuffled_old_new_pairing",
    "future_status_scrub": "future_status_scrub",
    "ethereum_only": "ethereum_only",
    "bitcoin_only": "bitcoin_only",
    "current_position_only": "current_position_only",
    "masked_semantic_embedding": "masked_semantic_embedding",
}


def _expected_artifact_paths() -> dict[str, Path]:
    return {
        "transition_ledger_2020": prereg.TRANSITION_LEDGER_PATH,
        "pca32_2020": prereg.PCA_PATH,
        "base_schedules_2021": prereg.SCHEDULE_PATH,
        "delayed_primary_schedules_2021": prereg.DELAYED_SCHEDULE_PATH,
        "schedule_manifest_2021": prereg.SCHEDULE_MANIFEST_PATH,
    }


@dataclass
class FittedPolicyFamily:
    policies: OrderedDict[str, Any]
    feature_names: dict[str, str]
    exact_memory_table: Mapping[tuple[str, str], np.ndarray]
    fitted_q_count: int
    ridge_fitted_q_count: int
    extra_trees_fitted_q_count: int


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
    runner = repository_path(prereg.RUNNER_PATH)
    runner_sha = _sha256_file(runner)
    committed_raw = subprocess.run(
        ["git", "show", f"{head}:{prereg.RUNNER_PATH.as_posix()}"],
        cwd=prereg.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if (
        head != origin_main
        or status
        or hashlib.sha256(committed_raw).hexdigest() != runner_sha
    ):
        raise RuntimeError(
            "PSIM S4 runner requires clean HEAD equal to origin/main"
        )
    return head, runner_sha


def validate_preregistration() -> dict[str, Any]:
    target = repository_path(prereg.DEFAULT_OUTPUT)
    if (
        target.is_symlink()
        or not target.is_file()
        or _sha256_file(target) != PREREGISTRATION_SHA256
    ):
        raise RuntimeError("PSIM S4 preregistration file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    if (
        payload.get("manifest_hash") != _canonical_hash(core)
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("candidate", {}).get("id") != prereg.STAGE_ID
        or payload.get("fitted_q_contract", {}).get("primary_policy_ids")
        != list(prereg.PRIMARY_POLICY_IDS)
        or payload.get("control_family", {}).get("policy_family_ids")
        != list(prereg.POLICY_FAMILY_IDS)
        or payload.get("authorized_2020_outcome_sources", {}).get(
            "2021_or_later_numeric_rows_authorized"
        )
        is not False
    ):
        raise RuntimeError("PSIM S4 preregistration contract changed")
    return payload


def _write_once(path: str | Path, raw: bytes) -> None:
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != raw:
            raise RuntimeError(f"PSIM S4 write-once artifact drift: {path}")
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


def _deterministic_npz_bytes(
    arrays: Mapping[str, np.ndarray],
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(
                buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def _deterministic_csv_gzip_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
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
        "outcome_authorization": {
            "2020_train_only": True,
            "2021_or_later": False,
        },
        "access_boundary_at_attempt": {
            "market_or_funding_paths_read": [],
            "market_or_funding_payload_bytes_hashed": False,
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "2020_outcomes_opened": False,
            "2021_or_later_outcomes_opened": False,
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


def _fit_policy_family(
    feature_family: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    reward_tensor: np.ndarray,
    terminal: np.ndarray,
    reachable: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
) -> FittedPolicyFamily:
    months = [str(row["decision_at"])[:7] for row in train_rows]
    policies: dict[str, Any] = {
        "always_flat": fqi.ConstantPolicy("TARGET_FLAT"),
        "always_long": fqi.ConstantPolicy("TARGET_LONG"),
        "always_short": fqi.ConstantPolicy("TARGET_SHORT"),
        "previous_target_persistence": fqi.PersistencePolicy(),
    }
    feature_names: dict[str, str] = {}
    fitted_q_count = 0
    ridge_count = 0
    extra_count = 0

    def fit_one(
        policy_id: str,
        feature_name: str,
        algorithm: str,
        *,
        reward_mode: str = "base",
    ) -> Any:
        nonlocal fitted_q_count, ridge_count, extra_count
        fitted = fqi.fit_fitted_q(
            feature_family[feature_name][train_indices],
            reward_tensor,
            terminal,
            reachable,
            algorithm=algorithm,
            reward_mode=reward_mode,
            months=months,
        )
        policies[policy_id] = fitted
        feature_names[policy_id] = feature_name
        fitted_q_count += 1
        if algorithm == "ridge":
            ridge_count += 1
        else:
            extra_count += 1
        return fitted

    primary_by_algorithm = {
        "ridge": fit_one(
            "semantic_ridge_fqi",
            "semantic",
            "ridge",
        ),
        "extra_trees": fit_one(
            "semantic_extra_trees_fqi",
            "semantic",
            "extra_trees",
        ),
    }
    for policy_id in (
        "metadata_frontmatter_only",
        "path_section_diff_size_only",
        "cadence_revision_topology_only",
        "shuffled_eip_bip_daily_relation",
        "shuffled_old_new_pairing",
        "future_status_scrub",
        "ethereum_only",
        "bitcoin_only",
        "current_position_only",
        "masked_semantic_embedding",
    ):
        fit_one(policy_id, FEATURE_NAME_BY_POLICY[policy_id], "ridge")
    for primary_id, algorithm in (
        ("semantic_ridge_fqi", "ridge"),
        ("semantic_extra_trees_fqi", "extra_trees"),
    ):
        for suffix, reward_mode in (
            ("circular_21_reward", "circular_21"),
            ("within_month_shuffled_reward", "within_month_shuffled"),
        ):
            fit_one(
                f"{primary_id}_{suffix}",
                "semantic",
                algorithm,
                reward_mode=reward_mode,
            )
        direction_id = f"{primary_id}_direction_flip"
        policies[direction_id] = fqi.DirectionFlipPolicy(
            primary_by_algorithm[algorithm]
        )
        feature_names[direction_id] = "semantic"
        permutation_id = f"{primary_id}_action_code_permutation"
        policies[permutation_id] = fqi.fit_action_code_permutation(
            feature_family["semantic"][train_indices],
            reward_tensor,
            terminal,
            reachable,
            algorithm=algorithm,
            months=months,
        )
        feature_names[permutation_id] = "semantic"
        fitted_q_count += 1
        if algorithm == "ridge":
            ridge_count += 1
        else:
            extra_count += 1
    signatures = [
        str(row["source_payload_sha256"]) for row in train_rows
    ]
    exact_memory_table = fqi.fit_exact_payload_memory(
        signatures,
        reward_tensor,
        reachable,
    )
    ordered = OrderedDict(
        (policy_id, policies[policy_id])
        for policy_id in prereg.POLICY_FAMILY_IDS
        if policy_id != "exact_redacted_payload_memory"
    )
    if (
        len(ordered) != len(prereg.POLICY_FAMILY_IDS) - 1
        or fitted_q_count != 18
        or ridge_count != 14
        or extra_count != 4
    ):
        raise RuntimeError("PSIM S4 fitted policy family changed")
    return FittedPolicyFamily(
        policies=ordered,
        feature_names=feature_names,
        exact_memory_table=exact_memory_table,
        fitted_q_count=fitted_q_count,
        ridge_fitted_q_count=ridge_count,
        extra_trees_fitted_q_count=extra_count,
    )


def _iso_z(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("PSIM S4 schedule timestamp must be timezone aware")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _schedule_for_policy(
    policy_id: str,
    family: FittedPolicyFamily,
    feature_family: Mapping[str, np.ndarray],
    row_indices: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if policy_id == "exact_redacted_payload_memory":
        signatures = [
            str(rows[index]["source_payload_sha256"])
            for index in row_indices
        ]
        targets = fqi.rollout_exact_payload_memory(
            family.exact_memory_table,
            signatures,
        )
    else:
        policy = family.policies[policy_id]
        if policy_id in {
            "always_flat",
            "always_long",
            "always_short",
            "previous_target_persistence",
        }:
            selected_features = np.zeros(
                (len(row_indices), 0),
                dtype=np.float32,
            )
        else:
            feature_name = family.feature_names[policy_id]
            selected_features = feature_family[feature_name][row_indices]
        targets = fqi.rollout_policy(policy, selected_features)
    return pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                "sequence_id": str(rows[index]["row_hash"]),
                "entry_time": _iso_z(rows[index]["decision_at"]),
                "target": target,
            }
            for index, target in zip(row_indices, targets)
        ],
        columns=SCHEDULE_COLUMNS,
    )


def build_schedule_family(
    family: FittedPolicyFamily,
    feature_family: Mapping[str, np.ndarray],
    row_indices: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frames = [
        _schedule_for_policy(
            policy_id,
            family,
            feature_family,
            row_indices,
            rows,
        )
        for policy_id in prereg.POLICY_FAMILY_IDS
    ]
    combined = pd.concat(frames, ignore_index=True)
    expected_rows = len(row_indices) * len(prereg.POLICY_FAMILY_IDS)
    if (
        tuple(combined.columns) != SCHEDULE_COLUMNS
        or len(combined) != expected_rows
        or combined.duplicated(["policy_id", "sequence_id"]).any()
        or not combined["target"].isin(fqi.ACTION_NAMES).all()
        or tuple(dict.fromkeys(combined["policy_id"]))
        != prereg.POLICY_FAMILY_IDS
    ):
        raise RuntimeError("PSIM S4 base schedule family changed")
    return combined


def build_delayed_primary_schedules(
    base_schedules: pd.DataFrame,
    *,
    stage_end: pd.Timestamp,
) -> pd.DataFrame:
    selected = base_schedules.loc[
        base_schedules["policy_id"].isin(prereg.PRIMARY_POLICY_IDS)
    ].copy()
    times = pd.to_datetime(selected["entry_time"], utc=True) + FIVE_MINUTES
    terminal = pd.Timestamp(stage_end).tz_convert("UTC") - FIVE_MINUTES
    keep = times < terminal
    selected = selected.loc[keep].copy()
    selected["entry_time"] = [_iso_z(value) for value in times[keep]]
    selected["sequence_id"] = (
        selected["sequence_id"].astype(str) + ":delay_5m"
    )
    selected = selected.reset_index(drop=True)
    if (
        tuple(selected.columns) != SCHEDULE_COLUMNS
        or tuple(dict.fromkeys(selected["policy_id"]))
        != prereg.PRIMARY_POLICY_IDS
    ):
        raise RuntimeError("PSIM S4 delayed schedule family changed")
    return selected


def _policy_schedule(
    combined: pd.DataFrame,
    policy_id: str,
) -> pd.DataFrame:
    return combined.loc[
        combined["policy_id"].eq(policy_id),
        list(SCHEDULE_COLUMNS),
    ].reset_index(drop=True)


def _trade_count(schedule: pd.DataFrame) -> int:
    prior = "TARGET_FLAT"
    count = 0
    for target in schedule["target"].astype(str):
        if target != prior:
            count += 1
        prior = target
    if prior != "TARGET_FLAT":
        count += 1
    return count


def _evaluate_2020_primaries(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    base_schedules: pd.DataFrame,
    delayed_schedules: pd.DataFrame,
) -> dict[str, Any]:
    spec = stage_sources.STAGE_SPECS["2020"]
    output: dict[str, Any] = {}
    for policy_id in prereg.PRIMARY_POLICY_IDS:
        base = _policy_schedule(base_schedules, policy_id)
        delayed = _policy_schedule(delayed_schedules, policy_id)
        metrics = {
            "base_6bp": evaluator._evaluate_one(
                market,
                funding,
                base,
                start=spec.start,
                end=spec.end,
                cost_rate=BASE_COST_RATE,
            ),
            "stress_10bp": evaluator._evaluate_one(
                market,
                funding,
                base,
                start=spec.start,
                end=spec.end,
                cost_rate=STRESS_COST_RATE,
            ),
            "delayed_5m_6bp": evaluator._evaluate_one(
                market,
                funding,
                delayed,
                start=spec.start,
                end=spec.end,
                cost_rate=BASE_COST_RATE,
            ),
        }
        metrics["base_6bp"]["trade_count"] = _trade_count(base)
        metrics["stress_10bp"]["trade_count"] = _trade_count(base)
        metrics["delayed_5m_6bp"]["trade_count"] = _trade_count(delayed)
        output[policy_id] = metrics
    return output


def _schedule_manifest(
    *,
    attempt_hash: str,
    base_artifact: Mapping[str, Any],
    delayed_artifact: Mapping[str, Any],
    pca_artifact: Mapping[str, Any],
    ledger_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": SCHEDULE_PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "fit_stage": "2020",
        "target_stage": "2021",
        "attempt_hash": attempt_hash,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "source_feature_binding": {
            "source_rows_sha256": prereg.S3_SOURCE_ROWS_SHA256,
            "embeddings_sha256": prereg.S3_EMBEDDINGS_SHA256,
            "relation_logits_sha256": prereg.S3_RELATION_LOGITS_SHA256,
            "source_row_roster_hash": prereg.S3_SOURCE_ROW_ROSTER_HASH,
        },
        "policy_family_ids": list(prereg.POLICY_FAMILY_IDS),
        "primary_policy_ids": list(prereg.PRIMARY_POLICY_IDS),
        "pca32_2020": dict(pca_artifact),
        "transition_ledger_2020": dict(ledger_artifact),
        "base_schedules": dict(base_artifact),
        "delayed_primary_schedules": dict(delayed_artifact),
        "outcome_boundary": {
            "2020_train_outcomes_used_for_fit": True,
            "2021_market_or_funding_payload_opened": False,
            "2021_reward_rows_created": 0,
            "2021_economic_metrics_computed": 0,
            "2022_or_2023_outcomes_opened": False,
        },
        "strategy_outcomes_calculated_for_target_stage": False,
        "selection_from_2020_training_metrics": False,
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def _verify_existing_result(
    attempt: Mapping[str, Any],
) -> dict[str, Any] | None:
    target = repository_path(prereg.RESULT_PATH)
    if not target.exists():
        return None
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("PSIM S4 result path is unsafe")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }
    artifacts = payload.get("artifacts")
    expected_paths = _expected_artifact_paths()
    if (
        payload.get("result_hash") != _canonical_hash(core)
        or payload.get("attempt_hash") != attempt["attempt_hash"]
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage_id") != prereg.STAGE_ID
        or payload.get("execution_commit") != attempt["execution_commit"]
        or payload.get("runner_sha256") != attempt["runner_sha256"]
        or payload.get("preregistration_manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
        or payload.get("decision") != "pass"
        or payload.get("open_2021_outcomes_authorized") is not True
        or payload.get("open_2022_or_later_outcomes_authorized") is not False
        or payload.get("qlora_authorized") is not False
        or payload.get("terminal_action")
        != (
            "ACCEPT_PSIM_D8_RLLM2_S4_SEALED_2021_SCHEDULES_"
            "AUTHORIZE_2021_EVALUATION_ONLY"
        )
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(expected_paths)
    ):
        raise RuntimeError("PSIM S4 existing result changed")
    for name, expected_path in expected_paths.items():
        artifact = artifacts[name]
        path = artifact.get("path")
        expected = artifact.get("sha256")
        if (
            path != expected_path.as_posix()
            or not isinstance(expected, str)
            or len(expected) != 64
            or _sha256_file(path) != expected
        ):
            raise RuntimeError("PSIM S4 existing artifact changed")
    manifest_path = expected_paths["schedule_manifest_2021"]
    manifest = json.loads(
        repository_path(manifest_path).read_text(encoding="utf-8")
    )
    manifest_core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    seal = payload.get("target_schedule_seal")
    boundary = payload.get("access_boundary")
    if (
        manifest.get("manifest_hash") != _canonical_hash(manifest_core)
        or manifest.get("protocol_version") != SCHEDULE_PROTOCOL_VERSION
        or manifest.get("stage_id") != prereg.STAGE_ID
        or manifest.get("attempt_hash") != attempt["attempt_hash"]
        or manifest.get("target_stage") != "2021"
        or manifest.get("policy_family_ids")
        != list(prereg.POLICY_FAMILY_IDS)
        or manifest.get("primary_policy_ids")
        != list(prereg.PRIMARY_POLICY_IDS)
        or manifest.get("transition_ledger_2020")
        != artifacts["transition_ledger_2020"]
        or manifest.get("pca32_2020") != artifacts["pca32_2020"]
        or manifest.get("base_schedules")
        != artifacts["base_schedules_2021"]
        or manifest.get("delayed_primary_schedules")
        != artifacts["delayed_primary_schedules_2021"]
        or not isinstance(seal, dict)
        or seal.get("target_stage") != "2021"
        or seal.get("manifest_hash") != manifest.get("manifest_hash")
        or not isinstance(boundary, dict)
        or boundary.get("2021_market_or_funding_paths_read") != []
        or boundary.get("2021_reward_rows_created") != 0
        or boundary.get("2021_economic_metrics_computed") != 0
        or boundary.get("2021_outcomes_opened") is not False
        or boundary.get("2022_or_2023_outcomes_opened") is not False
    ):
        raise RuntimeError("PSIM S4 existing schedule seal changed")
    return payload


def _validate_only() -> dict[str, Any]:
    payload = validate_preregistration()
    prereg.validate_s3_success()
    execution_commit, runner_sha = assert_committed_clean_runner()
    existing_outputs = [
        path.as_posix()
        for path in (
            prereg.ATTEMPT_PATH,
            prereg.RESULT_PATH,
            prereg.TRANSITION_LEDGER_PATH,
            prereg.PCA_PATH,
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
        "market_or_funding_paths_read": [],
        "2020_outcomes_opened": False,
        "2021_or_later_outcomes_opened": False,
    }


def execute() -> dict[str, Any]:
    validate_preregistration()
    predecessor = prereg.validate_s3_success()
    if (
        predecessor["result"]["source_roster"][
            "source_row_roster_hash"
        ]
        != prereg.S3_SOURCE_ROW_ROSTER_HASH
    ):
        raise RuntimeError("PSIM S4 source roster authority changed")
    execution_commit, runner_sha = assert_committed_clean_runner()
    attempt = _write_or_validate_attempt(execution_commit, runner_sha)
    existing = _verify_existing_result(attempt)
    if existing is not None:
        return existing

    bundle = feature_matrix.load_source_bundle(
        repository_path(prereg.s3.SOURCE_ROWS_PATH),
        repository_path(prereg.s3.EMBEDDINGS_PATH),
        repository_path(prereg.s3.RELATION_LOGITS_PATH),
        repository_path(prereg.s3.RELATION_ROWS_PATH),
    )
    train_indices = feature_matrix.year_indices(bundle.rows, 2020)
    target_indices = feature_matrix.year_indices(bundle.rows, 2021)
    if len(train_indices) != 366 or len(target_indices) != 365:
        raise RuntimeError("PSIM S4 annual source roster changed")
    train_rows = [bundle.rows[index] for index in train_indices]

    stage_manifest = stage_sources.prepare_stage_source(
        "2020",
        output_root=stage_sources.DEFAULT_OUTPUT_ROOT,
    )
    market, funding, stage_diagnostics = stage_sources.load_stage_source(
        "2020",
        output_root=stage_sources.DEFAULT_OUTPUT_ROOT,
    )
    spec = stage_sources.STAGE_SPECS["2020"]
    transition = transitions.build_reward_tensor(
        train_rows,
        market,
        funding,
        start=spec.start,
        end=spec.end,
        cost_rate=BASE_COST_RATE,
    )
    reward_tensor = np.asarray(
        transition["reward_tensor"],
        dtype=np.float64,
    )
    terminal = np.asarray(transition["terminal"], dtype=bool)
    reachable = np.asarray(
        transition["reachable_mask"],
        dtype=bool,
    )

    pca = feature_matrix.fit_pca(
        bundle.embeddings[train_indices],
        components=feature_matrix.PCA_COMPONENTS,
    )
    features = feature_matrix.build_feature_family(bundle, pca)
    family = _fit_policy_family(
        features,
        train_indices,
        reward_tensor,
        terminal,
        reachable,
        train_rows,
    )
    train_schedules = build_schedule_family(
        family,
        features,
        train_indices,
        bundle.rows,
    )
    target_schedules = build_schedule_family(
        family,
        features,
        target_indices,
        bundle.rows,
    )
    delayed_train = build_delayed_primary_schedules(
        train_schedules,
        stage_end=spec.end,
    )
    delayed_target = build_delayed_primary_schedules(
        target_schedules,
        stage_end=stage_sources.STAGE_SPECS["2021"].end,
    )
    train_metrics = _evaluate_2020_primaries(
        market,
        funding,
        train_schedules,
        delayed_train,
    )

    ledger_raw = bctp_labels.deterministic_gzip_csv_bytes(
        transition["ledger"]
    )
    pca_raw = _deterministic_npz_bytes(
        {
            "components": pca.components.astype(np.float64),
            "explained_variance": pca.explained_variance.astype(np.float64),
            "fit_row_index": train_indices.astype(np.int32),
            "mean": pca.mean.astype(np.float64),
            "singular_values": pca.singular_values.astype(np.float64),
        }
    )
    base_raw, base_decoded_sha = _deterministic_csv_gzip_bytes(
        target_schedules
    )
    delayed_raw, delayed_decoded_sha = _deterministic_csv_gzip_bytes(
        delayed_target
    )
    ledger_artifact = _artifact_record(
        prereg.TRANSITION_LEDGER_PATH,
        ledger_raw,
        extra={
            "rows": len(transition["ledger"]),
            "frame_hash": bctp_labels.ledger_frame_hash(
                transition["ledger"]
            ),
        },
    )
    pca_artifact = _artifact_record(
        prereg.PCA_PATH,
        pca_raw,
        extra={
            "fit_rows": pca.fit_row_count,
            "input_width": int(pca.mean.shape[0]),
            "components": int(pca.components.shape[0]),
            "sign_canonicalized": True,
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
            "rows": len(delayed_target),
            "policy_count": len(prereg.PRIMARY_POLICY_IDS),
        },
    )
    schedule_manifest = _schedule_manifest(
        attempt_hash=attempt["attempt_hash"],
        base_artifact=base_artifact,
        delayed_artifact=delayed_artifact,
        pca_artifact=pca_artifact,
        ledger_artifact=ledger_artifact,
    )
    schedule_manifest_raw = _canonical_bytes(
        schedule_manifest,
        pretty=True,
    )
    schedule_manifest_artifact = _artifact_record(
        prereg.SCHEDULE_MANIFEST_PATH,
        schedule_manifest_raw,
        extra={"manifest_hash": schedule_manifest["manifest_hash"]},
    )

    _write_once(prereg.TRANSITION_LEDGER_PATH, ledger_raw)
    _write_once(prereg.PCA_PATH, pca_raw)
    _write_once(prereg.SCHEDULE_PATH, base_raw)
    _write_once(prereg.DELAYED_SCHEDULE_PATH, delayed_raw)
    _write_once(prereg.SCHEDULE_MANIFEST_PATH, schedule_manifest_raw)

    artifacts = {
        "transition_ledger_2020": ledger_artifact,
        "pca32_2020": pca_artifact,
        "base_schedules_2021": base_artifact,
        "delayed_primary_schedules_2021": delayed_artifact,
        "schedule_manifest_2021": schedule_manifest_artifact,
    }
    for artifact in artifacts.values():
        if _sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("PSIM S4 published artifact hash changed")
    result_core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "decision": "pass",
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha,
        "attempt_hash": attempt["attempt_hash"],
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "stage_source_manifest_hash": stage_manifest["manifest_hash"],
        "stage_source_diagnostics": stage_diagnostics,
        "fit": {
            "fit_stage": "2020",
            "fit_source_rows": len(train_indices),
            "reward_rows": len(reward_tensor),
            "reachable_state_position_rows": int(reachable.sum()),
            "reward_values_created": int(
                np.isfinite(reward_tensor).sum()
            ),
            "pca_fit_rows": pca.fit_row_count,
            "pca_components": int(pca.components.shape[0]),
            "fitted_q_count": family.fitted_q_count,
            "ridge_fitted_q_count": family.ridge_fitted_q_count,
            "extra_trees_fitted_q_count": (
                family.extra_trees_fitted_q_count
            ),
            "primary_selected_from_2020": None,
        },
        "train_2020_diagnostics_only": train_metrics,
        "target_schedule_seal": {
            "target_stage": "2021",
            "source_rows": len(target_indices),
            "policy_family_ids": list(prereg.POLICY_FAMILY_IDS),
            "primary_policy_ids": list(prereg.PRIMARY_POLICY_IDS),
            "manifest_hash": schedule_manifest["manifest_hash"],
        },
        "artifacts": artifacts,
        "access_boundary": {
            "source_paths_read": [
                prereg.s3.SOURCE_ROWS_PATH.as_posix(),
                prereg.s3.EMBEDDINGS_PATH.as_posix(),
                prereg.s3.RELATION_LOGITS_PATH.as_posix(),
                prereg.s3.RELATION_ROWS_PATH.as_posix(),
                prereg.DEFAULT_OUTPUT.as_posix(),
            ],
            "outcome_stage_paths_read": [
                stage_manifest["market"]["path"],
                stage_manifest["funding"]["path"],
            ],
            "full_parent_market_or_funding_payload_hashed": False,
            "market_rows_parsed": len(market),
            "funding_rows_parsed": len(funding),
            "rewards_created": int(np.isfinite(reward_tensor).sum()),
            "economic_metric_sets_computed": 6,
            "2020_train_outcomes_opened": True,
            "2021_market_or_funding_paths_read": [],
            "2021_reward_rows_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_outcomes_opened": False,
            "2022_or_2023_outcomes_opened": False,
            "model_loaded": False,
            "model_forwards_started": 0,
        },
        "open_2021_outcomes_authorized": True,
        "open_2022_or_later_outcomes_authorized": False,
        "qlora_authorized": False,
        "terminal_action": (
            "ACCEPT_PSIM_D8_RLLM2_S4_SEALED_2021_SCHEDULES_"
            "AUTHORIZE_2021_EVALUATION_ONLY"
        ),
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
