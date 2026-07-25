"""Write-once BCTP transfer-year target schedule sealing.

The seal is deliberately upstream of outcome access: it validates in-memory
policy schedules, writes deterministic gzip CSV artifacts, and emits a canonical
manifest that binds only frozen evaluator/source-manifest contracts.  It never
opens or hashes market/funding payload bytes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from training import bctp_stage_sources as stage_sources
from training import freeze_block_clearing_target_position_evaluator as freeze

Stage = Literal["2021", "2022", "2023"]

DEFAULT_OUTPUT_ROOT: Final = Path("results/bctp_target_schedule_seals")
BASE_SCHEDULE_FILENAME: Final = "base_target_schedules.csv.gz"
DELAYED_SCHEDULE_FILENAME: Final = "delayed_primary_target_schedules.csv.gz"
MANIFEST_FILENAME: Final = "target_schedule_manifest.json"
ALLOWED_TRANSFER_STAGES: Final = ("2021", "2022", "2023")


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else stage_sources.REPOSITORY_ROOT / candidate


def _seal_paths(output_root: str | Path, stage: str) -> dict[str, Path]:
    root = _path(output_root) / str(stage)
    return {
        "root": root,
        "base": root / BASE_SCHEDULE_FILENAME,
        "delayed": root / DELAYED_SCHEDULE_FILENAME,
        "manifest": root / MANIFEST_FILENAME,
    }


def _coerce_ordered_mapping(
    schedules: Mapping[str, Any],
    *,
    expected_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(schedules, Mapping):
        raise TypeError(f"BCTP {label} schedules must be a mapping")
    observed = tuple(schedules.keys())
    if observed != expected_ids:
        raise ValueError(f"BCTP {label} policy order mismatch")
    return schedules


def _coerce_frame(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return pd.DataFrame(list(rows))
    raise TypeError("BCTP schedule must be a DataFrame or sequence of records")


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"BCTP {name} is NaT")
    if parsed.tzinfo is None:
        raise ValueError(f"BCTP {name} must be timezone aware")
    return parsed.tz_convert("UTC")


def _iso_z(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _normalise_schedule_frame(
    rows: Any,
    *,
    policy_id: str,
    stage: str,
    delayed: bool,
) -> pd.DataFrame:
    frame = _coerce_frame(rows)
    if tuple(frame.columns) != stage_sources.SCHEDULE_COLUMNS:
        raise ValueError("BCTP target schedule schema mismatch")
    if frame.empty:
        raise ValueError(f"BCTP target schedule is empty for {policy_id}")
    if not frame["policy_id"].eq(policy_id).all():
        raise ValueError(f"BCTP target schedule policy_id mismatch for {policy_id}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].astype(str).eq("").any():
        raise ValueError(f"BCTP target schedule sequence_id missing for {policy_id}")
    if frame["sequence_id"].duplicated().any():
        raise ValueError(f"BCTP target schedule duplicate sequence_id for {policy_id}")
    if not frame["target"].isin(freeze.MODEL_ACTION_NAMES).all():
        raise ValueError(f"BCTP target schedule unknown target for {policy_id}")

    spec = stage_sources.STAGE_SPECS[stage]
    terminal = spec.end - pd.Timedelta(minutes=5)
    normalised = frame.loc[:, stage_sources.SCHEDULE_COLUMNS].copy()
    times = [
        _utc_timestamp(value, name="entry_time")
        for value in normalised["entry_time"].tolist()
    ]
    index = pd.DatetimeIndex(times)
    if index.has_duplicates:
        raise ValueError(f"BCTP target schedule duplicate entry_time for {policy_id}")
    if not index.is_monotonic_increasing:
        raise ValueError(f"BCTP target schedule entry_time order mismatch for {policy_id}")
    if not ((index >= spec.start) & (index < terminal)).all():
        raise ValueError(f"BCTP target schedule out-of-stage executable range for {policy_id}")
    if delayed and (index < spec.start + pd.Timedelta(minutes=5)).any():
        raise ValueError("BCTP delayed primary schedule must be +5m or later")
    if any((timestamp - spec.start) % pd.Timedelta(minutes=5) != pd.Timedelta(0) for timestamp in index):
        raise ValueError(f"BCTP target schedule entry_time is off 5m grid for {policy_id}")
    normalised["policy_id"] = normalised["policy_id"].astype(str)
    normalised["sequence_id"] = normalised["sequence_id"].astype(str)
    normalised["entry_time"] = [_iso_z(timestamp) for timestamp in index]
    normalised["target"] = normalised["target"].astype(str)
    return normalised.reset_index(drop=True)


def _combined_frame(
    schedules: Mapping[str, Any],
    *,
    expected_ids: tuple[str, ...],
    stage: str,
    label: str,
    delayed: bool,
) -> pd.DataFrame:
    mapping = _coerce_ordered_mapping(schedules, expected_ids=expected_ids, label=label)
    frames = [
        _normalise_schedule_frame(rows, policy_id=policy_id, stage=stage, delayed=delayed)
        for policy_id, rows in mapping.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["policy_id", "sequence_id"]).any():
        raise ValueError(f"BCTP {label} schedules have duplicate policy/sequence ids")
    return combined.loc[:, stage_sources.SCHEDULE_COLUMNS]


def _csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def _artifact_binding(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": stage_sources._file_sha256(path),
        "frame_hash": stage_sources._schedule_frame_hash(frame),
        "rows": int(len(frame)),
        "columns": list(stage_sources.SCHEDULE_COLUMNS),
    }


def _validate_exact_primary_delays(
    base: pd.DataFrame,
    delayed: pd.DataFrame,
    *,
    stage: str,
) -> None:
    terminal = (
        stage_sources.STAGE_SPECS[stage].end
        - pd.Timedelta(minutes=5)
    )
    for policy_id in stage_sources.PROMOTABLE_PRIMARY_IDS:
        base_rows = base.loc[base["policy_id"].eq(policy_id)].copy()
        base_times = pd.DatetimeIndex(
            [
                _utc_timestamp(value, name="base entry_time")
                for value in base_rows["entry_time"]
            ]
        )
        keep = base_times + pd.Timedelta(minutes=5) < terminal
        expected = base_rows.loc[keep].reset_index(drop=True)
        expected["sequence_id"] = (
            expected["sequence_id"].astype(str) + ":delay_5m"
        )
        expected["entry_time"] = [
            _iso_z(value + pd.Timedelta(minutes=5))
            for value in base_times[keep]
        ]
        actual = delayed.loc[
            delayed["policy_id"].eq(policy_id)
        ].reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(
                "BCTP delayed primary schedule is not the exact +5m base shift"
            )


def _read_manifest(path: Path) -> dict[str, Any]:
    return stage_sources._read_json(path)


def _validate_stage(stage: str) -> Stage:
    if stage not in ALLOWED_TRANSFER_STAGES:
        raise ValueError(f"BCTP target schedule stage must be one of {ALLOWED_TRANSFER_STAGES}")
    return stage  # type: ignore[return-value]


def seal_transfer_year_schedule(
    target_stage: Stage,
    base_schedules: Mapping[str, Any],
    delayed_primary_schedules: Mapping[str, Any],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    evaluator_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and write-once seal target schedules before outcome access.

    ``base_schedules`` must be ordered exactly as ``freeze.FAMILY_IDS``.
    ``delayed_primary_schedules`` must be ordered exactly as
    ``stage_sources.PROMOTABLE_PRIMARY_IDS`` and contain only +5m-or-later rows.
    The function opens no market/funding payloads.
    """

    stage = _validate_stage(str(target_stage))
    base = _combined_frame(
        base_schedules,
        expected_ids=tuple(freeze.FAMILY_IDS),
        stage=stage,
        label="base",
        delayed=False,
    )
    delayed = _combined_frame(
        delayed_primary_schedules,
        expected_ids=tuple(stage_sources.PROMOTABLE_PRIMARY_IDS),
        stage=stage,
        label="delayed primary",
        delayed=True,
    )
    _validate_exact_primary_delays(base, delayed, stage=stage)

    # Manifest-only evaluator verification; this deliberately does not open
    # market/funding payloads or outcome tables.
    evaluator = stage_sources.verify_evaluator_freeze(evaluator_manifest_path)
    paths = _seal_paths(output_root, stage)
    paths["root"].mkdir(parents=True, exist_ok=True)

    stage_sources._write_gzip_atomic(paths["base"], _csv_text(base))
    stage_sources._write_gzip_atomic(
        paths["delayed"],
        _csv_text(delayed),
    )
    base_binding = _artifact_binding(paths["base"], base)
    delayed_binding = _artifact_binding(paths["delayed"], delayed)

    manifest_core: dict[str, Any] = {
        "protocol_version": stage_sources.SCHEDULE_MANIFEST_PROTOCOL,
        "target_stage": stage,
        "stage": stage,
        "start_inclusive": _iso_z(stage_sources.STAGE_SPECS[stage].start),
        "terminal_flat_time_exclusive": _iso_z(stage_sources.STAGE_SPECS[stage].end - pd.Timedelta(minutes=5)),
        "schedule_columns": list(stage_sources.SCHEDULE_COLUMNS),
        "evaluator_manifest_hash": evaluator["manifest_hash"],
        "family_ids": list(freeze.FAMILY_IDS),
        "promotable_primary_ids": list(stage_sources.PROMOTABLE_PRIMARY_IDS),
        "base_schedules": base_binding,
        "delayed_primary_schedules": delayed_binding,
        "stress_reuses_base_target_sequences": True,
        "strategy_outcomes_calculated": False,
        "outcome_payload_opened": False,
        "market_or_funding_payload_opened": False,
        "market_or_funding_payload_bytes_hashed": False,
    }
    manifest_core["manifest_canonical_hash"] = stage_sources._canonical_hash(manifest_core)
    manifest = {**manifest_core, "manifest_hash": stage_sources._canonical_hash(manifest_core)}
    stage_sources._write_json_once(paths["manifest"], manifest)
    return load_transfer_year_schedule_seal(stage, manifest_path=paths["manifest"], evaluator_manifest_path=evaluator_manifest_path)


def load_transfer_year_schedule_seal(
    target_stage: Stage | str,
    *,
    manifest_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    evaluator_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reload and fully validate a sealed transfer-year schedule manifest."""

    stage = _validate_stage(str(target_stage))
    stage_sources.verify_evaluator_freeze(evaluator_manifest_path)
    path = _path(manifest_path) if manifest_path is not None else _seal_paths(output_root, stage)["manifest"]
    payload = _read_manifest(path)
    if payload.get("protocol_version") != stage_sources.SCHEDULE_MANIFEST_PROTOCOL:
        raise ValueError("BCTP target schedule protocol mismatch")
    if payload.get("target_stage") != stage or payload.get("stage") != stage:
        raise ValueError("BCTP target schedule stage mismatch")
    if payload.get("strategy_outcomes_calculated") is not False:
        raise ValueError("BCTP target schedule contains outcomes")
    if payload.get("outcome_payload_opened") is not False:
        raise ValueError("BCTP target schedule opened outcome payload")
    if payload.get("market_or_funding_payload_opened") is not False:
        raise ValueError("BCTP target schedule opened market/funding payload")
    if payload.get("market_or_funding_payload_bytes_hashed") is not False:
        raise ValueError("BCTP target schedule hashed market/funding payload")
    if payload.get("family_ids") != list(freeze.FAMILY_IDS):
        raise ValueError("BCTP target schedule family ids drifted")
    if payload.get("promotable_primary_ids") != list(stage_sources.PROMOTABLE_PRIMARY_IDS):
        raise ValueError("BCTP target schedule primary ids drifted")
    if payload.get("stress_reuses_base_target_sequences") is not True:
        raise ValueError("BCTP target schedule stress reuse flag drifted")
    if payload.get("evaluator_manifest_hash") != stage_sources.EXPECTED_EVALUATOR_MANIFEST_HASH:
        raise ValueError("BCTP target schedule evaluator manifest drifted")
    embedded = dict(payload)
    manifest_hash = embedded.pop("manifest_hash", None)
    if manifest_hash != stage_sources._canonical_hash(embedded):
        raise ValueError("BCTP target schedule manifest hash mismatch")
    canonical = dict(embedded)
    manifest_canonical_hash = canonical.pop("manifest_canonical_hash", None)
    if manifest_canonical_hash != stage_sources._canonical_hash(canonical):
        raise ValueError("BCTP target schedule canonical hash mismatch")
    base = stage_sources._validate_schedule_artifact(
        payload.get("base_schedules"),
        stage=stage,
        expected_policy_ids=tuple(freeze.FAMILY_IDS),
    )
    delayed = stage_sources._validate_schedule_artifact(
        payload.get("delayed_primary_schedules"),
        stage=stage,
        expected_policy_ids=tuple(stage_sources.PROMOTABLE_PRIMARY_IDS),
    )
    delayed_frame = pd.read_csv(_path(delayed["path"]), compression="infer", dtype=str)
    start_plus_five = stage_sources.STAGE_SPECS[stage].start + pd.Timedelta(minutes=5)
    if any(_utc_timestamp(value, name="delayed entry_time") < start_plus_five for value in delayed_frame["entry_time"]):
        raise ValueError("BCTP delayed primary schedule +5m validation failed")
    base_frame = pd.read_csv(
        _path(base["path"]),
        compression="infer",
        dtype=str,
    )
    _validate_exact_primary_delays(
        base_frame,
        delayed_frame,
        stage=stage,
    )
    return {
        **payload,
        "path": str(path),
        "file_sha256": stage_sources._file_sha256(path),
        "canonical_hash": stage_sources._canonical_hash(payload),
        "base_schedules": base,
        "delayed_primary_schedules": delayed,
    }


# Backwards-compatible concise aliases for callers/tests.
seal_schedule = seal_transfer_year_schedule
load_schedule_seal = load_transfer_year_schedule_seal
