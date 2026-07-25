"""Build outcome-blind BCTP-12H source sequences and support evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import io
import json
from numbers import Integral
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import pandas as pd

from training import build_block_clearing_relational_topology_support as bcrt_s
from training import preregister_block_clearing_relational_topology as bcrt_p
from training import preregister_block_clearing_target_position_mdp as prereg


PROTOCOL_VERSION = "block_clearing_target_position_mdp_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_block_clearing_target_position_mdp_support.py"
)
TEST_PATH = Path(
    "tests/test_build_block_clearing_target_position_mdp_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/bctp-source-support-implementation-contract-2026-07-25.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "5d323839fd9ae6116bc6cefabcbf2c0585842a605aea17ff0f3b275c3cd95e9b"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_block_clearing_target_position_mdp.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "2bea5f3bfb5d0fd1985bf74ff7fe4cf7d43de7427378223af8c6f5cb65f2199c"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "dfdc18c61f578425ee4459ef30bdede97032c364af55f846365e0687694fdbc8"
)
PREREGISTRATION_MANIFEST_HASH = (
    "3c84d896c0d5e5c2917d06c9e34e786f6b0f8e396798971e1da35087f5d40635"
)
BCRT_SUPPORT_ARTIFACT = Path(
    "results/block_clearing_relational_topology_support_2026-07-24.json"
)
DEFAULT_SEQUENCE_OUTPUT = Path(prereg.SEQUENCE_OUTPUT)
DEFAULT_REPORT_OUTPUT = Path(prereg.SUPPORT_OUTPUT)

SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
DEVELOPMENT_END = pd.Timestamp("2023-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
WINDOWS = {
    "development": (SOURCE_START, DEVELOPMENT_END),
    "train": (SOURCE_START, pd.Timestamp("2022-01-01T00:00:00Z")),
    "2020": (SOURCE_START, pd.Timestamp("2021-01-01T00:00:00Z")),
    "2021": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2022": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        DEVELOPMENT_END,
    ),
    "2023": (DEVELOPMENT_END, SOURCE_END),
}
EXPECTED_REPLAY_COUNTS = {
    "formed_buckets": 2_918,
    "rank_complete_states": 2_792,
    "token_ready_states": 2_791,
}
BCRT_TOKEN_CANDIDATE_COLUMNS = (
    "signal_id",
    *bcrt_s.INTERNAL_TIME_COLUMNS,
    *bcrt_p.TOKEN_COLUMNS,
)
BCTP_TOKEN_CANDIDATE_COLUMNS = (
    "signal_id",
    "bucket_start",
    "confirmation_height",
    *(
        column
        for column in bcrt_s.INTERNAL_TIME_COLUMNS
        if column != "bucket_start"
    ),
    *bcrt_p.TOKEN_COLUMNS,
)
BATCH_SOURCE_COLUMNS = (
    "signal_id",
    "bucket_start",
    "confirmation_height",
    "entry_time",
    *bcrt_p.TOKEN_COLUMNS,
)
FORBIDDEN_COLUMN_FRAGMENTS = (
    "position",
    "action",
    "side",
    "price",
    "return",
    "funding",
    "reward",
    "pnl",
    "cagr",
    "mdd",
    "raw",
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH), str(IMPLEMENTATION_CONTRACT))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("BCTP source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("BCTP source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION_SOURCE) != PREREGISTRATION_SOURCE_SHA256:
        raise RuntimeError("BCTP preregistration source hash drift")
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("BCTP preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("BCTP preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("BCTP preregistration manifest hash drift")
    for field, value in payload["outcome_boundary"].items():
        if value not in (0, False):
            raise RuntimeError(f"BCTP preregistration boundary opened: {field}")
    if payload["report_only_2023"]["may_change_support_boolean"] is not False:
        raise RuntimeError("BCTP 2023 source incidence can alter support")
    return payload


def _source_dependencies() -> dict[str, str]:
    frozen = prereg.build_manifest()["immutable_bcrt_representation"]
    return {
        str(IMPLEMENTATION_CONTRACT): IMPLEMENTATION_CONTRACT_SHA256,
        str(PREREGISTRATION_SOURCE): PREREGISTRATION_SOURCE_SHA256,
        str(PREREGISTRATION): PREREGISTRATION_SHA256,
        prereg.BOUNDARY_DOCUMENT: prereg.BOUNDARY_DOCUMENT_SHA256,
        frozen["preregistration_source"]["path"]: frozen[
            "preregistration_source"
        ]["sha256"],
        frozen["support_source"]["path"]: frozen["support_source"]["sha256"],
        frozen["preregistration_artifact"]["path"]: frozen[
            "preregistration_artifact"
        ]["sha256"],
        frozen["support_artifact"]["path"]: frozen["support_artifact"][
            "sha256"
        ],
        frozen["retirement"]["path"]: frozen["retirement"]["sha256"],
        frozen["raw_source"]["path"]: frozen["raw_source"]["sha256"],
        frozen["source_manifest"]["path"]: frozen["source_manifest"]["sha256"],
        frozen["reference"]["path"]: frozen["reference"]["sha256"],
    }


def _without_manifest_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }


def verify_pre_source_bindings(
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    audit: dict[str, Any] = {}
    for path, expected in _source_dependencies().items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"BCTP frozen source binding changed: {path}")
        audit[path] = {"path": path, "sha256": actual}

    frozen = preregistration["immutable_bcrt_representation"]
    if frozen["retirement"] != {
        "path": prereg.BCRT_RETIREMENT_DOCUMENT,
        "sha256": prereg.BCRT_RETIREMENT_DOCUMENT_SHA256,
        "remains_terminal": True,
        "failed_gap_gate_not_changed": True,
    }:
        raise RuntimeError("BCTP changed terminal BCRT retirement")
    if frozen["expected_replay_counts"] != EXPECTED_REPLAY_COUNTS:
        raise RuntimeError("BCTP expected BCRT replay counts drift")

    bcrt_preregistration = bcrt_s.validate_preregistration()
    audit["bcrt_pre_source_bindings"] = bcrt_s.verify_pre_source_bindings(
        bcrt_preregistration
    )

    bcrt_support = json.loads(
        _path(BCRT_SUPPORT_ARTIFACT).read_text(encoding="utf-8")
    )
    if bcrt_support.get("manifest_hash") != canonical_hash(
        _without_manifest_hash(bcrt_support)
    ):
        raise RuntimeError("BCTP bound BCRT support manifest is invalid")
    if bcrt_support.get("manifest_hash") != prereg.BCRT_SUPPORT_MANIFEST_HASH:
        raise RuntimeError("BCTP bound BCRT support manifest drift")
    if bcrt_support.get("policy_id") != bcrt_p.POLICY_ID:
        raise RuntimeError("BCTP bound BCRT support policy drift")
    if bcrt_support.get("outcomes_opened") is not False:
        raise RuntimeError("BCTP bound BCRT support opened outcomes")
    forbidden = bcrt_support["outcome_boundary"]
    for field in (
        "BTC_market_rows_decoded",
        "funding_rows_decoded",
        "comparator_rows_decoded",
        "future_return_rows_decoded",
        "return_or_PnL_fields_decoded",
        "PnL_CAGR_MDD_values_decoded",
        "post_2023_rows_decoded",
        "model_labels_created",
        "model_training_runs",
        "network_calls",
    ):
        if forbidden[field] != 0:
            raise RuntimeError(f"BCTP bound BCRT artifact opened: {field}")
    audit["bcrt_support_artifact"] = {
        "path": str(BCRT_SUPPORT_ARTIFACT),
        "sha256": prereg.BCRT_SUPPORT_ARTIFACT_SHA256,
        "manifest_hash": bcrt_support["manifest_hash"],
        "terminal_decision": bcrt_support["decision"],
    }
    return audit, bcrt_support


def _candidate_bytes(rows: pd.DataFrame) -> bytes:
    if list(rows.columns) != list(BCRT_TOKEN_CANDIDATE_COLUMNS):
        raise RuntimeError("BCTP BCRT candidate projection schema drift")
    serialized = rows.sort_values(
        ["entry_time", "bucket_start", "signal_id"],
        kind="mergesort",
    ).copy()
    for column in bcrt_s.INTERNAL_TIME_COLUMNS:
        serialized[column] = serialized[column].map(bcrt_s._format_time)
    return serialized.to_csv(
        index=False,
        columns=BCRT_TOKEN_CANDIDATE_COLUMNS,
        lineterminator="\n",
    ).encode("utf-8")


def enrich_token_candidates(
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    candidates, feature_funnel = bcrt_s.build_token_candidates(ranked)
    if list(candidates.columns) != list(BCRT_TOKEN_CANDIDATE_COLUMNS):
        raise RuntimeError("BCTP frozen BCRT token candidate schema changed")
    complete = ranked.loc[ranked["rank_complete"]].reset_index(drop=True)
    predecessors_removed = complete.iloc[1:].reset_index(drop=True)
    if len(predecessors_removed) != len(candidates):
        raise RuntimeError("BCTP confirmation alignment count mismatch")
    if not pd.to_datetime(
        predecessors_removed["bucket_start"],
        utc=True,
    ).equals(pd.to_datetime(candidates["bucket_start"], utc=True)):
        raise RuntimeError("BCTP confirmation alignment bucket mismatch")

    confirmations = pd.to_numeric(
        predecessors_removed["confirmation_height"],
        errors="raise",
    )
    if (
        not pd.api.types.is_integer_dtype(confirmations.dtype)
        or confirmations.le(0).any()
    ):
        raise RuntimeError("BCTP confirmation heights are not positive integers")

    enriched = candidates.copy()
    enriched.insert(2, "confirmation_height", confirmations.astype("int64"))
    if list(enriched.columns) != list(BCTP_TOKEN_CANDIDATE_COLUMNS):
        raise RuntimeError("BCTP enriched token candidate schema drift")
    common = enriched.drop(columns=["confirmation_height"])
    candidate_bytes = _candidate_bytes(candidates)
    common_bytes = _candidate_bytes(common)
    replay_identical = common_bytes == candidate_bytes
    if not replay_identical:
        raise RuntimeError("BCTP changed a frozen BCRT token candidate")
    development_mask = pd.to_datetime(
        candidates["entry_time"],
        utc=True,
    ).lt(DEVELOPMENT_END)
    development_candidates = candidates.loc[
        development_mask
    ].reset_index(drop=True)
    development_common = common.loc[
        development_mask
    ].reset_index(drop=True)
    development_identical = (
        _candidate_bytes(development_candidates)
        == _candidate_bytes(development_common)
    )
    return enriched, feature_funnel, {
        "development_gate": {
            "common_projection_identical": development_identical,
            "common_projection_sha256": hashlib.sha256(
                _candidate_bytes(development_common)
            ).hexdigest(),
            "rows_compared": int(len(development_common)),
        },
        "full_source_report_only": {
            "common_projection_identical": replay_identical,
            "common_projection_sha256": hashlib.sha256(common_bytes).hexdigest(),
            "rows_compared": int(len(candidates)),
            "confirmation_height_added_only": True,
        },
    }


def batch_source_rows(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    if list(candidates.columns) != list(BCTP_TOKEN_CANDIDATE_COLUMNS):
        raise RuntimeError("BCTP token candidate input schema drift")
    frame = candidates.loc[:, list(BATCH_SOURCE_COLUMNS)].copy()
    frame["confirmation_height"] = frame["confirmation_height"].map(int)
    return frame.to_dict("records")


def _sequence_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=prereg.SOURCE_SEQUENCE_COLUMNS)
    if list(frame.columns) != list(prereg.SOURCE_SEQUENCE_COLUMNS):
        raise RuntimeError("BCTP sequence frame schema drift")
    return frame


def _entry_time(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("BCTP sequence entry time must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _report_only_batch_actionable_releases(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[pd.Timestamp, pd.Timestamp, int, str]] = set()
    for source_row in rows:
        entry = _entry_time(source_row.get("entry_time"))
        bucket = _entry_time(source_row.get("bucket_start"))
        if entry < DEVELOPMENT_END or entry >= SOURCE_END:
            raise RuntimeError("BCTP report-only release is outside 2023")
        confirmation_height = source_row.get("confirmation_height")
        if (
            isinstance(confirmation_height, bool)
            or not isinstance(confirmation_height, Integral)
            or int(confirmation_height) <= 0
        ):
            raise ValueError(
                "BCTP confirmation_height must be a positive integer"
            )
        signal_id = str(source_row.get("signal_id", "")).strip()
        if not signal_id:
            raise ValueError("BCTP source signal_id must be nonempty")
        if bucket >= entry:
            raise ValueError("BCTP bucket_start must precede entry_time")
        missing = [
            column
            for column in bcrt_p.TOKEN_COLUMNS
            if column not in source_row
        ]
        if missing:
            raise ValueError("BCTP report-only source token schema changed")
        identity = (entry, bucket, int(confirmation_height), signal_id)
        if identity in identities:
            raise ValueError("BCTP duplicate same-release source identity")
        identities.add(identity)
        normalized.append(
            {
                "signal_id": signal_id,
                "bucket_start": bcrt_s._format_time(bucket),
                "confirmation_height": int(confirmation_height),
                "entry_time": bcrt_s._format_time(entry),
                **{
                    column: str(source_row[column])
                    for column in bcrt_p.TOKEN_COLUMNS
                },
                "_entry": entry,
                "_bucket": bucket,
            }
        )

    normalized.sort(
        key=lambda row: (
            row["_entry"],
            row["_bucket"],
            int(row["confirmation_height"]),
            str(row["signal_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(normalized):
        entry = normalized[cursor]["_entry"]
        end = cursor + 1
        while end < len(normalized) and normalized[end]["_entry"] == entry:
            end += 1
        winner = max(
            normalized[cursor:end],
            key=lambda row: (
                row["_bucket"],
                int(row["confirmation_height"]),
                str(row["signal_id"]),
            ),
        )
        selected.append(
            {
                key: value
                for key, value in winner.items()
                if not key.startswith("_")
            }
        )
        cursor = end
    entries = [_entry_time(row["entry_time"]) for row in selected]
    if any(right <= left for left, right in zip(entries, entries[1:])):
        raise RuntimeError(
            "BCTP report-only batched releases are not strictly increasing"
        )
    return selected


def _tolerant_source_signature(
    entry_times: Sequence[Any],
    snapshots: Sequence[Mapping[str, str]],
) -> str:
    if len(entry_times) != prereg.Policy().sequence_states:
        raise ValueError("BCTP source sequence must contain exactly three times")
    parsed = [_entry_time(value) for value in entry_times]
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError("BCTP source sequence times must be strictly increasing")
    lines = [
        " | ".join(
            f"{column.upper()}={str(snapshot[column])}"
            for column in bcrt_p.TOKEN_COLUMNS
        )
        for snapshot in snapshots
    ]
    return canonical_hash({"oldest_first_source_snapshots": lines})


def _build_sequences_from_actionable(
    actionable: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries = [_entry_time(row["entry_time"]) for row in actionable]
    if any(right <= left for left, right in zip(entries, entries[1:])):
        raise RuntimeError("BCTP actionable releases are not strictly increasing")
    output: list[dict[str, Any]] = []
    for current_index in range(2, len(actionable)):
        window = actionable[current_index - 2 : current_index + 1]
        snapshots = [
            {
                column: str(row[column])
                for column in bcrt_p.TOKEN_COLUMNS
            }
            for row in window
        ]
        source_ids = [str(row["signal_id"]) for row in window]
        record: dict[str, Any] = {
            "sequence_id": canonical_hash(
                {
                    "policy_id": prereg.POLICY_ID,
                    "source_signal_ids": source_ids,
                }
            ),
            "entry_time": bcrt_s._format_time(
                _entry_time(window[-1]["entry_time"])
            ),
            "source_signal_id_m2": source_ids[0],
            "source_signal_id_m1": source_ids[1],
            "source_signal_id_s0": source_ids[2],
            "source_signature": _tolerant_source_signature(
                [row["entry_time"] for row in window],
                snapshots,
            ),
        }
        for label, snapshot in zip(prereg.SEQUENCE_LABELS, snapshots):
            for token in bcrt_p.TOKEN_COLUMNS:
                record[f"{label.lower()}__{token}"] = snapshot[token]
        if tuple(record) != prereg.SOURCE_SEQUENCE_COLUMNS:
            raise RuntimeError("BCTP source sequence schema changed")
        output.append(record)
    return output


def _future_append_audit(
    actionable: Sequence[Mapping[str, Any]],
    sequence_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    local_windows_checked = 0
    local_passed = True
    for index in range(2, len(actionable)):
        rebuilt = _build_sequences_from_actionable(
            actionable[index - 2 : index + 1]
        )
        if len(rebuilt) != 1 or rebuilt[0] != dict(sequence_records[index - 2]):
            local_passed = False
            break
        local_windows_checked += 1

    annual_prefixes_checked = 0
    annual_prefix_passed = True
    for cutoff in (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
        SOURCE_END,
    ):
        prefix = [
            row
            for row in actionable
            if _entry_time(row["entry_time"]) < cutoff
        ]
        rebuilt = _build_sequences_from_actionable(prefix)
        expected = [dict(row) for row in sequence_records[: len(rebuilt)]]
        if rebuilt != expected:
            annual_prefix_passed = False
            break
        annual_prefixes_checked += 1
    return {
        "completed_three_release_windows_checked": local_windows_checked,
        "completed_three_release_windows_passed": local_passed,
        "annual_completed_release_prefixes_checked": annual_prefixes_checked,
        "annual_completed_release_prefixes_passed": annual_prefix_passed,
        "future_append_invariance_passed": (
            local_passed and annual_prefix_passed
        ),
    }


def _batching_stats(
    release_count: int,
    actionable: Sequence[Mapping[str, Any]],
    sequence_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = [_entry_time(row["entry_time"]) for row in actionable]
    strictly_increasing = all(
        right > left for left, right in zip(entries, entries[1:])
    )
    expected_sequences = max(0, len(actionable) - 2)
    return {
        "token_ready_source_states": int(release_count),
        "actionable_releases": int(len(actionable)),
        "same_release_suppressed": int(release_count - len(actionable)),
        "actionable_entries_strictly_increasing": strictly_increasing,
        "warmup_actionable_releases": min(2, len(actionable)),
        "expected_sequence_rows": expected_sequences,
        "actual_sequence_rows": int(len(sequence_records)),
        "warmup_exact": len(sequence_records) == expected_sequences,
        "first_actionable_entry": (
            bcrt_s._format_time(entries[0]) if entries else None
        ),
        "last_actionable_entry": (
            bcrt_s._format_time(entries[-1]) if entries else None
        ),
    }


def build_sequence_artifacts(
    release_rows: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    eligible_release_rows = [
        row
        for row in release_rows
        if _entry_time(row["entry_time"]) < SOURCE_END
    ]
    post_2023_releases = len(release_rows) - len(eligible_release_rows)
    development_release_rows = [
        row
        for row in eligible_release_rows
        if _entry_time(row["entry_time"]) < DEVELOPMENT_END
    ]
    report_only_release_rows = [
        row
        for row in eligible_release_rows
        if _entry_time(row["entry_time"]) >= DEVELOPMENT_END
    ]
    development_actionable = prereg.batch_actionable_releases(
        development_release_rows
    )
    report_only_actionable = _report_only_batch_actionable_releases(
        report_only_release_rows
    )
    actionable = [*development_actionable, *report_only_actionable]
    records = _build_sequences_from_actionable(actionable)
    frame = _sequence_frame(records)
    development_records = prereg.build_source_sequences(
        development_release_rows
    )
    development_strict_replay = (
        _build_sequences_from_actionable(development_actionable)
        == development_records
    )
    full_development_records = [
        dict(row)
        for row in records
        if _entry_time(row["entry_time"]) < DEVELOPMENT_END
    ]
    development_matches_full = (
        development_records == full_development_records
    )
    full_source_report = _batching_stats(
        len(eligible_release_rows),
        actionable,
        records,
    )
    report_only_unknown = any(
        str(row[column]) not in bcrt_p.TOKEN_VOCABULARY[column]
        for row in report_only_actionable
        for column in bcrt_p.TOKEN_COLUMNS
    )
    strict_full_replay: bool | None = None
    if not report_only_unknown:
        strict_full_replay = (
            prereg.build_source_sequences(eligible_release_rows) == records
        )
    full_source_report[
        "development_sequences_match_full_prefix"
    ] = development_matches_full
    full_source_report["strict_full_replay_when_vocabulary_known"] = (
        strict_full_replay
    )
    full_source_report["post_2023_release_states_omitted"] = int(
        post_2023_releases
    )
    full_source_report["unknown_2023_vocabulary_present"] = (
        report_only_unknown
    )
    batching_audit = {
        "development_boolean": {
            **_batching_stats(
                len(development_release_rows),
                development_actionable,
                development_records,
            ),
            "strict_sequence_builder_replay_exact": (
                development_strict_replay
            ),
        },
        "full_source_report_only": full_source_report,
    }
    append_audit = {
        "development_boolean": _future_append_audit(
            development_actionable,
            development_records,
        ),
        "full_source_report_only": _future_append_audit(actionable, records),
    }
    return frame, batching_audit, append_audit


def _window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    entries = pd.to_datetime(rows["entry_time"], utc=True)
    return rows.loc[entries.ge(start) & entries.lt(end)].reset_index(drop=True)


def _maximum_gap_days(rows: pd.DataFrame) -> int | None:
    if len(rows) < 2:
        return None
    dates = [
        pd.Timestamp(value).date()
        for value in pd.to_datetime(
            rows["entry_time"],
            utc=True,
        ).sort_values(kind="mergesort")
    ]
    return max(
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
    )


def _unknown_vocabulary(rows: pd.DataFrame) -> dict[str, list[str]]:
    unknown: dict[str, list[str]] = {}
    for column in prereg.SOURCE_TOKEN_COLUMNS:
        token = column.split("__", maxsplit=1)[1]
        observed = set(rows[column].astype(str))
        invalid = sorted(observed - set(bcrt_p.TOKEN_VOCABULARY[token]))
        if invalid:
            unknown[column] = invalid
    return unknown


def sequence_report(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    entries = pd.to_datetime(rows["entry_time"], utc=True)
    months = Counter(value.strftime("%Y-%m") for value in entries)
    signatures = (
        rows["source_signature"].astype(str).value_counts()
        if total
        else pd.Series(dtype="int64")
    )
    return {
        "events": int(total),
        "first_entry": (
            bcrt_s._format_time(entries.min()) if total else None
        ),
        "last_entry": (
            bcrt_s._format_time(entries.max()) if total else None
        ),
        "active_months": int(len(months)),
        "maximum_month_share": (
            float(max(months.values()) / total) if total else None
        ),
        "maximum_gap_days_report_only": _maximum_gap_days(rows),
        "distinct_source_signatures": int(len(signatures)),
        "maximum_exact_source_signature_share": (
            float(signatures.max() / total) if total else None
        ),
        "unknown_vocabulary": _unknown_vocabulary(rows),
    }


def _development_tokens_valid(rows: pd.DataFrame) -> bool:
    return not _unknown_vocabulary(rows)


def support_checks(
    sequences: pd.DataFrame,
    *,
    batching_audit: Mapping[str, Any],
    append_audit: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    reports = {
        name: sequence_report(_window(sequences, name))
        for name in WINDOWS
    }
    development = _window(sequences, "development")
    exact_schema = list(sequences.columns) == list(
        prereg.SOURCE_SEQUENCE_COLUMNS
    )
    lowered_columns = [column.lower() for column in sequences.columns]
    outcome_free = exact_schema and not any(
        fragment in column
        for column in lowered_columns
        for fragment in FORBIDDEN_COLUMN_FRAGMENTS
    )
    development_batching = batching_audit.get("development_boolean", {})
    development_append = append_audit.get("development_boolean", {})
    development_replay = replay_audit.get("development_gate", {})
    train_2022_replay = replay_audit.get("train_2022_gate", {})
    formed = int(development_replay.get("formed_buckets", 0))
    ranked = int(development_replay.get("rank_complete_states", 0))
    token_ready = int(development_replay.get("token_ready_states", 0))
    checks = {
        "bcrt_development_replay_counts_ordered": (
            formed >= ranked >= token_ready > 0
        ),
        "bcrt_development_token_ready_states_min_1502": token_ready >= 1_502,
        "bcrt_development_common_projection_identical": (
            development_replay.get("common_projection_identical") is True
        ),
        "bcrt_development_prefix_replay_passed": (
            development_replay.get("prefix_replay_passed") is True
        ),
        "bcrt_train_2022_reports_exact": (
            train_2022_replay.get("reports_exact") is True
        ),
        "bcrt_train_2022_checks_exact": (
            train_2022_replay.get("checks_exact") is True
        ),
        "bcrt_train_2022_checks_all_true": (
            train_2022_replay.get("checks_all_true") is True
        ),
        "actionable_entries_strictly_increasing": (
            development_batching.get(
                "actionable_entries_strictly_increasing"
            ) is True
        ),
        "first_two_actionable_releases_are_warmup_only": (
            development_batching.get("warmup_exact") is True
            and development_batching.get("actual_sequence_rows")
            == max(
                0,
                int(development_batching.get("actionable_releases", 0)) - 2,
            )
        ),
        "development_sequence_builder_matches_frozen": (
            development_batching.get(
                "strict_sequence_builder_replay_exact"
            ) is True
        ),
        "future_append_invariance": (
            development_append.get("future_append_invariance_passed") is True
        ),
        "sequence_schema_exact": exact_schema,
        "sequence_schema_outcome_free": outcome_free,
        "development_tokens_valid": _development_tokens_valid(development),
        "year_2020_sequences_min_500": reports["2020"]["events"] >= 500,
        "year_2021_sequences_min_500": reports["2021"]["events"] >= 500,
        "year_2022_sequences_min_500": reports["2022"]["events"] >= 500,
        "year_2021_active_months_12": reports["2021"]["active_months"] == 12,
        "year_2022_active_months_12": reports["2022"]["active_months"] == 12,
        "development_signature_share_max_5pct": bool(
            reports["development"][
                "maximum_exact_source_signature_share"
            ]
            is not None
            and reports["development"][
                "maximum_exact_source_signature_share"
            ]
            <= prereg.Policy().source_signature_share_max
        ),
        "train_signature_share_max_5pct": bool(
            reports["train"]["maximum_exact_source_signature_share"] is not None
            and reports["train"]["maximum_exact_source_signature_share"]
            <= prereg.Policy().source_signature_share_max
        ),
        "selection_2022_signature_share_max_5pct": bool(
            reports["2022"]["maximum_exact_source_signature_share"] is not None
            and reports["2022"]["maximum_exact_source_signature_share"]
            <= prereg.Policy().source_signature_share_max
        ),
    }
    report_only_2023 = {
        **reports["2023"],
        "boolean_gate": False,
        "may_authorize_continue_retire_repair_or_selection": False,
        "unknown_vocabulary_operational_action": "TARGET_FLAT",
    }
    decision_reports = {
        key: value for key, value in reports.items() if key != "2023"
    }
    return decision_reports, checks, report_only_2023


def first_failure(
    checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    for name, passed in checks.items():
        if not passed:
            return "source_sequence_support", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    return "none", None


def deterministic_sequence_bytes(rows: pd.DataFrame) -> bytes:
    if list(rows.columns) != list(prereg.SOURCE_SEQUENCE_COLUMNS):
        raise RuntimeError("BCTP sequence schema drift")
    serialized = rows.sort_values(
        ["entry_time", "sequence_id"],
        kind="mergesort",
    ).copy()
    serialized["entry_time"] = serialized["entry_time"].map(
        bcrt_s._format_time
    )
    text = serialized.to_csv(
        index=False,
        columns=prereg.SOURCE_SEQUENCE_COLUMNS,
        lineterminator="\n",
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        mtime=0,
    ) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def _frame_hash(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values(
        ["entry_time", "sequence_id"],
        kind="mergesort",
    ).copy()
    ordered["entry_time"] = ordered["entry_time"].map(bcrt_s._format_time)
    return canonical_hash(
        ordered.loc[:, list(prereg.SOURCE_SEQUENCE_COLUMNS)].to_dict("records")
    )


def deterministic_report_bytes(report: Mapping[str, Any]) -> bytes:
    payload = dict(report)
    if payload.get("manifest_hash") != canonical_hash(
        _without_manifest_hash(payload)
    ):
        raise RuntimeError("BCTP support report manifest hash mismatch")
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _core_payload(
    sequences: pd.DataFrame,
    *,
    source_audit: Mapping[str, Any],
    bucket_audit: Mapping[str, Any],
    feature_funnel: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
    batching_audit: Mapping[str, Any],
    append_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    sequence_bytes: bytes,
) -> dict[str, Any]:
    artifact_eligible = False
    reports, checks, report_only_2023 = support_checks(
        sequences,
        batching_audit=batching_audit,
        append_audit=append_audit,
        replay_audit=replay_audit,
    )
    support_passed = all(checks.values())
    first_stage, first_check = first_failure(
        checks,
        artifact_eligible=artifact_eligible,
    )
    if not support_passed:
        decision = "retire_BCTP_12H_unchanged_before_market_access"
    else:
        decision = "synthetic_build_cannot_authorize_market_access"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "artifact_eligible": artifact_eligible,
        "source_sequence_incidence_opened": artifact_eligible,
        "outcomes_opened": False,
        "market_loaded": False,
        "funding_loaded": False,
        "post_2023_loaded": False,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "tests": str(TEST_PATH),
            "tests_sha256": sha256_file(TEST_PATH),
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "source_audit": dict(source_audit),
        "bucket_audit": dict(bucket_audit),
        "feature_funnel": dict(feature_funnel),
        "bcrt_replay_audit": dict(replay_audit),
        "batching_audit": dict(batching_audit),
        "future_append_audit": dict(append_audit),
        "development_sequence_reports": reports,
        "source_support_checks": checks,
        "source_support_passed": support_passed,
        "report_only_2023": report_only_2023,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
        "sequence_artifact": {
            "path": str(DEFAULT_SEQUENCE_OUTPUT),
            "sha256": hashlib.sha256(sequence_bytes).hexdigest(),
            "frame_hash": _frame_hash(sequences),
            "rows": int(len(sequences)),
            "columns": list(prereg.SOURCE_SEQUENCE_COLUMNS),
        },
        "decision": decision,
        "authorized_next_stage": (
            "freeze_economic_evaluator_and_cheap_policy_family"
            if support_passed and artifact_eligible
            else None
        ),
        "outcome_boundary": {
            "new_raw_source_rows_decoded": int(
                source_audit.get("source_rows_decoded", 0)
            ),
            "reference_rows_decoded": int(
                source_audit.get("reference_rows_decoded", 0)
            ),
            "bcrt_buckets_derived": int(
                bucket_audit.get("formed_buckets", 0)
            ),
            "bcrt_rank_rows_derived": int(
                feature_funnel.get("rank_complete_states", 0)
            ),
            "bcrt_token_rows_derived": int(
                feature_funnel.get("token_ready_states", 0)
            ),
            "bctp_sequences_derived": int(len(sequences)),
            "market_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "comparator_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_PnL_fields_decoded": 0,
            "PnL_CAGR_MDD_values_decoded": 0,
            "post_2023_rows_decoded": 0,
            "actions_or_labels_created": 0,
            "model_training_runs": 0,
            "network_calls": 0,
        },
        "binding_manifest_hash": preregistration["manifest_hash"],
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _synthetic_replay_audit() -> dict[str, Any]:
    return {
        "development_gate": {
            "formed_buckets": 2_190,
            "rank_complete_states": 2_064,
            "token_ready_states": 2_063,
            "common_projection_identical": True,
            "prefix_replay_passed": True,
        },
        "train_2022_gate": {
            "reports_exact": True,
            "checks_exact": True,
            "checks_all_true": True,
        },
        "full_clock_report_only": {
            **EXPECTED_REPLAY_COUNTS,
            "expected_replay_counts_exact": True,
            "common_projection_identical": True,
            "legacy_clock_sha_matches": True,
            "legacy_clock_frame_hash_matches": True,
            "legacy_clock_statistics_exact": True,
            "legacy_calendar_partitions_exact": True,
            "legacy_source_checks_exact": True,
            "legacy_eval_report_exact": True,
        },
        "synthetic_or_injected": True,
    }


def build_support_from_release_rows(
    release_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    sequences, batching_audit, append_audit = build_sequence_artifacts(
        release_rows
    )
    sequence_bytes = deterministic_sequence_bytes(sequences)
    payload = validate_preregistration()
    report = _core_payload(
        sequences,
        source_audit={
            "source_rows_decoded": 0,
            "reference_rows_decoded": 0,
            "synthetic_or_injected": True,
            "injected_token_ready_rows": int(len(release_rows)),
        },
        bucket_audit={"formed_buckets": 0, "synthetic_or_injected": True},
        feature_funnel={
            "rank_complete_states": 0,
            "token_ready_states": int(len(release_rows)),
            "synthetic_or_injected": True,
        },
        replay_audit=_synthetic_replay_audit(),
        batching_audit=batching_audit,
        append_audit=append_audit,
        preregistration=payload,
        sequence_bytes=sequence_bytes,
    )
    return report, sequence_bytes


def _real_replay_audit(
    candidates: pd.DataFrame,
    ranked: pd.DataFrame,
    feature_funnel: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    bucket_audit: Mapping[str, Any],
    frozen_support: Mapping[str, Any],
) -> dict[str, Any]:
    common = candidates.drop(columns=["confirmation_height"])
    reserved = bcrt_s.reserve_candidates(common)
    legacy_clock, reservation_funnel = bcrt_s.eligible_clock(reserved)
    legacy_clock_bytes = bcrt_s.deterministic_clock_bytes(legacy_clock)
    (
        statistics,
        partitions,
        token_reports,
        source_checks,
        token_checks,
        eval_report,
    ) = bcrt_s.support_checks(
        legacy_clock,
        development_replay_passed=bool(
            bucket_audit.get("development_prefix_replay_passed")
        ),
        eval_replay_report_only_passed=bool(
            bucket_audit.get("eval_prefix_replay_report_only_passed")
        ),
    )
    actual_clock_sha = hashlib.sha256(legacy_clock_bytes).hexdigest()
    actual_frame_hash = bcrt_s._frame_hash(legacy_clock)
    ranked_entries = pd.to_datetime(ranked["entry_time"], utc=True)
    development_ranked = ranked_entries.lt(DEVELOPMENT_END)
    candidate_entries = pd.to_datetime(candidates["entry_time"], utc=True)
    development_candidates = candidate_entries.lt(DEVELOPMENT_END)
    development_candidate_audit = candidate_audit["development_gate"]
    full_candidate_audit = candidate_audit["full_source_report_only"]
    full_counts = {
        "formed_buckets": int(feature_funnel["formed_buckets"]),
        "rank_complete_states": int(feature_funnel["rank_complete_states"]),
        "token_ready_states": int(feature_funnel["token_ready_states"]),
    }
    if full_counts != EXPECTED_REPLAY_COUNTS:
        raise RuntimeError("BCTP frozen full-period BCRT replay counts drift")
    return {
        "development_gate": {
            "formed_buckets": int(development_ranked.sum()),
            "rank_complete_states": int(
                (
                    development_ranked
                    & ranked["rank_complete"].astype(bool)
                ).sum()
            ),
            "token_ready_states": int(development_candidates.sum()),
            "common_projection_identical": development_candidate_audit[
                "common_projection_identical"
            ],
            "common_projection_sha256": development_candidate_audit[
                "common_projection_sha256"
            ],
            "rows_compared": development_candidate_audit["rows_compared"],
            "prefix_replay_passed": bool(
                bucket_audit.get("development_prefix_replay_passed")
            ),
        },
        "train_2022_gate": {
            "reports_exact": (
                token_reports == frozen_support["development_token_report"]
            ),
            "checks_exact": (
                token_checks == frozen_support["token_support_checks"]
            ),
            "checks_all_true": all(token_checks.values()),
        },
        "full_source_report_only": {
            **full_counts,
            "expected_replay_counts": dict(EXPECTED_REPLAY_COUNTS),
            "expected_replay_counts_exact": (
                full_counts == EXPECTED_REPLAY_COUNTS
            ),
            **dict(full_candidate_audit),
        },
        "full_clock_report_only": {
            "legacy_clock_sha256": actual_clock_sha,
            "legacy_clock_sha_matches": (
                actual_clock_sha == frozen_support["clock"]["sha256"]
            ),
            "legacy_clock_frame_hash": actual_frame_hash,
            "legacy_clock_frame_hash_matches": (
                actual_frame_hash == frozen_support["clock"]["frame_hash"]
            ),
            "legacy_clock_statistics_exact": (
                statistics == frozen_support["clock_statistics"]
            ),
            "legacy_calendar_partitions_exact": (
                partitions == frozen_support["calendar_partition_counts"]
            ),
            "legacy_source_checks_exact": (
                source_checks == frozen_support["source_support_checks"]
            ),
            "legacy_eval_report_exact": (
                eval_report == frozen_support["eval_source_report_only"]
            ),
            "legacy_reservation_funnel_exact": (
                reservation_funnel == frozen_support["reservation_funnel"]
            ),
        },
        "legacy_retired_decision_unchanged": frozen_support["decision"],
        "legacy_first_failure_unchanged": {
            "stage": frozen_support["first_failing_stage"],
            "check": frozen_support["first_failing_check"],
        },
    }


def build_real_support_payload() -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    payload = validate_preregistration()
    bindings, frozen_support = verify_pre_source_bindings(payload)
    source, reference = bcrt_s.load_source_frames()
    buckets, bucket_audit = bcrt_s.build_causal_buckets(source)
    ranked = bcrt_s.attach_strict_prior_ranks(buckets)
    candidates, feature_funnel, candidate_audit = enrich_token_candidates(
        ranked
    )
    replay_audit = _real_replay_audit(
        candidates,
        ranked,
        feature_funnel,
        candidate_audit,
        bucket_audit,
        frozen_support,
    )
    release_rows = batch_source_rows(candidates)
    sequences, batching_audit, append_audit = build_sequence_artifacts(
        release_rows
    )
    sequence_bytes = deterministic_sequence_bytes(sequences)
    source_audit = {
        "source": {
            "path": bcrt_p.SOURCE,
            "sha256": bcrt_p.SOURCE_SHA256,
            "header_sha256": bcrt_p.SOURCE_HEADER_SHA256,
            "allowlist": list(bcrt_p.SOURCE_ALLOWLIST),
        },
        "reference": {
            "path": bcrt_p.REFERENCE,
            "sha256": bcrt_p.REFERENCE_SHA256,
            "header_sha256": bcrt_p.REFERENCE_HEADER_SHA256,
            "allowlist": list(bcrt_p.REFERENCE_ALLOWLIST),
        },
        "source_rows_decoded": int(len(source)),
        "reference_rows_decoded": int(len(reference)),
        "pre_source_bindings": bindings,
        "synthetic_or_injected": False,
        "source_validation_passed": True,
        "reference_equality_passed": True,
    }
    if len(source) != 213_095 or len(reference) != 213_095:
        raise RuntimeError("BCTP real loaded-frame row evidence drift")
    if list(source.columns) != list(bcrt_p.SOURCE_ALLOWLIST):
        raise RuntimeError("BCTP real loaded source schema drift")
    if list(reference.columns) != list(bcrt_p.REFERENCE_ALLOWLIST):
        raise RuntimeError("BCTP real loaded reference schema drift")
    if not bindings or replay_audit.get("synthetic_or_injected") is True:
        raise RuntimeError("BCTP real verified evidence is unavailable")
    report = _core_payload(
        sequences,
        source_audit=source_audit,
        bucket_audit=bucket_audit,
        feature_funnel=feature_funnel,
        replay_audit=replay_audit,
        batching_audit=batching_audit,
        append_audit=append_audit,
        preregistration=payload,
        sequence_bytes=sequence_bytes,
    )
    authorized_core = _without_manifest_hash(report)
    authorized_core["artifact_eligible"] = True
    authorized_core["source_sequence_incidence_opened"] = True
    first_stage, first_check = first_failure(
        report["source_support_checks"],
        artifact_eligible=True,
    )
    authorized_core["first_failing_stage"] = first_stage
    authorized_core["first_failing_check"] = first_check
    if report["source_support_passed"]:
        authorized_core["decision"] = (
            "advance_to_frozen_economic_and_cheap_policy_evaluator"
        )
        authorized_core["authorized_next_stage"] = (
            "freeze_economic_evaluator_and_cheap_policy_family"
        )
    else:
        authorized_core["decision"] = (
            "retire_BCTP_12H_unchanged_before_market_access"
        )
        authorized_core["authorized_next_stage"] = None
    report = {
        **authorized_core,
        "manifest_hash": canonical_hash(authorized_core),
    }
    return report, sequence_bytes


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"BCTP noncanonical existing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != payload:
                raise RuntimeError(f"BCTP artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    sequence_output: str | Path = DEFAULT_SEQUENCE_OUTPUT,
) -> dict[str, Any]:
    if Path(report_output) != DEFAULT_REPORT_OUTPUT:
        raise RuntimeError("BCTP real report output path is frozen")
    if Path(sequence_output) != DEFAULT_SEQUENCE_OUTPUT:
        raise RuntimeError("BCTP real sequence output path is frozen")
    report, sequence_bytes = build_real_support_payload()
    sequence_status = _write_once(sequence_output, sequence_bytes)
    report_bytes = deterministic_report_bytes(report)
    report_status = _write_once(report_output, report_bytes)
    return {
        "report_status": report_status,
        "sequence_status": sequence_status,
        "report": str(report_output),
        "sequence": str(sequence_output),
        "source_support_passed": report["source_support_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT),
    )
    parser.add_argument(
        "--sequence-output",
        default=str(DEFAULT_SEQUENCE_OUTPUT),
    )
    args = parser.parse_args()
    result = write_support(args.report_output, args.sequence_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
