from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from training import build_bgp_routing_churn_relay_source as builder


def _mrt_record(timestamp: int, subtype: int, payload: bytes = b"x") -> bytes:
    return builder.protocol.MRT_COMMON_HEADER.pack(
        timestamp,
        builder.protocol.MRT_TYPE_BGP4MP,
        subtype,
        len(payload),
    ) + payload


def _fetched(label: datetime, *, body_suffix: bytes = b"") -> builder.FetchedObject:
    timestamp = int(label.timestamp())
    stream = (
        _mrt_record(timestamp, 0, b"state")
        + _mrt_record(timestamp + 1, 4, b"message" + body_suffix)
    )
    body = gzip.compress(stream, mtime=0)
    return builder.FetchedObject(
        body=body,
        last_modified=label.strftime("%a, %d %b %Y %H:05:10 GMT"),
        etag=f'"{timestamp:x}-{len(body):x}"',
        declared_content_length=len(body),
    )


def _entry(label: datetime, index: int, *, available: bool = True) -> dict[str, Any]:
    if not available:
        return builder.missing_object_entry(label)
    identity = f"{index + 1:064x}"
    compressed_sha = f"{index + 100_000:064x}"
    source_row = {
        "archive_label_utc": builder._utc_iso(label),
        "object_identity_sha256": identity,
    }
    return {
        "object": {
            **source_row,
            "status": "available",
            "compressed_sha256": compressed_sha,
            "replay_required": label in builder.REPLAY_AUDIT_LABELS,
            "replay_verified": label in builder.REPLAY_AUDIT_LABELS,
        },
        "rows": [source_row],
    }


def test_builder_is_bound_to_exact_v1_protocol() -> None:
    payload = builder.load_frozen_protocol()
    assert payload["protocol_version"] == "bgp_routing_churn_relay_source_v1"
    assert payload["manifest_hash"] == builder.PROTOCOL_MANIFEST_HASH
    assert builder.sha256_file(builder.PROTOCOL_PATH) == builder.PROTOCOL_FILE_SHA256


def test_process_object_retains_only_frozen_aggregate_source_fields() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    entry = builder.process_fetched_object(label, _fetched(label))
    object_row = entry["object"]
    source_row = entry["rows"][0]
    assert object_row["status"] == "available"
    assert object_row["replay_required"] is True
    assert object_row["replay_verified"] is False
    assert source_row["source_availability_utc"] == "2023-01-01T00:15:00Z"
    assert source_row["record_count"] == 2
    assert source_row["message_record_count"] == 1
    assert source_row["state_change_record_count"] == 1
    assert source_row["type_subtype_counts"] == {"16:0": 1, "16:4": 1}
    assert "transport" not in source_row
    assert "payload" not in json.dumps(source_row)


def test_process_object_rejects_metadata_drift_and_bad_gzip() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    fetched = _fetched(label)
    late = builder.FetchedObject(
        body=fetched.body,
        last_modified="Sun, 01 Jan 2023 01:00:00 GMT",
        etag=fetched.etag,
        declared_content_length=len(fetched.body),
    )
    with pytest.raises(builder.SourceContractError, match="source contract"):
        builder.process_fetched_object(label, late)

    non_gzip = builder.FetchedObject(
        body=b"not-gzip",
        last_modified=fetched.last_modified,
        etag='"1-8"',
        declared_content_length=8,
    )
    with pytest.raises(builder.SourceContractError, match="not a gzip"):
        builder.process_fetched_object(label, non_gzip)


def test_replay_requires_exact_body_and_headers() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    original = _fetched(label)
    builder.verify_replay(label, original, original)
    with pytest.raises(builder.SourceContractError, match="missing"):
        builder.verify_replay(label, original, None)
    with pytest.raises(builder.SourceContractError, match="differ"):
        builder.verify_replay(label, original, _fetched(label, body_suffix=b"changed"))


def test_checkpoint_is_resumable_only_as_contiguous_bound_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.sqlite3"
    checkpoint = builder.SourceCheckpoint(checkpoint_path, {"binding": "fixture"})
    completed, total = builder.collect_to_checkpoint(
        checkpoint,
        _fetched,
        max_new_objects=2,
    )
    assert completed == 2
    assert total == 5844
    checkpoint.close()

    resumed = builder.SourceCheckpoint(checkpoint_path, {"binding": "fixture"})
    completed, _ = builder.collect_to_checkpoint(
        resumed,
        _fetched,
        max_new_objects=1,
    )
    assert completed == 3
    resumed.close()
    with pytest.raises(RuntimeError, match="binding differs"):
        builder.SourceCheckpoint(checkpoint_path, {"binding": "changed"})


def test_checkpoint_payload_hash_detects_local_corruption(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.sqlite3"
    checkpoint = builder.SourceCheckpoint(path, {"binding": "fixture"})
    label = builder.protocol.archive_labels()[0]
    checkpoint.put(label, builder.missing_object_entry(label))
    checkpoint.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE objects SET payload = '{}' WHERE label = ?",
            (builder._utc_iso(label),),
        )
        connection.commit()
    reopened = builder.SourceCheckpoint(path, {"binding": "fixture"})
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        reopened.entries()
    reopened.close()


def test_parse_failure_requires_same_source_bytes_before_fatal_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(_: datetime, __: builder.FetchedObject) -> dict[str, object]:
        raise builder.SourceContractError("fixture parse failure")

    monkeypatch.setattr(builder, "process_fetched_object", reject)
    label = builder.protocol.archive_labels()[0]
    first = _fetched(label)
    changed = _fetched(label, body_suffix=b"changed")
    payloads = iter((first, changed))
    checkpoint = builder.SourceCheckpoint(tmp_path / "changed.sqlite3", {"v": 1})
    with pytest.raises(builder.SourceFetchError, match="changed"):
        builder.collect_to_checkpoint(
            checkpoint,
            lambda _: next(payloads),
            max_new_objects=1,
        )
    assert checkpoint.labels() == []
    checkpoint.close()

    checkpoint = builder.SourceCheckpoint(tmp_path / "same.sqlite3", {"v": 1})
    with pytest.raises(builder.SourceContractError, match="fixture parse failure"):
        builder.collect_to_checkpoint(
            checkpoint,
            lambda _: first,
            max_new_objects=1,
        )
    checkpoint.close()


def test_support_uses_full_denominators_replay_and_duplicate_gates() -> None:
    labels = builder.protocol.archive_labels()
    entries = [_entry(label, index) for index, label in enumerate(labels)]
    passed = builder.evaluate_support(entries)
    assert passed["all_gates_passed"] is True
    assert passed["source_row_count"] == 5844
    assert passed["replay_required_objects"] == 48
    assert passed["replay_verified_objects"] == 48

    for index in range(5):
        entries[index] = _entry(labels[index], index, available=False)
    rejected = builder.evaluate_support(entries)
    assert rejected["all_gates_passed"] is False
    streak = next(
        gate
        for gate in rejected["gates"]
        if gate["gate_id"] == "maximum_consecutive_missing_objects"
    )
    assert streak["observed"] == 5
    assert streak["passed"] is False

    entries = [_entry(label, index) for index, label in enumerate(labels)]
    entries[1]["object"]["compressed_sha256"] = entries[0]["object"][
        "compressed_sha256"
    ]
    duplicate = builder.evaluate_support(entries)
    duplicate_gate = next(
        gate
        for gate in duplicate["gates"]
        if gate["gate_id"] == "duplicate_sha_across_labels"
    )
    assert duplicate_gate["observed"] == 1
    assert duplicate_gate["passed"] is False


def test_final_artifacts_are_deterministic_write_once_and_outcome_blind(
    tmp_path: Path,
) -> None:
    labels = builder.protocol.archive_labels()
    entries = [_entry(label, index) for index, label in enumerate(labels)]
    paths = {
        "source_path": tmp_path / "source.jsonl.gz",
        "object_manifest_path": tmp_path / "objects.jsonl.gz",
        "source_manifest_path": tmp_path / "manifest.json",
        "support_path": tmp_path / "support.json",
    }
    first = builder.finalize_artifacts(entries, **paths)
    second = builder.finalize_artifacts(entries, **paths)
    assert first["status"] == "PASS_ADVANCE_TO_CHURN_FREEZE"
    assert second["artifact_statuses"] == {
        "source": "verified_existing",
        "object_manifest": "verified_existing",
        "support": "verified_existing",
        "source_manifest": "verified_existing",
    }
    support = json.loads(paths["support_path"].read_text())
    manifest = json.loads(paths["source_manifest_path"].read_text())
    assert support["outcomes_opened"] is False
    assert support["market_clocks_opened"] is False
    assert manifest["mechanism_parser_opened"] is False
    assert gzip.decompress(paths["source_path"].read_bytes()).count(b"\n") == 5844


def test_url_disk_and_fatal_rejection_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    builder.validate_official_archive_url(label, builder.protocol.archive_url(label))
    with pytest.raises(ValueError, match="official route"):
        builder.validate_official_archive_url(
            label,
            "https://example.com/updates.20230101.0000.gz",
        )
    monkeypatch.setattr(builder, "_used_gib", lambda _: builder.protocol.DISK_LIMIT_GIB)
    with pytest.raises(builder.SourceFetchError, match="disk guard"):
        builder.enforce_disk_guard(tmp_path / "checkpoint.sqlite3")

    rejection = tmp_path / "rejection.json"
    assert builder.write_fatal_rejection(rejection, "fixture corruption") == "created"
    payload = json.loads(rejection.read_text())
    assert payload["status"] == "REJECT_NO_REPAIR"
    assert payload["outcomes_opened"] is False
