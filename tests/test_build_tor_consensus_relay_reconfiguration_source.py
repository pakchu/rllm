from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from training import build_tor_consensus_relay_reconfiguration_source as builder


def _parsed(label: datetime, raw: bytes) -> dict[str, Any]:
    relays = [
        {
            "relay_identity": "11" * 20,
            "flags": ["Fast", "Guard", "Running", "Valid"],
            "consensus_bandwidth": 1000,
        },
        {
            "relay_identity": "22" * 20,
            "flags": ["Exit", "Fast", "Running", "Valid"],
            "consensus_bandwidth": 500,
        },
    ]
    summary = {
        "valid_after": builder._utc_iso(label),
        "availability_time": builder._utc_iso(
            builder.protocol.public_availability_time(label)
        ),
        "fresh_until": builder._utc_iso(label.replace(hour=label.hour + 1)),
        "valid_until": builder._utc_iso(label.replace(hour=label.hour + 3)),
        "consensus_method": 32,
        "voting_delay_seconds": [300, 300],
        "known_flags": ["Exit", "Fast", "Guard", "Running", "Valid"],
        "authority_identities": [f"{seed:040X}" for seed in range(1, 6)],
        "signature_headers": [
            {
                "identity": f"{seed:040X}",
                "signing_key": f"{seed + 100:040X}",
                "signature_bytes": 128,
            }
            for seed in range(1, 6)
        ],
        "signed_portion_sha1": "ab" * 20,
        "relays": relays,
    }
    return {
        "member_bytes": len(raw),
        "member_sha256": builder.sha256_bytes(raw),
        "relay_count": len(relays),
        "authority_count": 5,
        "signature_count": 5,
        "summary_identity_sha256": builder.canonical_hash(summary),
        **summary,
    }


def _archive_body(label: datetime, raw: bytes = b"fixture-consensus\n") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        info = tarfile.TarInfo(builder.protocol.expected_member_name(label))
        info.size = len(raw)
        info.mtime = int(label.timestamp())
        archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def _archive_row(month: str, body: bytes) -> dict[str, Any]:
    filename = f"consensuses-{month}.tar.xz"
    return {
        "month": month,
        "filename": filename,
        "url": f"{builder.protocol.ARCHIVE_BASE}/{filename}",
        "compressed_bytes": len(body),
        "compressed_sha256": builder.sha256_bytes(body),
    }


def _fetched(body: bytes) -> builder.FetchedArchive:
    return builder.FetchedArchive(
        body=body,
        declared_content_length=len(body),
        content_type="application/x-xz",
    )


def _support_entries() -> list[dict[str, Any]]:
    labels = builder.protocol.anchor_labels()
    rows_by_month: dict[str, list[dict[str, Any]]] = {
        row["month"]: [] for row in builder.protocol.archive_manifest()
    }
    for index, label in enumerate(labels):
        month = label.strftime("%Y-%m")
        rows_by_month[month].append(
            {
                "source_label_utc": builder._utc_iso(label),
                "member_sha256": f"{index + 1:064x}",
                "source_row_identity_sha256": f"{index + 10000:064x}",
                "relay_count": 6000 + index % 100,
                "authority_count": 8,
                "signature_count": 7,
            }
        )
    entries = []
    for manifest_row in builder.protocol.archive_manifest():
        month = manifest_row["month"]
        month_rows = rows_by_month[month]
        entries.append(
            {
                "archive": {
                    "month": month,
                    "compressed_sha256": manifest_row["compressed_sha256"],
                    "compressed_bytes": manifest_row["compressed_bytes"],
                    "target_member_count": len(month_rows),
                    "parse_twice_verified": True,
                },
                "rows": month_rows,
            }
        )
    return entries


def test_builder_is_bound_to_exact_v1_protocol() -> None:
    payload = builder.load_frozen_protocol()
    assert payload["protocol_version"] == (
        "tor_consensus_relay_reconfiguration_source_v1"
    )
    assert payload["manifest_hash"] == builder.PROTOCOL_MANIFEST_HASH
    assert builder.sha256_file(builder.PROTOCOL_PATH) == builder.PROTOCOL_FILE_SHA256


def test_process_archive_is_deterministic_and_outcome_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = b"fixture-consensus\n"
    body = _archive_body(label, raw)
    row = _archive_row("2023-01", body)
    monkeypatch.setattr(builder, "labels_for_month", lambda _: [label])
    calls = 0

    def parse(payload: bytes, *, expected_label: datetime) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert payload == raw
        assert expected_label == label
        return _parsed(label, raw)

    monkeypatch.setattr(builder.protocol, "parse_consensus_document", parse)
    entry = builder.process_archive(row, _fetched(body))
    assert calls == 2
    assert entry["archive"]["target_member_count"] == 1
    assert entry["archive"]["parse_twice_verified"] is True
    source = entry["rows"][0]
    assert source["relay_count"] == 2
    assert source["guard_count"] == 1
    assert source["exit_count"] == 1
    assert source["total_consensus_bandwidth"] == 1500
    assert source["flag_counts"] == {
        "Exit": 1,
        "Fast": 2,
        "Guard": 1,
        "Running": 2,
        "Valid": 2,
    }
    serialized = json.dumps(source)
    assert "relay_identity" not in serialized
    assert "BTC" not in serialized
    assert builder.process_archive(row, _fetched(body)) == entry


def test_process_archive_rejects_hash_drift_and_unsafe_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    body = _archive_body(label)
    row = _archive_row("2023-01", body)
    monkeypatch.setattr(builder, "labels_for_month", lambda _: [label])
    damaged = dict(row)
    damaged["compressed_sha256"] = "00" * 32
    with pytest.raises(builder.SourceContractError, match="SHA-256"):
        builder.process_archive(damaged, _fetched(body))

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        info = tarfile.TarInfo(builder.protocol.expected_member_name(label))
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape"
        archive.addfile(info)
    linked = buffer.getvalue()
    with pytest.raises(builder.SourceContractError, match="contract failed"):
        builder.process_archive(_archive_row("2023-01", linked), _fetched(linked))


def test_checkpoint_resume_requires_a_contiguous_bound_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = [
        {"month": month, "compressed_sha256": f"{index:064x}"}
        for index, month in enumerate(("2020-01", "2020-02", "2020-03"), start=1)
    ]
    monkeypatch.setattr(builder.protocol, "archive_manifest", lambda: manifest)
    monkeypatch.setattr(
        builder,
        "process_archive",
        lambda row, fetched: {"archive": {"month": row["month"]}, "rows": []},
    )
    checkpoint_path = tmp_path / "checkpoint.sqlite3"
    checkpoint = builder.SourceCheckpoint(checkpoint_path, {"binding": "fixture"})
    completed, total = builder.collect_to_checkpoint(
        checkpoint,
        lambda _: _fetched(b"fixture"),
        max_new_archives=2,
    )
    assert (completed, total) == (2, 3)
    checkpoint.close()

    resumed = builder.SourceCheckpoint(checkpoint_path, {"binding": "fixture"})
    assert builder.collect_to_checkpoint(
        resumed,
        lambda _: _fetched(b"fixture"),
    ) == (3, 3)
    resumed.close()
    with pytest.raises(RuntimeError, match="binding differs"):
        builder.SourceCheckpoint(checkpoint_path, {"binding": "changed"})


def test_checkpoint_payload_hash_detects_local_corruption(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.sqlite3"
    checkpoint = builder.SourceCheckpoint(path, {"binding": "fixture"})
    checkpoint.put("2020-01", {"archive": {"month": "2020-01"}, "rows": []})
    checkpoint.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE archives SET payload = '{}' WHERE month = '2020-01'"
        )
        connection.commit()
    reopened = builder.SourceCheckpoint(path, {"binding": "fixture"})
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        reopened.entries()
    reopened.close()


def test_parse_failure_requires_same_archive_before_terminal_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = [{"month": "2020-01"}]
    monkeypatch.setattr(builder.protocol, "archive_manifest", lambda: manifest)
    monkeypatch.setattr(
        builder,
        "process_archive",
        lambda row, fetched: (_ for _ in ()).throw(
            builder.SourceContractError("fixture parse failure")
        ),
    )
    first = _fetched(b"first")
    changed = _fetched(b"changed")
    payloads = iter((first, changed))
    checkpoint = builder.SourceCheckpoint(tmp_path / "changed.sqlite3", {"v": 1})
    with pytest.raises(builder.SourceFetchError, match="changed"):
        builder.collect_to_checkpoint(checkpoint, lambda _: next(payloads))
    assert checkpoint.months() == []
    checkpoint.close()

    checkpoint = builder.SourceCheckpoint(tmp_path / "same.sqlite3", {"v": 1})
    with pytest.raises(builder.SourceContractError, match="fixture parse failure"):
        builder.collect_to_checkpoint(checkpoint, lambda _: first)
    checkpoint.close()


def test_support_uses_full_frozen_denominators() -> None:
    entries = _support_entries()
    support = builder.evaluate_support(entries)
    assert support["all_gates_passed"] is True
    assert support["source_row_count"] == 5844
    assert support["expected_archive_count"] == 48
    assert support["yearly_source_summary"]["2020"]["target_documents"] == 1464

    entries[0]["rows"][1]["member_sha256"] = entries[0]["rows"][0][
        "member_sha256"
    ]
    with pytest.raises(RuntimeError, match="duplicate member"):
        builder.evaluate_support(entries)


def test_final_artifacts_are_deterministic_write_once_and_outcome_blind(
    tmp_path: Path,
) -> None:
    entries = _support_entries()
    paths = {
        "source_path": tmp_path / "source.jsonl.gz",
        "archive_manifest_path": tmp_path / "archives.jsonl.gz",
        "source_manifest_path": tmp_path / "manifest.json",
        "support_path": tmp_path / "support.json",
    }
    first = builder.finalize_artifacts(entries, **paths)
    second = builder.finalize_artifacts(entries, **paths)
    assert first["status"] == "PASS_ADVANCE_TO_RECONFIGURATION_FREEZE"
    assert second["artifact_statuses"] == {
        "source": "verified_existing",
        "archive_manifest": "verified_existing",
        "support": "verified_existing",
        "source_manifest": "verified_existing",
    }
    support = json.loads(paths["support_path"].read_text())
    manifest = json.loads(paths["source_manifest_path"].read_text())
    assert support["outcomes_opened"] is False
    assert support["market_clocks_opened"] is False
    assert manifest["mechanism_features_opened"] is False


def test_url_disk_and_fatal_rejection_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = builder.protocol.archive_manifest()[0]
    builder.validate_official_archive_url(row)
    damaged = dict(row)
    damaged["url"] = "https://example.com/consensuses-2020-01.tar.xz"
    with pytest.raises(ValueError, match="official route"):
        builder.validate_official_archive_url(damaged)

    monkeypatch.setattr(builder, "_used_gib", lambda _: builder.protocol.DISK_LIMIT_GIB)
    with pytest.raises(builder.SourceFetchError, match="disk guard"):
        builder.enforce_disk_guard(tmp_path / "checkpoint.sqlite3")

    rejection = tmp_path / "rejection.json"
    assert builder.write_fatal_rejection(rejection, "fixture corruption") == "created"
    payload = json.loads(rejection.read_text())
    assert payload["status"] == "REJECT_NO_REPAIR"
    assert payload["outcomes_opened"] is False
