from __future__ import annotations

import gzip
import io
import json
import sqlite3
from collections.abc import Set as AbstractSet
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import netcdf_file

from training import build_texas_metar_thermal_relay_source as builder


def _char_row(value: str, width: int) -> np.ndarray:
    raw = value.encode("ascii")
    if len(raw) > width:
        raise ValueError("fixture text exceeds width")
    return np.frombuffer(raw.ljust(width, b"\x00"), dtype="S1")


def _write_netcdf(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    station_width: int = 5,
) -> bytes:
    with netcdf_file(path, "w", version=1) as dataset:
        dataset.createDimension("recNum", None)
        dataset.createDimension("maxStaNamLen", station_width)
        dataset.createDimension("maxRepLen", 6)
        dataset.createDimension("maxMETARLen", 256)
        station = dataset.createVariable(
            "stationName",
            "c",
            ("recNum", "maxStaNamLen"),
        )
        report = dataset.createVariable(
            "reportType",
            "c",
            ("recNum", "maxRepLen"),
        )
        raw_metar = dataset.createVariable(
            "rawMETAR",
            "c",
            ("recNum", "maxMETARLen"),
        )
        observation = dataset.createVariable("timeObs", "d", ("recNum",))
        receipt = dataset.createVariable("timeReceived", "d", ("recNum",))
        correction = dataset.createVariable("correction", "i", ("recNum",))
        observation._FillValue = np.float64(1.7976931348623157e308)
        receipt._FillValue = np.float64(1.7976931348623157e308)
        correction._FillValue = np.int32(-2147483647)
        temperature = dataset.createVariable("temperature", "f", ("recNum",))

        for index, row in enumerate(rows):
            station[index, :] = _char_row(str(row["station"]), station_width)
            report[index, :] = _char_row(str(row.get("report_type", "METAR")), 6)
            raw_metar[index, :] = _char_row(str(row["raw_metar"]), 256)
            observation[index] = int(row["observation_time"].timestamp())
            receipt[index] = int(row["receipt_time"].timestamp())
            correction[index] = int(row.get("correction", 0))
            temperature[index] = 0.0
    return path.read_bytes()


def _rows() -> list[dict[str, Any]]:
    observed = datetime(2022, 12, 31, 23, 53, tzinfo=timezone.utc)
    received = datetime(2022, 12, 31, 23, 58, tzinfo=timezone.utc)
    return [
        {
            "station": station,
            "raw_metar": (f"{station} 312353Z AUTO 24012KT 10SM CLR 18/03 A2985"),
            "observation_time": observed,
            "receipt_time": received,
        }
        for station in builder.protocol.STATIONS
    ]


def _entry(
    label: datetime,
    stations: AbstractSet[str] | None,
) -> dict[str, object]:
    if stations is None:
        return builder.missing_object_entry(label)
    rows = [
        {"stationName": station}
        for station in builder.protocol.STATIONS
        if station in stations
    ]
    return {
        "object": {
            "archive_label_utc": label.isoformat().replace("+00:00", "Z"),
            "archive_url": builder.protocol.archive_url(label),
            "status": "available",
            "eligible_stations": [row["stationName"] for row in rows],
        },
        "rows": rows,
    }


def test_builder_is_bound_to_exact_v3_protocol() -> None:
    payload = builder.load_frozen_protocol()
    assert payload["protocol_version"] == "texas_metar_thermal_relay_source_v3"
    assert payload["manifest_hash"] == builder.PROTOCOL_MANIFEST_HASH
    assert builder.sha256_file(builder.PROTOCOL_PATH) == builder.PROTOCOL_FILE_SHA256


def test_exact_netcdf_parser_retains_only_four_causal_raw_rows(tmp_path: Path) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = _write_netcdf(tmp_path / "fixture.nc", _rows())
    parsed = builder.parse_netcdf_object(label, raw)
    assert parsed.record_count == 4
    assert parsed.ineligible_target_rows == 0
    assert [row["stationName"] for row in parsed.rows] == list(
        builder.protocol.STATIONS
    )
    assert {row["source_availability_utc"] for row in parsed.rows} == {
        "2023-01-01T01:00:00Z"
    }
    assert all(row["reportType"] == "METAR" for row in parsed.rows)
    assert all(row["correction"] == 0 for row in parsed.rows)
    assert all("temperature" not in row for row in parsed.rows)


def test_process_object_replays_twice_and_binds_object_hash(tmp_path: Path) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    netcdf = _write_netcdf(tmp_path / "fixture.nc", _rows())
    compressed = gzip.compress(netcdf, mtime=0)
    entry = builder.process_compressed_object(label, compressed)
    assert entry["object"]["status"] == "available"
    assert entry["object"]["eligible_station_count"] == 4
    assert entry["object"]["compressed_sha256"] == builder.sha256_bytes(compressed)
    assert {row["compressed_object_sha256"] for row in entry["rows"]} == {
        builder.sha256_bytes(compressed)
    }


def test_parser_excludes_timestamp_mismatch_and_rejects_required_schema_drift(
    tmp_path: Path,
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = _rows()
    rows[0]["raw_metar"] = "KMAF 312352Z AUTO 24012KT 10SM CLR 18/03 A2985"
    parsed = builder.parse_netcdf_object(
        label,
        _write_netcdf(tmp_path / "mismatch.nc", rows),
    )
    assert [row["stationName"] for row in parsed.rows] == ["KABI", "KLBB", "KACT"]
    assert parsed.ineligible_target_rows == 1

    with pytest.raises(builder.SourceContractError, match="schema"):
        builder.parse_netcdf_object(
            label,
            _write_netcdf(
                tmp_path / "bad-schema.nc",
                _rows(),
                station_width=6,
            ),
        )


def test_exact_duplicate_is_deduped_but_distinct_eligible_row_is_fatal(
    tmp_path: Path,
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    duplicate_rows = _rows() + [_rows()[0]]
    parsed = builder.parse_netcdf_object(
        label,
        _write_netcdf(tmp_path / "duplicate.nc", duplicate_rows),
    )
    assert len(parsed.rows) == 4

    conflicting = _rows() + [
        {
            **_rows()[0],
            "raw_metar": "KMAF 312353Z AUTO 25012KT 10SM CLR 18/03 A2985",
        }
    ]
    with pytest.raises(builder.SourceContractError, match="cardinality"):
        builder.parse_netcdf_object(
            label,
            _write_netcdf(tmp_path / "conflict.nc", conflicting),
        )


def test_gzip_and_two_pass_replay_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(builder.SourceContractError, match="gzip"):
        builder.decompress_gzip_bounded(b"not-gzip")

    calls = 0

    def divergent(_: datetime, __: bytes) -> builder.ParsedObject:
        nonlocal calls
        calls += 1
        return builder.ParsedObject(calls, (), 0, ())

    monkeypatch.setattr(builder, "parse_netcdf_object", divergent)
    with pytest.raises(builder.SourceContractError, match="parse passes differ"):
        builder.process_compressed_object(
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            gzip.compress(b"payload", mtime=0),
        )


def test_checkpoint_is_resumable_only_as_contiguous_bound_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.sqlite3"
    checkpoint = builder.SourceCheckpoint(checkpoint_path, {"binding": "fixture"})
    completed, total = builder.collect_to_checkpoint(
        checkpoint,
        lambda _: None,
        max_new_objects=2,
    )
    assert completed == 2
    assert total == 5844
    checkpoint.close()

    resumed = builder.SourceCheckpoint(checkpoint_path, {"binding": "fixture"})
    completed, _ = builder.collect_to_checkpoint(
        resumed,
        lambda _: None,
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
            (label.isoformat().replace("+00:00", "Z"),),
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
    calls: list[bytes] = []

    def process(_: datetime, payload: bytes) -> dict[str, object]:
        calls.append(payload)
        if payload == b"bad":
            raise builder.SourceContractError("fixture parse failure")
        return builder.missing_object_entry(builder.protocol.archive_labels()[0])

    monkeypatch.setattr(builder, "process_compressed_object", process)
    payloads = iter((b"bad", b"good"))
    checkpoint = builder.SourceCheckpoint(tmp_path / "changed.sqlite3", {"v": 1})
    with pytest.raises(builder.SourceFetchError, match="source bytes changed"):
        builder.collect_to_checkpoint(
            checkpoint,
            lambda _: next(payloads),
            max_new_objects=1,
        )
    assert checkpoint.labels() == []
    assert calls == [b"bad"]
    checkpoint.close()

    checkpoint = builder.SourceCheckpoint(tmp_path / "same.sqlite3", {"v": 1})
    with pytest.raises(builder.SourceContractError, match="fixture parse failure"):
        builder.collect_to_checkpoint(
            checkpoint,
            lambda _: b"bad",
            max_new_objects=1,
        )
    checkpoint.close()


def test_support_uses_full_expected_denominators_and_rejects_long_gap() -> None:
    labels = builder.protocol.archive_labels()
    complete = set(builder.protocol.STATIONS)
    entries = [_entry(label, complete) for label in labels]
    passed = builder.evaluate_support(entries)
    assert passed["all_gates_passed"] is True
    assert passed["source_row_count"] == 4 * 5844

    for index in range(5):
        entries[index] = _entry(labels[index], None)
    rejected = builder.evaluate_support(entries)
    assert rejected["all_gates_passed"] is False
    streak = next(
        gate
        for gate in rejected["gates"]
        if gate["gate_id"] == "maximum_consecutive_incomplete_panels"
    )
    assert streak["observed"] == 5
    assert streak["passed"] is False


def test_final_artifacts_are_deterministic_write_once_and_outcome_blind(
    tmp_path: Path,
) -> None:
    labels = builder.protocol.archive_labels()
    entries = [_entry(label, set(builder.protocol.STATIONS)) for label in labels]
    paths = {
        "source_path": tmp_path / "source.jsonl.gz",
        "object_manifest_path": tmp_path / "objects.jsonl.gz",
        "source_manifest_path": tmp_path / "manifest.json",
        "support_path": tmp_path / "support.json",
    }
    first = builder.finalize_artifacts(entries, **paths)
    second = builder.finalize_artifacts(entries, **paths)
    assert first["status"] == "PASS_ADVANCE_TO_THERMAL_FREEZE"
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
    assert manifest["thermal_parser_opened"] is False
    assert gzip.decompress(paths["source_path"].read_bytes()).count(b"\n") == 4 * 5844


def test_url_disk_and_fatal_rejection_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    builder.validate_official_archive_url(label, builder.protocol.archive_url(label))
    with pytest.raises(ValueError, match="official route"):
        builder.validate_official_archive_url(
            label,
            "https://example.com/20230101_0000.gz",
        )
    monkeypatch.setattr(builder, "_used_gib", lambda _: builder.protocol.DISK_LIMIT_GIB)
    with pytest.raises(builder.SourceFetchError, match="disk guard"):
        builder.enforce_disk_guard(tmp_path / "checkpoint.sqlite3")

    rejection = tmp_path / "rejection.json"
    assert builder.write_fatal_rejection(rejection, "fixture corruption") == "created"
    payload = json.loads(rejection.read_text())
    assert payload["status"] == "REJECT_NO_REPAIR"
    assert payload["outcomes_opened"] is False
    assert builder.write_fatal_rejection(rejection, "fixture corruption") == (
        "verified_existing"
    )


def test_transport_length_and_artifact_set_preflight_fail_before_partial_write(
    tmp_path: Path,
) -> None:
    response = io.BytesIO(b"abc")
    response.headers = {"Content-Length": "4"}  # type: ignore[attr-defined]
    with pytest.raises(builder.SourceFetchError, match="Content-Length"):
        builder._read_bounded_response(response)

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    second.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="overwrite"):
        builder.write_artifact_set(
            (("first", first, b"new"), ("second", second, b"different"))
        )
    assert not first.exists()
