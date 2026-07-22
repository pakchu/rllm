from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import preregister_bgp_routing_churn_relay as brcr


def _rehash(payload: dict[str, object]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = brcr.canonical_hash(core)


def _record(timestamp: int, subtype: int, payload: bytes = b"x") -> bytes:
    return brcr.MRT_COMMON_HEADER.pack(
        timestamp,
        brcr.MRT_TYPE_BGP4MP,
        subtype,
        len(payload),
    ) + payload


def _valid_stream(label: datetime) -> bytes:
    timestamp = int(label.timestamp())
    return b"".join(
        [
            _record(timestamp, 0, b"state"),
            _record(timestamp + 2, 4, b"message-a"),
            _record(timestamp + 299, 1, b"message-b"),
        ]
    )


def test_archive_envelope_and_monthly_replay_audit_are_exact() -> None:
    labels = brcr.archive_labels()
    assert len(labels) == brcr.EXPECTED_OBJECTS == 5844
    assert labels[0] == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert labels[-1] == datetime(2023, 12, 31, 18, tzinfo=timezone.utc)
    assert {label.hour for label in labels} == set(brcr.ARCHIVE_HOURS)
    assert len(set(labels)) == len(labels)

    audit = brcr.replay_audit_labels()
    assert len(audit) == 48
    assert all(label.day == 1 and label.hour == 0 for label in audit)
    assert len({(label.year, label.month) for label in audit}) == 48


def test_archive_url_and_availability_are_frozen() -> None:
    label = datetime(2023, 1, 1, 6, tzinfo=timezone.utc)
    assert brcr.archive_url(label) == (
        "https://data.ris.ripe.net/rrc00/2023.01/"
        "updates.20230101.0600.gz"
    )
    assert brcr.public_availability_time(label) == label + timedelta(minutes=15)

    with pytest.raises(ValueError, match="hour set"):
        brcr.archive_url(label.replace(hour=5))
    with pytest.raises(ValueError, match="interval"):
        brcr.archive_url(label.replace(year=2024))
    with pytest.raises(ValueError, match="UTC-aware"):
        brcr.archive_url(label.replace(tzinfo=None))
    with pytest.raises(ValueError, match="exact hour"):
        brcr.archive_url(label.replace(minute=5))


def test_http_metadata_gate_is_transport_only_and_fail_closed() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    evidence = brcr.validate_transport_headers(
        label=label,
        last_modified="Sun, 01 Jan 2023 00:05:13 GMT",
        etag='"63b0ce39-36d05e"',
        declared_content_length=3592286,
        actual_content_length=3592286,
    )
    assert evidence["last_modified"] == "2023-01-01T00:05:13Z"

    with pytest.raises(ValueError, match="publication window"):
        brcr.validate_last_modified(label, "Sun, 01 Jan 2023 00:04:59 GMT")
    with pytest.raises(ValueError, match="publication window"):
        brcr.validate_last_modified(label, "Sun, 01 Jan 2023 00:30:01 GMT")
    with pytest.raises(ValueError, match="ETag"):
        brcr.validate_transport_headers(
            label=label,
            last_modified="Sun, 01 Jan 2023 00:05:13 GMT",
            etag='W/"63b0ce39-36d05e"',
            declared_content_length=10,
            actual_content_length=10,
        )
    with pytest.raises(ValueError, match="differs"):
        brcr.validate_transport_headers(
            label=label,
            last_modified="Sun, 01 Jan 2023 00:05:13 GMT",
            etag='"63b0ce39-36d05e"',
            declared_content_length=10,
            actual_content_length=9,
        )
    with pytest.raises(ValueError, match="must be an integer"):
        brcr.validate_transport_headers(
            label=label,
            last_modified="Sun, 01 Jan 2023 00:05:13 GMT",
            etag='"63b0ce39-36d05e"',
            declared_content_length=10.0,  # type: ignore[arg-type]
            actual_content_length=10,
        )


def test_mrt_parser_counts_only_frozen_bgp4mp_forms() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    parsed = brcr.parse_mrt_stream(_valid_stream(label), label=label)
    assert parsed == {
        "decompressed_bytes": len(_valid_stream(label)),
        "record_count": 3,
        "message_record_count": 2,
        "state_change_record_count": 1,
        "minimum_timestamp": int(label.timestamp()),
        "maximum_timestamp": int(label.timestamp()) + 299,
        "type_subtype_counts": {"16:0": 1, "16:1": 1, "16:4": 1},
    }


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"", "empty"),
        (b"short", "truncated common header"),
    ],
)
def test_mrt_parser_rejects_empty_or_truncated_streams(raw: bytes, match: str) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=match):
        brcr.parse_mrt_stream(raw, label=label)


def test_mrt_parser_rejects_bad_length_type_subtype_and_time() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    timestamp = int(label.timestamp())
    bad_length = brcr.MRT_COMMON_HEADER.pack(timestamp, 16, 4, 9) + b"x"
    with pytest.raises(ValueError, match="payload length"):
        brcr.parse_mrt_stream(bad_length, label=label)

    non_bgp = brcr.MRT_COMMON_HEADER.pack(timestamp, 13, 4, 1) + b"x"
    with pytest.raises(ValueError, match="non-BGP4MP"):
        brcr.parse_mrt_stream(non_bgp, label=label)

    unknown_subtype = brcr.MRT_COMMON_HEADER.pack(timestamp, 16, 9, 1) + b"x"
    with pytest.raises(ValueError, match="unknown"):
        brcr.parse_mrt_stream(unknown_subtype, label=label)

    outside = _record(timestamp + 300, 4)
    with pytest.raises(ValueError, match="outside"):
        brcr.parse_mrt_stream(outside, label=label)

    nonmonotone = _record(timestamp + 2, 4) + _record(timestamp + 1, 4)
    with pytest.raises(ValueError, match="not monotone"):
        brcr.parse_mrt_stream(nonmonotone, label=label)


def test_mrt_parser_requires_at_least_one_message_record() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    state_changes_only = _record(int(label.timestamp()), 0)
    with pytest.raises(ValueError, match="no BGP update messages"):
        brcr.parse_mrt_stream(state_changes_only, label=label)


def test_gzip_parser_checks_magic_crc_hash_and_identity() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = gzip.compress(_valid_stream(label), mtime=0)
    parsed = brcr.parse_gzip_mrt_object(raw, label=label)
    assert parsed["compressed_bytes"] == len(raw)
    assert parsed["compressed_sha256"] == brcr.sha256_bytes(raw)
    assert parsed["record_count"] == 3
    assert brcr.canonical_object_identity(label=label, parsed=parsed) == (
        brcr.canonical_object_identity(label=label, parsed=dict(parsed))
    )

    with pytest.raises(ValueError, match="not gzip"):
        brcr.parse_gzip_mrt_object(b"not-gzip", label=label)
    damaged = bytearray(raw)
    damaged[-1] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        brcr.parse_gzip_mrt_object(bytes(damaged), label=label)


def test_object_identity_rejects_unfrozen_fields() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    parsed = brcr.parse_gzip_mrt_object(
        gzip.compress(_valid_stream(label), mtime=0),
        label=label,
    )
    parsed["http_etag"] = '"forbidden-feature"'
    with pytest.raises(ValueError, match="fields differ"):
        brcr.canonical_object_identity(label=label, parsed=parsed)


def test_coverage_denominators_include_every_expected_label() -> None:
    labels = brcr.archive_labels()
    available = {label: True for label in labels}
    missing = labels[100:103]
    for label in missing:
        available[label] = False
    summary = brcr.summarize_expected_coverage(available)
    assert summary["expected_labels"] == 5844
    assert summary["maximum_consecutive_missing_objects"] == 3
    assert sum(
        year["expected_objects"] for year in summary["years"].values()
    ) == 5844
    assert sum(
        year["available_objects"] for year in summary["years"].values()
    ) == 5841

    with pytest.raises(ValueError, match="out-of-contract"):
        brcr.summarize_expected_coverage(
            {datetime(2024, 1, 1, tzinfo=timezone.utc): True}
        )
    with pytest.raises(ValueError, match="must be booleans"):
        brcr.summarize_expected_coverage({labels[0]: 1})  # type: ignore[dict-item]


def test_manifest_keeps_source_only_boundary_and_binds_files() -> None:
    payload = brcr.build_manifest()
    brcr.validate_manifest(payload)
    assert payload["source_id"] == "BRCR"
    assert payload["outcomes_opened"] is False
    assert payload["market_clocks_opened"] is False
    assert payload["historical_source_incidence_opened"] is False
    assert payload["mechanism_parser_opened"] is False
    assert payload["source_only_probe"]["object_bodies_opened"] == 4
    assert payload["decision_binding"]["sha256"] == brcr.DECISION_SHA256
    assert payload["implementation_binding"]["sha256"] == brcr.sha256_file(
        brcr.SCRIPT_PATH
    )
    assert payload["source_contract"]["archive_interval"]["expected_objects"] == 5844
    assert payload["source_quality_gates"]["replay"]["expected_objects"] == 48
    forbidden = " ".join(payload["source_contract"]["forbidden_fields"])
    assert "BTC bars" in forbidden
    assert "bview" in forbidden


def test_manifest_tampering_is_rejected_even_after_rehash() -> None:
    payload = brcr.build_manifest()
    payload["outcomes_opened"] = True
    _rehash(payload)
    with pytest.raises(RuntimeError, match="outcomes_opened=false"):
        brcr.validate_manifest(payload)

    payload = brcr.build_manifest()
    payload["source_quality_gates"]["transport"][
        "minimum_object_fraction_each_year"
    ] = 0.5
    _rehash(payload)
    with pytest.raises(RuntimeError, match="differs from code"):
        brcr.validate_manifest(payload)


def test_write_once_creates_then_verifies_and_refuses_conflict(tmp_path: Path) -> None:
    output = tmp_path / "source_protocol.json"
    payload = brcr.build_manifest()
    assert brcr.write_manifest_once(output, payload) == "created"
    assert brcr.write_manifest_once(output, payload) == "verified_existing"

    existing = json.loads(output.read_text(encoding="utf-8"))
    existing["manifest_hash"] = "0" * 64
    output.write_text(json.dumps(existing), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        brcr.write_manifest_once(output, payload)


def test_checked_in_manifest_matches_executable_freeze() -> None:
    path = brcr.REPO_ROOT / brcr.DEFAULT_OUTPUT
    if not path.exists():
        pytest.skip("checked-in BRCR source protocol not generated yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    brcr.validate_manifest(payload)
    assert payload == brcr.build_manifest()
