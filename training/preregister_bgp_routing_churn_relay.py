"""Freeze BRCR source support before full RIPE incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import operator
import re
import struct
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/bgp_routing_churn_relay_source_protocol_2026-07-22.json")
DECISION_PATH = Path(
    "docs/bgp-routing-churn-relay-source-axis-decision-2026-07-22.md"
)
DECISION_SHA256 = "9f26353a78884cb027b79560f48ef37b8c8427df08a62eeedc24ef6c990b3005"
SCRIPT_PATH = Path("training/preregister_bgp_routing_churn_relay.py")

SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SOURCE_END_EXCLUSIVE = datetime(2024, 1, 1, tzinfo=timezone.utc)
ARCHIVE_HOURS = (0, 6, 12, 18)
COLLECTOR = "rrc00"
ARCHIVE_BASE = f"https://data.ris.ripe.net/{COLLECTOR}"
EXPECTED_OBJECTS = 5844
MRT_INTERVAL_MINUTES = 5
PUBLICATION_FLOOR_MINUTES = 15
MINIMUM_LAST_MODIFIED_DELAY_MINUTES = 5
MAXIMUM_LAST_MODIFIED_DELAY_MINUTES = 30
DISK_LIMIT_GIB = 300
MAXIMUM_COMPRESSED_BYTES = 128 * 1024 * 1024
MAXIMUM_DECOMPRESSED_BYTES = 1024 * 1024 * 1024

GZIP_MAGIC = b"\x1f\x8b"
MRT_COMMON_HEADER = struct.Struct("!IHHI")
MRT_TYPE_BGP4MP = 16
MRT_ALLOWED_SUBTYPES = frozenset({0, 1, 4, 5, 6, 7})
MRT_MESSAGE_SUBTYPES = frozenset({1, 4, 6, 7})
MRT_STATE_CHANGE_SUBTYPES = frozenset({0, 5})
ETAG_PATTERN = re.compile(r'"[0-9a-f]+-[0-9a-f]+"', re.ASCII)

PROBE_OBJECTS: tuple[dict[str, Any], ...] = (
    {
        "archive_label": "2020-01-01T00:00:00Z",
        "compressed_bytes": 3953511,
        "compressed_sha256": (
            "84ea79cd40424754a77530bcaf98eff988f6582712ef5199d53cc1ec159b3162"
        ),
        "decompressed_bytes": 19967733,
        "record_count": 138891,
        "last_modified": "Wed, 01 Jan 2020 00:05:15 GMT",
        "etag": '"5e0be23b-3c5367"',
    },
    {
        "archive_label": "2021-01-01T00:00:00Z",
        "compressed_bytes": 7573375,
        "compressed_sha256": (
            "ed199898df7cb1df6388315990cb0dd69492b3b144046cf1e226c97efe88c929"
        ),
        "decompressed_bytes": 44790599,
        "record_count": 295941,
        "last_modified": "Fri, 01 Jan 2021 00:05:44 GMT",
        "etag": '"5fee6758-738f7f"',
    },
    {
        "archive_label": "2022-01-01T00:00:00Z",
        "compressed_bytes": 3085355,
        "compressed_sha256": (
            "6815f3dd52a507ec202f774ee56ecc18ca02c2a0681606517e65d345b7d8d11a"
        ),
        "decompressed_bytes": 18301516,
        "record_count": 114035,
        "last_modified": "Sat, 01 Jan 2022 00:05:08 GMT",
        "etag": '"61cf9ab4-2f142b"',
    },
    {
        "archive_label": "2023-01-01T00:00:00Z",
        "compressed_bytes": 3592286,
        "compressed_sha256": (
            "e7233b1424bfee50dfa5bd3990c03ebf53f9662e120c343d5ed26f3e51d3c3a3"
        ),
        "decompressed_bytes": 24584606,
        "record_count": 138274,
        "last_modified": "Sun, 01 Jan 2023 00:05:13 GMT",
        "etag": '"63b0ce39-36d05e"',
    },
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return sha256_bytes(candidate.read_bytes())


def canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def archive_labels() -> list[datetime]:
    labels: list[datetime] = []
    current = SOURCE_START
    while current < SOURCE_END_EXCLUSIVE:
        if current.hour in ARCHIVE_HOURS:
            labels.append(current)
        current += timedelta(hours=1)
    if len(labels) != EXPECTED_OBJECTS:
        raise RuntimeError("BRCR archive-label envelope is not exactly 5,844 objects")
    return labels


def _validate_label(label: datetime) -> None:
    if label.tzinfo is None or label.utcoffset() != timedelta(0):
        raise ValueError("BRCR archive label must be UTC-aware")
    if label.minute != 0 or label.second != 0 or label.microsecond != 0:
        raise ValueError("BRCR archive label must be an exact hour")
    if label.hour not in ARCHIVE_HOURS:
        raise ValueError("BRCR archive label is outside the frozen hour set")
    if not SOURCE_START <= label < SOURCE_END_EXCLUSIVE:
        raise ValueError("BRCR archive label is outside the frozen interval")


def archive_url(label: datetime) -> str:
    _validate_label(label)
    return (
        f"{ARCHIVE_BASE}/{label:%Y.%m}/"
        f"updates.{label:%Y%m%d.%H%M}.gz"
    )


def public_availability_time(label: datetime) -> datetime:
    _validate_label(label)
    return label + timedelta(minutes=PUBLICATION_FLOOR_MINUTES)


def parse_http_datetime(value: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("BRCR HTTP datetime is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError("BRCR HTTP datetime is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def validate_last_modified(label: datetime, value: str) -> datetime:
    _validate_label(label)
    modified = parse_http_datetime(value)
    earliest = label + timedelta(minutes=MINIMUM_LAST_MODIFIED_DELAY_MINUTES)
    latest = label + timedelta(minutes=MAXIMUM_LAST_MODIFIED_DELAY_MINUTES)
    if not earliest <= modified <= latest:
        raise ValueError("BRCR Last-Modified is outside the frozen publication window")
    return modified


def validate_transport_headers(
    *,
    label: datetime,
    last_modified: str,
    etag: str,
    declared_content_length: int,
    actual_content_length: int,
) -> dict[str, Any]:
    modified = validate_last_modified(label, last_modified)
    if ETAG_PATTERN.fullmatch(etag) is None:
        raise ValueError("BRCR ETag differs from the frozen strong Apache form")
    if isinstance(declared_content_length, bool) or isinstance(actual_content_length, bool):
        raise ValueError("BRCR content length must be an integer")
    try:
        declared_length = operator.index(declared_content_length)
        actual_length = operator.index(actual_content_length)
    except TypeError as exc:
        raise ValueError("BRCR content length must be an integer") from exc
    if declared_length <= 0 or declared_length != actual_length:
        raise ValueError("BRCR declared content length differs from received bytes")
    if actual_length > MAXIMUM_COMPRESSED_BYTES:
        raise ValueError("BRCR compressed object exceeds the frozen size guard")
    return {
        "last_modified": modified.isoformat().replace("+00:00", "Z"),
        "etag": etag,
        "declared_content_length": declared_length,
    }


def parse_mrt_stream(raw: bytes, *, label: datetime) -> dict[str, Any]:
    _validate_label(label)
    if not raw:
        raise ValueError("BRCR MRT stream is empty")
    if len(raw) > MAXIMUM_DECOMPRESSED_BYTES:
        raise ValueError("BRCR decompressed object exceeds the frozen size guard")

    offset = 0
    previous_timestamp: int | None = None
    minimum_timestamp: int | None = None
    maximum_timestamp: int | None = None
    counts: Counter[tuple[int, int]] = Counter()
    interval_start = int(label.timestamp())
    interval_end = int((label + timedelta(minutes=MRT_INTERVAL_MINUTES)).timestamp())

    while offset < len(raw):
        remaining = len(raw) - offset
        if remaining < MRT_COMMON_HEADER.size:
            raise ValueError("BRCR MRT stream has a truncated common header")
        timestamp, record_type, subtype, payload_length = MRT_COMMON_HEADER.unpack_from(
            raw, offset
        )
        offset += MRT_COMMON_HEADER.size
        if payload_length > len(raw) - offset:
            raise ValueError("BRCR MRT payload length exceeds remaining bytes")
        if record_type != MRT_TYPE_BGP4MP:
            raise ValueError("BRCR MRT stream contains a non-BGP4MP record")
        if subtype not in MRT_ALLOWED_SUBTYPES:
            raise ValueError("BRCR MRT stream contains an unknown BGP4MP subtype")
        if not interval_start <= timestamp < interval_end:
            raise ValueError("BRCR MRT timestamp is outside the frozen file interval")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("BRCR MRT timestamps are not monotone")

        counts[(record_type, subtype)] += 1
        minimum_timestamp = timestamp if minimum_timestamp is None else minimum_timestamp
        maximum_timestamp = timestamp
        previous_timestamp = timestamp
        offset += payload_length

    record_count = sum(counts.values())
    message_count = sum(
        count for (_, subtype), count in counts.items() if subtype in MRT_MESSAGE_SUBTYPES
    )
    state_change_count = sum(
        count
        for (_, subtype), count in counts.items()
        if subtype in MRT_STATE_CHANGE_SUBTYPES
    )
    if record_count == 0 or message_count == 0:
        raise ValueError("BRCR MRT stream contains no BGP update messages")
    return {
        "decompressed_bytes": len(raw),
        "record_count": record_count,
        "message_record_count": message_count,
        "state_change_record_count": state_change_count,
        "minimum_timestamp": minimum_timestamp,
        "maximum_timestamp": maximum_timestamp,
        "type_subtype_counts": {
            f"{record_type}:{subtype}": counts[(record_type, subtype)]
            for record_type, subtype in sorted(counts)
        },
    }


def parse_gzip_mrt_object(raw: bytes, *, label: datetime) -> dict[str, Any]:
    _validate_label(label)
    if len(raw) > MAXIMUM_COMPRESSED_BYTES:
        raise ValueError("BRCR compressed object exceeds the frozen size guard")
    if raw[:2] != GZIP_MAGIC:
        raise ValueError("BRCR object is not gzip")
    try:
        decompressed = gzip.decompress(raw)
    except (EOFError, OSError) as exc:
        raise ValueError("BRCR gzip object failed decompression or CRC validation") from exc
    parsed = parse_mrt_stream(decompressed, label=label)
    return {
        "compressed_bytes": len(raw),
        "compressed_sha256": sha256_bytes(raw),
        **parsed,
    }


def canonical_object_identity(*, label: datetime, parsed: Mapping[str, Any]) -> str:
    _validate_label(label)
    required = {
        "compressed_bytes",
        "compressed_sha256",
        "decompressed_bytes",
        "record_count",
        "message_record_count",
        "state_change_record_count",
        "minimum_timestamp",
        "maximum_timestamp",
        "type_subtype_counts",
    }
    if set(parsed) != required:
        raise ValueError("BRCR parsed object fields differ from the frozen identity")
    payload = {
        "archive_label": label.isoformat().replace("+00:00", "Z"),
        "availability": public_availability_time(label)
        .isoformat()
        .replace("+00:00", "Z"),
        "url": archive_url(label),
        **dict(parsed),
    }
    return canonical_hash(payload)


def replay_audit_labels() -> list[datetime]:
    labels = [
        label
        for label in archive_labels()
        if label.day == 1 and label.hour == 0
    ]
    if len(labels) != 48:
        raise RuntimeError("BRCR replay audit is not exactly one object per month")
    return labels


def summarize_expected_coverage(
    available: Mapping[datetime, bool],
) -> dict[str, Any]:
    expected = archive_labels()
    expected_set = set(expected)
    if set(available).difference(expected_set):
        raise ValueError("BRCR coverage input contains an out-of-contract label")
    if any(type(value) is not bool for value in available.values()):
        raise ValueError("BRCR coverage availability values must be booleans")

    years: dict[str, dict[str, int]] = {}
    months: dict[str, dict[str, int]] = {}
    missing_streak = 0
    maximum_missing_streak = 0
    for label in expected:
        year = years.setdefault(
            str(label.year), {"expected_objects": 0, "available_objects": 0}
        )
        month = months.setdefault(
            label.strftime("%Y-%m"),
            {"expected_objects": 0, "available_objects": 0},
        )
        year["expected_objects"] += 1
        month["expected_objects"] += 1
        if available.get(label, False):
            year["available_objects"] += 1
            month["available_objects"] += 1
            missing_streak = 0
        else:
            missing_streak += 1
            maximum_missing_streak = max(maximum_missing_streak, missing_streak)
    return {
        "expected_labels": len(expected),
        "years": years,
        "months": months,
        "maximum_consecutive_missing_objects": maximum_missing_streak,
    }


def _source_contract() -> dict[str, Any]:
    return {
        "authority": {
            "operator": "RIPE NCC Routing Information Service",
            "collector": COLLECTOR,
            "collector_scope": "multihop global",
            "archive_base": ARCHIVE_BASE,
            "transport": "public HTTPS gzip-compressed MRT",
            "raw_archive_documentation": "https://ris.ripe.net/docs/mrt/",
            "collector_documentation": (
                "https://ris.ripe.net/docs/route-collectors/"
            ),
            "mrt_specification": "https://www.rfc-editor.org/rfc/rfc6396",
        },
        "archive_interval": {
            "start_inclusive": SOURCE_START.isoformat().replace("+00:00", "Z"),
            "end_exclusive": SOURCE_END_EXCLUSIVE.isoformat().replace("+00:00", "Z"),
            "hours_utc": list(ARCHIVE_HOURS),
            "file_interval_minutes": MRT_INTERVAL_MINUTES,
            "expected_objects": EXPECTED_OBJECTS,
            "filename": "updates.YYYYMMDD.HHmm.gz",
        },
        "mrt_schema": {
            "common_header": "!IHHI",
            "common_header_bytes": MRT_COMMON_HEADER.size,
            "record_type": MRT_TYPE_BGP4MP,
            "allowed_subtypes": sorted(MRT_ALLOWED_SUBTYPES),
            "message_subtypes": sorted(MRT_MESSAGE_SUBTYPES),
            "state_change_subtypes": sorted(MRT_STATE_CHANGE_SUBTYPES),
            "timestamps": "monotone and in [label, label + 5 minutes)",
            "payload_length": "must terminate exactly at gzip stream boundary",
            "unknown_type_or_subtype": "fatal",
        },
        "retained_fields": [
            "archive label, conservative availability, and official URL",
            "compressed bytes and SHA-256",
            "decompressed bytes",
            "MRT record, message-record, and state-change-record counts",
            "minimum and maximum embedded MRT timestamps",
            "exact type/subtype counts",
            "canonical object identity SHA-256",
            "transport-only Last-Modified, ETag, and Content-Length evidence",
        ],
        "forbidden_fields": [
            "raw BGP payloads in normalized source output",
            "prefixes, AS paths, peer IPs, peer ASNs, communities, or next hops",
            "bview files, another collector, RIPEstat, RIPE Atlas, or IODA fallback",
            "HTTP metadata as a signal feature",
            "BTC bars/returns/funding/PnL",
            "prior alpha clocks or portfolio outcomes",
        ],
    }


def _quality_contract() -> dict[str, Any]:
    return {
        "transport": {
            "minimum_object_fraction_each_year": 0.995,
            "minimum_object_fraction_each_month": 0.98,
            "maximum_consecutive_missing_objects": 4,
            "gzip_crc_required": True,
            "compressed_sha256_fraction": 1.0,
            "declared_content_length_must_match": True,
            "strong_apache_etag_required": True,
            "last_modified_delay_minutes_inclusive": [
                MINIMUM_LAST_MODIFIED_DELAY_MINUTES,
                MAXIMUM_LAST_MODIFIED_DELAY_MINUTES,
            ],
            "duplicate_sha_across_labels_allowed": 0,
            "two_in_memory_parse_passes_required": True,
            "exact_cross_pass_equality_required": True,
        },
        "mrt": {
            "valid_common_header_fraction": 1.0,
            "valid_payload_boundary_fraction": 1.0,
            "valid_timestamp_interval_fraction": 1.0,
            "valid_type_subtype_fraction": 1.0,
            "minimum_message_records_each_object": 1,
            "imputation_or_forward_fill_allowed": False,
        },
        "replay": {
            "labels": "00:00 UTC on the first day of every source month",
            "expected_objects": 48,
            "second_fetch_sha256_equality_fraction": 1.0,
            "second_fetch_header_equality_fraction": 1.0,
        },
        "denominators": {
            "object_year": "all expected frozen labels in that calendar year",
            "object_month": "all expected frozen labels in that calendar month",
            "missing_or_unparseable_object": "counts missing in its expected slot",
            "missing_streak": "walk every expected label in chronological order",
            "fetched_or_successful_objects_never_define_a_denominator": True,
        },
        "failure_effect": "REJECT_NO_REPAIR",
    }


def build_manifest() -> dict[str, Any]:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("BRCR source-axis decision hash differs from the freeze")
    labels = archive_labels()
    audit_labels = replay_audit_labels()
    core: dict[str, Any] = {
        "protocol_version": "bgp_routing_churn_relay_source_v1",
        "source_id": "BRCR",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "historical_source_incidence_opened": False,
        "mechanism_parser_opened": False,
        "source_only_probe_opened": True,
        "source_only_probe": {
            "objects": list(PROBE_OBJECTS),
            "object_bodies_opened": 4,
            "directory_listings_opened": [
                "https://data.ris.ripe.net/rrc00/2023.01/"
            ],
            "coordinate_rule": "January 1 00:00 UTC, one object per source year",
            "all_streams_gzip_crc_valid": True,
            "all_streams_exact_mrt_boundary": True,
            "all_streams_timestamp_monotone": True,
            "all_embedded_timestamps": "00:00:00 through 00:04:59 UTC",
            "observed_type_subtypes": ["16:0", "16:1", "16:4", "16:5"],
            "full_interval_incidence_counted": False,
            "market_or_outcomes_opened": False,
        },
        "decision_binding": {
            "path": str(DECISION_PATH),
            "sha256": DECISION_SHA256,
        },
        "implementation_binding": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_contract": _source_contract(),
        "availability_contract": {
            "effective_point_in_time_field": "archive_label + 15 minutes",
            "documented_creation_cadence_minutes": 5,
            "conservative_publication_floor_minutes_after_label": (
                PUBLICATION_FLOOR_MINUTES
            ),
            "http_last_modified_is_availability": False,
            "http_etag_is_availability": False,
            "historical_archive_alone_proves_immutable_bytes": False,
            "bview_recopy_incident_disclosed": True,
            "later_signal_floor": (
                "availability plus separately frozen execution delay"
            ),
        },
        "source_quality_gates": _quality_contract(),
        "disk_contract": {
            "abort_before_download_when_used_gib_at_least": DISK_LIMIT_GIB,
            "stream_one_object_then_release": True,
            "maximum_concurrent_objects": 1,
            "maximum_compressed_bytes_each": MAXIMUM_COMPRESSED_BYTES,
            "maximum_decompressed_bytes_each": MAXIMUM_DECOMPRESSED_BYTES,
            "deterministic_output_gzip_mtime": 0,
        },
        "later_mechanism_boundary": {
            "authorized_now": False,
            "eligible_inputs": [
                "prior-only aggregate message-record churn",
                "prior-only aggregate peer-state-change churn",
                "prior-only decompressed-byte churn",
            ],
            "must_freeze_before_market": [
                "exact churn algebra and prior-only baseline",
                "event clock and direction",
                "hold, execution delay, leverage, and support controls",
                "clock novelty comparators",
            ],
            "llm_or_rllm_may_not": [
                "create, delete, retime, relabel, or impute source objects",
                "decode forbidden BGP identity fields without a new protocol",
                "select direction or threshold from eval outcomes",
            ],
        },
        "next_stage_artifacts": {
            "archive_labels_first": labels[0].isoformat().replace("+00:00", "Z"),
            "archive_labels_last": labels[-1].isoformat().replace("+00:00", "Z"),
            "replay_labels_first": audit_labels[0]
            .isoformat()
            .replace("+00:00", "Z"),
            "replay_labels_last": audit_labels[-1]
            .isoformat()
            .replace("+00:00", "Z"),
            "source_rows": "data/bgp_routing_churn_relay_2020_2023.jsonl.gz",
            "object_manifest": (
                "results/bgp_routing_churn_relay_object_manifest_2026-07-22.json.gz"
            ),
            "source_manifest": (
                "results/bgp_routing_churn_relay_source_manifest_2026-07-22.json"
            ),
            "source_support_result": (
                "results/bgp_routing_churn_relay_source_support_2026-07-22.json"
            ),
            "write_once": True,
        },
        "rejection_contract": (
            "any transport, metadata-age, gzip, MRT boundary, timestamp, type/subtype, "
            "coverage, duplicate, replay, disk, or source-support failure retires BRCR "
            "without changing collector, archive hours, interval, accepted fields, or "
            "thresholds"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("BRCR source protocol hash mismatch")
    for field in (
        "outcomes_opened",
        "market_clocks_opened",
        "historical_source_incidence_opened",
        "mechanism_parser_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"BRCR source protocol must keep {field}=false")
    if payload.get("source_only_probe_opened") is not True:
        raise RuntimeError("BRCR source protocol must disclose its bounded probe")
    expected = build_manifest()
    expected_core = {
        key: value for key, value in expected.items() if key != "manifest_hash"
    }
    if core != expected_core:
        raise RuntimeError("BRCR frozen source contract differs from code")


def write_manifest_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_manifest(payload)
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError("BRCR existing source protocol is not an object")
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen BRCR source protocol")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "manifest_hash": payload["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
