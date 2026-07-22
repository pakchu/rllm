"""Freeze TMTR source support before full MADIS incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/texas_metar_thermal_relay_source_protocol_2026-07-22.json")
DECISION_PATH = Path(
    "docs/texas-metar-thermal-relay-source-axis-decision-2026-07-22.md"
)
DECISION_SHA256 = "ae95b6211ad701e638984e1739b23fd8b08d2a3da01a29b572dc15820d6acb5d"
SCRIPT_PATH = Path("training/preregister_texas_metar_thermal_relay.py")

SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SOURCE_END_EXCLUSIVE = datetime(2024, 1, 1, tzinfo=timezone.utc)
ARCHIVE_HOURS = (0, 6, 12, 18)
STATIONS = ("KMAF", "KABI", "KLBB", "KACT")
DISK_LIMIT_GIB = 300
ARCHIVE_BASE = (
    "https://madis-data.ncep.noaa.gov/madisPublic/data/archive"
)

VISIBLE_ASCII = re.compile(r"[\x20-\x7e]+", re.ASCII)
METAR_PREFIX = re.compile(
    r"\A(?:METAR )?(?P<station>KMAF|KABI|KLBB|KACT) "
    r"(?P<day>0[1-9]|[12][0-9]|3[01])"
    r"(?P<hour>[01][0-9]|2[0-3])"
    r"(?P<minute>[0-5][0-9])Z(?: [!-~]+)+\Z",
    re.ASCII,
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
    if len(labels) != 5844:
        raise RuntimeError("TMTR archive-label envelope is not exactly 5,844 objects")
    return labels


def archive_url(label: datetime) -> str:
    if label.tzinfo is None or label.utcoffset() != timedelta(0):
        raise ValueError("TMTR archive label must be UTC-aware")
    if label.minute != 0 or label.second != 0 or label.microsecond != 0:
        raise ValueError("TMTR archive label must be an exact hour")
    if label.hour not in ARCHIVE_HOURS:
        raise ValueError("TMTR archive label is outside the frozen hour set")
    if not SOURCE_START <= label < SOURCE_END_EXCLUSIVE:
        raise ValueError("TMTR archive label is outside the frozen interval")
    stamp = label.strftime("%Y%m%d_%H00.gz")
    return (
        f"{ARCHIVE_BASE}/{label:%Y/%m/%d}/point/metar/netcdf/{stamp}"
    )


def parse_raw_metar(station: str, raw_metar: str) -> dict[str, Any]:
    if station not in STATIONS:
        raise ValueError("TMTR station is outside the frozen panel")
    raw_metar.encode("ascii", errors="strict")
    if not 20 <= len(raw_metar) <= 255:
        raise ValueError("TMTR raw METAR length is outside the frozen boundary")
    if VISIBLE_ASCII.fullmatch(raw_metar) is None:
        raise ValueError("TMTR raw METAR contains a control character")
    match = METAR_PREFIX.fullmatch(raw_metar)
    if match is None or match.group("station") != station:
        raise ValueError("TMTR raw METAR identity or timestamp token is malformed")
    return {
        "station": station,
        "day": int(match.group("day")),
        "hour": int(match.group("hour")),
        "minute": int(match.group("minute")),
    }


def validate_temporal_envelope(
    archive_label: datetime,
    observation_time: datetime,
    receipt_time: datetime,
) -> None:
    archive_url(archive_label)
    for value in (observation_time, receipt_time):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("TMTR observation and receipt times must be UTC-aware")
    if not archive_label - timedelta(minutes=75) < observation_time <= archive_label:
        raise ValueError("TMTR observation is outside the frozen archive window")
    if not observation_time <= receipt_time <= archive_label + timedelta(minutes=15):
        raise ValueError("TMTR receipt is outside the frozen causal window")
    if receipt_time - observation_time > timedelta(minutes=30):
        raise ValueError("TMTR observation-to-receipt delay exceeds 30 minutes")


def _source_contract() -> dict[str, Any]:
    return {
        "authority": {
            "operator": "NOAA/NWS NCEP MADIS",
            "archive_base": ARCHIVE_BASE,
            "dataset_path": "point/metar/netcdf",
            "transport": "public HTTPS gzip-compressed classic netCDF",
            "historical_documentation": (
                "https://madis.ncep.noaa.gov/faq_historicaldata.shtml"
            ),
        },
        "archive_interval": {
            "start_inclusive": SOURCE_START.isoformat().replace("+00:00", "Z"),
            "end_exclusive": SOURCE_END_EXCLUSIVE.isoformat().replace("+00:00", "Z"),
            "hours_utc": list(ARCHIVE_HOURS),
            "expected_objects": 5844,
        },
        "station_panel": [
            {"station": "KMAF", "name": "Midland International", "role": "west"},
            {"station": "KABI", "name": "Abilene Regional", "role": "west-central"},
            {"station": "KLBB", "name": "Lubbock Preston Smith", "role": "northwest"},
            {"station": "KACT", "name": "Waco Regional", "role": "central"},
        ],
        "membership": {
            "report_type": "METAR",
            "correction_field": "empty",
            "raw_metar_fullmatch": METAR_PREFIX.pattern,
            "observation_window": "(archive_label - 75 minutes, archive_label]",
            "receipt_window": "[timeObs, archive_label + 15 minutes]",
            "maximum_observation_to_receipt_minutes": 30,
            "exact_duplicate_rows": "deduplicate before cardinality check",
            "conflicting_or_multiple_rows_per_station_object": "fatal",
        },
        "retained_fields": [
            "archive label and official URL",
            "compressed object SHA-256 and byte count",
            "stationName",
            "timeObs",
            "timeReceived",
            "reportType",
            "correction",
            "rawMETAR",
            "canonical row identity SHA-256",
        ],
        "forbidden_fields": [
            "decoded MADIS meteorological variables or QC summaries",
            "latitude/longitude/elevation",
            "GHCNh, ISD CSV, forecast, or unofficial mirror fallback",
            "BTC bars/returns/funding/PnL",
            "prior alpha clocks or portfolio outcomes",
        ],
    }


def _quality_contract() -> dict[str, Any]:
    return {
        "transport": {
            "minimum_object_fraction_each_year": 0.995,
            "minimum_object_fraction_each_month": 0.98,
            "gzip_crc_required": True,
            "classic_netcdf_magic_required": True,
            "compressed_sha256_fraction": 1.0,
            "two_in_memory_parse_passes_required": True,
            "exact_cross_pass_equality_required": True,
        },
        "panel": {
            "minimum_station_fraction_each_year": 0.95,
            "minimum_all_four_fraction_each_year": 0.90,
            "minimum_all_four_fraction_each_month": 0.85,
            "maximum_consecutive_missing_all_four_anchors": 4,
            "retained_raw_syntax_fraction": 1.0,
            "retained_temporal_envelope_fraction": 1.0,
            "conflicting_target_reports_allowed": 0,
            "imputation_or_forward_fill_allowed": False,
        },
        "failure_effect": "REJECT_NO_REPAIR",
    }


def build_manifest() -> dict[str, Any]:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("TMTR source-axis decision hash differs from the freeze")
    labels = archive_labels()
    core: dict[str, Any] = {
        "protocol_version": "texas_metar_thermal_relay_source_v1",
        "source_id": "TMTR",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "historical_source_incidence_opened": False,
        "thermal_parser_opened": False,
        "source_only_probe_opened": True,
        "source_only_probe": {
            "archive_label": "2023-01-01T00:00:00Z",
            "url": (
                f"{ARCHIVE_BASE}/2023/01/01/point/metar/netcdf/"
                "20230101_0000.gz"
            ),
            "compressed_bytes": 1219946,
            "compressed_sha256": (
                "01b595a18291d75d94851c0ed2de2ae1725f8fadd7d10aaffaa27e9ea5528ffb"
            ),
            "decompressed_bytes": 12263204,
            "decompressed_sha256": (
                "ec97587646a1a2492143cbc22a2d891640af1cfd2905630cb9b4f48f2f6bb8c1"
            ),
            "netcdf_record_count": 8243,
            "target_station_rows": 4,
            "target_stations": list(STATIONS),
            "target_receipt_time": "2022-12-31T23:58:00Z",
            "full_interval_incidence_counted": False,
            "thermal_values_parsed": False,
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
            "causal_field": "timeReceived",
            "archive_label_is_availability": False,
            "http_last_modified_is_availability": False,
            "later_signal_floor": "max retained timeReceived plus frozen execution delay",
            "negative_or_over_30_minute_receipt_delay": "exclude and count as audit failure",
        },
        "source_quality_gates": _quality_contract(),
        "disk_contract": {
            "abort_before_download_when_used_gib_at_least": DISK_LIMIT_GIB,
            "stream_one_object_then_delete": True,
            "maximum_concurrent_objects": 8,
            "deterministic_output_gzip_mtime": 0,
        },
        "later_mechanism_boundary": {
            "authorized_now": False,
            "eligible_input": "rawMETAR only",
            "must_freeze_before_market": [
                "exact weather-token parser",
                "thermal state algebra",
                "event clock and direction",
                "hold, execution delay, leverage, and support controls",
                "clock novelty comparators",
            ],
            "llm_or_rllm_may_not": [
                "create, delete, retime, or relabel source reports",
                "repair missing stations",
                "select direction or threshold from eval outcomes",
            ],
        },
        "next_stage_artifacts": {
            "archive_labels_first": labels[0].isoformat().replace("+00:00", "Z"),
            "archive_labels_last": labels[-1].isoformat().replace("+00:00", "Z"),
            "source_rows": "data/texas_metar_thermal_relay_2020_2023.jsonl.gz",
            "object_manifest": (
                "results/texas_metar_thermal_relay_object_manifest_2026-07-22.json.gz"
            ),
            "source_manifest": (
                "results/texas_metar_thermal_relay_source_manifest_2026-07-22.json"
            ),
            "source_support_result": (
                "results/texas_metar_thermal_relay_source_support_2026-07-22.json"
            ),
            "write_once": True,
        },
        "rejection_contract": (
            "any transport, gzip, netCDF, schema, object-coverage, station-coverage, "
            "receipt-time, raw-identity, duplicate, conflict, replay, disk, or source-"
            "support failure retires TMTR without changing stations, archive hours, "
            "interval, accepted fields, or thresholds"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("TMTR source protocol hash mismatch")
    for field in (
        "outcomes_opened",
        "market_clocks_opened",
        "historical_source_incidence_opened",
        "thermal_parser_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"TMTR source protocol must keep {field}=false")
    if payload.get("source_only_probe_opened") is not True:
        raise RuntimeError("TMTR source protocol must disclose its bounded probe")
    expected = build_manifest()
    expected_core = {
        key: value for key, value in expected.items() if key != "manifest_hash"
    }
    if core != expected_core:
        raise RuntimeError("TMTR frozen source contract differs from code")


def write_manifest_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_manifest(payload)
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError("TMTR existing source protocol is not an object")
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen TMTR source protocol")
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
            {"status": status, "output": str(args.output), "manifest_hash": payload["manifest_hash"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
