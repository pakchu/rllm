"""Freeze TMTR source support before full MADIS incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import operator
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/texas_metar_thermal_relay_source_protocol_2026-07-22.json")
DECISION_PATH = Path(
    "docs/texas-metar-thermal-relay-source-axis-decision-2026-07-22.md"
)
DECISION_SHA256 = "5839a46b1531a3ab569bf94dc9f3b639b24c1f63aadab2f86e00f53eb6f0c6e5"
SCRIPT_PATH = Path("training/preregister_texas_metar_thermal_relay.py")

SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SOURCE_END_EXCLUSIVE = datetime(2024, 1, 1, tzinfo=timezone.utc)
ARCHIVE_HOURS = (0, 6, 12, 18)
STATIONS = ("KMAF", "KABI", "KLBB", "KACT")
DISK_LIMIT_GIB = 300
ARCHIVE_BASE = (
    "https://madis-data.ncep.noaa.gov/madisPublic/data/archive"
)
REALTIME_BASE = (
    "https://madis-data.ncep.noaa.gov/madisPublic/data/point/metar/netcdf"
)
PUBLICATION_FLOOR_MINUTES = 60
NETCDF_MAGIC = b"CDF\x01"
NETCDF_REQUIRED_DIMENSIONS: dict[str, int | None] = {
    "recNum": None,
    "maxStaNamLen": 5,
    "maxRepLen": 6,
    "maxMETARLen": 256,
}
NETCDF_REQUIRED_VARIABLES: dict[str, dict[str, Any]] = {
    "stationName": {
        "dimensions": ("recNum", "maxStaNamLen"),
        "typecode": "c",
        "fill_value": None,
    },
    "reportType": {
        "dimensions": ("recNum", "maxRepLen"),
        "typecode": "c",
        "fill_value": None,
    },
    "rawMETAR": {
        "dimensions": ("recNum", "maxMETARLen"),
        "typecode": "c",
        "fill_value": None,
    },
    "timeObs": {
        "dimensions": ("recNum",),
        "typecode": "d",
        "fill_value": 1.7976931348623157e308,
    },
    "timeReceived": {
        "dimensions": ("recNum",),
        "typecode": "d",
        "fill_value": 1.7976931348623157e308,
    },
    "correction": {
        "dimensions": ("recNum",),
        "typecode": "i",
        "fill_value": -2147483647,
    },
}

VISIBLE_ASCII = re.compile(r"[\x20-\x7e]+", re.ASCII)
FORBIDDEN_RAW_TOKENS = frozenset({"COR", "NIL", "SPECI"})
SHA256_HEX = re.compile(r"[0-9a-f]{64}", re.ASCII)
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


def realtime_url(label: datetime) -> str:
    if label.tzinfo is None or label.utcoffset() != timedelta(0):
        raise ValueError("TMTR real-time label must be UTC-aware")
    if label.minute != 0 or label.second != 0 or label.microsecond != 0:
        raise ValueError("TMTR real-time label must be an exact hour")
    if label.hour not in ARCHIVE_HOURS:
        raise ValueError("TMTR real-time label is outside the frozen hour set")
    return f"{REALTIME_BASE}/{label:%Y%m%d_%H00.gz}"


def validate_netcdf_magic(raw: bytes) -> None:
    if raw[:4] != NETCDF_MAGIC:
        raise ValueError("TMTR object is not classic CDF1 netCDF")


def validate_netcdf_schema(
    dimensions: Mapping[str, int | None],
    variables: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, expected_size in NETCDF_REQUIRED_DIMENSIONS.items():
        if name not in dimensions or dimensions[name] != expected_size:
            raise ValueError(f"TMTR netCDF dimension {name} differs from the freeze")
    for name, expected in NETCDF_REQUIRED_VARIABLES.items():
        actual = variables.get(name)
        if actual is None:
            raise ValueError(f"TMTR netCDF variable {name} is missing")
        normalized = {
            "dimensions": tuple(actual.get("dimensions", ())),
            "typecode": actual.get("typecode"),
            "fill_value": actual.get("fill_value"),
        }
        if normalized != expected:
            raise ValueError(f"TMTR netCDF variable {name} differs from the freeze")


def decode_fixed_ascii(raw: bytes, *, width: int) -> str:
    if len(raw) != width:
        raise ValueError("TMTR fixed-width ASCII field has an unexpected width")
    trimmed = raw.rstrip(b"\x00 ")
    if not trimmed:
        raise ValueError("TMTR fixed-width ASCII field is empty")
    if b"\x00" in trimmed:
        raise ValueError("TMTR fixed-width ASCII field contains embedded NUL")
    try:
        decoded = trimmed.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("TMTR fixed-width field is not ASCII") from exc
    if VISIBLE_ASCII.fullmatch(decoded) is None:
        raise ValueError("TMTR fixed-width ASCII field contains a control character")
    return decoded


def epoch_seconds_to_utc(value: Any, *, fill_value: float) -> datetime:
    if isinstance(value, bool):
        raise ValueError("TMTR epoch value must be a finite integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("TMTR epoch value must be numeric") from exc
    if not math.isfinite(numeric) or numeric == fill_value or not numeric.is_integer():
        raise ValueError("TMTR epoch value must be a finite non-fill integer")
    try:
        return datetime.fromtimestamp(int(numeric), tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("TMTR epoch value is outside the UTC datetime range") from exc


def validate_correction(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("TMTR correction must be the integer zero")
    try:
        correction = operator.index(value)
    except TypeError as exc:
        raise ValueError("TMTR correction must be an integer") from exc
    if correction != 0:
        raise ValueError("TMTR correction must be the integer zero")
    return correction


def parse_raw_metar(station: str, raw_metar: str) -> dict[str, Any]:
    if station not in STATIONS:
        raise ValueError("TMTR station is outside the frozen panel")
    raw_metar.encode("ascii", errors="strict")
    if FORBIDDEN_RAW_TOKENS.intersection(raw_metar.split()):
        raise ValueError("TMTR raw METAR contains a forbidden report token")
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


def resolve_raw_observation_time(
    archive_label: datetime,
    parsed_raw_metar: Mapping[str, Any],
) -> datetime:
    archive_url(archive_label)
    candidates: list[datetime] = []
    for month_offset in (-1, 0, 1):
        month_index = archive_label.year * 12 + archive_label.month - 1 + month_offset
        year, zero_based_month = divmod(month_index, 12)
        month = zero_based_month + 1
        day = int(parsed_raw_metar["day"])
        if day > calendar.monthrange(year, month)[1]:
            continue
        candidate = datetime(
            year,
            month,
            day,
            int(parsed_raw_metar["hour"]),
            int(parsed_raw_metar["minute"]),
            tzinfo=timezone.utc,
        )
        if archive_label - timedelta(minutes=15) < candidate <= archive_label:
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("TMTR raw METAR timestamp does not resolve inside its archive window")
    return candidates[0]


def validate_temporal_envelope(
    archive_label: datetime,
    observation_time: datetime,
    receipt_time: datetime,
) -> None:
    archive_url(archive_label)
    for value in (observation_time, receipt_time):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("TMTR observation and receipt times must be UTC-aware")
    if not archive_label - timedelta(minutes=15) < observation_time <= archive_label:
        raise ValueError("TMTR observation is outside the frozen archive window")
    if not observation_time <= receipt_time <= archive_label + timedelta(minutes=15):
        raise ValueError("TMTR receipt is outside the frozen causal window")


def validate_metar_row_membership(
    *,
    archive_label: datetime,
    station: str,
    raw_metar: str,
    observation_time: datetime,
    receipt_time: datetime,
    report_type: str,
    correction: Any,
) -> None:
    if report_type != "METAR":
        raise ValueError("TMTR reportType must be exactly METAR")
    validate_correction(correction)
    parsed = parse_raw_metar(station, raw_metar)
    validate_temporal_envelope(archive_label, observation_time, receipt_time)
    if resolve_raw_observation_time(archive_label, parsed) != observation_time:
        raise ValueError("TMTR raw DDHHMMZ timestamp differs from timeObs")


def public_availability_time(
    archive_label: datetime,
    receipt_time: datetime,
) -> datetime:
    archive_url(archive_label)
    if receipt_time.tzinfo is None or receipt_time.utcoffset() != timedelta(0):
        raise ValueError("TMTR receipt time must be UTC-aware")
    publication_floor = archive_label + timedelta(minutes=PUBLICATION_FLOOR_MINUTES)
    return max(receipt_time, publication_floor)


def canonical_row_identity(
    *,
    archive_label: datetime,
    station: str,
    observation_time: datetime,
    receipt_time: datetime,
    report_type: str,
    correction: Any,
    raw_metar: str,
) -> str:
    validate_metar_row_membership(
        archive_label=archive_label,
        station=station,
        raw_metar=raw_metar,
        observation_time=observation_time,
        receipt_time=receipt_time,
        report_type=report_type,
        correction=correction,
    )
    identity = {
        "archive_label": archive_label.isoformat().replace("+00:00", "Z"),
        "stationName": station,
        "timeObs": int(observation_time.timestamp()),
        "timeReceived": int(receipt_time.timestamp()),
        "reportType": report_type,
        "correction": 0,
        "rawMETAR": raw_metar,
    }
    return canonical_hash(identity)


def resolve_object_row_identities(
    rows: Sequence[tuple[str, str]],
) -> dict[str, str]:
    identities_by_station: dict[str, set[str]] = defaultdict(set)
    for station, identity in rows:
        if station not in STATIONS:
            raise ValueError("TMTR object row has a station outside the frozen panel")
        if SHA256_HEX.fullmatch(identity) is None:
            raise ValueError("TMTR object row identity is not a SHA-256 hex digest")
        identities_by_station[station].add(identity)
    for station, identities in identities_by_station.items():
        if len(identities) > 1:
            raise ValueError(
                f"TMTR object has conflicting eligible rows for station {station}"
            )
    return {
        station: next(iter(identities_by_station[station]))
        for station in STATIONS
        if station in identities_by_station
    }


def summarize_expected_coverage(
    panels: Mapping[datetime, Collection[str] | None],
) -> dict[str, Any]:
    expected = archive_labels()
    expected_set = set(expected)
    if set(panels).difference(expected_set):
        raise ValueError("TMTR coverage input contains an out-of-contract label")

    years: dict[str, dict[str, Any]] = {}
    months: dict[str, dict[str, Any]] = {}
    incomplete_streak = 0
    maximum_incomplete_streak = 0
    for label in expected:
        year_key = str(label.year)
        month_key = label.strftime("%Y-%m")
        year = years.setdefault(
            year_key,
            {
                "expected_objects": 0,
                "available_objects": 0,
                "complete_panels": 0,
                "station_counts": {station: 0 for station in STATIONS},
            },
        )
        month = months.setdefault(
            month_key,
            {
                "expected_objects": 0,
                "available_objects": 0,
                "complete_panels": 0,
            },
        )
        year["expected_objects"] += 1
        month["expected_objects"] += 1

        stations_raw = panels.get(label)
        if stations_raw is None:
            station_set: set[str] = set()
        else:
            station_set = set(stations_raw)
            if not station_set.issubset(STATIONS):
                raise ValueError("TMTR coverage input contains an unknown station")
            year["available_objects"] += 1
            month["available_objects"] += 1
        for station in station_set:
            year["station_counts"][station] += 1
        if station_set == set(STATIONS):
            year["complete_panels"] += 1
            month["complete_panels"] += 1
            incomplete_streak = 0
        else:
            incomplete_streak += 1
            maximum_incomplete_streak = max(
                maximum_incomplete_streak,
                incomplete_streak,
            )

    return {
        "expected_labels": len(expected),
        "years": years,
        "months": months,
        "maximum_consecutive_incomplete_panels": maximum_incomplete_streak,
    }


def _source_contract() -> dict[str, Any]:
    return {
        "authority": {
            "operator": "NOAA/NWS NCEP MADIS",
            "archive_base": ARCHIVE_BASE,
            "real_time_base": REALTIME_BASE,
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
            "correction_integer_must_equal": 0,
            "raw_metar_fullmatch": METAR_PREFIX.pattern,
            "forbidden_raw_tokens": sorted(FORBIDDEN_RAW_TOKENS),
            "auto_token_allowed": True,
            "observation_window": "(archive_label - 15 minutes, archive_label]",
            "receipt_window": "[timeObs, archive_label + 15 minutes]",
            "maximum_observation_to_receipt_minutes_exclusive": 30,
            "raw_ddhhmmz_must_equal_time_obs": True,
            "month_and_year_rollover_resolved_against_archive_label": True,
            "exact_duplicate_rows": "deduplicate before cardinality check",
            "conflicting_or_multiple_rows_per_station_object": "fatal",
        },
        "netcdf_schema": {
            "magic_hex": NETCDF_MAGIC.hex(),
            "required_dimensions": NETCDF_REQUIRED_DIMENSIONS,
            "required_variables": {
                name: {**spec, "dimensions": list(spec["dimensions"])}
                for name, spec in NETCDF_REQUIRED_VARIABLES.items()
            },
            "aliases_allowed": False,
            "additional_dataset_fields_permitted_but_unread": True,
            "extractor_reads_only_required_variables": True,
            "fixed_ascii_decode": (
                "C-order bytes; trim trailing NUL/space only; reject empty, "
                "embedded NUL, control, and non-ASCII"
            ),
            "epoch_decode": "finite integer seconds since Unix epoch in UTC; fill rejected",
            "schema_corruption_effect": "fatal",
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
            "netcdf_schema_corruption_is_fatal": True,
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
        "denominators": {
            "object_year": "all expected frozen archive labels in that calendar year",
            "object_month": "all expected frozen archive labels in that calendar month",
            "station_year": (
                "all expected frozen archive labels in that year for each station"
            ),
            "all_four_year": "all expected frozen archive labels in that year",
            "all_four_month": "all expected frozen archive labels in that month",
            "missing_or_unparseable_object": (
                "counts missing for all four stations and as an incomplete panel"
            ),
            "panel_streak": "walk every expected label in chronological order",
            "fetched_or_successful_rows_never_define_a_denominator": True,
        },
        "failure_effect": "REJECT_NO_REPAIR",
    }


def build_manifest() -> dict[str, Any]:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("TMTR source-axis decision hash differs from the freeze")
    labels = archive_labels()
    core: dict[str, Any] = {
        "protocol_version": "texas_metar_thermal_relay_source_v3",
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
            "target_report_type": "METAR",
            "target_correction_value": 0,
            "netcdf_magic_hex": NETCDF_MAGIC.hex(),
            "required_dimension_values": {
                "recNum": None,
                "maxStaNamLen": 5,
                "maxRepLen": 6,
                "maxMETARLen": 256,
            },
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
            "receipt_field": "timeReceived",
            "effective_point_in_time_field": "source_row_availability",
            "archive_label_is_availability": False,
            "http_last_modified_is_availability": False,
            "real_time_object_route": f"{REALTIME_BASE}/YYYYMMDD_HH00.gz",
            "official_processing_cadence_minutes": 5,
            "nominal_file_window_end_minutes_after_label": 44,
            "conservative_publication_floor_minutes_after_label": (
                PUBLICATION_FLOOR_MINUTES
            ),
            "source_row_availability": "max(timeReceived, archive_label + 60 minutes)",
            "later_signal_floor": (
                "source_row_availability plus separately frozen execution delay"
            ),
            "historical_archive_alone_proves_point_in_time_availability": False,
            "negative_or_at_least_30_minute_receipt_delay": (
                "exclude and count as audit failure"
            ),
        },
        "source_quality_gates": _quality_contract(),
        "disk_contract": {
            "abort_before_download_when_used_gib_at_least": DISK_LIMIT_GIB,
            "stream_one_object_then_delete": True,
            "maximum_concurrent_objects": 1,
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
