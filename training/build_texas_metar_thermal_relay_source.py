"""Build and gate the outcome-blind TMTR 2020-2023 MADIS source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import operator
import os
import sqlite3
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scipy.io import netcdf_file

from training import preregister_texas_metar_thermal_relay as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_texas_metar_thermal_relay_source.py")
PROTOCOL_PATH = protocol.DEFAULT_OUTPUT
PROTOCOL_FILE_SHA256 = (
    "16d01d0d1e8f617b78c054e81e9407a9986625071ec77ff3584398fb39525f7d"
)
PROTOCOL_MANIFEST_HASH = (
    "e24d1a8c30efa5142b9f81ddd78f47a1673c5df2e6dc610072a06cca87c5e4eb"
)

DEFAULT_SOURCE = Path("data/texas_metar_thermal_relay_2020_2023.jsonl.gz")
DEFAULT_OBJECT_MANIFEST = Path(
    "results/texas_metar_thermal_relay_object_manifest_2026-07-22.json.gz"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "results/texas_metar_thermal_relay_source_manifest_2026-07-22.json"
)
DEFAULT_SUPPORT = Path(
    "results/texas_metar_thermal_relay_source_support_2026-07-22.json"
)
DEFAULT_CHECKPOINT = Path("checkpoints/tmtr_source_checkpoint.sqlite3")

USER_AGENT = "rllm-tmtr-source-freeze/1.0"
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
STATION_BYTES = frozenset(station.encode("ascii") for station in protocol.STATIONS)


class SourceContractError(RuntimeError):
    """A frozen source invariant failed and TMTR must not be repaired."""


class SourceFetchError(RuntimeError):
    """A transient or non-missing transport failure interrupted collection."""


@dataclass(frozen=True)
class ParsedObject:
    record_count: int
    rows: tuple[dict[str, Any], ...]
    ineligible_target_rows: int
    exclusion_counts: tuple[tuple[str, int], ...]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = _repo_path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).rstrip(b"\n"))


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("TMTR timestamp must be UTC-aware")
    return value.isoformat().replace("+00:00", "Z")


def load_frozen_protocol() -> dict[str, Any]:
    path = _repo_path(PROTOCOL_PATH)
    if sha256_file(path) != PROTOCOL_FILE_SHA256:
        raise RuntimeError(
            "TMTR source protocol file hash differs from builder binding"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("TMTR source protocol artifact is not an object")
    protocol.validate_manifest(payload)
    if payload.get("manifest_hash") != PROTOCOL_MANIFEST_HASH:
        raise RuntimeError(
            "TMTR source protocol manifest hash differs from builder binding"
        )
    return payload


def _used_gib(path: Path) -> int:
    stats = os.statvfs(path)
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    return used // (1024**3)


def enforce_disk_guard(path: str | Path) -> int:
    candidate = _repo_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    used_gib = _used_gib(candidate.parent)
    if used_gib >= protocol.DISK_LIMIT_GIB:
        raise SourceFetchError(
            f"TMTR disk guard rejected {used_gib} GiB used at "
            f"{protocol.DISK_LIMIT_GIB} GiB"
        )
    return used_gib


def validate_official_archive_url(label: datetime, url: str) -> None:
    expected = protocol.archive_url(label)
    parsed = urllib.parse.urlsplit(url)
    if (
        url != expected
        or parsed.scheme != "https"
        or parsed.hostname != "madis-data.ncep.noaa.gov"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("TMTR archive URL differs from the frozen official route")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        raise SourceFetchError("TMTR archive transport attempted a redirect")


def _read_bounded_response(response: Any) -> bytes:
    content_length = response.headers.get("Content-Length")
    advertised: int | None = None
    if content_length is not None:
        try:
            advertised = int(content_length)
        except ValueError as exc:
            raise SourceFetchError(
                "TMTR response has malformed Content-Length"
            ) from exc
        if advertised <= 0 or advertised > MAX_COMPRESSED_BYTES:
            raise SourceFetchError("TMTR response Content-Length is outside the bound")
    payload = bytearray()
    while True:
        chunk = response.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > MAX_COMPRESSED_BYTES:
            raise SourceFetchError("TMTR compressed object exceeds the memory bound")
    if not payload:
        raise SourceFetchError("TMTR archive returned an empty object")
    if advertised is not None and len(payload) != advertised:
        raise SourceFetchError("TMTR response body differs from Content-Length")
    return bytes(payload)


def fetch_archive_object(
    label: datetime,
    *,
    retries: int,
    timeout_seconds: float,
    retry_base_seconds: float,
) -> bytes | None:
    url = protocol.archive_url(label)
    validate_official_archive_url(label, url)
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        method="GET",
    )
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                if response.status != 200 or response.geturl() != url:
                    raise SourceFetchError(
                        "TMTR archive returned an unexpected response"
                    )
                return _read_bounded_response(response)
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 410}:
                return None
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == retries:
                raise SourceFetchError(
                    f"TMTR archive HTTP failure {exc.code} for {url}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise SourceFetchError(f"TMTR archive fetch failed for {url}") from exc
        except SourceFetchError:
            if attempt == retries:
                raise
        time.sleep(retry_base_seconds * (2**attempt))
    raise AssertionError("TMTR retry loop terminated unexpectedly")


def decompress_gzip_bounded(compressed: bytes) -> bytes:
    if not compressed.startswith(b"\x1f\x8b\x08"):
        raise SourceContractError("TMTR object is not a gzip stream")
    output = bytearray()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
            while True:
                chunk = handle.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > MAX_DECOMPRESSED_BYTES:
                    raise SourceContractError(
                        "TMTR decompressed object exceeds the memory bound"
                    )
    except (EOFError, OSError) as exc:
        raise SourceContractError("TMTR gzip CRC or stream validation failed") from exc
    return bytes(output)


def _normalize_fill(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _schema_snapshot(dataset: Any) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name in protocol.NETCDF_REQUIRED_VARIABLES:
        variable = dataset.variables.get(name)
        if variable is None:
            continue
        snapshot[name] = {
            "dimensions": tuple(variable.dimensions),
            "typecode": variable.typecode(),
            "fill_value": _normalize_fill(getattr(variable, "_FillValue", None)),
        }
    return snapshot


def _decode_correction(value: Any) -> int:
    try:
        correction = operator.index(value)
    except TypeError as exc:
        raise SourceContractError("TMTR correction scalar is not integer") from exc
    fill = protocol.NETCDF_REQUIRED_VARIABLES["correction"]["fill_value"]
    if correction == fill:
        raise SourceContractError("TMTR correction scalar is fill")
    return correction


def parse_netcdf_object(label: datetime, decompressed: bytes) -> ParsedObject:
    try:
        protocol.validate_netcdf_magic(decompressed)
        with netcdf_file(io.BytesIO(decompressed), "r", mmap=False) as dataset:
            protocol.validate_netcdf_schema(
                dataset.dimensions,
                _schema_snapshot(dataset),
            )
            variables = {
                name: dataset.variables[name]
                for name in protocol.NETCDF_REQUIRED_VARIABLES
            }
            record_count = int(variables["stationName"].shape[0])
            if record_count <= 0:
                raise SourceContractError("TMTR netCDF has no records")
            if any(
                int(variable.shape[0]) != record_count
                for variable in variables.values()
            ):
                raise SourceContractError("TMTR netCDF record dimensions disagree")

            candidates: list[dict[str, Any]] = []
            exclusions: Counter[str] = Counter()
            for index in range(record_count):
                station_raw = variables["stationName"][index].tobytes(order="C")
                if station_raw.rstrip(b"\x00 ") not in STATION_BYTES:
                    continue
                station = protocol.decode_fixed_ascii(
                    station_raw,
                    width=protocol.NETCDF_REQUIRED_DIMENSIONS["maxStaNamLen"] or 0,
                )
                report_type = protocol.decode_fixed_ascii(
                    variables["reportType"][index].tobytes(order="C"),
                    width=protocol.NETCDF_REQUIRED_DIMENSIONS["maxRepLen"] or 0,
                )
                raw_metar = protocol.decode_fixed_ascii(
                    variables["rawMETAR"][index].tobytes(order="C"),
                    width=protocol.NETCDF_REQUIRED_DIMENSIONS["maxMETARLen"] or 0,
                )
                observation_time = protocol.epoch_seconds_to_utc(
                    variables["timeObs"][index],
                    fill_value=protocol.NETCDF_REQUIRED_VARIABLES["timeObs"][
                        "fill_value"
                    ],
                )
                receipt_time = protocol.epoch_seconds_to_utc(
                    variables["timeReceived"][index],
                    fill_value=protocol.NETCDF_REQUIRED_VARIABLES["timeReceived"][
                        "fill_value"
                    ],
                )
                correction = _decode_correction(variables["correction"][index])
                try:
                    protocol.validate_metar_row_membership(
                        archive_label=label,
                        station=station,
                        raw_metar=raw_metar,
                        observation_time=observation_time,
                        receipt_time=receipt_time,
                        report_type=report_type,
                        correction=correction,
                    )
                except ValueError as exc:
                    exclusions[str(exc)] += 1
                    continue
                identity = protocol.canonical_row_identity(
                    archive_label=label,
                    station=station,
                    observation_time=observation_time,
                    receipt_time=receipt_time,
                    report_type=report_type,
                    correction=correction,
                    raw_metar=raw_metar,
                )
                candidates.append(
                    {
                        "archive_label_utc": _utc_iso(label),
                        "source_availability_utc": _utc_iso(
                            protocol.public_availability_time(label, receipt_time)
                        ),
                        "stationName": station,
                        "timeObs": int(observation_time.timestamp()),
                        "timeObs_utc": _utc_iso(observation_time),
                        "timeReceived": int(receipt_time.timestamp()),
                        "timeReceived_utc": _utc_iso(receipt_time),
                        "reportType": report_type,
                        "correction": correction,
                        "rawMETAR": raw_metar,
                        "row_identity_sha256": identity,
                    }
                )
    except SourceContractError:
        raise
    except Exception as exc:
        raise SourceContractError(
            "TMTR netCDF decode or schema validation failed"
        ) from exc

    try:
        identities = protocol.resolve_object_row_identities(
            [(row["stationName"], row["row_identity_sha256"]) for row in candidates]
        )
    except ValueError as exc:
        raise SourceContractError(
            "TMTR eligible target-row cardinality failed"
        ) from exc
    by_identity = {row["row_identity_sha256"]: row for row in candidates}
    rows = tuple(
        by_identity[identities[station]]
        for station in protocol.STATIONS
        if station in identities
    )
    return ParsedObject(
        record_count=record_count,
        rows=rows,
        ineligible_target_rows=sum(exclusions.values()),
        exclusion_counts=tuple(sorted(exclusions.items())),
    )


def process_compressed_object(label: datetime, compressed: bytes) -> dict[str, Any]:
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise SourceContractError("TMTR compressed object exceeds the memory bound")
    compressed_sha = sha256_bytes(compressed)
    decompressed = decompress_gzip_bounded(compressed)
    first = parse_netcdf_object(label, decompressed)
    second = parse_netcdf_object(label, decompressed)
    if first != second:
        raise SourceContractError("TMTR two in-memory parse passes differ")
    url = protocol.archive_url(label)
    rows = [
        {
            **row,
            "archive_url": url,
            "compressed_object_sha256": compressed_sha,
        }
        for row in first.rows
    ]
    return {
        "object": {
            "archive_label_utc": _utc_iso(label),
            "archive_url": url,
            "status": "available",
            "compressed_bytes": len(compressed),
            "compressed_sha256": compressed_sha,
            "decompressed_bytes": len(decompressed),
            "decompressed_sha256": sha256_bytes(decompressed),
            "netcdf_record_count": first.record_count,
            "eligible_stations": [row["stationName"] for row in rows],
            "eligible_station_count": len(rows),
            "ineligible_target_rows": first.ineligible_target_rows,
            "exclusion_counts": dict(first.exclusion_counts),
            "row_identity_sha256_by_station": {
                row["stationName"]: row["row_identity_sha256"] for row in rows
            },
        },
        "rows": rows,
    }


def missing_object_entry(label: datetime) -> dict[str, Any]:
    return {
        "object": {
            "archive_label_utc": _utc_iso(label),
            "archive_url": protocol.archive_url(label),
            "status": "missing",
        },
        "rows": [],
    }


class SourceCheckpoint:
    def __init__(self, path: str | Path, binding: Mapping[str, Any]) -> None:
        self.path = _repo_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS objects ("
            "label TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL)"
        )
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        expected = {
            key: canonical_json(value).decode("utf-8").rstrip("\n")
            for key, value in binding.items()
        }
        if existing and existing != expected:
            self.close()
            raise RuntimeError("TMTR checkpoint binding differs from current builder")
        object_count = self.connection.execute(
            "SELECT COUNT(*) FROM objects"
        ).fetchone()[0]
        if not existing and object_count:
            self.close()
            raise RuntimeError("TMTR checkpoint has objects without a binding")
        if not existing:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(expected.items()),
            )
            self.connection.commit()

    def labels(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT label FROM objects ORDER BY label"
            )
        ]

    def put(self, label: datetime, payload: Mapping[str, Any]) -> None:
        label_text = _utc_iso(label)
        payload_text = canonical_json(payload).decode("utf-8").rstrip("\n")
        payload_sha = sha256_bytes(payload_text.encode("utf-8"))
        existing = self.connection.execute(
            "SELECT payload, payload_sha256 FROM objects WHERE label = ?",
            (label_text,),
        ).fetchone()
        if existing is not None:
            if existing != (payload_text, payload_sha):
                raise RuntimeError("TMTR checkpoint label has conflicting payload")
            return
        self.connection.execute(
            "INSERT INTO objects(label, payload, payload_sha256) VALUES (?, ?, ?)",
            (label_text, payload_text, payload_sha),
        )
        self.connection.commit()

    def entries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload, payload_sha256 FROM objects ORDER BY label"
        )
        entries: list[dict[str, Any]] = []
        for payload, expected_sha in rows:
            if sha256_bytes(payload.encode("utf-8")) != expected_sha:
                raise RuntimeError("TMTR checkpoint payload hash mismatch")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise RuntimeError("TMTR checkpoint payload is not an object")
            entries.append(parsed)
        return entries

    def close(self) -> None:
        self.connection.close()

    def delete(self) -> None:
        self.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            if candidate.exists():
                candidate.unlink()


def checkpoint_binding() -> dict[str, Any]:
    return {
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
        "source_start": _utc_iso(protocol.SOURCE_START),
        "source_end_exclusive": _utc_iso(protocol.SOURCE_END_EXCLUSIVE),
        "expected_objects": len(protocol.archive_labels()),
    }


def collect_to_checkpoint(
    checkpoint: SourceCheckpoint,
    fetcher: Callable[[datetime], bytes | None],
    *,
    max_new_objects: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    labels = protocol.archive_labels()
    processed = checkpoint.labels()
    expected_prefix = [_utc_iso(label) for label in labels[: len(processed)]]
    if processed != expected_prefix:
        raise RuntimeError("TMTR checkpoint labels are not a contiguous frozen prefix")
    added = 0
    for index, label in enumerate(labels[len(processed) :], start=len(processed)):
        compressed = fetcher(label)
        try:
            entry = (
                missing_object_entry(label)
                if compressed is None
                else process_compressed_object(label, compressed)
            )
        except SourceContractError as exc:
            if compressed is None:
                raise AssertionError(
                    "TMTR missing objects cannot fail parsing"
                ) from exc
            confirmation = fetcher(label)
            if confirmation is None:
                raise SourceFetchError(
                    f"TMTR failed object disappeared during confirmation: {_utc_iso(label)}"
                ) from exc
            original_sha = sha256_bytes(compressed)
            confirmation_sha = sha256_bytes(confirmation)
            if confirmation_sha != original_sha:
                raise SourceFetchError(
                    "TMTR source bytes changed while confirming a parse failure at "
                    f"{_utc_iso(label)}: {original_sha} != {confirmation_sha}"
                ) from exc
            raise SourceContractError(
                f"{_utc_iso(label)} [{original_sha}]: {exc}"
            ) from exc
        checkpoint.put(label, entry)
        added += 1
        if progress is not None:
            progress(index + 1, len(labels))
        if max_new_objects > 0 and added >= max_new_objects:
            break
    return len(checkpoint.labels()), len(labels)


def _gate(
    gate_id: str,
    observed: float | int,
    threshold: float | int,
    operator_name: str,
) -> dict[str, Any]:
    if operator_name == ">=":
        passed = observed >= threshold
    elif operator_name == "<=":
        passed = observed <= threshold
    elif operator_name == "==":
        passed = observed == threshold
    else:
        raise ValueError("TMTR gate operator is unsupported")
    return {
        "gate_id": gate_id,
        "observed": observed,
        "threshold": threshold,
        "operator": operator_name,
        "passed": passed,
    }


def evaluate_support(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = protocol.archive_labels()
    if len(entries) != len(labels):
        raise RuntimeError("TMTR support requires every expected archive label")
    panels: dict[datetime, set[str] | None] = {}
    row_count = 0
    for label, entry in zip(labels, entries):
        object_row = entry["object"]
        if object_row["archive_label_utc"] != _utc_iso(label):
            raise RuntimeError("TMTR support entries are out of frozen order")
        if object_row["status"] == "missing":
            panels[label] = None
        elif object_row["status"] == "available":
            stations = set(object_row["eligible_stations"])
            if stations != {row["stationName"] for row in entry["rows"]}:
                raise RuntimeError(
                    "TMTR object station summary differs from source rows"
                )
            panels[label] = stations
            row_count += len(entry["rows"])
        else:
            raise RuntimeError("TMTR object has an unknown status")

    coverage = protocol.summarize_expected_coverage(panels)
    quality = protocol.build_manifest()["source_quality_gates"]
    gates: list[dict[str, Any]] = []
    for year, values in coverage["years"].items():
        expected = values["expected_objects"]
        gates.append(
            _gate(
                f"object_fraction_year_{year}",
                values["available_objects"] / expected,
                quality["transport"]["minimum_object_fraction_each_year"],
                ">=",
            )
        )
        gates.append(
            _gate(
                f"all_four_fraction_year_{year}",
                values["complete_panels"] / expected,
                quality["panel"]["minimum_all_four_fraction_each_year"],
                ">=",
            )
        )
        for station in protocol.STATIONS:
            gates.append(
                _gate(
                    f"station_fraction_{station}_{year}",
                    values["station_counts"][station] / expected,
                    quality["panel"]["minimum_station_fraction_each_year"],
                    ">=",
                )
            )
    for month, values in coverage["months"].items():
        expected = values["expected_objects"]
        gates.append(
            _gate(
                f"object_fraction_month_{month}",
                values["available_objects"] / expected,
                quality["transport"]["minimum_object_fraction_each_month"],
                ">=",
            )
        )
        gates.append(
            _gate(
                f"all_four_fraction_month_{month}",
                values["complete_panels"] / expected,
                quality["panel"]["minimum_all_four_fraction_each_month"],
                ">=",
            )
        )
    gates.append(
        _gate(
            "maximum_consecutive_incomplete_panels",
            coverage["maximum_consecutive_incomplete_panels"],
            quality["panel"]["maximum_consecutive_missing_all_four_anchors"],
            "<=",
        )
    )
    gates.extend(
        (
            _gate("compressed_sha256_fraction", 1.0, 1.0, "=="),
            _gate("exact_cross_pass_equality_fraction", 1.0, 1.0, "=="),
            _gate("retained_raw_syntax_fraction", 1.0, 1.0, "=="),
            _gate("retained_temporal_envelope_fraction", 1.0, 1.0, "=="),
            _gate("conflicting_target_reports", 0, 0, "=="),
            _gate("imputed_or_forward_filled_rows", 0, 0, "=="),
        )
    )
    all_passed = all(gate["passed"] for gate in gates)
    return {
        "source_id": "TMTR",
        "status": "PASS_ADVANCE_TO_THERMAL_FREEZE"
        if all_passed
        else "REJECT_NO_REPAIR",
        "all_gates_passed": all_passed,
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "thermal_parser_opened": False,
        "expected_object_count": len(labels),
        "source_row_count": row_count,
        "ineligible_target_row_count": sum(
            int(entry["object"].get("ineligible_target_rows", 0)) for entry in entries
        ),
        "coverage": coverage,
        "gates": gates,
    }


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return gzip.compress(
        b"".join(canonical_json(row) for row in rows),
        compresslevel=9,
        mtime=0,
    )


def _pretty_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_once(path: str | Path, raw: bytes) -> str:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite frozen TMTR artifact {output}")
        return "verified_existing"
    with output.open("xb") as handle:
        handle.write(raw)
    return "created"


def write_artifact_set(
    artifacts: Sequence[tuple[str, str | Path, bytes]],
) -> dict[str, str]:
    for _, path, raw in artifacts:
        output = _repo_path(path)
        if output.exists() and output.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite frozen TMTR artifact {output}")
    return {name: write_once(path, raw) for name, path, raw in artifacts}


def finalize_artifacts(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_path: str | Path = DEFAULT_SOURCE,
    object_manifest_path: str | Path = DEFAULT_OBJECT_MANIFEST,
    source_manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST,
    support_path: str | Path = DEFAULT_SUPPORT,
) -> dict[str, Any]:
    labels = protocol.archive_labels()
    if len(entries) != len(labels):
        raise RuntimeError("TMTR cannot finalize an incomplete checkpoint")
    rows = [row for entry in entries for row in entry["rows"]]
    object_rows = [entry["object"] for entry in entries]
    source_raw = _gzip_jsonl(rows)
    object_raw = _gzip_jsonl(object_rows)
    support_core = {
        **evaluate_support(entries),
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
    }
    support = {**support_core, "manifest_hash": canonical_hash(support_core)}
    support_raw = _pretty_json(support)

    source_manifest_core = {
        "source_id": "TMTR",
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_path": str(BUILDER_PATH),
        "builder_sha256": sha256_file(BUILDER_PATH),
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "thermal_parser_opened": False,
        "source": {
            "path": str(source_path),
            "rows": len(rows),
            "bytes": len(source_raw),
            "sha256": sha256_bytes(source_raw),
        },
        "object_manifest": {
            "path": str(object_manifest_path),
            "objects": len(object_rows),
            "bytes": len(object_raw),
            "sha256": sha256_bytes(object_raw),
        },
        "support": {
            "path": str(support_path),
            "bytes": len(support_raw),
            "sha256": sha256_bytes(support_raw),
            "status": support["status"],
        },
    }
    source_manifest = {
        **source_manifest_core,
        "manifest_hash": canonical_hash(source_manifest_core),
    }
    source_manifest_raw = _pretty_json(source_manifest)
    statuses = write_artifact_set(
        (
            ("source", source_path, source_raw),
            ("object_manifest", object_manifest_path, object_raw),
            ("support", support_path, support_raw),
            ("source_manifest", source_manifest_path, source_manifest_raw),
        )
    )
    return {
        "status": support["status"],
        "all_gates_passed": support["all_gates_passed"],
        "source_rows": len(rows),
        "objects": len(object_rows),
        "artifact_statuses": statuses,
        "source_manifest_hash": source_manifest["manifest_hash"],
    }


def write_fatal_rejection(path: str | Path, reason: str) -> str:
    core = {
        "source_id": "TMTR",
        "status": "REJECT_NO_REPAIR",
        "all_gates_passed": False,
        "fatal_source_contract_failure": reason,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "thermal_parser_opened": False,
    }
    return write_once(
        path, _pretty_json({**core, "manifest_hash": canonical_hash(core)})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--object-manifest", type=Path, default=DEFAULT_OBJECT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--max-new-objects", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--request-pace-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_frozen_protocol()
    enforce_disk_guard(args.checkpoint)
    checkpoint = SourceCheckpoint(args.checkpoint, checkpoint_binding())

    def fetcher(label: datetime) -> bytes | None:
        enforce_disk_guard(args.checkpoint)
        payload = fetch_archive_object(
            label,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            retry_base_seconds=args.retry_base_seconds,
        )
        if args.request_pace_seconds > 0:
            time.sleep(args.request_pace_seconds)
        return payload

    def progress(completed: int, total: int) -> None:
        if completed == total or completed % 25 == 0:
            print(json.dumps({"completed": completed, "total": total}), flush=True)

    try:
        completed, total = collect_to_checkpoint(
            checkpoint,
            fetcher,
            max_new_objects=args.max_new_objects,
            progress=progress,
        )
    except SourceContractError as exc:
        checkpoint.close()
        status = write_fatal_rejection(args.support, str(exc))
        raise SystemExit(f"TMTR REJECT_NO_REPAIR ({status}): {exc}") from exc

    if completed < total:
        checkpoint.close()
        print(
            json.dumps(
                {"status": "checkpointed", "completed": completed, "total": total},
                indent=2,
            )
        )
        return
    entries = checkpoint.entries()
    result = finalize_artifacts(
        entries,
        source_path=args.source,
        object_manifest_path=args.object_manifest,
        source_manifest_path=args.source_manifest,
        support_path=args.support,
    )
    checkpoint.delete()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
