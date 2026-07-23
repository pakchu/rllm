"""Run the frozen, outcome-blind Bybit public-trade source probe.

This module is intentionally not an alpha builder.  It inspects only the
official directory metadata and enough of three gzip objects to validate one
CSV header and one first record at each frozen boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/probe_bybit_public_trade_sequence_source.py")
DECISION_PATH = Path(
    "docs/bybit-public-trade-sequence-source-axis-decision-2026-07-23.md"
)
DEFAULT_OUTPUT = Path(
    "results/bybit_public_trade_sequence_source_feasibility_2026-07-23.json"
)

ARCHIVE_ROOT = "https://public.bybit.com/trading/BTCUSDT"
DIRECTORY_URL = f"{ARCHIVE_ROOT}/"
PROBE_DAYS = (
    date(2020, 3, 25),
    date(2023, 1, 1),
    date(2026, 7, 22),
)
DISK_LIMIT_GIB = 300
READ_CHUNK_BYTES = 16 * 1024
MAX_COMPRESSED_PREFIX_BYTES = 4 * 1024 * 1024
MAX_DECOMPRESSED_PREFIX_BYTES = 8 * 1024 * 1024
MAX_DIRECTORY_BYTES = 4 * 1024 * 1024
MAX_DAILY_CONTENT_LENGTH = 1024 * 1024 * 1024
USER_AGENT = "rllm-bybit-source-feasibility/1.0"

CANONICAL_ALIASES: dict[str, frozenset[str]] = {
    "timestamp": frozenset({"timestamp", "time", "tradetime", "exectime"}),
    "symbol": frozenset({"symbol"}),
    "side": frozenset({"side", "takerside"}),
    "size": frozenset({"size", "qty", "quantity"}),
    "price": frozenset({"price"}),
    "execution_id": frozenset(
        {"trdmatchid", "tradeid", "execid", "executionid", "id"}
    ),
}


class SourceProbeError(RuntimeError):
    """A frozen source or transport invariant failed."""


@dataclass(frozen=True)
class ProbeConfig:
    output: Path = DEFAULT_OUTPUT
    timeout_seconds: float = 30.0


OpenUrl = Callable[[str, float], Any]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def archive_url(day: date) -> str:
    return f"{ARCHIVE_ROOT}/BTCUSDT{day.isoformat()}.csv.gz"


def validate_official_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "public.bybit.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/trading/BTCUSDT/")
    ):
        raise SourceProbeError("URL differs from frozen official Bybit route")


def _default_open_url(url: str, timeout_seconds: float) -> Any:
    validate_official_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        method="GET",
    )
    context = ssl.create_default_context()
    return urllib.request.urlopen(request, timeout=timeout_seconds, context=context)


def used_gib(path: str | Path) -> int:
    candidate = _repo_path(path)
    stats = os.statvfs(candidate.parent if candidate.suffix else candidate)
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    return used // (1024**3)


def enforce_disk_guard(output: str | Path) -> int:
    current = used_gib(output)
    if current >= DISK_LIMIT_GIB:
        raise SourceProbeError(
            f"disk guard rejected {current} GiB used at {DISK_LIMIT_GIB} GiB"
        )
    return current


def _response_metadata(response: Any, expected_url: str) -> dict[str, Any]:
    status = int(getattr(response, "status", response.getcode()))
    final_url = str(response.geturl())
    if status != 200 or final_url != expected_url:
        raise SourceProbeError("unexpected HTTP status or redirect")
    headers = response.headers
    raw_length = headers.get("Content-Length")
    content_length: int | None = None
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise SourceProbeError("malformed Content-Length") from exc
        if content_length <= 0 or content_length > MAX_DAILY_CONTENT_LENGTH:
            raise SourceProbeError("daily Content-Length outside frozen bound")
    return {
        "status": status,
        "final_url": final_url,
        "content_length": content_length,
        "content_type": headers.get("Content-Type"),
        "last_modified": headers.get("Last-Modified"),
        "etag": headers.get("ETag"),
        "accept_ranges": headers.get("Accept-Ranges"),
    }


def _read_bounded(response: Any, limit: int) -> bytes:
    payload = bytearray()
    while True:
        chunk = response.read(min(READ_CHUNK_BYTES, limit + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > limit:
            raise SourceProbeError("response exceeds frozen metadata bound")
    if not payload:
        raise SourceProbeError("empty response")
    return bytes(payload)


def expected_2023_names() -> tuple[str, ...]:
    current = date(2023, 1, 1)
    end = date(2024, 1, 1)
    names: list[str] = []
    while current < end:
        names.append(f"BTCUSDT{current.isoformat()}.csv.gz")
        current += timedelta(days=1)
    return tuple(names)


def inspect_directory(
    *, timeout_seconds: float, open_url: OpenUrl = _default_open_url
) -> dict[str, Any]:
    validate_official_url(DIRECTORY_URL)
    with open_url(DIRECTORY_URL, timeout_seconds) as response:
        metadata = _response_metadata(response, DIRECTORY_URL)
        raw = _read_bounded(response, MAX_DIRECTORY_BYTES)
    text = raw.decode("utf-8", errors="strict")
    observed = set(re.findall(r"BTCUSDT2023-\d{2}-\d{2}\.csv\.gz", text))
    expected = set(expected_2023_names())
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return {
        "url": DIRECTORY_URL,
        "metadata": metadata,
        "response_bytes": len(raw),
        "response_sha256": sha256_bytes(raw),
        "expected_2023_files": len(expected),
        "observed_2023_files": len(observed),
        "missing_2023_files": missing,
        "unexpected_2023_files": unexpected,
        "complete_2023": not missing and not unexpected,
    }


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def map_required_fields(header: list[str]) -> dict[str, str]:
    normalized = {_normalize_header(value): value for value in header}
    mapping: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        matches = [normalized[alias] for alias in aliases if alias in normalized]
        if len(matches) != 1:
            raise SourceProbeError(
                f"required field {canonical!r} has {len(matches)} matches"
            )
        mapping[canonical] = matches[0]
    return mapping


def _positive_decimal(value: str, label: str) -> None:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise SourceProbeError(f"first record {label} is not decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise SourceProbeError(f"first record {label} is not finite positive")


def validate_first_record(
    header: list[str], row: list[str], mapping: Mapping[str, str]
) -> dict[str, bool]:
    if len(header) != len(row):
        raise SourceProbeError("first record field count differs from header")
    record = dict(zip(header, row, strict=True))
    _positive_decimal(record[mapping["timestamp"]], "timestamp")
    _positive_decimal(record[mapping["size"]], "size")
    _positive_decimal(record[mapping["price"]], "price")
    if record[mapping["symbol"]].strip().upper() != "BTCUSDT":
        raise SourceProbeError("first record symbol is not BTCUSDT")
    if record[mapping["side"]].strip().lower() not in {"buy", "sell"}:
        raise SourceProbeError("first record side is not Buy/Sell")
    if not record[mapping["execution_id"]].strip():
        raise SourceProbeError("first record execution ID is empty")
    return {
        "timestamp_positive_decimal": True,
        "symbol_btcusdt": True,
        "side_buy_or_sell": True,
        "size_positive_decimal": True,
        "price_positive_decimal": True,
        "execution_id_nonempty": True,
    }


def _read_two_csv_records(response: Any) -> tuple[bytes, bytes]:
    digest_input = bytearray()
    decompressed = bytearray()
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    while decompressed.count(b"\n") < 2:
        remaining = MAX_COMPRESSED_PREFIX_BYTES - len(digest_input)
        if remaining <= 0:
            raise SourceProbeError("gzip prefix lacks two complete CSV records")
        chunk = response.read(min(READ_CHUNK_BYTES, remaining))
        if not chunk:
            raise SourceProbeError("gzip object ended before two CSV records")
        digest_input.extend(chunk)
        try:
            inflated = inflater.decompress(chunk)
        except zlib.error as exc:
            raise SourceProbeError("invalid gzip prefix") from exc
        decompressed.extend(inflated)
        if len(decompressed) > MAX_DECOMPRESSED_PREFIX_BYTES:
            raise SourceProbeError("decompressed prefix exceeds frozen bound")
    lines = decompressed.splitlines(keepends=True)
    if len(lines) < 2:
        raise SourceProbeError("gzip prefix lacks complete lines")
    return bytes(digest_input), b"".join(lines[:2])


def inspect_archive_day(
    day: date, *, timeout_seconds: float, open_url: OpenUrl = _default_open_url
) -> dict[str, Any]:
    if day not in PROBE_DAYS:
        raise SourceProbeError("day is outside the frozen three-day probe")
    url = archive_url(day)
    validate_official_url(url)
    with open_url(url, timeout_seconds) as response:
        metadata = _response_metadata(response, url)
        compressed_prefix, csv_prefix = _read_two_csv_records(response)
    try:
        decoded = csv_prefix.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceProbeError("CSV prefix is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(decoded)))
    if len(rows) != 2:
        raise SourceProbeError("CSV prefix does not contain exactly two records")
    header, first = rows
    if len(header) != len(set(header)) or any(not value.strip() for value in header):
        raise SourceProbeError("CSV header is empty or duplicated")
    mapping = map_required_fields(header)
    validations = validate_first_record(header, first, mapping)
    return {
        "day": day.isoformat(),
        "url": url,
        "metadata": metadata,
        "compressed_prefix_bytes_consumed": len(compressed_prefix),
        "compressed_prefix_sha256": sha256_bytes(compressed_prefix),
        "csv_two_record_sha256": sha256_bytes(csv_prefix),
        "header": header,
        "field_count": len(header),
        "canonical_field_mapping": mapping,
        "first_record_sha256": sha256_bytes(
            (",".join(first) + "\n").encode("utf-8")
        ),
        "first_record_values_retained": False,
        "first_record_validations": validations,
    }


def build_artifact(
    cfg: ProbeConfig,
    *,
    open_url: OpenUrl = _default_open_url,
    disk_used_gib: int | None = None,
) -> dict[str, Any]:
    current_disk = enforce_disk_guard(cfg.output) if disk_used_gib is None else disk_used_gib
    if current_disk >= DISK_LIMIT_GIB:
        raise SourceProbeError(
            f"disk guard rejected {current_disk} GiB used at {DISK_LIMIT_GIB} GiB"
        )
    directory: dict[str, Any] | None = None
    probes: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        directory = inspect_directory(
            timeout_seconds=cfg.timeout_seconds,
            open_url=open_url,
        )
        if not directory["complete_2023"]:
            failures.append("official directory lacks exact 2023 daily coverage")
        for day in PROBE_DAYS:
            try:
                probes.append(
                    inspect_archive_day(
                        day,
                        timeout_seconds=cfg.timeout_seconds,
                        open_url=open_url,
                    )
                )
            except SourceProbeError as exc:
                failures.append(f"{day.isoformat()}: {exc}")
                break
    except SourceProbeError as exc:
        failures.append(f"directory: {exc}")

    if len(probes) == len(PROBE_DAYS):
        mappings = [probe["canonical_field_mapping"] for probe in probes]
        if any(mapping != mappings[0] for mapping in mappings[1:]):
            failures.append("canonical historical field mapping drifts across boundaries")

    artifact: dict[str, Any] = {
        "protocol_version": "bybit_public_trade_sequence_source_feasibility_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "SOURCE_FEASIBILITY_PASS" if not failures else "REJECT_NO_REPAIR",
        "source_axis": "Bybit BTCUSDT linear individual public trades",
        "provisional_candidate": "BSEA-24",
        "source_values_opened": bool(probes),
        "opened_source_days": [probe["day"] for probe in probes],
        "candidate_incidence_opened": False,
        "binance_comparator_opened": False,
        "market_outcomes_opened": False,
        "returns_or_pnl_opened": False,
        "directory": directory,
        "probes": probes,
        "failures": failures,
        "disk": {
            "used_gib_before_probe": current_disk,
            "limit_gib": DISK_LIMIT_GIB,
            "raw_archives_persisted": False,
        },
        "bindings": {
            "decision_path": str(DECISION_PATH),
            "decision_sha256": sha256_file(DECISION_PATH),
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
    }
    artifact["manifest_hash"] = canonical_hash(artifact)
    return artifact


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(canonical_json(artifact))
    temporary.replace(output)


def parse_args(argv: list[str] | None = None) -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    return ProbeConfig(output=args.output, timeout_seconds=args.timeout_seconds)


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    artifact = build_artifact(cfg)
    write_artifact(cfg.output, artifact)
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0 if artifact["decision"] == "SOURCE_FEASIBILITY_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
