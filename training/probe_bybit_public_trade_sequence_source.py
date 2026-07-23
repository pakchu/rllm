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
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/probe_bybit_public_trade_sequence_source.py")
DECISION_PATH = Path(
    "docs/bybit-public-trade-sequence-source-axis-decision-2026-07-23.md"
)
DEFAULT_OUTPUT = Path(
    "results/bybit_public_trade_sequence_source_feasibility_v2_2026-07-23.json"
)
CORRECTION_PATH = Path(
    "docs/bybit-public-trade-source-probe-v1-invalidation-v2-correction-"
    "2026-07-23.md"
)
CORRECTION_SHA256 = (
    "73996f714ce74e1bc81268b545be04177375d61554ba33a219980b3b04ca0bda"
)
V1_INVALID_PATH = Path(
    "results/bybit_public_trade_sequence_source_feasibility_v1_invalid_"
    "2026-07-23.json"
)
V1_INVALID_FILE_SHA256 = (
    "3e2872467acebfd07f91ff8b9ff0079eb9dc518f6f37ad79f83a6a47cf413536"
)
V1_INVALID_MANIFEST_HASH = (
    "fdc57f5ecfb34f516726fbd1d323faf1d397cd492c81ae23ef006354b3554227"
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
MAX_DECOMPRESSED_PREFIX_BYTES = 128 * 1024
MAX_DIRECTORY_BYTES = 4 * 1024 * 1024
MAX_DAILY_CONTENT_LENGTH = 1024 * 1024 * 1024
USER_AGENT = "rllm-bybit-source-feasibility/2.0"

V1_PREFIX_SHA256 = {
    "2020-03-25": "dfde89d406b7d179c03fef8116d2668ab52999199e39e07cdd55175a6e749821",
    "2023-01-01": "e91c088cdf75a06fcaecb42be75d59178706798029cdec098ef67e49cc2e7455",
    "2026-07-22": "ad70af6e771252b3c5331234882d7a60defece0813e9ca189c91b64bc93e6039",
}
FROZEN_BASE_HEADER = (
    "timestamp",
    "symbol",
    "side",
    "size",
    "price",
    "tickDirection",
    "trdMatchID",
    "grossValue",
    "homeNotional",
    "foreignNotional",
)
ALLOWED_OPTIONAL_SUFFIX = ("RPI",)

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


class SourcePrefixMismatch(SourceProbeError):
    """The replay prefix differs before any source record is decompressed."""

    def __init__(self, day: date, observed_sha256: str) -> None:
        self.day = day
        self.observed_sha256 = observed_sha256
        super().__init__(f"{day.isoformat()}: compressed prefix differs from v1")


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
        del req, fp, msg, headers
        raise SourceProbeError(
            f"redirect rejected before target access: HTTP {code} to {newurl}"
        )


def _default_open_url(url: str, timeout_seconds: float) -> Any:
    validate_official_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        method="GET",
    )
    context = ssl.create_default_context()
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=context),
    )
    try:
        return opener.open(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        raise SourceProbeError(f"HTTP {exc.code} for frozen Bybit URL") from exc
    except urllib.error.URLError as exc:
        raise SourceProbeError("frozen Bybit URL transport failed") from exc


def used_gib(path: str | Path) -> int:
    candidate = _repo_path(path)
    stats = os.statvfs(candidate.parent if candidate.suffix else candidate)
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    return used // (1024**3)


def enforce_disk_guard() -> int:
    current = used_gib(REPO_ROOT)
    if current >= DISK_LIMIT_GIB:
        raise SourceProbeError(
            f"disk guard rejected {current} GiB used at {DISK_LIMIT_GIB} GiB"
        )
    return current


def _response_metadata(response: Any, expected_url: str) -> dict[str, Any]:
    status_value = getattr(response, "status", None)
    status = int(response.getcode() if status_value is None else status_value)
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


class _AnchorHrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "a":
            return
        href_values = [value for key, value in attrs if key.lower() == "href"]
        if len(href_values) != 1 or href_values[0] is None:
            raise SourceProbeError("directory anchor lacks one exact href")
        self.hrefs.append(href_values[0])


def inspect_directory(
    *, timeout_seconds: float, open_url: OpenUrl = _default_open_url
) -> dict[str, Any]:
    validate_official_url(DIRECTORY_URL)
    with open_url(DIRECTORY_URL, timeout_seconds) as response:
        metadata = _response_metadata(response, DIRECTORY_URL)
        raw = _read_bounded(response, MAX_DIRECTORY_BYTES)
    text = raw.decode("utf-8", errors="strict")
    parser = _AnchorHrefParser()
    parser.feed(text)
    parser.close()
    pattern = re.compile(r"BTCUSDT2023-\d{2}-\d{2}\.csv\.gz")
    observed_counts = Counter(
        href for href in parser.hrefs if pattern.fullmatch(href) is not None
    )
    observed = set(observed_counts)
    expected = set(expected_2023_names())
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(name for name, count in observed_counts.items() if count != 1)
    return {
        "url": DIRECTORY_URL,
        "metadata": metadata,
        "response_bytes": len(raw),
        "response_sha256": sha256_bytes(raw),
        "expected_2023_files": len(expected),
        "observed_2023_files": len(observed),
        "missing_2023_files": missing,
        "unexpected_2023_files": unexpected,
        "duplicate_2023_hrefs": duplicates,
        "complete_2023": not missing and not unexpected and not duplicates,
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


@dataclass
class _CsvBoundaryState:
    in_quotes: bool = False
    pending_quote: bool = False
    record_ends: list[int] | None = None

    def __post_init__(self) -> None:
        if self.record_ends is None:
            self.record_ends = []

    def feed(self, value: int, offset_after_byte: int) -> None:
        if self.in_quotes:
            if self.pending_quote:
                if value == ord('"'):
                    self.pending_quote = False
                    return
                self.in_quotes = False
                self.pending_quote = False
                self._outside(value, offset_after_byte)
                return
            if value == ord('"'):
                self.pending_quote = True
            return
        self._outside(value, offset_after_byte)

    def _outside(self, value: int, offset_after_byte: int) -> None:
        if value == ord('"'):
            self.in_quotes = True
        elif value == ord("\n"):
            assert self.record_ends is not None
            self.record_ends.append(offset_after_byte)


def _read_compressed_prefix(response: Any) -> bytes:
    payload = response.read(READ_CHUNK_BYTES)
    if not payload:
        raise SourceProbeError("empty gzip prefix")
    if len(payload) > READ_CHUNK_BYTES:
        raise SourceProbeError("response ignored the frozen prefix bound")
    return bytes(payload)


def _decompress_two_csv_records(
    compressed_prefix: bytes,
) -> tuple[bytes, bytes, bytes]:
    decompressed = bytearray()
    boundaries = _CsvBoundaryState()
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = compressed_prefix
    while len(boundaries.record_ends or []) < 2:
        if not pending:
            raise SourceProbeError("frozen gzip prefix lacks two CSV records")
        previous_pending_length = len(pending)
        try:
            inflated = inflater.decompress(pending, max_length=1)
        except zlib.error as exc:
            raise SourceProbeError("invalid gzip prefix") from exc
        pending = inflater.unconsumed_tail
        if inflated:
            decompressed.extend(inflated)
            if len(decompressed) > MAX_DECOMPRESSED_PREFIX_BYTES:
                raise SourceProbeError("decompressed prefix exceeds frozen bound")
            boundaries.feed(inflated[0], len(decompressed))
            continue
        if inflater.eof:
            raise SourceProbeError("gzip object ended before two CSV records")
        if pending and len(pending) == previous_pending_length:
            raise SourceProbeError("gzip decompressor made no bounded progress")

    record_ends = boundaries.record_ends
    assert record_ends is not None and len(record_ends) == 2
    first_end, second_end = record_ends
    exact_prefix = bytes(decompressed[:second_end])
    raw_header = exact_prefix[:first_end]
    raw_first_record = exact_prefix[first_end:second_end]
    return exact_prefix, raw_header, raw_first_record


def inspect_archive_day(
    day: date,
    *,
    expected_prefix_sha256: str,
    timeout_seconds: float,
    open_url: OpenUrl = _default_open_url,
) -> dict[str, Any]:
    if day not in PROBE_DAYS:
        raise SourceProbeError("day is outside the frozen three-day probe")
    url = archive_url(day)
    validate_official_url(url)
    with open_url(url, timeout_seconds) as response:
        metadata = _response_metadata(response, url)
        compressed_prefix = _read_compressed_prefix(response)
    observed_prefix_sha256 = sha256_bytes(compressed_prefix)
    if observed_prefix_sha256 != expected_prefix_sha256:
        raise SourcePrefixMismatch(day, observed_prefix_sha256)
    csv_prefix, raw_header, raw_first = _decompress_two_csv_records(
        compressed_prefix
    )
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
        "compressed_prefix_sha256": observed_prefix_sha256,
        "csv_two_record_sha256": sha256_bytes(csv_prefix),
        "raw_header_sha256": sha256_bytes(raw_header),
        "first_record_raw_sha256": sha256_bytes(raw_first),
        "logical_csv_records_decompressed": 2,
        "decompressed_bytes": len(csv_prefix),
        "bytes_decompressed_after_first_record": 0,
        "header": header,
        "field_count": len(header),
        "canonical_field_mapping": mapping,
        "first_record_values_retained": False,
        "first_record_validations": validations,
    }


def classify_schema_drift(probes: list[dict[str, Any]]) -> dict[str, Any]:
    by_day: list[dict[str, Any]] = []
    acceptable = True
    base = list(FROZEN_BASE_HEADER)
    allowed_recent = base + list(ALLOWED_OPTIONAL_SUFFIX)
    for item in probes:
        day = str(item["day"])
        header = list(item["header"])
        if header == base:
            added: list[str] = []
            classification = "frozen_base_header"
        elif day == "2026-07-22" and header == allowed_recent:
            added = list(ALLOWED_OPTIONAL_SUFFIX)
            classification = "explicit_recent_optional_suffix"
        else:
            added = [field for field in header if field not in base]
            classification = "unclassified_drift"
            acceptable = False
        removed = [field for field in base if field not in header]
        common = [field for field in header if field in base]
        reordered = common != [field for field in base if field in header]
        if removed or reordered:
            acceptable = False
        by_day.append(
            {
                "day": day,
                "classification": classification,
                "added_fields": added,
                "removed_fields": removed,
                "base_fields_reordered": reordered,
                "accepted": classification != "unclassified_drift"
                and not removed
                and not reordered,
            }
        )
    return {
        "observed": any(row["added_fields"] for row in by_day),
        "frozen_base_header": base,
        "allowed_optional_suffix": list(ALLOWED_OPTIONAL_SUFFIX),
        "optional_fields_excluded_from_primary": list(ALLOWED_OPTIONAL_SUFFIX),
        "by_day": by_day,
        "explicitly_classified": acceptable,
    }


def validate_correction_bindings() -> None:
    if sha256_file(CORRECTION_PATH) != CORRECTION_SHA256:
        raise SourceProbeError("v2 correction document hash differs")
    if sha256_file(V1_INVALID_PATH) != V1_INVALID_FILE_SHA256:
        raise SourceProbeError("invalid v1 audit artifact hash differs")


def build_artifact(
    cfg: ProbeConfig,
    *,
    open_url: OpenUrl = _default_open_url,
) -> dict[str, Any]:
    validate_correction_bindings()
    current_disk = enforce_disk_guard()
    directory: dict[str, Any] | None = None
    probes: list[dict[str, Any]] = []
    prefix_rejections: list[dict[str, str]] = []
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
                probe = inspect_archive_day(
                    day,
                    expected_prefix_sha256=V1_PREFIX_SHA256[day.isoformat()],
                    timeout_seconds=cfg.timeout_seconds,
                    open_url=open_url,
                )
                probes.append(probe)
            except SourcePrefixMismatch as exc:
                prefix_rejections.append(
                    {
                        "day": exc.day.isoformat(),
                        "observed_compressed_prefix_sha256": exc.observed_sha256,
                        "expected_compressed_prefix_sha256": V1_PREFIX_SHA256[
                            exc.day.isoformat()
                        ],
                        "source_schema_decompressed": "false",
                    }
                )
                failures.append(str(exc))
                break
            except SourceProbeError as exc:
                failures.append(f"{day.isoformat()}: {exc}")
                break
    except SourceProbeError as exc:
        failures.append(f"directory: {exc}")

    schema_drift: dict[str, Any] | None = None
    if len(probes) == len(PROBE_DAYS):
        mappings = [probe["canonical_field_mapping"] for probe in probes]
        if any(mapping != mappings[0] for mapping in mappings[1:]):
            failures.append("canonical historical field mapping drifts across boundaries")
        schema_drift = classify_schema_drift(probes)
        if not schema_drift["explicitly_classified"]:
            failures.append("full historical schema drift is not explicitly classified")

    artifact: dict[str, Any] = {
        "protocol_version": "bybit_public_trade_sequence_source_feasibility_v2",
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
        "prefix_rejections": prefix_rejections,
        "prefix_binding_enforced": True,
        "schema_drift": schema_drift,
        "failures": failures,
        "disk": {
            "used_gib_before_probe": current_disk,
            "limit_gib": DISK_LIMIT_GIB,
            "guard_filesystem": str(REPO_ROOT),
            "guard_enforced": True,
            "raw_archives_persisted": False,
        },
        "bindings": {
            "decision_path": str(DECISION_PATH),
            "decision_sha256": sha256_file(DECISION_PATH),
            "correction_path": str(CORRECTION_PATH),
            "correction_sha256": CORRECTION_SHA256,
            "invalid_v1_path": str(V1_INVALID_PATH),
            "invalid_v1_file_sha256": V1_INVALID_FILE_SHA256,
            "invalid_v1_manifest_hash": V1_INVALID_MANIFEST_HASH,
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
        "v1_decision_authoritative": False,
        "manifest_hash_scope": "canonical_json_excluding_manifest_hash_without_self",
    }
    artifact["manifest_hash_without_self"] = canonical_hash(artifact)
    return artifact


def validate_manifest_hash(artifact: Mapping[str, Any]) -> None:
    mutable = dict(artifact)
    observed = mutable.pop("manifest_hash_without_self", None)
    if mutable.get("manifest_hash_scope") != (
        "canonical_json_excluding_manifest_hash_without_self"
    ):
        raise SourceProbeError("unknown manifest hash scope")
    if not isinstance(observed, str) or observed != canonical_hash(mutable):
        raise SourceProbeError("manifest hash verification failed")


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_manifest_hash(artifact)
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
