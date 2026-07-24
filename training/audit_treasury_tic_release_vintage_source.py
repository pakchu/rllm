"""One-shot source-only audit for Treasury TIC release-vintage archives.

The runner is intentionally unable to inspect TIC table semantics or market
outcomes.  It freezes the official archive identities, retrieves each release
exactly once through the documented redirect, validates the ZIP container at
the byte level, and publishes only source-support metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import http.client
import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import shutil
import ssl
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.parse
import uuid
import zipfile
from zoneinfo import ZoneInfo
import zlib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = Path(
    "docs/treasury-tic-release-vintage-source-axis-decision-2026-07-24.md"
)
SCRIPT_PATH = Path("training/audit_treasury_tic_release_vintage_source.py")
TEST_PATH = Path("tests/test_audit_treasury_tic_release_vintage_source.py")
BOUNDARY_SHA256 = (
    "ed7008daff527d0bdfe13dd03686ca7113e39dfff8ab4a7056100cbd36799419"
)
PROTOCOL_VERSION = "TICRV-v1-source-audit-2026-07-24"

ARCHIVE_INDEX_URL = (
    "https://home.treasury.gov/archives-of-tic-monthly-data-releases"
)
LIVE_METADATA_URL = (
    "https://home.treasury.gov/data/treasury-international-capital-tic-system/"
    "tic-press-releases-by-topic"
)
START_ARCHIVE_PREFIX = (
    "https://www.treasury.gov/resource-center/data-chart-center/tic/Documents/"
)
FINAL_ARCHIVE_PREFIX = (
    "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/"
    "Documents/"
)
ARCHIVE_FILENAME_RE = re.compile(r"ticrel_(\d{8})\.zip\Z", re.ASCII)
LABEL_RE = re.compile(r"\d{2}/\d{2}/\d{4}\Z", re.ASCII)
PUBLIC_RELEASE_RE = re.compile(
    r"\(released (\d{2}-\d{2}-\d{4})\)", re.ASCII
)

EXPECTED_RELEASE_ROWS: tuple[tuple[str, str, str | None], ...] = (
    ("01/18/2022", "ticrel_20220118.zip", None),
    ("02/15/2022", "ticrel_20220222.zip", None),
    ("03/15/2022", "ticrel_20220315.zip", None),
    ("04/15/2022", "ticrel_20220415.zip", None),
    ("05/16/2022", "ticrel_20220516.zip", None),
    ("06/15/2022", "ticrel_20220615.zip", None),
    ("07/18/2022", "ticrel_20220718.zip", None),
    ("08/15/2022", "ticrel_20220815.zip", None),
    ("09/16/2022", "ticrel_20220916.zip", None),
    ("10/18/2022", "ticrel_20221018.zip", None),
    ("11/16/2022", "ticrel_20221116.zip", None),
    ("12/15/2022", "ticrel_20221215.zip", None),
    ("01/18/2023", "ticrel_20230118.zip", None),
    ("02/15/2023", "ticrel_20230215.zip", None),
    ("03/15/2023", "ticrel_20230315.zip", None),
    ("04/17/2023", "ticrel_20230417.zip", None),
    ("05/15/2023", "ticrel_20230515.zip", None),
    ("06/15/2023", "ticrel_20230615.zip", None),
    ("07/18/2023", "ticrel_20230718.zip", None),
    ("08/15/2023", "ticrel_20230815.zip", None),
    ("09/18/2023", "ticrel_20230918.zip", None),
    ("10/18/2023", "ticrel_20231018.zip", None),
    ("11/16/2023", "ticrel_20231116.zip", None),
    ("12/19/2023", "ticrel_20231219.zip", None),
    ("01/19/2024", "ticrel_20240119.zip", None),
    ("02/15/2024", "ticrel_20240215.zip", None),
    ("03/19/2024", "ticrel_20240319.zip", None),
    ("04/17/2024", "ticrel_20240417.zip", None),
    ("05/15/2024", "ticrel_20240515.zip", None),
    ("06/18/2024", "ticrel_20240618.zip", None),
    ("07/18/2024", "ticrel_20240718.zip", None),
    ("08/15/2024", "ticrel_20240815.zip", None),
    ("09/18/2024", "ticrel_20240918.zip", None),
    ("10/17/2024", "ticrel_20241017.zip", None),
    ("11/18/2024", "ticrel_20241118.zip", None),
    ("12/19/2024", "ticrel_20241219.zip", None),
    ("01/17/2025", "ticrel_20250117.zip", None),
    ("02/18/2025", "ticrel_20250218.zip", None),
    ("03/19/2025", "ticrel_20250319.zip", None),
    ("04/16/2025", "ticrel_20250416.zip", None),
    ("05/16/2025", "ticrel_20250516.zip", None),
    ("06/18/2025", "ticrel_20250618.zip", None),
    ("07/17/2025", "ticrel_20250717.zip", None),
    ("08/15/2025", "ticrel_20250815.zip", None),
    ("09/18/2025", "ticrel_20250918.zip", None),
    ("10/17/2025", "ticrel_20251017.zip", "11-18-2025"),
    ("11/18/2025", "ticrel_20251118.zip", None),
    ("12/18/2025", "ticrel_20251218.zip", None),
    ("01/15/2026", "ticrel_20260115.zip", None),
    ("02/18/2026", "ticrel_20260218.zip", None),
    ("03/18/2026", "ticrel_20260318.zip", None),
    ("04/15/2026", "ticrel_20260415.zip", None),
    ("05/18/2026", "ticrel_20260518.zip", None),
    ("06/18/2026", "ticrel_20260618.zip", None),
)

EXPECTED_RELEASES = 54
EXPECTED_REQUESTS = 109
INDEX_BODY_CAP = 4 * 1024 * 1024
REDIRECT_BODY_CAP = 1 * 1024 * 1024
ARCHIVE_BODY_CAP = 32 * 1024 * 1024
MAX_MEMBERS = 512
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100
DISK_USED_LIMIT = 300 * 1024**3
DISK_FREE_FLOOR = 2 * 1024**3
REQUEST_INTERVAL_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30.0
CONTACT_USER_AGENT = (
    "rllm-ticrv-source-audit/1.0 "
    "(contact: https://github.com/pakchu/rllm)"
)
ALLOWED_REDIRECT_STATUSES = frozenset({301, 302, 307, 308})
ALLOWED_ARCHIVE_CONTENT_TYPES = frozenset(
    {"application/zip", "application/octet-stream"}
)
ALLOWED_MEMBER_SUFFIXES = frozenset({".txt", ".csv", ".htm", ".html", ".pdf"})
MACHINE_MEMBER_SUFFIXES = frozenset({".txt", ".csv", ".htm", ".html"})
ALLOWED_COMPRESSION_METHODS = frozenset({0, 8})
ALLOWED_GENERAL_FLAGS = frozenset({0, 0x0800})
RECOGNIZED_TIMESTAMP_EXTRA_IDS = frozenset({0x5455, 0x000A, 0x5855})
NEW_YORK = ZoneInfo("America/New_York")

DEFAULT_SENTINEL = Path(
    "results/.treasury_tic_release_vintage_source_2026-07-24.started"
)
DEFAULT_MANIFEST = Path(
    "results/.treasury_tic_release_vintage_source_2026-07-24.manifest.ndjson"
)
DEFAULT_RAW_DIR = Path(
    "results/.treasury_tic_release_vintage_source_2026-07-24.raw"
)
DEFAULT_REPORT = Path(
    "results/treasury_tic_release_vintage_source_2026-07-24.json"
)


class TicrvError(RuntimeError):
    """Base class for frozen source-audit failures."""


class ProtocolError(TicrvError):
    """The committed one-shot protocol precondition failed."""


class DiskGuardError(ProtocolError):
    """The repository filesystem is outside the frozen storage boundary."""


class TransportError(TicrvError):
    """A direct HTTPS response violated the frozen transport contract."""


class IndexContractError(TicrvError):
    """The official release index no longer matches the frozen identities."""


class ZipContractError(TicrvError):
    """A ZIP container violated the frozen byte-level contract."""


class SupportError(TicrvError):
    """The frozen release panel lacks source-only support."""


class PublicationError(TicrvError):
    """A one-shot artifact could not be published atomically."""


@dataclass(frozen=True, slots=True)
class ReleaseIdentity:
    index_label_date: date
    filename_date: date
    explicit_public_release_date: date | None
    release_identity_date: date
    starting_url: str
    final_url: str

    @property
    def available_at_utc(self) -> datetime:
        local_midnight = datetime.combine(
            self.release_identity_date + timedelta(days=1),
            datetime_time.min,
            tzinfo=NEW_YORK,
        )
        return local_midnight.astimezone(timezone.utc)

    def private_dict(self) -> dict[str, Any]:
        return {
            "release_identity_date": self.release_identity_date.isoformat(),
            "index_label_date": self.index_label_date.isoformat(),
            "filename_date": self.filename_date.isoformat(),
            "explicit_public_release_date": (
                self.explicit_public_release_date.isoformat()
                if self.explicit_public_release_date is not None
                else None
            ),
            "starting_url": self.starting_url,
            "final_url": self.final_url,
            "available_at_utc": canonical_utc(self.available_at_utc),
        }


@dataclass(frozen=True, slots=True)
class IdentityAudit:
    identities: tuple[ReleaseIdentity, ...]
    strictly_later_transitions: int
    same_day_transitions: int


@dataclass(frozen=True, slots=True)
class HttpPayload:
    requested_url: str
    status: int
    headers: tuple[tuple[str, str], ...]
    raw: bytes
    request_started_at_utc: datetime
    response_completed_at_utc: datetime


@dataclass(frozen=True, slots=True)
class RawZipEntry:
    name: str
    raw_name: bytes
    suffix: str
    flags: int
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    data_offset: int
    data_end: int
    timestamp_date: date
    external_attributes: int


@dataclass(frozen=True, slots=True)
class MemberAudit:
    basename: str
    suffix: str
    sha256: str
    uncompressed_bytes: int
    timestamp_lag_days: int


@dataclass(frozen=True, slots=True)
class ArchiveAudit:
    archive_sha256: str
    archive_bytes: int
    members: tuple[MemberAudit, ...]
    suffix_counts: tuple[tuple[str, int], ...]
    declared_uncompressed_bytes: int
    realized_uncompressed_bytes: int
    machine_family_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class AuditPaths:
    sentinel: Path
    manifest: Path
    raw_dir: Path
    report: Path


PRODUCTION_PATHS = AuditPaths(
    sentinel=DEFAULT_SENTINEL,
    manifest=DEFAULT_MANIFEST,
    raw_dir=DEFAULT_RAW_DIR,
    report=DEFAULT_REPORT,
)


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return rendered + (b"\n" if newline else b"")


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _deduplicating_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("JSON object contains a duplicate key")
        result[key] = value
    return result


def parse_json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_deduplicating_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("JSON is not canonical ASCII object data") from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSON root is not an object")
    return value


def _parse_label(value: str) -> date:
    if LABEL_RE.fullmatch(value) is None:
        raise IndexContractError("release label is not exact MM/DD/YYYY")
    try:
        return datetime.strptime(value, "%m/%d/%Y").date()
    except ValueError as exc:
        raise IndexContractError("release label is not a calendar date") from exc


def _parse_filename_date(filename: str) -> date:
    match = ARCHIVE_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise IndexContractError("archive filename is not canonical")
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError as exc:
        raise IndexContractError("archive filename date is invalid") from exc


def _parse_public_release(value: str) -> date:
    try:
        return datetime.strptime(value, "%m-%d-%Y").date()
    except ValueError as exc:
        raise IndexContractError("explicit public release date is invalid") from exc


def _strict_url(
    value: str,
    *,
    exact_host: str,
    exact_path: str | None = None,
) -> urllib.parse.SplitResult:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TransportError("URL is not ASCII") from exc
    if (
        not value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        or "%" in value
        or "\\" in value
    ):
        raise TransportError("URL has forbidden encoding or control bytes")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != exact_host
        or parsed.hostname != exact_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TransportError("URL authority is outside the frozen contract")
    if exact_path is not None and parsed.path != exact_path:
        raise TransportError("URL path changed")
    if (
        not parsed.path.startswith("/")
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise TransportError("URL path is not canonical")
    return parsed


def _starting_url(filename: str) -> str:
    value = START_ARCHIVE_PREFIX + filename
    _strict_url(
        value,
        exact_host="www.treasury.gov",
        exact_path=urllib.parse.urlsplit(START_ARCHIVE_PREFIX).path + filename,
    )
    return value


def _final_url(filename: str) -> str:
    value = FINAL_ARCHIVE_PREFIX + filename
    _strict_url(
        value,
        exact_host="ticdata.treasury.gov",
        exact_path=urllib.parse.urlsplit(FINAL_ARCHIVE_PREFIX).path + filename,
    )
    return value


@dataclass(slots=True)
class _PendingIndexRecord:
    href: str
    label: str
    description_parts: list[str]


class _ArchiveIndexParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._anchor_href: str | None = None
        self._anchor_parts: list[str] = []
        self._pending: _PendingIndexRecord | None = None
        self.records: list[tuple[str, str, str]] = []

    def _finish_pending(self) -> None:
        if self._pending is None:
            return
        self.records.append(
            (
                self._pending.href,
                self._pending.label,
                "".join(self._pending.description_parts).strip(),
            )
        )
        self._pending = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        normalized = tag.lower()
        if normalized == "a":
            self._finish_pending()
            values = dict(attrs)
            self._anchor_href = values.get("href")
            self._anchor_parts = []
        elif normalized == "br":
            self._finish_pending()

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._anchor_href is None:
            return
        label = "".join(self._anchor_parts).strip()
        if LABEL_RE.fullmatch(label) is not None:
            parsed = _parse_label(label)
            if date(2022, 1, 1) <= parsed <= date(2026, 6, 30):
                self._pending = _PendingIndexRecord(
                    href=self._anchor_href,
                    label=label,
                    description_parts=[],
                )
        self._anchor_href = None
        self._anchor_parts = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_parts.append(data)
        elif self._pending is not None:
            self._pending.description_parts.append(data)

    def close(self) -> None:
        super().close()
        self._finish_pending()


def parse_archive_index(raw: bytes) -> IdentityAudit:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IndexContractError("archive index is not UTF-8") from exc
    if "\x00" in text:
        raise IndexContractError("archive index contains NUL")

    parser = _ArchiveIndexParser()
    parser.feed(text)
    parser.close()

    by_label: dict[str, tuple[str, str]] = {}
    for href, label, description in parser.records:
        if label in by_label:
            raise IndexContractError("archive index duplicates a frozen label")
        by_label[label] = (href, description)

    expected_labels = {row[0] for row in EXPECTED_RELEASE_ROWS}
    if set(by_label) != expected_labels:
        raise IndexContractError("archive index label set changed")

    identities: list[ReleaseIdentity] = []
    mismatches: list[tuple[date, date]] = []
    delayed: list[tuple[date, date]] = []
    for label, filename, expected_public_literal in EXPECTED_RELEASE_ROWS:
        href, description = by_label[label]
        expected_url = _starting_url(filename)
        if href != expected_url:
            raise IndexContractError("archive index URL changed")

        markers = PUBLIC_RELEASE_RE.findall(description)
        released_tokens = re.findall(
            r"\([^)]*release[^)]*\)",
            description,
            re.ASCII | re.IGNORECASE,
        )
        if len(markers) != len(released_tokens):
            raise IndexContractError("archive release annotation format changed")
        if expected_public_literal is None:
            if markers:
                raise IndexContractError("unexpected delayed-release annotation")
            explicit_public_release_date = None
        else:
            if markers != [expected_public_literal]:
                raise IndexContractError("delayed-release annotation changed")
            explicit_public_release_date = _parse_public_release(markers[0])

        index_label_date = _parse_label(label)
        filename_date = _parse_filename_date(filename)
        if index_label_date != filename_date:
            mismatches.append((index_label_date, filename_date))
        if explicit_public_release_date is not None:
            delayed.append((index_label_date, explicit_public_release_date))
        release_identity_date = max(
            value
            for value in (
                index_label_date,
                filename_date,
                explicit_public_release_date,
            )
            if value is not None
        )
        identities.append(
            ReleaseIdentity(
                index_label_date=index_label_date,
                filename_date=filename_date,
                explicit_public_release_date=explicit_public_release_date,
                release_identity_date=release_identity_date,
                starting_url=expected_url,
                final_url=_final_url(filename),
            )
        )

    if mismatches != [(date(2022, 2, 15), date(2022, 2, 22))]:
        raise IndexContractError("label/filename exception changed")
    if delayed != [(date(2025, 10, 17), date(2025, 11, 18))]:
        raise IndexContractError("delayed-public-release exception changed")

    ordered = tuple(
        sorted(
            identities,
            key=lambda row: (
                row.release_identity_date,
                row.index_label_date,
                row.filename_date,
                row.explicit_public_release_date or date.min,
                row.starting_url,
            ),
        )
    )
    if len(ordered) != EXPECTED_RELEASES:
        raise IndexContractError("archive identity count changed")
    identity_rows = [row.private_dict() for row in ordered]
    if len({canonical_json_bytes(row, newline=False) for row in identity_rows}) != len(
        ordered
    ):
        raise IndexContractError("archive identity is duplicated")

    strictly_later = 0
    same_day = 0
    for previous, current in zip(ordered, ordered[1:]):
        if current.available_at_utc > previous.available_at_utc:
            strictly_later += 1
        elif current.available_at_utc == previous.available_at_utc:
            same_day += 1
        else:
            raise IndexContractError("availability clock moved backward")
    if strictly_later != 52 or same_day != 1:
        raise IndexContractError("availability transition contract changed")
    if (
        ordered[-1].available_at_utc - ordered[0].available_at_utc
        <= timedelta(days=4 * 365)
    ):
        raise IndexContractError("causal source span is not longer than four years")
    return IdentityAudit(
        identities=ordered,
        strictly_later_transitions=strictly_later,
        same_day_transitions=same_day,
    )


def _header_values(
    headers: Sequence[tuple[str, str]], name: str
) -> tuple[str, ...]:
    lowered = name.lower()
    return tuple(value for key, value in headers if key.lower() == lowered)


def _single_header(
    headers: Sequence[tuple[str, str]],
    name: str,
    *,
    required: bool,
) -> str | None:
    values = _header_values(headers, name)
    if len(values) > 1 or (required and len(values) != 1):
        raise TransportError(f"{name} header cardinality changed")
    return values[0] if values else None


def _media_type(value: str) -> str:
    rendered = value.split(";", 1)[0].strip().lower()
    if not rendered or any(character.isspace() for character in rendered):
        raise TransportError("Content-Type is malformed")
    return rendered


def validate_http_payload(
    payload: HttpPayload,
    *,
    expected_statuses: frozenset[int],
    body_cap: int,
    allowed_content_types: frozenset[str] | None,
    location_required: bool = False,
) -> None:
    if type(payload.status) is not int or payload.status not in expected_statuses:
        raise TransportError("HTTP status is outside the frozen contract")
    if len(payload.raw) > body_cap:
        raise TransportError("HTTP body exceeds the frozen cap")
    if payload.response_completed_at_utc < payload.request_started_at_utc:
        raise TransportError("HTTP receipt clock moved backward")

    content_encoding = _single_header(
        payload.headers, "Content-Encoding", required=False
    )
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        raise TransportError("HTTP Content-Encoding is not identity")
    content_range = _single_header(payload.headers, "Content-Range", required=False)
    if content_range is not None:
        raise TransportError("partial HTTP response is forbidden")

    content_length = _single_header(
        payload.headers, "Content-Length", required=False
    )
    transfer_encoding = _single_header(
        payload.headers, "Transfer-Encoding", required=False
    )
    if content_length is not None:
        if re.fullmatch(r"0|[1-9]\d*", content_length) is None:
            raise TransportError("HTTP Content-Length is malformed")
        if int(content_length) != len(payload.raw):
            raise TransportError("HTTP Content-Length does not match the body")
    if transfer_encoding is not None:
        if transfer_encoding.strip().lower() != "chunked" or content_length is not None:
            raise TransportError("HTTP Transfer-Encoding is ambiguous")

    if allowed_content_types is not None:
        content_type = _single_header(
            payload.headers, "Content-Type", required=True
        )
        assert content_type is not None
        if _media_type(content_type) not in allowed_content_types:
            raise TransportError("HTTP Content-Type changed")
    _single_header(payload.headers, "Location", required=location_required)


class _RateLimiter:
    def __init__(
        self,
        *,
        interval_seconds: float = REQUEST_INTERVAL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            delay = self._interval_seconds - (now - self._last_request)
            if delay > 0:
                self._sleep(delay)
                now = self._monotonic()
        self._last_request = now


def _direct_request_plan(url: str) -> tuple[str, str, dict[str, str]]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc not in {
        "home.treasury.gov",
        "www.treasury.gov",
        "ticdata.treasury.gov",
    }:
        raise TransportError("request host is outside the Treasury allowlist")
    _strict_url(url, exact_host=parsed.netloc, exact_path=parsed.path)
    return (
        parsed.netloc,
        parsed.path,
        {
            "User-Agent": CONTACT_USER_AGENT,
            "Accept-Encoding": "identity",
            "Accept": "*/*",
            "Connection": "close",
        },
    )


class _OfflineFixtureTransport:
    """Exact in-memory response ledger with no callable I/O capability."""

    def __init__(self, responses: dict[str, HttpPayload]) -> None:
        if type(responses) is not dict or len(responses) != EXPECTED_REQUESTS:
            raise ProtocolError("fixture response ledger must contain exactly 109 rows")
        copied: dict[str, HttpPayload] = {}
        for url, payload in responses.items():
            if type(url) is not str or type(payload) is not HttpPayload:
                raise ProtocolError("fixture response ledger has an invalid row")
            if (
                type(payload.requested_url) is not str
                or type(payload.status) is not int
                or type(payload.headers) is not tuple
                or any(
                    type(row) is not tuple
                    or len(row) != 2
                    or type(row[0]) is not str
                    or type(row[1]) is not str
                    for row in payload.headers
                )
                or type(payload.raw) is not bytes
                or type(payload.request_started_at_utc) is not datetime
                or type(payload.response_completed_at_utc) is not datetime
            ):
                raise ProtocolError("fixture response row is not inert")
            if payload.requested_url != url:
                raise ProtocolError("fixture response identity changed")
            copied[url] = payload
        self._responses = copied
        self._used: set[str] = set()

    def get(self, url: str, body_cap: int) -> HttpPayload:
        del body_cap
        if url in self._used:
            raise ProtocolError("fixture response was requested twice")
        try:
            payload = self._responses[url]
        except KeyError as exc:
            raise ProtocolError("fixture response is not preloaded") from exc
        self._used.add(url)
        return payload


_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL = struct.Struct("<4s6H3I5H2I")
_LOCAL = struct.Struct("<4s5H3I2H")
_EXTRA_HEADER = struct.Struct("<HH")
_NTFS_TAG = struct.Struct("<HH")
_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_NESTED_ARCHIVE_MAGICS = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
    b"BZh",
    b"\xfd7zXZ\x00",
)


def _parse_extra_fields(raw: bytes) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    cursor = 0
    while cursor < len(raw):
        if len(raw) - cursor < _EXTRA_HEADER.size:
            raise ZipContractError("ZIP extra field is truncated")
        field_id, field_length = _EXTRA_HEADER.unpack_from(raw, cursor)
        cursor += _EXTRA_HEADER.size
        end = cursor + field_length
        if end > len(raw):
            raise ZipContractError("ZIP extra field length is invalid")
        if field_id in fields:
            raise ZipContractError("ZIP extra field is duplicated")
        fields[field_id] = raw[cursor:end]
        cursor = end
    if 0x0001 in fields:
        raise ZipContractError("ZIP64 extra field is forbidden")
    return fields


def _dos_datetime(raw_date: int, raw_time: int) -> datetime:
    year = 1980 + ((raw_date >> 9) & 0x7F)
    month = (raw_date >> 5) & 0x0F
    day = raw_date & 0x1F
    hour = (raw_time >> 11) & 0x1F
    minute = (raw_time >> 5) & 0x3F
    second = (raw_time & 0x1F) * 2
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ZipContractError("DOS timestamp is invalid") from exc


def _unix_timestamp_date(raw_seconds: int) -> date:
    if raw_seconds <= 0:
        raise ZipContractError("Unix timestamp is zero or negative")
    try:
        value = datetime.fromtimestamp(raw_seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ZipContractError("Unix timestamp is out of range") from exc
    if value.year < 1980:
        raise ZipContractError("Unix timestamp predates 1980")
    return value.date()


def _ntfs_filetime_date(raw_filetime: int) -> date:
    if raw_filetime <= 116444736000000000:
        raise ZipContractError("NTFS timestamp predates Unix epoch")
    seconds, remainder = divmod(
        raw_filetime - 116444736000000000,
        10_000_000,
    )
    try:
        value = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
            microsecond=remainder // 10
        )
    except (OverflowError, OSError, ValueError) as exc:
        raise ZipContractError("NTFS timestamp is out of range") from exc
    if value.year < 1980:
        raise ZipContractError("NTFS timestamp predates 1980")
    return value.date()


def _timestamp_dates_from_fields(
    fields: Mapping[int, bytes],
    *,
    central_directory: bool,
) -> tuple[date, ...]:
    dates: list[date] = []
    if 0x5455 in fields:
        payload = fields[0x5455]
        if not payload:
            raise ZipContractError("extended timestamp field is empty")
        flags = payload[0]
        if flags & ~0x07 or not flags & 0x01:
            raise ZipContractError("extended timestamp flags are unsupported")
        expected_length = 1 + 4 * int(flags.bit_count())
        permitted_lengths = (
            {5, expected_length}
            if central_directory
            else {expected_length}
        )
        if len(payload) not in permitted_lengths:
            raise ZipContractError("extended timestamp length is ambiguous")
        modification_time = struct.unpack_from("<I", payload, 1)[0]
        dates.append(_unix_timestamp_date(modification_time))

    if 0x5855 in fields:
        payload = fields[0x5855]
        if len(payload) not in {8, 12}:
            raise ZipContractError("Info-ZIP Unix timestamp length is invalid")
        modification_time = struct.unpack_from("<I", payload, 4)[0]
        dates.append(_unix_timestamp_date(modification_time))

    if 0x000A in fields:
        payload = fields[0x000A]
        if len(payload) < 8 or payload[:4] != b"\x00\x00\x00\x00":
            raise ZipContractError("NTFS timestamp header is malformed")
        cursor = 4
        seen_timestamp = False
        while cursor < len(payload):
            if len(payload) - cursor < _NTFS_TAG.size:
                raise ZipContractError("NTFS timestamp tag is truncated")
            tag, length = _NTFS_TAG.unpack_from(payload, cursor)
            cursor += _NTFS_TAG.size
            end = cursor + length
            if end > len(payload):
                raise ZipContractError("NTFS timestamp tag length is invalid")
            if tag != 0x0001 or seen_timestamp or length != 24:
                raise ZipContractError("NTFS timestamp tag is unsupported")
            modification_time = struct.unpack_from("<Q", payload, cursor)[0]
            dates.append(_ntfs_filetime_date(modification_time))
            seen_timestamp = True
            cursor = end
        if not seen_timestamp:
            raise ZipContractError("NTFS modification timestamp is missing")
    return tuple(dates)


def _validate_unicode_path(
    fields: Mapping[int, bytes],
    *,
    raw_name: bytes,
    decoded_name: str,
) -> None:
    payload = fields.get(0x7075)
    if payload is None:
        return
    if len(payload) < 5 or payload[0] != 1:
        raise ZipContractError("Unicode path extra field is malformed")
    expected_crc = struct.unpack_from("<I", payload, 1)[0]
    if expected_crc != zlib.crc32(raw_name) & 0xFFFFFFFF:
        raise ZipContractError("Unicode path name CRC changed")
    try:
        unicode_name = payload[5:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ZipContractError("Unicode path is not UTF-8") from exc
    if unicode_name != decoded_name:
        raise ZipContractError("Unicode path disagrees with raw filename")


def _validate_raw_member_name(
    raw_name: bytes,
    *,
    central_fields: Mapping[int, bytes],
    local_fields: Mapping[int, bytes],
) -> tuple[str, str]:
    if (
        not raw_name
        or any(value < 0x20 or value > 0x7E for value in raw_name)
        or b"/" in raw_name
        or b"\\" in raw_name
        or b":" in raw_name
    ):
        raise ZipContractError("ZIP member filename is not a safe ASCII basename")
    name = raw_name.decode("ascii")
    if name in {".", ".."}:
        raise ZipContractError("ZIP member filename is a dot path")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        posix.name != name
        or len(posix.parts) != 1
        or windows.name != name
        or len(windows.parts) != 1
        or windows.drive
        or windows.root
    ):
        raise ZipContractError("ZIP member filename has path semantics")
    _validate_unicode_path(central_fields, raw_name=raw_name, decoded_name=name)
    _validate_unicode_path(local_fields, raw_name=raw_name, decoded_name=name)
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in ALLOWED_MEMBER_SUFFIXES:
        raise ZipContractError("ZIP member suffix is unsupported")
    return name, suffix


def _validate_member_type(entry: RawZipEntry) -> None:
    dos_attributes = entry.external_attributes & 0xFF
    if dos_attributes & 0x58:
        raise ZipContractError(
            "ZIP member is a directory, volume label, or device"
        )
    mode = (entry.external_attributes >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG}:
        raise ZipContractError("ZIP member is not a regular file")


def parse_zip_entries(raw: bytes, *, release_identity_date: date) -> tuple[RawZipEntry, ...]:
    if len(raw) < _EOCD.size or raw[-_EOCD.size : -_EOCD.size + 4] != _EOCD_SIGNATURE:
        raise ZipContractError("ZIP EOCD is not exact or archive comment is present")
    eocd_offset = len(raw) - _EOCD.size
    if (
        eocd_offset >= 20
        and raw[eocd_offset - 20 : eocd_offset - 16]
        == _ZIP64_LOCATOR_SIGNATURE
    ):
        raise ZipContractError("ZIP64 structure is forbidden")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = _EOCD.unpack_from(raw, len(raw) - _EOCD.size)
    if signature != _EOCD_SIGNATURE or comment_length != 0:
        raise ZipContractError("ZIP EOCD changed")
    if disk_number != 0 or central_disk != 0 or entries_on_disk != total_entries:
        raise ZipContractError("multi-disk ZIP is forbidden")
    if total_entries < 1 or total_entries > MAX_MEMBERS:
        raise ZipContractError("ZIP member count is outside the frozen bounds")
    if central_offset in {0xFFFFFFFF} or central_size in {0xFFFFFFFF}:
        raise ZipContractError("ZIP64 sentinel is forbidden")
    central_end = central_offset + central_size
    if central_end != len(raw) - _EOCD.size:
        raise ZipContractError("ZIP central directory does not end at EOCD")
    if central_offset <= 0 or raw[:4] != _LOCAL_SIGNATURE:
        raise ZipContractError("ZIP has bytes before the first local header")

    cursor = central_offset
    entries: list[RawZipEntry] = []
    exact_names: set[bytes] = set()
    lowercase_names: set[str] = set()
    local_offsets: set[int] = set()
    total_declared = 0
    total_compressed = 0

    for _ in range(total_entries):
        if cursor + _CENTRAL.size > central_end:
            raise ZipContractError("central directory entry is truncated")
        values = _CENTRAL.unpack_from(raw, cursor)
        (
            central_signature,
            _version_made,
            version_needed,
            flags,
            compression_method,
            mod_time,
            mod_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            member_comment_length,
            member_disk,
            _internal_attributes,
            external_attributes,
            local_header_offset,
        ) = values
        if central_signature != _CENTRAL_SIGNATURE:
            raise ZipContractError("central directory signature changed")
        if (
            name_length == 0
            or member_comment_length != 0
            or member_disk != 0
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            raise ZipContractError("central directory uses unsupported structure")
        record_end = (
            cursor
            + _CENTRAL.size
            + name_length
            + extra_length
            + member_comment_length
        )
        if record_end > central_end:
            raise ZipContractError("central directory variable fields are truncated")
        raw_name = raw[cursor + _CENTRAL.size : cursor + _CENTRAL.size + name_length]
        central_extra_start = cursor + _CENTRAL.size + name_length
        central_extra = raw[central_extra_start : central_extra_start + extra_length]
        central_fields = _parse_extra_fields(central_extra)

        if flags not in ALLOWED_GENERAL_FLAGS:
            raise ZipContractError("ZIP general-purpose flag is unsupported")
        if compression_method not in ALLOWED_COMPRESSION_METHODS:
            raise ZipContractError("ZIP compression method is unsupported")
        if version_needed > 20:
            raise ZipContractError("ZIP required-version is unsupported")
        if local_header_offset in local_offsets:
            raise ZipContractError("ZIP local header is duplicated")
        if raw_name in exact_names:
            raise ZipContractError("ZIP member filename is duplicated")
        local_offsets.add(local_header_offset)
        exact_names.add(raw_name)

        if local_header_offset + _LOCAL.size > central_offset:
            raise ZipContractError("local header overlaps the central directory")
        local_values = _LOCAL.unpack_from(raw, local_header_offset)
        (
            local_signature,
            local_version_needed,
            local_flags,
            local_compression_method,
            local_mod_time,
            local_mod_date,
            local_crc32,
            local_compressed_size,
            local_uncompressed_size,
            local_name_length,
            local_extra_length,
        ) = local_values
        if local_signature != _LOCAL_SIGNATURE:
            raise ZipContractError("local header signature changed")
        local_variable_end = (
            local_header_offset
            + _LOCAL.size
            + local_name_length
            + local_extra_length
        )
        if local_variable_end > central_offset:
            raise ZipContractError("local header variable fields are truncated")
        local_name_start = local_header_offset + _LOCAL.size
        local_raw_name = raw[local_name_start : local_name_start + local_name_length]
        local_extra_start = local_name_start + local_name_length
        local_extra = raw[local_extra_start : local_extra_start + local_extra_length]
        local_fields = _parse_extra_fields(local_extra)
        if (
            version_needed != local_version_needed
            or flags != local_flags
            or compression_method != local_compression_method
            or mod_time != local_mod_time
            or mod_date != local_mod_date
            or crc32 != local_crc32
            or compressed_size != local_compressed_size
            or uncompressed_size != local_uncompressed_size
            or raw_name != local_raw_name
        ):
            raise ZipContractError("central and local ZIP headers disagree")

        name, suffix = _validate_raw_member_name(
            raw_name,
            central_fields=central_fields,
            local_fields=local_fields,
        )
        lowered = name.lower()
        if lowered in lowercase_names:
            raise ZipContractError("ZIP member ASCII-lowercase name is duplicated")
        lowercase_names.add(lowered)

        dos_timestamp = _dos_datetime(mod_date, mod_time)
        represented_dates = (
            dos_timestamp.date(),
            *_timestamp_dates_from_fields(
                central_fields,
                central_directory=True,
            ),
            *_timestamp_dates_from_fields(
                local_fields,
                central_directory=False,
            ),
        )
        if len(set(represented_dates)) != 1:
            raise ZipContractError("ZIP timestamp representations disagree")
        timestamp_date = represented_dates[0]
        if timestamp_date > release_identity_date + timedelta(days=7):
            raise ZipContractError("ZIP timestamp exceeds release-plus-seven days")

        data_offset = local_variable_end
        data_end = data_offset + compressed_size
        if data_end > central_offset:
            raise ZipContractError("compressed member overlaps central directory")
        if uncompressed_size > MAX_MEMBER_BYTES:
            raise ZipContractError("ZIP member exceeds the uncompressed size cap")
        if compressed_size == 0:
            if uncompressed_size != 0:
                raise ZipContractError("ZIP member compression ratio is infinite")
        elif uncompressed_size > MAX_COMPRESSION_RATIO * compressed_size:
            raise ZipContractError("ZIP member compression ratio exceeds the cap")
        total_declared += uncompressed_size
        total_compressed += compressed_size
        if total_declared > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ZipContractError("ZIP declared uncompressed total exceeds the cap")

        entry = RawZipEntry(
            name=name,
            raw_name=raw_name,
            suffix=suffix,
            flags=flags,
            compression_method=compression_method,
            crc32=crc32,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
            data_offset=data_offset,
            data_end=data_end,
            timestamp_date=timestamp_date,
            external_attributes=external_attributes,
        )
        _validate_member_type(entry)
        entries.append(entry)
        cursor = record_end

    if cursor != central_end:
        raise ZipContractError("central directory has unaccounted bytes")
    if total_compressed == 0:
        if total_declared != 0:
            raise ZipContractError("aggregate compression ratio is infinite")
    elif total_declared > MAX_COMPRESSION_RATIO * total_compressed:
        raise ZipContractError("aggregate compression ratio exceeds the cap")

    by_local_offset = sorted(entries, key=lambda row: row.local_header_offset)
    expected_offset = 0
    for entry in by_local_offset:
        if entry.local_header_offset != expected_offset:
            raise ZipContractError("ZIP local records have overlap or unaccounted gaps")
        expected_offset = entry.data_end
    if expected_offset != central_offset:
        raise ZipContractError("ZIP local records do not reach the central directory")
    return tuple(entries)


def _has_nested_archive_magic(prefix: bytes) -> bool:
    return any(prefix.startswith(magic) for magic in _NESTED_ARCHIVE_MAGICS)


def _audit_member_stream(
    raw: bytes,
    entry: RawZipEntry,
) -> tuple[str, int, bytes]:
    compressed = raw[entry.data_offset : entry.data_end]
    digest = hashlib.sha256()
    crc32 = 0
    realized = 0
    prefix = b""

    def consume(chunk: bytes) -> None:
        nonlocal crc32, prefix, realized
        if not chunk:
            return
        if len(prefix) < 8:
            prefix += chunk[: 8 - len(prefix)]
        digest.update(chunk)
        crc32 = zlib.crc32(chunk, crc32)
        realized += len(chunk)
        if realized > MAX_MEMBER_BYTES:
            raise ZipContractError("realized ZIP member exceeds the size cap")

    if entry.compression_method == 0:
        consume(compressed)
    else:
        decompressor = zlib.decompressobj(-15)
        try:
            for offset in range(0, len(compressed), 1 << 16):
                pending = compressed[offset : offset + (1 << 16)]
                while pending:
                    output_limit = min(
                        1 << 20,
                        MAX_MEMBER_BYTES + 1 - realized,
                    )
                    if output_limit <= 0:
                        raise ZipContractError(
                            "realized ZIP member exceeds the size cap"
                        )
                    consume(decompressor.decompress(pending, output_limit))
                    pending = decompressor.unconsumed_tail
            consume(
                decompressor.flush(
                    max(1, MAX_MEMBER_BYTES + 1 - realized)
                )
            )
        except zlib.error as exc:
            raise ZipContractError("raw DEFLATE stream is malformed") from exc
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise ZipContractError("raw DEFLATE stream has trailing or missing bytes")
    if realized != entry.uncompressed_size:
        raise ZipContractError("realized ZIP member length changed")
    if crc32 & 0xFFFFFFFF != entry.crc32:
        raise ZipContractError("ZIP member CRC changed")
    return digest.hexdigest(), realized, prefix


def audit_zip_archive(
    raw: bytes,
    *,
    release_identity_date: date,
) -> ArchiveAudit:
    entries = parse_zip_entries(raw, release_identity_date=release_identity_date)
    entries_by_name = {entry.name: entry for entry in entries}
    members: list[MemberAudit] = []
    suffix_counts: Counter[str] = Counter()
    realized_total = 0
    machine_family_keys: set[str] = set()

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), mode="r", allowZip64=False)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ZipContractError("stdlib ZIP parser rejected the archive") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) != len(entries) or {info.filename for info in infos} != set(
            entries_by_name
        ):
            raise ZipContractError("stdlib ZIP inventory disagrees with raw parser")
        for info in infos:
            entry = entries_by_name[info.filename]
            if (
                info.flag_bits != entry.flags
                or info.compress_type != entry.compression_method
                or info.CRC != entry.crc32
                or info.compress_size != entry.compressed_size
                or info.file_size != entry.uncompressed_size
                or info.header_offset != entry.local_header_offset
            ):
                raise ZipContractError("stdlib ZIP metadata disagrees with raw parser")
            digest, realized, prefix = _audit_member_stream(raw, entry)
            realized_total += realized
            if realized_total > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ZipContractError(
                    "realized ZIP total exceeds the size cap"
                )
            if _has_nested_archive_magic(prefix):
                raise ZipContractError("nested archive payload is forbidden")
            suffix_counts[entry.suffix] += 1
            if realized > 0 and entry.suffix in MACHINE_MEMBER_SUFFIXES:
                machine_family_keys.add(entry.name.lower())
            members.append(
                MemberAudit(
                    basename=entry.name,
                    suffix=entry.suffix,
                    sha256=digest,
                    uncompressed_bytes=realized,
                    timestamp_lag_days=(
                        entry.timestamp_date - release_identity_date
                    ).days,
                )
            )

    if realized_total != sum(entry.uncompressed_size for entry in entries):
        raise ZipContractError("realized ZIP total disagrees with declarations")
    if not machine_family_keys:
        raise SupportError("release has no non-empty machine-readable member")
    return ArchiveAudit(
        archive_sha256=sha256_bytes(raw),
        archive_bytes=len(raw),
        members=tuple(members),
        suffix_counts=tuple(sorted(suffix_counts.items())),
        declared_uncompressed_bytes=sum(
            entry.uncompressed_size for entry in entries
        ),
        realized_uncompressed_bytes=realized_total,
        machine_family_keys=frozenset(machine_family_keys),
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def assert_protocol_committed() -> tuple[str, str]:
    if sha256_file(BOUNDARY_PATH) != BOUNDARY_SHA256:
        raise ProtocolError("boundary hash changed")
    for path in (BOUNDARY_PATH, SCRIPT_PATH, TEST_PATH):
        tracked = _git("ls-files", "--error-unmatch", str(path))
        if tracked.returncode != 0:
            raise ProtocolError("required protocol path is not committed")
        clean = _git("diff", "--quiet", "HEAD", "--", str(path))
        if clean.returncode != 0:
            raise ProtocolError("required protocol path differs from HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0 or status.stdout:
        raise ProtocolError("repository is not HEAD-clean")
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}\n?", head.stdout) is None:
        raise ProtocolError("HEAD commit is unavailable")
    blob = _git("rev-parse", f"HEAD:{SCRIPT_PATH}")
    if blob.returncode != 0 or re.fullmatch(r"[0-9a-f]{40,64}\n?", blob.stdout) is None:
        raise ProtocolError("runner Git blob is unavailable")
    return head.stdout.strip(), blob.stdout.strip()


def assert_disk_guard(path: Path = REPOSITORY_ROOT) -> tuple[int, int]:
    usage = shutil.disk_usage(path)
    if usage.used >= DISK_USED_LIMIT or usage.free < DISK_FREE_FLOOR:
        raise DiskGuardError("filesystem is outside the frozen disk guard")
    return usage.used, usage.free


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = absolute
    while True:
        if current.is_symlink():
            raise ProtocolError("one-shot artifact path traverses a symlink")
        if current == current.parent:
            break
        current = current.parent


def _canonical_artifact_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise ProtocolError("one-shot artifact path must be a Path")
    lexical = repository_path(path)
    if ".." in lexical.parts:
        raise ProtocolError("one-shot artifact path contains a parent segment")
    _assert_no_symlink_components(lexical)
    canonical = lexical.resolve(strict=False)
    _assert_no_symlink_components(canonical)
    return canonical


def _absolute_audit_paths(paths: AuditPaths) -> AuditPaths:
    return AuditPaths(
        sentinel=_canonical_artifact_path(paths.sentinel),
        manifest=_canonical_artifact_path(paths.manifest),
        raw_dir=_canonical_artifact_path(paths.raw_dir),
        report=_canonical_artifact_path(paths.report),
    )


def _validate_artifact_paths(paths: AuditPaths) -> AuditPaths:
    absolute = _absolute_audit_paths(paths)
    values = (
        absolute.sentinel,
        absolute.manifest,
        absolute.raw_dir,
        absolute.report,
    )
    if len(set(values)) != len(values):
        raise ProtocolError("one-shot artifact paths overlap")
    for path in values:
        _assert_no_symlink_components(path)
    return absolute


def _exclusive_write(path: Path, payload: bytes, *, mode: int) -> None:
    _assert_no_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
    except FileExistsError as exc:
        raise ProtocolError("one-shot artifact already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    except BaseException:
        raise


def _atomic_publish(path: Path, payload: bytes) -> None:
    _assert_no_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _exclusive_write(temporary, payload, mode=0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PublicationError("one-shot report already exists") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(slots=True)
class AttemptGuard:
    sentinel_path: Path
    manifest_path: Path
    sentinel_bytes: bytes
    sentinel_sha256: str
    manifest_ordinal: int = 0
    manifest_last_hash: str = "0" * 64
    manifest_prefix_sha256: str = hashlib.sha256(b"").hexdigest()
    request_count: int = 0
    identity_finalized: bool = False

    def _validate_sentinel(self) -> None:
        _assert_no_symlink_components(self.sentinel_path)
        try:
            current = self.sentinel_path.read_bytes()
        except OSError as exc:
            raise ProtocolError("one-shot sentinel is unavailable") from exc
        if current != self.sentinel_bytes or sha256_bytes(current) != self.sentinel_sha256:
            raise ProtocolError("one-shot sentinel changed")

    def _validate_manifest(self) -> None:
        _assert_no_symlink_components(self.manifest_path)
        try:
            raw = self.manifest_path.read_bytes()
        except OSError as exc:
            raise ProtocolError("retrieval manifest is unavailable") from exc
        if sha256_bytes(raw) != self.manifest_prefix_sha256:
            raise ProtocolError("retrieval manifest exact prefix changed")
        previous_hash = "0" * 64
        ordinal = 0
        if raw and not raw.endswith(b"\n"):
            raise ProtocolError("retrieval manifest has a partial record")
        for line in raw.splitlines():
            record = parse_json_object(line)
            expected_keys = {
                "event",
                "ordinal",
                "payload",
                "previous_sha256",
                "record_sha256",
            }
            if set(record) != expected_keys:
                raise ProtocolError("retrieval manifest record shape changed")
            ordinal += 1
            if record["ordinal"] != ordinal or record["previous_sha256"] != previous_hash:
                raise ProtocolError("retrieval manifest chain changed")
            record_without_hash = dict(record)
            claimed_hash = record_without_hash.pop("record_sha256")
            if not isinstance(claimed_hash, str):
                raise ProtocolError("retrieval manifest hash is malformed")
            computed_hash = sha256_bytes(
                canonical_json_bytes(record_without_hash, newline=False)
            )
            if claimed_hash != computed_hash:
                raise ProtocolError("retrieval manifest record hash changed")
            if line != canonical_json_bytes(record, newline=False):
                raise ProtocolError("retrieval manifest record is not canonical")
            previous_hash = computed_hash
        if ordinal != self.manifest_ordinal or previous_hash != self.manifest_last_hash:
            raise ProtocolError("retrieval manifest prefix changed")

    def validate(self) -> None:
        self._validate_sentinel()
        self._validate_manifest()

    def append(self, event: str, payload: Mapping[str, Any]) -> str:
        self.validate()
        record_without_hash = {
            "event": event,
            "ordinal": self.manifest_ordinal + 1,
            "payload": dict(payload),
            "previous_sha256": self.manifest_last_hash,
        }
        record_hash = sha256_bytes(
            canonical_json_bytes(record_without_hash, newline=False)
        )
        record = dict(record_without_hash)
        record["record_sha256"] = record_hash
        encoded = canonical_json_bytes(record)
        descriptor = os.open(self.manifest_path, os.O_WRONLY | os.O_APPEND)
        try:
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            pass
        self.manifest_ordinal += 1
        self.manifest_last_hash = record_hash
        self.manifest_prefix_sha256 = sha256_file(self.manifest_path)
        return record_hash

    def begin_request(self, *, kind: str, url: str) -> int:
        if kind != "index" and not self.identity_finalized:
            raise ProtocolError("archive request precedes identity finalization")
        if self.request_count >= EXPECTED_REQUESTS:
            raise ProtocolError("request count exceeds the frozen sequence")
        self.append(
            "request_intent",
            {
                "kind": kind,
                "method": "GET",
                "request_ordinal": self.request_count + 1,
                "url": url,
            },
        )
        self.request_count += 1
        self.validate()
        return self.request_count

    def finalize_identities(self, audit: IdentityAudit) -> None:
        if self.identity_finalized or self.request_count != 1:
            raise ProtocolError("identity finalization is out of sequence")
        self.append(
            "identity_finalization",
            {
                "identities": [row.private_dict() for row in audit.identities],
                "same_day_transitions": audit.same_day_transitions,
                "strictly_later_transitions": audit.strictly_later_transitions,
            },
        )
        self.identity_finalized = True


def reserve_attempt(
    *,
    paths: AuditPaths,
    verifier_commit: str,
    runner_blob: str,
    started_at_utc: datetime | None = None,
    run_id: str | None = None,
) -> AttemptGuard:
    absolute = _validate_artifact_paths(paths)
    for path in (
        absolute.sentinel,
        absolute.manifest,
        absolute.raw_dir,
        absolute.report,
    ):
        if path.exists():
            raise ProtocolError("one-shot path already exists")
    started = started_at_utc or datetime.now(timezone.utc)
    identifier = run_id or str(uuid.uuid4())
    try:
        uuid.UUID(identifier)
    except ValueError as exc:
        raise ProtocolError("run ID is not a UUID") from exc
    sentinel_bytes = canonical_json_bytes(
        {
            "boundary_sha256": BOUNDARY_SHA256,
            "protocol_version": PROTOCOL_VERSION,
            "reserved_at_utc": canonical_utc(started),
            "run_id": identifier,
            "runner_git_blob": runner_blob,
            "verifier_commit": verifier_commit,
        }
    )
    _exclusive_write(absolute.sentinel, sentinel_bytes, mode=0o400)
    try:
        _exclusive_write(absolute.manifest, b"", mode=0o600)
        absolute.raw_dir.mkdir(mode=0o700)
        _fsync_directory(absolute.raw_dir.parent)
    except BaseException:
        # The sentinel deliberately survives any partial reservation.
        raise
    return AttemptGuard(
        sentinel_path=absolute.sentinel,
        manifest_path=absolute.manifest,
        sentinel_bytes=sentinel_bytes,
        sentinel_sha256=sha256_bytes(sentinel_bytes),
    )


def _selected_headers(headers: Sequence[tuple[str, str]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name in (
        "Content-Type",
        "Content-Length",
        "Content-Encoding",
        "Transfer-Encoding",
        "Location",
        "ETag",
        "Last-Modified",
        "Date",
    ):
        values = _header_values(headers, name)
        if values:
            selected[name.lower()] = list(values)
    return selected


def _store_raw_response(
    *,
    raw_dir: Path,
    request_ordinal: int,
    kind: str,
    raw: bytes,
) -> tuple[Path, str]:
    digest = sha256_bytes(raw)
    suffix = {
        "index": ".html",
        "archive_redirect": ".redirect.bin",
        "archive": ".zip",
    }[kind]
    path = raw_dir / f"{request_ordinal:03d}-{kind}-{digest}{suffix}"
    _exclusive_write(path, raw, mode=0o600)
    return path, digest


_PRODUCTION_EXECUTION = object()
_FIXTURE_EXECUTION = object()


def _validate_index_response(payload: HttpPayload) -> None:
    _strict_url(
        payload.requested_url,
        exact_host="home.treasury.gov",
        exact_path="/archives-of-tic-monthly-data-releases",
    )
    validate_http_payload(
        payload,
        expected_statuses=frozenset({200}),
        body_cap=INDEX_BODY_CAP,
        allowed_content_types=frozenset({"text/html"}),
    )


def _validate_redirect_response(
    payload: HttpPayload,
    *,
    identity: ReleaseIdentity,
) -> None:
    _strict_url(
        payload.requested_url,
        exact_host="www.treasury.gov",
        exact_path=urllib.parse.urlsplit(identity.starting_url).path,
    )
    validate_http_payload(
        payload,
        expected_statuses=ALLOWED_REDIRECT_STATUSES,
        body_cap=REDIRECT_BODY_CAP,
        allowed_content_types=None,
        location_required=True,
    )
    location = _single_header(payload.headers, "Location", required=True)
    if location != identity.final_url:
        raise TransportError("archive redirect target changed")
    _strict_url(
        location,
        exact_host="ticdata.treasury.gov",
        exact_path=urllib.parse.urlsplit(identity.final_url).path,
    )


def _validate_archive_response(
    payload: HttpPayload,
    *,
    identity: ReleaseIdentity,
) -> None:
    _strict_url(
        payload.requested_url,
        exact_host="ticdata.treasury.gov",
        exact_path=urllib.parse.urlsplit(identity.final_url).path,
    )
    validate_http_payload(
        payload,
        expected_statuses=frozenset({200}),
        body_cap=ARCHIVE_BODY_CAP,
        allowed_content_types=ALLOWED_ARCHIVE_CONTENT_TYPES,
    )


def _outcome_boundary() -> dict[str, bool]:
    return {
        "candidate_incidence_opened": False,
        "market_or_price_opened": False,
        "funding_premium_or_open_interest_opened": False,
        "return_pnl_cagr_or_mdd_opened": False,
        "portfolio_reward_checkpoint_or_model_opened": False,
        "tic_table_header_row_or_value_emitted": False,
    }


def _bindings(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
) -> dict[str, Any]:
    return {
        "boundary_path": str(BOUNDARY_PATH),
        "boundary_sha256": BOUNDARY_SHA256,
        "manifest_path": str(guard.manifest_path),
        "manifest_sha256": sha256_file(guard.manifest_path),
        "protocol_version": PROTOCOL_VERSION,
        "runner_git_blob": runner_blob,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "sentinel_path": str(guard.sentinel_path),
        "sentinel_sha256": guard.sentinel_sha256,
        "test_path": str(TEST_PATH),
        "test_sha256": sha256_file(TEST_PATH),
        "verifier_commit": verifier_commit,
    }


def _pass_report(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
    index_sha256: str,
    identity_audit: IdentityAudit,
    archive_audits: Sequence[ArchiveAudit],
    disk_used_before: int,
    disk_free_before: int,
    authoritative: bool,
) -> dict[str, Any]:
    family_support: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    lag_counts: Counter[int] = Counter()
    member_count = 0
    member_bytes = 0
    for audit in archive_audits:
        family_support.update(audit.machine_family_keys)
        suffix_counts.update(dict(audit.suffix_counts))
        member_count += len(audit.members)
        member_bytes += audit.realized_uncompressed_bytes
        lag_counts.update(member.timestamp_lag_days for member in audit.members)
    stable_families = sorted(
        family
        for family, support in family_support.items()
        if support == EXPECTED_RELEASES
    )
    report: dict[str, Any] = {
        "archive_support": {
            "archive_bytes_total": sum(row.archive_bytes for row in archive_audits),
            "archive_hashes": [row.archive_sha256 for row in archive_audits],
            "archives": len(archive_audits),
            "machine_family_support_counts": dict(sorted(family_support.items())),
            "member_basenames": sorted(family_support),
            "member_count": member_count,
            "realized_uncompressed_bytes": member_bytes,
            "stable_machine_families": stable_families,
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "timestamp_lag_days": {
                str(key): value for key, value in sorted(lag_counts.items())
            },
        },
        "bindings": _bindings(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
        ),
        "created_at_utc": canonical_utc(datetime.now(timezone.utc)),
        "decision": "SOURCE_SUPPORT_PASS",
        "execution_authority": (
            "production_one_shot" if authoritative else "offline_fixture"
        ),
        "disk": {
            "free_bytes_before": disk_free_before,
            "free_floor_bytes": DISK_FREE_FLOOR,
            "used_bytes_before": disk_used_before,
            "used_limit_bytes": DISK_USED_LIMIT,
        },
        "index": {
            "bytes": None,
            "sha256": index_sha256,
        },
        "mechanism_preregistration_authorized": authoritative,
        "outcome_boundary": _outcome_boundary(),
        "request_contract": {
            "expected_get_requests": EXPECTED_REQUESTS,
            "observed_get_requests": guard.request_count,
            "retry_or_resume_used": False,
        },
        "source_clock": {
            "earliest_available_at_utc": canonical_utc(
                identity_audit.identities[0].available_at_utc
            ),
            "latest_available_at_utc": canonical_utc(
                identity_audit.identities[-1].available_at_utc
            ),
            "release_identities": len(identity_audit.identities),
            "same_day_transitions": identity_audit.same_day_transitions,
            "strictly_later_transitions": (
                identity_audit.strictly_later_transitions
            ),
        },
        "source_gates": {
            "archive_hashes_unique": True,
            "exact_54_release_identity": True,
            "exact_delayed_public_release_exception": True,
            "exact_label_filename_exception": True,
            "exact_one_hop_redirects": True,
            "machine_family_intersection_nonempty": bool(stable_families),
            "member_crc_and_structure": True,
            "source_span_over_four_years": True,
        },
        "source_audit_authoritative": authoritative,
    }
    report_without_hash = dict(report)
    report["manifest_hash_without_self"] = sha256_bytes(
        canonical_json_bytes(report_without_hash, newline=False)
    )
    return report


def _reject_report(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
    stage: str,
    exception: BaseException,
    disk_used_before: int,
    disk_free_before: int,
    authoritative: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "bindings": _bindings(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
        ),
        "created_at_utc": canonical_utc(datetime.now(timezone.utc)),
        "decision": "TERMINAL_REJECT",
        "execution_authority": (
            "production_one_shot" if authoritative else "offline_fixture"
        ),
        "disk": {
            "free_bytes_before": disk_free_before,
            "free_floor_bytes": DISK_FREE_FLOOR,
            "used_bytes_before": disk_used_before,
            "used_limit_bytes": DISK_USED_LIMIT,
        },
        "failure": {
            "exception_class": type(exception).__name__,
            "stage": stage,
        },
        "mechanism_preregistration_authorized": False,
        "outcome_boundary": _outcome_boundary(),
        "retry_or_resume_authorized": False,
        "source_audit_authoritative": authoritative,
    }
    report_without_hash = dict(report)
    report["manifest_hash_without_self"] = sha256_bytes(
        canonical_json_bytes(report_without_hash, newline=False)
    )
    return report


def _run_bound_source_audit(
    *,
    execution_mode: object,
    paths: AuditPaths | None = None,
    fixture_responses: dict[str, HttpPayload] | None = None,
    fixture_verifier_commit: str | None = None,
    fixture_runner_blob: str | None = None,
    isolated_authority: str | None = None,
) -> dict[str, Any]:
    if execution_mode is _PRODUCTION_EXECUTION:
        if (
            paths is not None
            or fixture_responses is not None
            or fixture_verifier_commit is not None
            or fixture_runner_blob is not None
        ):
            raise ProtocolError("production execution binding changed")
        if (
            sys.flags.isolated != 1
            or __name__ != "__main__"
            or __spec__ is not None
            or not isinstance(isolated_authority, str)
            or re.fullmatch(r"[0-9a-f]{64}", isolated_authority) is None
            or os.environ.get("TICRV_ISOLATED_CHILD_TOKEN")
            != isolated_authority
            or os.environ.get("TICRV_ISOLATED_PARENT_PID")
            != str(os.getppid())
        ):
            raise ProtocolError(
                "production requires the isolated CLI child"
            )
        literal_paths = AuditPaths(
            sentinel=Path(
                "results/.treasury_tic_release_vintage_source_2026-07-24.started"
            ),
            manifest=Path(
                "results/.treasury_tic_release_vintage_source_2026-07-24."
                "manifest.ndjson"
            ),
            raw_dir=Path(
                "results/.treasury_tic_release_vintage_source_2026-07-24.raw"
            ),
            report=Path(
                "results/treasury_tic_release_vintage_source_2026-07-24.json"
            ),
        )
        absolute_paths = _validate_artifact_paths(literal_paths)
        production_index_url = (
            "https://home.treasury.gov/archives-of-tic-monthly-data-releases"
        )
        production_index_plan = _direct_request_plan(production_index_url)
        if production_index_plan != (
            "home.treasury.gov",
            "/archives-of-tic-monthly-data-releases",
            {
                "User-Agent": CONTACT_USER_AGENT,
                "Accept-Encoding": "identity",
                "Accept": "*/*",
                "Connection": "close",
            },
        ):
            raise ProtocolError("production index request plan changed")
        verifier_commit, runner_blob = assert_protocol_committed()
        rate_limiter = _RateLimiter()
        tls_context = ssl.create_default_context()

        def fetch_payload(url: str, body_cap: int) -> HttpPayload:
            if url == production_index_url:
                host, path, headers = production_index_plan
            else:
                host, path, headers = _direct_request_plan(url)
            rate_limiter.wait()
            request_started = datetime.now(timezone.utc)
            connection = http.client.HTTPSConnection(
                host,
                443,
                timeout=HTTP_TIMEOUT_SECONDS,
                context=tls_context,
            )
            try:
                connection.request("GET", path, headers=headers)
                response = connection.getresponse()
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(
                        min(1 << 16, body_cap + 1 - total)
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > body_cap:
                        raise TransportError(
                            "HTTP body exceeds the frozen cap"
                        )
                raw = b"".join(chunks)
                response_headers = tuple(
                    (str(key), str(value))
                    for key, value in response.getheaders()
                )
                status = response.status
            except (
                OSError,
                ssl.SSLError,
                http.client.HTTPException,
                TimeoutError,
            ) as exc:
                raise TransportError(
                    "direct HTTPS request failed"
                ) from exc
            finally:
                connection.close()
            return HttpPayload(
                requested_url=url,
                status=status,
                headers=response_headers,
                raw=raw,
                request_started_at_utc=request_started,
                response_completed_at_utc=datetime.now(timezone.utc),
            )

        disk_guard = assert_disk_guard
        authoritative = True
        index_url = production_index_url
    elif execution_mode is _FIXTURE_EXECUTION:
        if (
            isolated_authority is not None
            or not isinstance(paths, AuditPaths)
            or type(fixture_responses) is not dict
            or not isinstance(fixture_verifier_commit, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", fixture_verifier_commit) is None
            or not isinstance(fixture_runner_blob, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", fixture_runner_blob) is None
        ):
            raise ProtocolError("fixture execution binding changed")
        absolute_paths = _fixture_paths_are_disjoint(paths)
        verifier_commit = fixture_verifier_commit
        runner_blob = fixture_runner_blob
        fixture_transport = _OfflineFixtureTransport(fixture_responses)
        fetch_payload = fixture_transport.get

        def fixture_disk_guard() -> tuple[int, int]:
            return 0, 1 << 50

        disk_guard = fixture_disk_guard
        authoritative = False
        index_url = ARCHIVE_INDEX_URL
    else:
        raise ProtocolError("audit execution mode is not authorized")

    disk_used_before, disk_free_before = disk_guard()
    guard = reserve_attempt(
        paths=absolute_paths,
        verifier_commit=verifier_commit,
        runner_blob=runner_blob,
    )
    absolute_raw_dir = absolute_paths.raw_dir

    def fetch_and_record(
        *,
        kind: str,
        url: str,
        body_cap: int,
    ) -> tuple[HttpPayload, str]:
        disk_guard()
        request_ordinal = guard.begin_request(kind=kind, url=url)
        payload = fetch_payload(url, body_cap)
        if payload.requested_url != url:
            raise TransportError("fetcher returned a different request identity")
        raw_path, digest = _store_raw_response(
            raw_dir=absolute_raw_dir,
            request_ordinal=request_ordinal,
            kind=kind,
            raw=payload.raw,
        )
        guard.append(
            "response_receipt",
            {
                "body_bytes": len(payload.raw),
                "body_sha256": digest,
                "headers": _selected_headers(payload.headers),
                "kind": kind,
                "raw_path": str(raw_path),
                "request_ordinal": request_ordinal,
                "request_started_at_utc": canonical_utc(
                    payload.request_started_at_utc
                ),
                "requested_url": url,
                "response_completed_at_utc": canonical_utc(
                    payload.response_completed_at_utc
                ),
                "status": payload.status,
            },
        )
        return payload, digest

    stage = "index"
    try:
        index_payload, index_sha256 = fetch_and_record(
            kind="index",
            url=index_url,
            body_cap=INDEX_BODY_CAP,
        )
        _validate_index_response(index_payload)
        identity_audit = parse_archive_index(index_payload.raw)
        guard.finalize_identities(identity_audit)

        archive_audits_list: list[ArchiveAudit] = []
        archive_hashes: set[str] = set()
        for identity in identity_audit.identities:
            redirect_payload, _ = fetch_and_record(
                kind="archive_redirect",
                url=identity.starting_url,
                body_cap=REDIRECT_BODY_CAP,
            )
            _validate_redirect_response(redirect_payload, identity=identity)

            archive_payload, _ = fetch_and_record(
                kind="archive",
                url=identity.final_url,
                body_cap=ARCHIVE_BODY_CAP,
            )
            _validate_archive_response(archive_payload, identity=identity)
            archive_audit = audit_zip_archive(
                archive_payload.raw,
                release_identity_date=identity.release_identity_date,
            )
            if archive_audit.archive_sha256 in archive_hashes:
                raise SupportError("archive hash is duplicated")
            archive_hashes.add(archive_audit.archive_sha256)
            archive_audits_list.append(archive_audit)
            guard.append(
                "archive_member_audit",
                {
                    "archive_sha256": archive_audit.archive_sha256,
                    "machine_family_keys": sorted(
                        archive_audit.machine_family_keys
                    ),
                    "members": [
                        {
                            "basename": member.basename,
                            "sha256": member.sha256,
                            "suffix": member.suffix,
                            "timestamp_lag_days": member.timestamp_lag_days,
                            "uncompressed_bytes": member.uncompressed_bytes,
                        }
                        for member in archive_audit.members
                    ],
                },
            )

        if len(archive_audits_list) != EXPECTED_RELEASES:
            raise SupportError("archive count changed")
        stable_families = set(archive_audits_list[0].machine_family_keys)
        for archive_audit in archive_audits_list[1:]:
            stable_families.intersection_update(
                archive_audit.machine_family_keys
            )
        if not stable_families:
            raise SupportError(
                "stable exact machine-readable family is absent"
            )
        if guard.request_count != EXPECTED_REQUESTS:
            raise ProtocolError(
                "request count differs from the frozen sequence"
            )
        guard.append(
            "source_support_pass",
            {
                "archive_count": len(archive_audits_list),
                "request_count": guard.request_count,
                "stable_machine_families": sorted(stable_families),
            },
        )
        guard.validate()
        archive_audits = tuple(archive_audits_list)

        stage = "support"
        report = _pass_report(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
            index_sha256=index_sha256,
            identity_audit=identity_audit,
            archive_audits=archive_audits,
            disk_used_before=disk_used_before,
            disk_free_before=disk_free_before,
            authoritative=authoritative,
        )
        report["index"]["bytes"] = next(
            path.stat().st_size
            for path in absolute_raw_dir.iterdir()
            if "-index-" in path.name
        )
        report_without_hash = dict(report)
        report_without_hash.pop("manifest_hash_without_self", None)
        report["manifest_hash_without_self"] = sha256_bytes(
            canonical_json_bytes(report_without_hash, newline=False)
        )
    except BaseException as exc:
        if isinstance(exc, IndexContractError):
            stage = "index"
        elif isinstance(exc, TransportError):
            stage = "transport"
        elif isinstance(exc, ZipContractError):
            stage = "zip"
        elif isinstance(exc, SupportError):
            stage = "support"
        elif isinstance(exc, (ProtocolError, DiskGuardError)):
            stage = "protocol"
        else:
            stage = "source"
        try:
            guard.append(
                "terminal_reject",
                {
                    "exception_class": type(exc).__name__,
                    "stage": stage,
                },
            )
        except BaseException:
            pass
        report = _reject_report(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
            stage=stage,
            exception=exc,
            disk_used_before=disk_used_before,
            disk_free_before=disk_free_before,
            authoritative=authoritative,
        )
    _atomic_publish(absolute_paths.report, canonical_json_bytes(report))
    return report


def _fixture_paths_are_disjoint(paths: AuditPaths) -> AuditPaths:
    absolute = _validate_artifact_paths(paths)
    production_root = _canonical_artifact_path(Path("results"))
    for path in (
        absolute.sentinel,
        absolute.manifest,
        absolute.raw_dir,
        absolute.report,
    ):
        if (
            path == production_root
            or production_root in path.parents
            or path in production_root.parents
        ):
            raise ProtocolError(
                "fixture artifacts must be disjoint from the production tree"
            )
    return absolute


def run_fixture_audit(
    *,
    paths: AuditPaths,
    responses: dict[str, HttpPayload],
    verifier_commit: str,
    runner_blob: str,
) -> dict[str, Any]:
    """Run the protocol against an exact preloaded in-memory response ledger."""

    return _run_bound_source_audit(
        execution_mode=_FIXTURE_EXECUTION,
        paths=paths,
        fixture_responses=responses,
        fixture_verifier_commit=verifier_commit,
        fixture_runner_blob=runner_blob,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--_isolated-child-token",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.report != DEFAULT_REPORT:
        print("TICRV source audit: PRECHECK_REJECT")
        return 2
    if args._isolated_child_token is None:
        if __name__ != "__main__" or __spec__ is not None:
            print("TICRV source audit: PRECHECK_REJECT")
            return 2
        token = secrets.token_hex(32)
        environment = dict(os.environ)
        environment["TICRV_ISOLATED_CHILD_TOKEN"] = token
        environment["TICRV_ISOLATED_PARENT_PID"] = str(os.getpid())
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--_isolated-child-token",
            token,
        ]
        try:
            completed = subprocess.run(
                command,
                env=environment,
                check=False,
                close_fds=True,
            )
        except OSError:
            print("TICRV source audit: PRECHECK_REJECT")
            return 2
        return completed.returncode

    try:
        report = _run_bound_source_audit(
            execution_mode=_PRODUCTION_EXECUTION,
            isolated_authority=args._isolated_child_token,
        )
    except BaseException:
        consumed = repository_path(
            Path(
                "results/.treasury_tic_release_vintage_source_2026-07-24.started"
            )
        ).exists()
        print(
            "TICRV source audit: "
            + ("TERMINAL_REJECT" if consumed else "PRECHECK_REJECT")
        )
        return 2
    if (
        report["decision"] != "SOURCE_SUPPORT_PASS"
        or report["execution_authority"] != "production_one_shot"
        or report["source_audit_authoritative"] is not True
        or report["mechanism_preregistration_authorized"] is not True
    ):
        print("TICRV source audit: TERMINAL_REJECT")
        return 1
    print("TICRV source audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
