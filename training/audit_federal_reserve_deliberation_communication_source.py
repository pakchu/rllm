"""One-shot source-only audit for the frozen FOMC communication ledger.

The module intentionally uses only the Python standard library. It can retrieve
and structurally validate official Federal Reserve statements and minutes, but
it has no market, model, portfolio, database, or live-trading dependency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import html.parser
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = Path(
    "docs/federal-reserve-deliberation-communication-source-axis-decision-"
    "2026-07-25.md"
)
LEDGER_PATH = Path(
    "data/federal_reserve_deliberation_communication_identity_2012_2020.csv"
)
SCRIPT_PATH = Path(
    "training/audit_federal_reserve_deliberation_communication_source.py"
)
TEST_PATH = Path(
    "tests/test_audit_federal_reserve_deliberation_communication_source.py"
)
BOUNDARY_SHA256 = (
    "5ebfe75e2e7d0f5f8186376712e6a551290effc1b9bfd8061f37e9426df636a7"
)
LEDGER_SHA256 = (
    "a8586f749e2e1d3f3a83fb14de579f3c286fd1a8077af8fb9a5b67d247012bea"
)
LEDGER_BYTES = 29_677
PROTOCOL_VERSION = "FRDCL-D1-source-audit-2026-07-25"

SOURCE_START = date(2012, 1, 1)
SOURCE_END = date(2020, 12, 31)
NEW_YORK = ZoneInfo("America/New_York")
ALLOWED_HOST = "www.federalreserve.gov"
INDEX_YEARS = tuple(range(2011, 2021))
INDEX_URLS = tuple(
    f"https://{ALLOWED_HOST}/monetarypolicy/fomchistorical{year}.htm"
    for year in INDEX_YEARS
)

REQUEST_HEADERS: tuple[tuple[str, str], ...] = (
    (
        "User-Agent",
        "rllm-frdcl-source-audit/1.0 (https://github.com/pakchu/rllm)",
    ),
    ("Accept", "text/html"),
    ("Accept-Encoding", "identity"),
)
HTTP_TIMEOUT_SECONDS = 30.0
FIRST_ATTEMPT_DELAY_SECONDS = 0.10
RETRY_DELAYS = (1.0, 2.0)
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
HTML_BODY_CAP = 2 * 1024 * 1024
TOTAL_RAW_SOURCE_CAP = 256 * 1024 * 1024
DISK_USED_LIMIT = 300 * 1024**3
DISK_FREE_FLOOR = 8 * 1024**3
MANIFEST_CAP = 64 * 1024 * 1024

EXPECTED_FIELDS = (
    "document_class",
    "meeting_date",
    "release_date",
    "last_update_date",
    "encoding",
    "source_eligible",
    "official_url",
    "index_url",
)
EXPECTED_CANDIDATE_CLASS_COUNTS = {
    "fomc_statement": 75,
    "fomc_minutes": 72,
}
EXPECTED_ELIGIBLE_CLASS_COUNTS = {
    "fomc_statement": 73,
    "fomc_minutes": 72,
}
EXPECTED_RELEASE_YEAR_COUNTS = {
    2012: 16,
    2013: 16,
    2014: 16,
    2015: 16,
    2016: 16,
    2017: 16,
    2018: 16,
    2019: 17,
    2020: 18,
}
EXPECTED_QUARANTINES = frozenset(
    {
        (
            date(2018, 1, 31),
            date(2018, 2, 1),
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20180131a.htm",
        ),
        (
            date(2019, 10, 11),
            date(2019, 10, 15),
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20191011a.htm",
        ),
    }
)
EXPECTED_WINDOWS_1252_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20111213.htm"
)

STATEMENT_PATH_RE = re.compile(
    r"/newsevents/pressreleases/monetary(?P<stamp>\d{8})a\.htm\Z",
    re.ASCII,
)
MINUTES_PATH_RE = re.compile(
    r"/monetarypolicy/fomcminutes(?P<stamp>\d{8})\.htm\Z",
    re.ASCII,
)
INDEX_PATH_RE = re.compile(
    r"/monetarypolicy/fomchistorical(?P<year>201[1-9]|2020)\.htm\Z",
    re.ASCII,
)
RELEASED_RE = re.compile(
    r"\(\s*Released\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+"
    r"(?P<day>[1-9]|[12]\d|3[01]),\s+"
    r"(?P<year>\d{4})\s*\)",
    re.ASCII,
)
LAST_UPDATE_RE = re.compile(
    r"Last\s+[Uu]pdate:\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+"
    r"(?P<day>[1-9]|[12]\d|3[01]),\s+"
    r"(?P<year>\d{4})\Z",
    re.ASCII,
)
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
SAFE_KEY_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z", re.ASCII)

CONTAINER_TAGS = frozenset(
    {"div", "section", "article", "main", "ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "tr"}
)
BLOCK_TAGS = frozenset(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "blockquote",
        "dt",
        "dd",
        "caption",
        "th",
        "td",
    }
)
INLINE_TAGS = frozenset(
    {
        "a",
        "em",
        "strong",
        "span",
        "sup",
        "sub",
        "b",
        "i",
        "u",
        "abbr",
        "cite",
        "code",
        "q",
        "small",
        "time",
    }
)
SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "form",
        "noscript",
        "svg",
        "template",
        "picture",
    }
)
VOID_TAGS = frozenset({"br", "hr", "img", "source"})
ZERO_WIDTH = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"})

DEFAULT_SENTINEL = REPOSITORY_ROOT / (
    "results/.federal_reserve_deliberation_communication_source_"
    "2026-07-25.started"
)
DEFAULT_MANIFEST = REPOSITORY_ROOT / (
    "results/.federal_reserve_deliberation_communication_source_"
    "2026-07-25.manifest.ndjson"
)
DEFAULT_RAW_DIR = REPOSITORY_ROOT / (
    "results/.federal_reserve_deliberation_communication_source_"
    "2026-07-25.raw"
)
DEFAULT_REPORT = REPOSITORY_ROOT / (
    "results/federal_reserve_deliberation_communication_source_2026-07-25.json"
)


class AuditError(RuntimeError):
    """Base class for the frozen source-audit protocol."""


class ProtocolError(AuditError):
    """A local one-shot protocol precondition failed."""


class DiskGuardError(ProtocolError):
    """The repository filesystem violated the frozen disk boundary."""


class TransportError(AuditError):
    """An HTTP request or response violated the frozen transport."""


class RetryableTransportError(TransportError):
    """A request failed under one of the explicitly retryable conditions."""


class SourceContractError(AuditError):
    """An official source identity or page violated the source contract."""


class LedgerError(SourceContractError):
    """The committed identity ledger drifted."""


class IndexError(SourceContractError):
    """A historical index could not reproduce the frozen ledger."""


class PageMetadataError(SourceContractError):
    """A document's first-party metadata violated its frozen identity."""


class ArticleError(SourceContractError):
    """An eligible article violated the deterministic extraction grammar."""


class SupportError(AuditError):
    """The complete source panel failed a frozen support gate."""


class PublicationError(AuditError):
    """An immutable local audit artifact could not be published."""


@dataclass(frozen=True, slots=True)
class LedgerRow:
    document_class: str
    meeting_date: date
    release_date: date
    last_update_date: date
    encoding: str
    source_eligible: bool
    official_url: str
    index_url: str

    def identity_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.document_class,
            self.meeting_date.isoformat(),
            self.release_date.isoformat(),
            self.official_url,
            self.index_url,
        )


@dataclass(frozen=True, slots=True)
class IndexEntry:
    document_class: str
    meeting_date: date
    release_date: date
    official_url: str
    index_url: str

    def identity_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.document_class,
            self.meeting_date.isoformat(),
            self.release_date.isoformat(),
            self.official_url,
            self.index_url,
        )


@dataclass(frozen=True, slots=True)
class PageMetadata:
    article_count: int
    last_update_date: date


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    document_class: str
    source_eligible: bool
    raw_bytes: int
    raw_sha256: str
    article_count: int
    canonical_bytes: int | None
    canonical_characters: int | None
    canonical_blocks: int | None
    canonical_sha256: str | None

    def aggregate_dict(self) -> dict[str, Any]:
        return {
            "article_count": self.article_count,
            "canonical_blocks": self.canonical_blocks,
            "canonical_bytes": self.canonical_bytes,
            "canonical_characters": self.canonical_characters,
            "canonical_sha256": self.canonical_sha256,
            "document_class": self.document_class,
            "raw_bytes": self.raw_bytes,
            "raw_sha256": self.raw_sha256,
            "source_eligible": self.source_eligible,
        }


@dataclass(frozen=True, slots=True)
class RequestIntent:
    method: str
    url: str
    body: bytes
    headers: tuple[tuple[str, str], ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class HttpPayload:
    intent: RequestIntent
    status: int
    headers: tuple[tuple[str, str], ...]
    raw: bytes
    request_started_at_utc: datetime
    response_completed_at_utc: datetime


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


@dataclass(slots=True)
class _IndexAnchor:
    href: str
    label_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _IndexBlock:
    tag: str
    text_parts: list[str] = field(default_factory=list)
    anchors: list[_IndexAnchor] = field(default_factory=list)


@dataclass(slots=True)
class _ArticleFrame:
    tag: str
    block_parts: list[str] | None
    skipped: bool
    article_root: bool = False


def repository_path(path: Path) -> Path:
    return REPOSITORY_ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def available_at_utc(release_date: date) -> datetime:
    local = datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        0,
        5,
        tzinfo=NEW_YORK,
    ) + timedelta(days=1)
    return local.astimezone(timezone.utc)


def _parse_iso(value: str, *, label: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise LedgerError(f"{label} is not an ISO date") from exc
    if parsed.isoformat() != value:
        raise LedgerError(f"{label} is not canonical")
    return parsed


def _parse_named_date(match: re.Match[str], *, label: str) -> date:
    value = (
        f"{match.group('month')} {match.group('day')}, {match.group('year')}"
    )
    try:
        return datetime.strptime(value, "%B %d, %Y").date()
    except ValueError as exc:
        raise SourceContractError(f"{label} is not a valid date") from exc


def _date_from_stamp(stamp: str, *, label: str) -> date:
    try:
        return datetime.strptime(stamp, "%Y%m%d").date()
    except ValueError as exc:
        raise SourceContractError(f"{label} path date is invalid") from exc


def _normalize_space(value: str) -> str:
    transformed: list[str] = []
    for character in unicodedata.normalize("NFC", value):
        if character in ZERO_WIDTH:
            continue
        if character in "\r\n\t\f\v" or unicodedata.category(character) == "Zs":
            transformed.append(" ")
        else:
            transformed.append(character)
    return re.sub(r" +", " ", "".join(transformed)).strip()


def _attributes(
    attrs: Sequence[tuple[str, str | None]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in attrs:
        if key in result:
            raise SourceContractError(f"duplicate HTML attribute: {key}")
        result[key] = value or ""
    return result


def _strict_absolute_url(value: str) -> urllib.parse.SplitResult:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TransportError("URL is not ASCII") from exc
    if (
        not value
        or "\\" in value
        or "%" in value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise TransportError("URL has forbidden encoding or control bytes")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_HOST
        or parsed.netloc != ALLOWED_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise TransportError("URL is outside the frozen authority")
    return parsed


def _classify_item_url(value: str) -> tuple[str, date]:
    parsed = _strict_absolute_url(value)
    statement = STATEMENT_PATH_RE.fullmatch(parsed.path)
    if statement is not None:
        return (
            "fomc_statement",
            _date_from_stamp(statement.group("stamp"), label="statement"),
        )
    minutes = MINUTES_PATH_RE.fullmatch(parsed.path)
    if minutes is not None:
        return (
            "fomc_minutes",
            _date_from_stamp(minutes.group("stamp"), label="minutes"),
        )
    raise TransportError("item URL path is unauthorized")


def _index_year(value: str) -> int:
    parsed = _strict_absolute_url(value)
    match = INDEX_PATH_RE.fullmatch(parsed.path)
    if match is None:
        raise TransportError("index URL path is unauthorized")
    year = int(match.group("year"))
    if value not in INDEX_URLS:
        raise TransportError("index URL is outside the frozen set")
    return year


def make_get_intent(url: str) -> RequestIntent:
    if url not in INDEX_URLS:
        raise TransportError("GET endpoint is not a frozen historical index")
    _index_year(url)
    return RequestIntent(
        method="GET",
        url=url,
        body=b"",
        headers=REQUEST_HEADERS,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
    )


def make_document_intent(row: LedgerRow) -> RequestIntent:
    frozen = load_ledger()
    if row not in frozen:
        raise TransportError("document row is absent from the frozen ledger")
    document_class, meeting_date = _classify_item_url(row.official_url)
    if (
        document_class != row.document_class
        or meeting_date != row.meeting_date
        or not SOURCE_START <= row.release_date <= SOURCE_END
    ):
        raise TransportError("document row identity is inconsistent")
    return RequestIntent(
        method="GET",
        url=row.official_url,
        body=b"",
        headers=REQUEST_HEADERS,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
    )


def load_ledger(path: Path | None = None) -> tuple[LedgerRow, ...]:
    target = repository_path(LEDGER_PATH) if path is None else path
    raw = target.read_bytes()
    if len(raw) != LEDGER_BYTES or sha256_bytes(raw) != LEDGER_SHA256:
        raise LedgerError("identity ledger bytes or digest changed")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LedgerError("identity ledger is not UTF-8") from exc
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
        raise LedgerError("identity ledger schema changed")
    rows: list[LedgerRow] = []
    for raw_row in reader:
        if set(raw_row) != set(EXPECTED_FIELDS) or any(
            value is None for value in raw_row.values()
        ):
            raise LedgerError("identity ledger row is malformed")
        eligible_text = raw_row["source_eligible"]
        if eligible_text not in {"true", "false"}:
            raise LedgerError("source_eligible is not canonical")
        row = LedgerRow(
            document_class=raw_row["document_class"],
            meeting_date=_parse_iso(raw_row["meeting_date"], label="meeting_date"),
            release_date=_parse_iso(raw_row["release_date"], label="release_date"),
            last_update_date=_parse_iso(
                raw_row["last_update_date"], label="last_update_date"
            ),
            encoding=raw_row["encoding"],
            source_eligible=eligible_text == "true",
            official_url=raw_row["official_url"],
            index_url=raw_row["index_url"],
        )
        rows.append(row)
    _validate_ledger(rows)
    return tuple(rows)


def _validate_ledger(rows: Sequence[LedgerRow]) -> None:
    if len(rows) != 147:
        raise LedgerError("identity ledger row count changed")
    if len({row.official_url for row in rows}) != len(rows):
        raise LedgerError("identity ledger repeats an official URL")
    if list(rows) != sorted(
        rows,
        key=lambda row: (
            row.release_date,
            row.document_class,
            row.official_url,
        ),
    ):
        raise LedgerError("identity ledger order changed")
    if Counter(row.document_class for row in rows) != Counter(
        EXPECTED_CANDIDATE_CLASS_COUNTS
    ):
        raise LedgerError("candidate class counts changed")
    eligible = [row for row in rows if row.source_eligible]
    if len(eligible) != 145 or Counter(
        row.document_class for row in eligible
    ) != Counter(EXPECTED_ELIGIBLE_CLASS_COUNTS):
        raise LedgerError("eligible class counts changed")
    if Counter(row.release_date.year for row in rows) != Counter(
        EXPECTED_RELEASE_YEAR_COUNTS
    ):
        raise LedgerError("release-year counts changed")
    if any(not SOURCE_START <= row.release_date <= SOURCE_END for row in rows):
        raise LedgerError("release date escaped the frozen source window")
    quarantines = {
        (row.release_date, row.last_update_date, row.official_url)
        for row in rows
        if not row.source_eligible
    }
    if quarantines != EXPECTED_QUARANTINES:
        raise LedgerError("quarantine set changed")
    if any(
        row.source_eligible and row.last_update_date != row.release_date
        for row in rows
    ):
        raise LedgerError("eligible row has a later update")
    if Counter(row.encoding for row in rows) != Counter(
        {"utf-8": 146, "windows-1252": 1}
    ):
        raise LedgerError("encoding ledger changed")
    if [
        row.official_url for row in rows if row.encoding == "windows-1252"
    ] != [EXPECTED_WINDOWS_1252_URL]:
        raise LedgerError("windows-1252 identity changed")
    for row in rows:
        index_year = _index_year(row.index_url)
        document_class, path_date = _classify_item_url(row.official_url)
        if document_class != row.document_class:
            raise LedgerError("document class disagrees with URL")
        if index_year != row.meeting_date.year:
            raise LedgerError("historical index year disagrees with meeting")
        if path_date != row.meeting_date:
            raise LedgerError("item URL date disagrees with meeting")
        if row.document_class == "fomc_statement":
            if row.meeting_date != row.release_date:
                raise LedgerError("statement meeting and release dates differ")
        elif row.release_date <= row.meeting_date:
            raise LedgerError("minutes release is not after its meeting")


class _HistoricalIndexParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._blocks: list[_IndexBlock] = []
        self._anchor: _IndexAnchor | None = None
        self._records: list[tuple[_IndexAnchor, str]] = []

    @property
    def records(self) -> tuple[tuple[_IndexAnchor, str], ...]:
        if self._blocks or self._anchor is not None:
            raise IndexError("historical index has unclosed candidate structure")
        return tuple(self._records)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = _attributes(attrs)
        if tag in {"p", "li", "td", "div", "section"}:
            self._blocks.append(_IndexBlock(tag=tag))
        elif tag == "a":
            if self._anchor is not None:
                raise IndexError("historical index nests anchors")
            self._anchor = _IndexAnchor(href=attributes.get("href", ""))
        elif tag == "br":
            for block in self._blocks:
                block.text_parts.append(" ")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        for block in self._blocks:
            block.text_parts.append(data)
        if self._anchor is not None:
            self._anchor.label_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            if self._anchor is None:
                return
            if not self._blocks:
                href = self._anchor.href
                self._anchor = None
                try:
                    parsed = urllib.parse.urlsplit(href)
                except ValueError:
                    return
                if STATEMENT_PATH_RE.fullmatch(parsed.path) or MINUTES_PATH_RE.fullmatch(
                    parsed.path
                ):
                    raise IndexError("candidate anchor is outside a supported block")
                return
            self._blocks[-1].anchors.append(self._anchor)
            self._anchor = None
        elif tag in {"p", "li", "td", "div", "section"}:
            if not self._blocks:
                return
            if self._blocks[-1].tag != tag:
                if any(block.anchors for block in self._blocks):
                    raise IndexError("historical index candidate nesting is malformed")
                return
            if self._anchor is not None:
                raise IndexError("candidate block closes inside an anchor")
            block = self._blocks.pop()
            context = _normalize_space("".join(block.text_parts))
            self._records.extend((anchor, context) for anchor in block.anchors)


def parse_historical_index(raw: bytes, *, index_url: str) -> tuple[IndexEntry, ...]:
    index_year = _index_year(index_url)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise IndexError("historical index is not strict UTF-8") from exc
    parser = _HistoricalIndexParser()
    try:
        parser.feed(text)
        parser.close()
        records = parser.records
    except SourceContractError as exc:
        if isinstance(exc, IndexError):
            raise
        raise IndexError("historical index HTML is malformed") from exc

    entries: list[IndexEntry] = []
    seen: set[str] = set()
    for anchor, context in records:
        if not anchor.href:
            continue
        absolute = urllib.parse.urljoin(index_url, anchor.href)
        parsed = urllib.parse.urlsplit(absolute)
        statement_match = STATEMENT_PATH_RE.fullmatch(parsed.path)
        minutes_match = MINUTES_PATH_RE.fullmatch(parsed.path)
        if statement_match is None and minutes_match is None:
            continue
        document_class, meeting_date = _classify_item_url(absolute)
        if meeting_date.year != index_year:
            raise IndexError("candidate item is listed under the wrong year")
        label = _normalize_space("".join(anchor.label_parts))
        if document_class == "fomc_statement":
            if label != "Statement":
                raise IndexError("statement anchor label changed")
            release_date = meeting_date
        else:
            if label != "Minutes":
                raise IndexError("minutes anchor label changed")
            matches = list(RELEASED_RE.finditer(context))
            if len(matches) != 1:
                raise IndexError("minutes release date is absent or ambiguous")
            release_date = _parse_named_date(matches[0], label="minutes release")
            if release_date <= meeting_date:
                raise IndexError("minutes release date is not after the meeting")
        if not SOURCE_START <= release_date <= SOURCE_END:
            continue
        if absolute in seen:
            raise IndexError("historical index repeats a candidate URL")
        seen.add(absolute)
        entries.append(
            IndexEntry(
                document_class=document_class,
                meeting_date=meeting_date,
                release_date=release_date,
                official_url=absolute,
                index_url=index_url,
            )
        )
    return tuple(
        sorted(
            entries,
            key=lambda row: (
                row.release_date,
                row.document_class,
                row.official_url,
            ),
        )
    )


class _PageMetadataParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_count = 0
        self.last_update_count = 0
        self._last_update_depth = 0
        self._last_update_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = _attributes(attrs)
        identifier = attributes.get("id")
        if tag == "div" and identifier == "article":
            self.article_count += 1
        if self._last_update_depth:
            if tag not in VOID_TAGS:
                self._last_update_depth += 1
        elif identifier == "lastUpdate":
            if tag != "div":
                raise PageMetadataError("lastUpdate is not a div")
            self.last_update_count += 1
            self._last_update_depth = 1

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if self._last_update_depth and tag not in VOID_TAGS:
            self._last_update_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._last_update_depth:
            self._last_update_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._last_update_depth:
            self._last_update_depth -= 1
            if self._last_update_depth < 0:
                raise PageMetadataError("lastUpdate nesting underflowed")

    def result(self) -> PageMetadata:
        if self.article_count != 1:
            raise PageMetadataError("document does not have exactly one article")
        if self.last_update_count != 1 or self._last_update_depth != 0:
            raise PageMetadataError("document does not have one closed lastUpdate")
        normalized = _normalize_space("".join(self._last_update_parts))
        match = LAST_UPDATE_RE.fullmatch(normalized)
        if match is None:
            raise PageMetadataError("lastUpdate text is malformed")
        return PageMetadata(
            article_count=self.article_count,
            last_update_date=_parse_named_date(match, label="lastUpdate"),
        )


def parse_page_metadata(text: str) -> PageMetadata:
    parser = _PageMetadataParser()
    try:
        parser.feed(text)
        parser.close()
        return parser.result()
    except SourceContractError as exc:
        if isinstance(exc, PageMetadataError):
            raise
        raise PageMetadataError("document metadata HTML is malformed") from exc


class _ArticleParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_count = 0
        self.article_active = False
        self.article_closed = False
        self.stack: list[_ArticleFrame] = []
        self.skip_depth = 0
        self.blocks: list[str] = []

    def _active_block(self) -> _ArticleFrame | None:
        for frame in reversed(self.stack):
            if frame.block_parts is not None and not frame.skipped:
                return frame
        return None

    def _flush(self, frame: _ArticleFrame) -> None:
        if frame.block_parts is None:
            return
        normalized = _normalize_space("".join(frame.block_parts))
        frame.block_parts.clear()
        if normalized:
            self.blocks.append(normalized)

    def _start_article(self) -> None:
        self.article_count += 1
        self.article_active = True
        self.article_closed = False
        self.stack.append(
            _ArticleFrame(
                tag="div",
                block_parts=None,
                skipped=False,
                article_root=True,
            )
        )

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = _attributes(attrs)
        identifier = attributes.get("id")
        if not self.article_active:
            if tag == "div" and identifier == "article":
                self._start_article()
            return

        if tag == "div" and identifier == "article":
            self.article_count += 1
            raise ArticleError("document repeats div#article")

        if self.skip_depth:
            if tag not in VOID_TAGS:
                self.stack.append(
                    _ArticleFrame(
                        tag=tag,
                        block_parts=None,
                        skipped=True,
                    )
                )
                self.skip_depth += 1
            return

        if identifier == "lastUpdate" or tag in SKIP_TAGS:
            self.stack.append(
                _ArticleFrame(tag=tag, block_parts=None, skipped=True)
            )
            self.skip_depth = 1
            return

        if tag in VOID_TAGS:
            if tag == "br":
                block = self._active_block()
                if block is None:
                    raise ArticleError("br is outside a retained block")
                self._flush(block)
            return

        if tag not in CONTAINER_TAGS | BLOCK_TAGS | INLINE_TAGS:
            raise ArticleError(f"unknown article tag: {tag}")
        if tag == "table" and any(frame.tag == "table" for frame in self.stack):
            raise ArticleError("nested table is forbidden")
        if tag == "tr":
            if not any(frame.tag == "table" for frame in self.stack):
                raise ArticleError("table row is outside a table")
            if any(frame.tag == "tr" for frame in self.stack):
                raise ArticleError("nested table row is forbidden")
        if tag in {"th", "td"}:
            if not self.stack or self.stack[-1].tag != "tr":
                raise ArticleError("table cell is outside a row")
            if any(frame.tag in {"th", "td"} for frame in self.stack):
                raise ArticleError("nested table cell is forbidden")
        if tag in BLOCK_TAGS:
            parent = self._active_block()
            if parent is not None:
                self._flush(parent)
            self.stack.append(
                _ArticleFrame(tag=tag, block_parts=[], skipped=False)
            )
        else:
            self.stack.append(
                _ArticleFrame(tag=tag, block_parts=None, skipped=False)
            )

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in VOID_TAGS:
            self.handle_starttag(tag, attrs)
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self.article_active or self.skip_depth:
            return
        block = self._active_block()
        if block is None:
            if _normalize_space(data):
                raise ArticleError("visible article text is outside a retained block")
            return
        assert block.block_parts is not None
        block.block_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.article_active:
            return
        if not self.stack or self.stack[-1].tag != tag:
            raise ArticleError("article tag nesting is malformed")
        frame = self.stack.pop()
        if frame.skipped:
            self.skip_depth -= 1
            if self.skip_depth < 0:
                raise ArticleError("article skip nesting underflowed")
        elif frame.block_parts is not None:
            self._flush(frame)
        if frame.article_root:
            if self.stack:
                raise ArticleError("article root closed with open descendants")
            self.article_active = False
            self.article_closed = True

    def canonical(self) -> tuple[str, int]:
        if self.article_active or self.stack or self.skip_depth:
            raise ArticleError("article has unclosed tags")
        if self.article_count != 1 or not self.article_closed:
            raise ArticleError("document does not have exactly one closed article")
        if not self.blocks:
            raise ArticleError("article canonical text is empty")
        return "\n".join(self.blocks) + "\n", len(self.blocks)


def canonicalize_article(text: str) -> tuple[str, int]:
    parser = _ArticleParser()
    try:
        parser.feed(text)
        parser.close()
        return parser.canonical()
    except SourceContractError as exc:
        if isinstance(exc, ArticleError):
            raise
        raise ArticleError("article HTML is malformed") from exc


def decode_document(raw: bytes, *, encoding: str) -> str:
    if encoding not in {"utf-8", "windows-1252"}:
        raise PageMetadataError("document encoding is unauthorized")
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise PageMetadataError("document fails strict frozen decoding") from exc


def summarize_document(
    row: LedgerRow,
    raw: bytes,
    *,
    canonicalizer: Callable[[str], tuple[str, int]] = canonicalize_article,
) -> tuple[DocumentSummary, bytes | None]:
    if len(raw) > HTML_BODY_CAP:
        raise PageMetadataError("document exceeds the raw body cap")
    text = decode_document(raw, encoding=row.encoding)
    metadata = parse_page_metadata(text)
    if metadata.last_update_date != row.last_update_date:
        raise PageMetadataError("embedded Last Update disagrees with ledger")
    if row.source_eligible and metadata.last_update_date != row.release_date:
        raise PageMetadataError("eligible page was updated after release")
    raw_hash = sha256_bytes(raw)
    if not row.source_eligible:
        return (
            DocumentSummary(
                document_class=row.document_class,
                source_eligible=False,
                raw_bytes=len(raw),
                raw_sha256=raw_hash,
                article_count=metadata.article_count,
                canonical_bytes=None,
                canonical_characters=None,
                canonical_blocks=None,
                canonical_sha256=None,
            ),
            None,
        )
    canonical, block_count = canonicalizer(text)
    encoded = canonical.encode("utf-8")
    return (
        DocumentSummary(
            document_class=row.document_class,
            source_eligible=True,
            raw_bytes=len(raw),
            raw_sha256=raw_hash,
            article_count=metadata.article_count,
            canonical_bytes=len(encoded),
            canonical_characters=len(canonical),
            canonical_blocks=block_count,
            canonical_sha256=sha256_bytes(encoded),
        ),
        encoded,
    )


def reconcile_indexes(
    ledger: Sequence[LedgerRow],
    entries: Sequence[IndexEntry],
) -> None:
    expected = [row.identity_tuple() for row in ledger]
    actual = [
        row.identity_tuple()
        for row in sorted(
            entries,
            key=lambda row: (
                row.release_date,
                row.document_class,
                row.official_url,
            ),
        )
    ]
    if actual != expected:
        raise IndexError("historical indexes do not reproduce the frozen ledger")


def evaluate_support(
    ledger: Sequence[LedgerRow],
    summaries: Sequence[DocumentSummary],
) -> dict[str, Any]:
    if len(summaries) != len(ledger):
        raise SupportError("document summary count changed")
    if any(
        summary.document_class != row.document_class
        or summary.source_eligible != row.source_eligible
        for row, summary in zip(ledger, summaries)
    ):
        raise SupportError("document summaries do not align with the ledger")
    candidate_counts = Counter(summary.document_class for summary in summaries)
    eligible_summaries = [summary for summary in summaries if summary.source_eligible]
    eligible_counts = Counter(
        summary.document_class for summary in eligible_summaries
    )
    encoding_counts = Counter(row.encoding for row in ledger)
    release_year_counts = Counter(row.release_date.year for row in ledger)
    if candidate_counts != Counter(EXPECTED_CANDIDATE_CLASS_COUNTS):
        raise SupportError("candidate class support changed")
    if eligible_counts != Counter(EXPECTED_ELIGIBLE_CLASS_COUNTS):
        raise SupportError("eligible class support changed")
    if encoding_counts != Counter({"utf-8": 146, "windows-1252": 1}):
        raise SupportError("encoding support changed")
    if release_year_counts != Counter(EXPECTED_RELEASE_YEAR_COUNTS):
        raise SupportError("release-year support changed")

    class_hashes: dict[str, set[str]] = {
        "fomc_statement": set(),
        "fomc_minutes": set(),
    }
    total_canonical_bytes = 0
    total_canonical_characters = 0
    total_blocks = 0
    for summary in eligible_summaries:
        if (
            summary.canonical_sha256 is None
            or summary.canonical_bytes is None
            or summary.canonical_characters is None
            or summary.canonical_blocks is None
        ):
            raise SupportError("eligible document lacks canonical support")
        if summary.document_class == "fomc_statement":
            lower, upper = 500, 50_000
        else:
            lower, upper = 10_000, 1_000_000
        if not lower <= summary.canonical_characters <= upper:
            raise SupportError("canonical document length is outside its class gate")
        hashes = class_hashes[summary.document_class]
        if summary.canonical_sha256 in hashes:
            raise SupportError("canonical text repeats within a document class")
        hashes.add(summary.canonical_sha256)
        total_canonical_bytes += summary.canonical_bytes
        total_canonical_characters += summary.canonical_characters
        total_blocks += summary.canonical_blocks
    quarantines = [summary for summary in summaries if not summary.source_eligible]
    if len(quarantines) != 2 or any(
        summary.canonical_sha256 is not None
        or summary.canonical_bytes is not None
        or summary.canonical_characters is not None
        or summary.canonical_blocks is not None
        for summary in quarantines
    ):
        raise SupportError("quarantine canonicalization boundary changed")

    transitions = {
        document_class: count - 1
        for document_class, count in sorted(eligible_counts.items())
    }
    if transitions != {"fomc_minutes": 71, "fomc_statement": 72}:
        raise SupportError("same-class transition support changed")

    return {
        "candidate_class_counts": dict(sorted(candidate_counts.items())),
        "candidate_count": len(summaries),
        "decision": "SOURCE_SUPPORT_PASS",
        "eligible_class_counts": dict(sorted(eligible_counts.items())),
        "eligible_count": len(eligible_summaries),
        "encoding_counts": dict(sorted(encoding_counts.items())),
        "quarantine_count": len(quarantines),
        "release_year_counts": {
            str(year): release_year_counts[year]
            for year in sorted(release_year_counts)
        },
        "same_class_transition_count": sum(transitions.values()),
        "same_class_transitions": transitions,
        "total_canonical_blocks": total_blocks,
        "total_canonical_bytes": total_canonical_bytes,
        "total_canonical_characters": total_canonical_characters,
        "total_raw_bytes": sum(summary.raw_bytes for summary in summaries),
    }


def _header_values(
    headers: Sequence[tuple[str, str]],
    name: str,
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


def _validate_success_payload(
    payload: HttpPayload,
    intent: RequestIntent,
) -> None:
    if payload.intent != intent:
        raise TransportError("transport changed the exact request intent")
    if payload.status != 200:
        raise TransportError("HTTP status is not 200")
    if payload.response_completed_at_utc < payload.request_started_at_utc:
        raise TransportError("HTTP receipt clock moved backward")
    if len(payload.raw) > HTML_BODY_CAP:
        raise TransportError("HTTP body exceeds the frozen cap")
    content_type = _single_header(payload.headers, "Content-Type", required=True)
    assert content_type is not None
    if content_type.split(";", 1)[0].strip().lower() != "text/html":
        raise TransportError("HTTP content type is not text/html")
    content_encoding = _single_header(
        payload.headers, "Content-Encoding", required=False
    )
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        raise TransportError("HTTP content encoding is not identity")
    if _single_header(payload.headers, "Content-Range", required=False) is not None:
        raise TransportError("partial HTTP response is forbidden")
    content_length = _single_header(
        payload.headers, "Content-Length", required=False
    )
    if content_length is not None:
        if not content_length.isdigit() or int(content_length) != len(payload.raw):
            raise TransportError("HTTP content length disagrees with body")


class Transport(Protocol):
    def request(self, intent: RequestIntent) -> HttpPayload: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class StdlibTransport:
    def __init__(self) -> None:
        context = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=context),
        )

    def request(self, intent: RequestIntent) -> HttpPayload:
        started = datetime.now(timezone.utc)
        request = urllib.request.Request(
            intent.url,
            data=None,
            headers=dict(intent.headers),
            method=intent.method,
        )
        try:
            with self._opener.open(
                request, timeout=intent.timeout_seconds
            ) as response:
                raw = response.read(HTML_BODY_CAP + 1)
                status = int(response.status)
                headers = tuple(response.headers.items())
        except urllib.error.HTTPError as exc:
            raw = exc.read(HTML_BODY_CAP + 1)
            status = int(exc.code)
            headers = tuple(exc.headers.items())
        except (socket.timeout, TimeoutError, http.client.RemoteDisconnected) as exc:
            raise RetryableTransportError("retryable HTTP transport failure") from exc
        except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
            cause = getattr(exc, "reason", None)
            if isinstance(cause, (socket.timeout, TimeoutError)):
                raise RetryableTransportError(
                    "retryable HTTP transport failure"
                ) from exc
            raise TransportError("non-retryable HTTP transport failure") from exc
        return HttpPayload(
            intent=intent,
            status=status,
            headers=headers,
            raw=raw,
            request_started_at_utc=started,
            response_completed_at_utc=datetime.now(timezone.utc),
        )


def request_with_retries(
    transport: Transport,
    intent: RequestIntent,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_response: Callable[[HttpPayload, int], None] | None = None,
    on_retry: Callable[[str, int, float], None] | None = None,
) -> HttpPayload:
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            payload = transport.request(intent)
        except RetryableTransportError as exc:
            last_error = exc
            if attempt >= 2:
                break
            delay = RETRY_DELAYS[attempt]
            if on_retry is not None:
                on_retry(type(exc).__name__, attempt + 1, delay)
            sleep(delay)
            continue
        if len(payload.raw) > HTML_BODY_CAP:
            raise TransportError("HTTP body exceeds the frozen cap")
        if on_response is not None:
            on_response(payload, attempt + 1)
        if payload.intent != intent:
            raise TransportError("transport changed the exact request intent")
        if payload.status in RETRYABLE_STATUSES:
            last_error = RetryableTransportError("retryable HTTP status")
            if attempt >= 2:
                break
            delay = RETRY_DELAYS[attempt]
            if on_retry is not None:
                on_retry(f"http_{payload.status}", attempt + 1, delay)
            sleep(delay)
            continue
        _validate_success_payload(payload, intent)
        return payload
    raise RetryableTransportError("retry budget exhausted") from last_error


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def validate_fixture_paths(paths: AuditPaths) -> AuditPaths:
    values = (paths.sentinel, paths.manifest, paths.raw_dir, paths.report)
    if any(not path.is_absolute() for path in values):
        raise ProtocolError("fixture paths must be absolute")
    production = (
        PRODUCTION_PATHS.sentinel,
        PRODUCTION_PATHS.manifest,
        PRODUCTION_PATHS.raw_dir,
        PRODUCTION_PATHS.report,
    )
    if any(
        _same_path(candidate, frozen)
        for candidate in values
        for frozen in production
    ):
        raise ProtocolError("fixture path aliases a production artifact")
    if len({path.resolve(strict=False) for path in values}) != len(values):
        raise ProtocolError("fixture paths alias each other")
    return paths


@dataclass(slots=True)
class AttemptGuard:
    paths: AuditPaths
    sentinel_sha256: str
    manifest_device: int
    manifest_inode: int
    sequence: int = 0
    previous_hash: str = "0" * 64
    raw_bytes_stored: int = 0
    response_count: int = 0

    def append(self, event: str, payload: Mapping[str, Any]) -> str:
        record_without_hash = {
            "event": event,
            "payload": dict(payload),
            "previous_hash": self.previous_hash,
            "protocol_version": PROTOCOL_VERSION,
            "sequence": self.sequence,
        }
        record_hash = sha256_bytes(
            canonical_json_bytes(record_without_hash, newline=False)
        )
        record = {**record_without_hash, "record_hash": record_hash}
        encoded = canonical_json_bytes(record)
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.paths.manifest, flags)
        metadata = os.fstat(descriptor)
        if (
            metadata.st_dev != self.manifest_device
            or metadata.st_ino != self.manifest_inode
        ):
            os.close(descriptor)
            raise PublicationError("manifest inode changed")
        with os.fdopen(descriptor, "ab", buffering=0) as handle:
            handle.write(encoded)
            os.fsync(handle.fileno())
        self.previous_hash = record_hash
        self.sequence += 1
        return record_hash


def _exclusive_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise ProtocolError("one-shot artifact already exists") from exc
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(payload)
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def reserve_attempt(
    *,
    paths: AuditPaths,
    verifier_commit: str,
    runner_blob: str,
) -> AttemptGuard:
    if not HEX40_RE.fullmatch(verifier_commit) or not HEX40_RE.fullmatch(runner_blob):
        raise ProtocolError("verifier binding is not a Git object identity")
    for value in (paths.sentinel, paths.manifest, paths.raw_dir, paths.report):
        if value.exists() or value.is_symlink():
            raise ProtocolError("one-shot path is already consumed")
    paths.sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel = canonical_json_bytes(
        {
            "protocol_version": PROTOCOL_VERSION,
            "runner_git_blob": runner_blob,
            "verifier_commit": verifier_commit,
        }
    )
    _exclusive_write(paths.sentinel, sentinel, mode=0o600)
    _exclusive_write(paths.manifest, b"", mode=0o600)
    paths.raw_dir.mkdir(mode=0o700)
    (paths.raw_dir / "raw").mkdir(mode=0o700)
    (paths.raw_dir / "canonical").mkdir(mode=0o700)
    manifest_metadata = paths.manifest.stat(follow_symlinks=False)
    return AttemptGuard(
        paths=paths,
        sentinel_sha256=sha256_bytes(sentinel),
        manifest_device=manifest_metadata.st_dev,
        manifest_inode=manifest_metadata.st_ino,
    )


def _artifact_path(
    guard: AttemptGuard,
    *,
    category: str,
    key: str,
    suffix: str,
) -> Path:
    if category not in {"raw", "canonical"} or SAFE_KEY_RE.fullmatch(key) is None:
        raise PublicationError("artifact identity is unsafe")
    if suffix not in {".html", ".txt"}:
        raise PublicationError("artifact suffix is unsafe")
    return guard.paths.raw_dir / category / f"{key}{suffix}"


def _persist_response(
    guard: AttemptGuard,
    payload: HttpPayload,
    *,
    kind: str,
    key: str,
    attempt: int,
) -> Path:
    if attempt not in {1, 2, 3}:
        raise PublicationError("HTTP attempt number is outside the retry budget")
    if len(payload.raw) > HTML_BODY_CAP:
        raise TransportError("HTTP body exceeds the frozen cap")
    if guard.raw_bytes_stored + len(payload.raw) > TOTAL_RAW_SOURCE_CAP:
        raise DiskGuardError("raw source bytes exceed 256 MiB")
    path = _artifact_path(
        guard,
        category="raw",
        key=f"{kind}-{key}-attempt-{attempt}",
        suffix=".html",
    )
    _exclusive_write(path, payload.raw)
    guard.raw_bytes_stored += len(payload.raw)
    guard.response_count += 1
    guard.append(
        "response_stored",
        {
            "attempt": attempt,
            "completed_at_utc": canonical_utc(payload.response_completed_at_utc),
            "headers_sha256": sha256_bytes(
                canonical_json_bytes(payload.headers, newline=False)
            ),
            "key": key,
            "kind": kind,
            "raw_bytes": len(payload.raw),
            "raw_path": path.relative_to(guard.paths.raw_dir).as_posix(),
            "raw_sha256": sha256_bytes(payload.raw),
            "started_at_utc": canonical_utc(payload.request_started_at_utc),
            "status": payload.status,
            "url": payload.intent.url,
        },
    )
    return path


def _persist_canonical(
    guard: AttemptGuard,
    canonical: bytes,
    *,
    key: str,
) -> Path:
    path = _artifact_path(
        guard,
        category="canonical",
        key=f"document-{key}",
        suffix=".txt",
    )
    _exclusive_write(path, canonical)
    return path


def _request_and_store(
    *,
    guard: AttemptGuard,
    transport: Transport,
    intent: RequestIntent,
    kind: str,
    key: str,
    first_request: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    if not first_request:
        sleep(FIRST_ATTEMPT_DELAY_SECONDS)
    guard.append(
        "request_intent",
        {
            "body_sha256": sha256_bytes(intent.body),
            "headers_sha256": sha256_bytes(
                canonical_json_bytes(intent.headers, newline=False)
            ),
            "key": key,
            "kind": kind,
            "method": intent.method,
            "url": intent.url,
        },
    )
    response_paths: list[Path] = []

    def observe_response(payload: HttpPayload, attempt: int) -> None:
        response_paths.append(
            _persist_response(
                guard,
                payload,
                kind=kind,
                key=key,
                attempt=attempt,
            )
        )

    def observe_retry(reason: str, attempt: int, delay: float) -> None:
        guard.append(
            "transport_retry",
            {
                "attempt": attempt,
                "delay_seconds": delay,
                "key": key,
                "kind": kind,
                "reason": reason,
            },
        )

    request_with_retries(
        transport,
        intent,
        sleep=sleep,
        on_response=observe_response,
        on_retry=observe_retry,
    )
    if not response_paths:
        raise TransportError("HTTP request completed without a response artifact")
    return response_paths[-1]


def _deduplicating_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("manifest JSON duplicates a key")
        result[key] = value
    return result


def _safe_manifest_artifact(root: Path, relative: str) -> Path:
    if not relative or relative.startswith("/") or "\\" in relative or "%" in relative:
        raise ProtocolError("manifest artifact path is unsafe")
    parts = Path(relative).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ProtocolError("manifest artifact path is unsafe")
    path = root.joinpath(*parts)
    if not path.resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise ProtocolError("manifest artifact escapes the raw directory")
    return path


def validate_manifest_chain(
    path: Path,
    *,
    raw_dir: Path,
    require_complete: bool,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("manifest is not a regular file")
    if raw_dir.is_symlink() or not raw_dir.is_dir():
        raise ProtocolError("raw artifact directory is invalid")
    raw = path.read_bytes()
    if len(raw) > MANIFEST_CAP:
        raise ProtocolError("manifest exceeds the frozen cap")
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ProtocolError("manifest is empty or lacks terminal newlines")

    previous_hash = "0" * 64
    raw_paths: set[Path] = set()
    canonical_paths: set[Path] = set()
    terminal_events: list[str] = []
    for sequence, line in enumerate(lines):
        try:
            record = json.loads(
                line,
                object_pairs_hook=_deduplicating_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ProtocolError(f"non-finite manifest value: {value}")
                ),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProtocolError("manifest JSON is invalid") from exc
        if not isinstance(record, dict):
            raise ProtocolError("manifest record is not an object")
        record_hash = record.pop("record_hash", None)
        if (
            record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("sequence") != sequence
            or record.get("previous_hash") != previous_hash
            or not isinstance(record_hash, str)
            or not HEX64_RE.fullmatch(record_hash)
            or sha256_bytes(canonical_json_bytes(record, newline=False))
            != record_hash
        ):
            raise ProtocolError("manifest hash chain is invalid")
        previous_hash = record_hash
        event = record.get("event")
        payload = record.get("payload")
        if not isinstance(event, str) or not isinstance(payload, dict):
            raise ProtocolError("manifest event structure is invalid")
        if event == "response_stored":
            artifact = _safe_manifest_artifact(raw_dir, payload.get("raw_path", ""))
            encoded = artifact.read_bytes()
            if (
                len(encoded) != payload.get("raw_bytes")
                or sha256_bytes(encoded) != payload.get("raw_sha256")
            ):
                raise ProtocolError("manifest raw artifact binding failed")
            raw_paths.add(artifact.resolve())
        elif event == "document_parsed":
            canonical_path = payload.get("canonical_path")
            if payload.get("source_eligible") is True:
                if not isinstance(canonical_path, str):
                    raise ProtocolError("eligible parser event lacks canonical path")
                artifact = _safe_manifest_artifact(raw_dir, canonical_path)
                encoded = artifact.read_bytes()
                if (
                    len(encoded) != payload.get("canonical_bytes")
                    or sha256_bytes(encoded) != payload.get("canonical_sha256")
                ):
                    raise ProtocolError("manifest canonical artifact binding failed")
                canonical_paths.add(artifact.resolve())
            elif canonical_path is not None:
                raise ProtocolError("quarantine parser event has canonical artifact")
        elif event in {"source_support_complete", "terminal_reject"}:
            terminal_events.append(event)

    if require_complete:
        if len(terminal_events) != 1 or terminal_events[0] != "source_support_complete":
            raise ProtocolError("manifest does not end in source support")
        if lines and json.loads(lines[-1])["event"] != "source_support_complete":
            raise ProtocolError("source-support event is not terminal")
    actual_raw = {
        path.resolve()
        for path in (raw_dir / "raw").glob("*")
        if path.is_file()
    }
    actual_canonical = {
        path.resolve()
        for path in (raw_dir / "canonical").glob("*")
        if path.is_file()
    }
    if actual_raw != raw_paths or actual_canonical != canonical_paths:
        raise ProtocolError("manifest does not bind the exact artifact set")
    return {
        "final_record_hash": previous_hash,
        "record_count": len(lines),
        "raw_artifact_count": len(raw_paths),
        "canonical_artifact_count": len(canonical_paths),
    }


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PublicationError("aggregate report already exists")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    _exclusive_write(temporary, payload)
    try:
        os.link(temporary, path, follow_symlinks=False)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as exc:
        raise PublicationError("aggregate report raced with another writer") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(values: Sequence[Any]) -> str:
    normalized = [
        value.aggregate_dict() if hasattr(value, "aggregate_dict") else value
        for value in values
    ]
    return sha256_bytes(canonical_json_bytes(normalized, newline=False))


def _outcome_boundary() -> dict[str, bool]:
    return {
        "article_text_emitted": False,
        "database_opened": False,
        "market_price_return_or_funding_opened": False,
        "model_tokenizer_adapter_prompt_or_checkpoint_opened": False,
        "portfolio_reward_or_performance_opened": False,
        "post_2020_document_body_opened": False,
        "semantic_label_embedding_or_inference_called": False,
    }


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(report)
    unsigned.pop("report_hash_without_self", None)
    report["report_hash_without_self"] = sha256_bytes(
        canonical_json_bytes(unsigned, newline=False)
    )
    return report


def build_report(
    *,
    execution_authority: str,
    source_audit_authoritative: bool,
    verifier_commit: str,
    runner_blob: str,
    manifest_sha256: str,
    sentinel_sha256: str,
    index_fingerprint: str,
    document_fingerprint: str,
    canonical_fingerprint: str,
    support: Mapping[str, Any],
) -> dict[str, Any]:
    decision = str(support["decision"])
    return _finalize_report(
        {
            "bindings": {
                "boundary_path": str(BOUNDARY_PATH),
                "boundary_sha256": BOUNDARY_SHA256,
                "ledger_path": str(LEDGER_PATH),
                "ledger_sha256": LEDGER_SHA256,
                "manifest_sha256": manifest_sha256,
                "protocol_version": PROTOCOL_VERSION,
                "runner_git_blob": runner_blob,
                "script_path": str(SCRIPT_PATH),
                "script_sha256": sha256_file(repository_path(SCRIPT_PATH)),
                "sentinel_sha256": sentinel_sha256,
                "test_path": str(TEST_PATH),
                "test_sha256": sha256_file(repository_path(TEST_PATH)),
                "verifier_commit": verifier_commit,
            },
            "decision": decision,
            "execution_authority": execution_authority,
            "mechanism_preregistration_authorized": (
                source_audit_authoritative and decision == "SOURCE_SUPPORT_PASS"
            ),
            "outcome_boundary": _outcome_boundary(),
            "retry_or_resume_authorized": False,
            "source_audit_authoritative": source_audit_authoritative,
            "source_hashes": {
                "canonical_fingerprint": canonical_fingerprint,
                "document_fingerprint": document_fingerprint,
                "index_fingerprint": index_fingerprint,
            },
            "support": dict(support),
        }
    )


def _failure_report(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
    stage: str,
    exception_class: str,
) -> dict[str, Any]:
    return _finalize_report(
        {
            "bindings": {
                "boundary_path": str(BOUNDARY_PATH),
                "boundary_sha256": BOUNDARY_SHA256,
                "ledger_path": str(LEDGER_PATH),
                "ledger_sha256": LEDGER_SHA256,
                "manifest_sha256": sha256_file(guard.paths.manifest),
                "protocol_version": PROTOCOL_VERSION,
                "runner_git_blob": runner_blob,
                "script_path": str(SCRIPT_PATH),
                "script_sha256": sha256_file(repository_path(SCRIPT_PATH)),
                "sentinel_sha256": guard.sentinel_sha256,
                "test_path": str(TEST_PATH),
                "test_sha256": sha256_file(repository_path(TEST_PATH)),
                "verifier_commit": verifier_commit,
            },
            "decision": "TERMINAL_REJECT",
            "execution_authority": "production_one_shot",
            "failure": {
                "exception_class": exception_class,
                "stage": stage,
            },
            "mechanism_preregistration_authorized": False,
            "outcome_boundary": _outcome_boundary(),
            "retry_or_resume_authorized": False,
            "source_audit_authoritative": True,
        }
    )


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(REPOSITORY_ROOT), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    return completed.stdout.decode("utf-8", errors="strict").strip()


def assert_protocol_committed() -> tuple[str, str]:
    if sha256_file(repository_path(BOUNDARY_PATH)) != BOUNDARY_SHA256:
        raise ProtocolError("boundary digest changed")
    if sha256_file(repository_path(LEDGER_PATH)) != LEDGER_SHA256:
        raise ProtocolError("ledger digest changed")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProtocolError("production source audit requires a clean worktree")
    verifier_commit = _git("rev-parse", "HEAD")
    if HEX40_RE.fullmatch(verifier_commit) is None:
        raise ProtocolError("HEAD is not a full Git commit")
    entries: dict[str, str] = {}
    for path in (BOUNDARY_PATH, LEDGER_PATH, SCRIPT_PATH, TEST_PATH):
        output = _git("ls-tree", "HEAD", "--", str(path))
        fields = output.split()
        if len(fields) < 4 or fields[1] != "blob":
            raise ProtocolError(f"committed protocol file missing: {path}")
        entries[str(path)] = fields[2]
    if any(HEX40_RE.fullmatch(value) is None for value in entries.values()):
        raise ProtocolError("protocol Git blob identity is invalid")
    for path in (BOUNDARY_PATH, LEDGER_PATH, SCRIPT_PATH, TEST_PATH):
        committed = subprocess.run(
            ["/usr/bin/git", "-C", str(REPOSITORY_ROOT), "show", f"HEAD:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
        ).stdout
        if committed != repository_path(path).read_bytes():
            raise ProtocolError(f"working protocol file differs from HEAD: {path}")
    return verifier_commit, entries[str(SCRIPT_PATH)]


def _disk_guard() -> tuple[int, int]:
    usage = shutil.disk_usage(REPOSITORY_ROOT)
    if usage.used >= DISK_USED_LIMIT:
        raise DiskGuardError("filesystem use is at or above 300 GiB")
    if usage.free < DISK_FREE_FLOOR:
        raise DiskGuardError("filesystem free space is below 8 GiB")
    return usage.used, usage.free


def run_production_audit() -> dict[str, Any]:
    verifier_commit, runner_blob = assert_protocol_committed()
    ledger = load_ledger()
    _disk_guard()
    guard = reserve_attempt(
        paths=PRODUCTION_PATHS,
        verifier_commit=verifier_commit,
        runner_blob=runner_blob,
    )
    transport = StdlibTransport()
    first_request = True
    stage = "historical_indexes"
    try:
        index_paths: list[tuple[str, Path]] = []
        entries: list[IndexEntry] = []
        for year, index_url in zip(INDEX_YEARS, INDEX_URLS):
            path = _request_and_store(
                guard=guard,
                transport=transport,
                intent=make_get_intent(index_url),
                kind="index",
                key=str(year),
                first_request=first_request,
            )
            first_request = False
            parsed = parse_historical_index(path.read_bytes(), index_url=index_url)
            index_paths.append((index_url, path))
            entries.extend(parsed)
            guard.append(
                "index_parsed",
                {
                    "candidate_count": len(parsed),
                    "index_year": year,
                    "raw_sha256": sha256_file(path),
                },
            )
        reconcile_indexes(ledger, entries)

        stage = "documents"
        document_paths: list[tuple[LedgerRow, Path, Path | None]] = []
        summaries: list[DocumentSummary] = []
        for position, row in enumerate(ledger, start=1):
            _disk_guard()
            key = f"{position:03d}"
            path = _request_and_store(
                guard=guard,
                transport=transport,
                intent=make_document_intent(row),
                kind="document",
                key=key,
                first_request=first_request,
            )
            first_request = False
            summary, canonical = summarize_document(row, path.read_bytes())
            canonical_path: Path | None = None
            if canonical is not None:
                canonical_path = _persist_canonical(guard, canonical, key=key)
            guard.append(
                "document_parsed",
                {
                    "article_count": summary.article_count,
                    "canonical_blocks": summary.canonical_blocks,
                    "canonical_bytes": summary.canonical_bytes,
                    "canonical_characters": summary.canonical_characters,
                    "canonical_path": (
                        canonical_path.relative_to(guard.paths.raw_dir).as_posix()
                        if canonical_path is not None
                        else None
                    ),
                    "canonical_sha256": summary.canonical_sha256,
                    "document_class": summary.document_class,
                    "key": key,
                    "last_update_matches_ledger": True,
                    "raw_sha256": summary.raw_sha256,
                    "source_eligible": summary.source_eligible,
                },
            )
            document_paths.append((row, path, canonical_path))
            summaries.append(summary)

        stage = "offline_replay"
        replay_entries: list[IndexEntry] = []
        for index_url, path in index_paths:
            replay_entries.extend(
                parse_historical_index(path.read_bytes(), index_url=index_url)
            )
        reconcile_indexes(ledger, replay_entries)
        replay_summaries: list[DocumentSummary] = []
        for row, path, canonical_path in document_paths:
            summary, canonical = summarize_document(row, path.read_bytes())
            if canonical_path is None:
                if canonical is not None:
                    raise SourceContractError("quarantine replay produced text")
            elif canonical != canonical_path.read_bytes():
                raise SourceContractError("canonical offline replay changed")
            replay_summaries.append(summary)
        if _fingerprint(replay_summaries) != _fingerprint(summaries):
            raise SourceContractError("document summary replay changed")

        stage = "support"
        support = evaluate_support(ledger, summaries)
        guard.append(
            "source_support_complete",
            {
                "candidate_count": support["candidate_count"],
                "decision": support["decision"],
                "eligible_count": support["eligible_count"],
                "same_class_transition_count": support[
                    "same_class_transition_count"
                ],
            },
        )
        manifest_audit = validate_manifest_chain(
            guard.paths.manifest,
            raw_dir=guard.paths.raw_dir,
            require_complete=True,
        )
        if (
            manifest_audit["final_record_hash"] != guard.previous_hash
            or manifest_audit["record_count"] != guard.sequence
            or manifest_audit["raw_artifact_count"] != guard.response_count
            or not 157 <= guard.response_count <= 471
            or manifest_audit["canonical_artifact_count"] != 145
        ):
            raise ProtocolError("manifest replay disagrees with writer state")
        report = build_report(
            execution_authority="production_one_shot",
            source_audit_authoritative=True,
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            manifest_sha256=sha256_file(guard.paths.manifest),
            sentinel_sha256=guard.sentinel_sha256,
            index_fingerprint=_fingerprint(
                [
                    {
                        "candidate_count": len(
                            parse_historical_index(
                                path.read_bytes(), index_url=index_url
                            )
                        ),
                        "raw_sha256": sha256_file(path),
                    }
                    for index_url, path in index_paths
                ]
            ),
            document_fingerprint=_fingerprint(summaries),
            canonical_fingerprint=_fingerprint(
                [
                    {
                        "canonical_bytes": summary.canonical_bytes,
                        "canonical_sha256": summary.canonical_sha256,
                    }
                    for summary in summaries
                    if summary.source_eligible
                ]
            ),
            support=support,
        )
    except BaseException as exc:
        guard.append(
            "terminal_reject",
            {
                "exception_class": type(exc).__name__,
                "stage": stage,
            },
        )
        report = _failure_report(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
            stage=stage,
            exception_class=type(exc).__name__,
        )
    _atomic_publish(PRODUCTION_PATHS.report, canonical_json_bytes(report))
    return report


def _sealed_child_environment(token: str) -> dict[str, str]:
    return {
        "FRDCL_ISOLATED_CHILD_TOKEN": token,
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen FOMC communication source-only audit."
    )
    parser.add_argument("--_isolated-child-token", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args._isolated_child_token is None:
        token = secrets.token_hex(32)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(repository_path(SCRIPT_PATH)),
                "--_isolated-child-token",
                token,
            ],
            cwd=REPOSITORY_ROOT,
            env=_sealed_child_environment(token),
        )
        return completed.returncode
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not HEX64_RE.fullmatch(args._isolated_child_token)
        or os.environ.get("FRDCL_ISOLATED_CHILD_TOKEN")
        != args._isolated_child_token
    ):
        raise ProtocolError("production requires the isolated CLI child")
    report = run_production_audit()
    print(f"FRDCL-D1 source audit: {report['decision']}")
    return 0 if report["decision"] == "SOURCE_SUPPORT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
