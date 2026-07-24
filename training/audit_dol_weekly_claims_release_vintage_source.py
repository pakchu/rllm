"""One-shot source-only audit for DOL weekly-claims release vintages.

The module uses only the Python standard library.  It validates the official
DOL newsroom teaser history and date-coded state-table archive while remaining
unable to inspect market, model, portfolio, or live-trading data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import html
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
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = Path(
    "docs/dol-weekly-claims-release-vintage-source-axis-decision-2026-07-24.md"
)
SCRIPT_PATH = Path(
    "training/audit_dol_weekly_claims_release_vintage_source.py"
)
TEST_PATH = Path(
    "tests/test_audit_dol_weekly_claims_release_vintage_source.py"
)
BOUNDARY_SHA256 = (
    "7119044d3278a4fc152131e90756cacd1cd28c5dc4b1614a98ff17b6cc8289fd"
)
PROTOCOL_VERSION = "DOL-WCRV-D1-source-audit-2026-07-24"

SCHEDULE_URL = "https://oui.doleta.gov/unemploy/claims_arch.asp"
ARCHIVE_URL = "https://oui.doleta.gov/unemploy/archive.asp"
NEWSROOM_URL = "https://www.dol.gov/newsroom/releases/eta"
ALLOWED_HOSTS = frozenset({"oui.doleta.gov", "www.dol.gov"})
SOURCE_START = date(2012, 1, 1)
SOURCE_END = date(2026, 1, 1)
NEW_YORK = ZoneInfo("America/New_York")

REQUEST_HEADERS: tuple[tuple[str, str], ...] = (
    (
        "User-Agent",
        "rllm-dol-wcrv-source-audit/1.0 "
        "(https://github.com/pakchu/rllm)",
    ),
    ("Accept", "text/html"),
    ("Accept-Encoding", "identity"),
)
HTTP_TIMEOUT_SECONDS = 30.0
RETRY_DELAYS = (1.0, 2.0)
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
HTML_BODY_CAP = 2 * 1024 * 1024
TOTAL_RAW_SOURCE_CAP = 256 * 1024 * 1024
DISK_USED_LIMIT = 300 * 1024**3
DISK_FREE_FLOOR = 8 * 1024**3

FROZEN_MALFORMED_PATHS: dict[int, tuple[str, ...]] = {
    2012: ("/unemploy/page8/2012/022312.html",),
    2023: ("/unemploy/page8/2023/0223.html",),
}
FROZEN_MALFORMED_2023_PATH = FROZEN_MALFORMED_PATHS[2023][0]
EXPECTED_STATE_TABLE_COUNTS: dict[int, int] = {
    2012: 52,
    2013: 52,
    2014: 52,
    2015: 52,
    2016: 53,
    2017: 52,
    2018: 52,
    2019: 23,
    2020: 52,
    2021: 52,
    2022: 52,
    2023: 52,
    2024: 52,
    2025: 44,
}

JURISDICTIONS: tuple[str, ...] = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Puerto Rico",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virgin Islands",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)
JURISDICTION_SET = frozenset(JURISDICTIONS)
RATE_COLUMN = 6
NONNEGATIVE_COLUMNS = frozenset({0, 3, 4, 5, 9, 10, 11})
SUMMED_COLUMNS = tuple(index for index in range(12) if index != RATE_COLUMN)

AUTHORIZED_TITLES = frozenset(
    {
        "Unemployment Insurance Weekly Claims Report",
        "ETA Press Release: Unemployment Insurance Weekly Claims Report",
    }
)

DEFAULT_SENTINEL = REPOSITORY_ROOT / (
    "results/.dol_weekly_claims_release_vintage_source_2026-07-24.started"
)
DEFAULT_MANIFEST = REPOSITORY_ROOT / (
    "results/.dol_weekly_claims_release_vintage_source_2026-07-24.manifest.ndjson"
)
DEFAULT_RAW_DIR = REPOSITORY_ROOT / (
    "results/.dol_weekly_claims_release_vintage_source_2026-07-24.raw"
)
DEFAULT_REPORT = REPOSITORY_ROOT / (
    "results/dol_weekly_claims_release_vintage_source_2026-07-24.json"
)

HEX40_RE = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
STATE_PATH_RE = re.compile(
    r"/unemploy/page8/(?P<year>\d{4})/(?P<stamp>\d{6})\.html\Z",
    re.ASCII,
)
NEWSROOM_PATH_RE = re.compile(
    r"/newsroom/releases/eta/eta(?P<stamp>\d{8})(?:-(?P<suffix>0|[1-9]\d*))?\Z",
    re.ASCII,
)
INT_RE = re.compile(r"-?(?:0|[1-9]\d{0,2}(?:,\d{3})*|[1-9]\d*)\Z", re.ASCII)
RATE_RE = re.compile(r"(?:0|[1-9]\d{0,2})(?:\.\d+)?\Z", re.ASCII)


class AuditError(RuntimeError):
    """Base class for frozen DOL source-audit failures."""


class ProtocolError(AuditError):
    """The committed one-shot protocol precondition failed."""


class DiskGuardError(ProtocolError):
    """The repository filesystem is outside the frozen storage boundary."""


class TransportError(AuditError):
    """An HTTP response violated the frozen transport contract."""


class RetryableTransportError(TransportError):
    """A frozen retry condition occurred."""


class SourceContractError(AuditError):
    """An official source page violated its frozen source contract."""


class NewsroomError(SourceContractError):
    """A newsroom page or teaser identity was invalid."""


class ArithmeticContractError(NewsroomError):
    """Disclosed national arithmetic did not reconcile."""


class StateInventoryError(SourceContractError):
    """A state-table archive calendar violated the frozen inventory."""


class StateTableError(SourceContractError):
    """A date-coded state table violated the frozen schema."""


class SupportError(AuditError):
    """The frozen source panel failed a support gate."""


class PublicationError(AuditError):
    """A local immutable artifact could not be published safely."""


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
class NationalRelease:
    node_id: int
    path: str
    release_date: date
    availability_at_utc: datetime
    week_ending: date
    current_initial_claims: int
    signed_change: int
    prior_status: str
    prior_before: int | None
    prior_after: int
    current_four_week_average: int
    signed_average_change: int
    prior_average_before: int | None
    prior_average_after: int
    teaser_sha256: str
    source_page_sha256: str
    grammar_variant: str

    def private_dict(self) -> dict[str, Any]:
        return {
            "availability_at_utc": canonical_utc(self.availability_at_utc),
            "current_four_week_average": self.current_four_week_average,
            "current_initial_claims": self.current_initial_claims,
            "grammar_variant": self.grammar_variant,
            "node_id": self.node_id,
            "path": self.path,
            "prior_after": self.prior_after,
            "prior_average_after": self.prior_average_after,
            "prior_average_before": self.prior_average_before,
            "prior_before": self.prior_before,
            "prior_status": self.prior_status,
            "release_date": self.release_date.isoformat(),
            "signed_average_change": self.signed_average_change,
            "signed_change": self.signed_change,
            "source_page_sha256": self.source_page_sha256,
            "teaser_sha256": self.teaser_sha256,
            "week_ending": self.week_ending.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class NewsroomPage:
    page_number: int
    raw_sha256: str
    record_count: int
    opaque_postwindow_count: int
    selected_count: int
    minimum_displayed_date: date
    maximum_displayed_date: date
    next_page: int | None
    releases: tuple[NationalRelease, ...]

    def aggregate_dict(self) -> dict[str, Any]:
        return {
            "opaque_postwindow_count": self.opaque_postwindow_count,
            "page_number": self.page_number,
            "raw_sha256": self.raw_sha256,
            "record_count": self.record_count,
            "selected_count": self.selected_count,
        }


@dataclass(frozen=True, slots=True)
class StateArtifact:
    year: int
    week_ending: date
    path: str
    url: str


@dataclass(frozen=True, slots=True)
class StateInventory:
    year: int
    artifacts: tuple[StateArtifact, ...]
    malformed_link_count: int
    raw_sha256: str
    structural_sha256: str


@dataclass(frozen=True, slots=True)
class StateTableSummary:
    week_ending: date
    insured_week_ending: date
    path: str
    raw_sha256: str
    structural_sha256: str
    jurisdiction_count: int
    arithmetic_columns_reconciled: int
    grammar_variant: str

    def private_dict(self) -> dict[str, Any]:
        return {
            "arithmetic_columns_reconciled": self.arithmetic_columns_reconciled,
            "grammar_variant": self.grammar_variant,
            "insured_week_ending": self.insured_week_ending.isoformat(),
            "jurisdiction_count": self.jurisdiction_count,
            "path": self.path,
            "raw_sha256": self.raw_sha256,
            "structural_sha256": self.structural_sha256,
            "week_ending": self.week_ending.isoformat(),
        }


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


def historical_availability(release_date: date) -> datetime:
    local = datetime.combine(
        release_date,
        datetime_time(hour=9),
        tzinfo=NEW_YORK,
    )
    return local.astimezone(timezone.utc)


def previous_saturday(value: date) -> date:
    return value - timedelta(days=(value.weekday() - 5) % 7)


def _strict_utf8(raw: bytes, *, label: str) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceContractError(f"{label} is not strict UTF-8") from exc
    if "\x00" in text or "\ufffd" in text:
        raise SourceContractError(f"{label} contains a forbidden code point")
    return text


def _validate_utf8_bytes(raw: bytes, *, label: str) -> None:
    """Validate RFC 3629 UTF-8 without decoding opaque body spans to text."""

    index = 0
    length = len(raw)
    while index < length:
        first = raw[index]
        if first == 0:
            raise SourceContractError(f"{label} contains NUL")
        if first <= 0x7F:
            index += 1
            continue
        if 0xC2 <= first <= 0xDF:
            width = 2
            second_bounds = (0x80, 0xBF)
        elif first == 0xE0:
            width = 3
            second_bounds = (0xA0, 0xBF)
        elif 0xE1 <= first <= 0xEC or 0xEE <= first <= 0xEF:
            width = 3
            second_bounds = (0x80, 0xBF)
        elif first == 0xED:
            width = 3
            second_bounds = (0x80, 0x9F)
        elif first == 0xF0:
            width = 4
            second_bounds = (0x90, 0xBF)
        elif 0xF1 <= first <= 0xF3:
            width = 4
            second_bounds = (0x80, 0xBF)
        elif first == 0xF4:
            width = 4
            second_bounds = (0x80, 0x8F)
        else:
            raise SourceContractError(f"{label} is not strict UTF-8")
        if index + width > length:
            raise SourceContractError(f"{label} is truncated UTF-8")
        second = raw[index + 1]
        if not second_bounds[0] <= second <= second_bounds[1]:
            raise SourceContractError(f"{label} is not strict UTF-8")
        if any(not 0x80 <= raw[offset] <= 0xBF for offset in range(index + 2, index + width)):
            raise SourceContractError(f"{label} is not strict UTF-8")
        index += width
    if b"\xef\xbf\xbd" in raw:
        raise SourceContractError(f"{label} contains a replacement code point")


_TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
_SPACE_RE = re.compile(r"\s+")


def _normalize_html_fragment(raw: bytes, *, label: str) -> str:
    text = _strict_utf8(raw, label=label)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = text.replace("\u2019", "'").replace("\u00a0", " ")
    return _SPACE_RE.sub(" ", text).strip()


def parse_schedule_page(raw: bytes) -> dict[str, str]:
    text = _normalize_html_fragment(raw, label="schedule page")
    required = (
        "The UI Weekly Claims News Release is published each week on "
        "Thursday morning at 8:30am EST."
    )
    holiday = (
        "Exceptions to this schedule occur when a Thursday falls on a "
        "Federal Holiday."
    )
    if required not in text or holiday not in text:
        raise SourceContractError("official release schedule statement changed")
    return {
        "normal_release_time": "08:30 America/New_York",
        "raw_sha256": sha256_bytes(raw),
    }


def _parse_full_date(value: str) -> date:
    normalized = _SPACE_RE.sub(" ", value.replace(",", " ")).strip()
    for format_value in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(normalized, format_value).date()
        except ValueError:
            pass
    raise NewsroomError("displayed date is not an exact English calendar date")


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _parse_observation_date(value: str, *, floor: date) -> date:
    normalized = (
        value.replace(".", "").replace(",", " ").replace("\u00a0", " ")
    )
    parts = _SPACE_RE.sub(" ", normalized).strip().split(" ")
    if len(parts) not in {2, 3}:
        raise SourceContractError("observation date grammar changed")
    month = MONTHS.get(parts[0].lower())
    if month is None or not parts[1].isdigit():
        raise SourceContractError("observation date is invalid")
    year = int(parts[2]) if len(parts) == 3 and parts[2].isdigit() else floor.year
    try:
        result = date(year, month, int(parts[1]))
    except ValueError as exc:
        raise SourceContractError("observation date is not Gregorian") from exc
    if len(parts) == 2 and result >= floor:
        result = date(year - 1, month, int(parts[1]))
    if not (timedelta(days=1) <= floor - result <= timedelta(days=14)):
        raise SourceContractError("observation date is outside the causal release lag")
    return result


def _parse_integer(value: str, *, allow_negative: bool) -> int:
    normalized = value.strip()
    if INT_RE.fullmatch(normalized) is None:
        raise SourceContractError("integer lexeme is not canonical")
    result = int(normalized.replace(",", ""))
    if not allow_negative and result < 0:
        raise SourceContractError("non-negative integer is negative")
    return result


def _signed_phrase(direction: str, magnitude: str | None) -> int:
    normalized = direction.lower()
    if normalized == "unchanged":
        if magnitude not in {None, "0"}:
            raise ArithmeticContractError("unchanged phrase has a magnitude")
        return 0
    if magnitude is None:
        raise ArithmeticContractError("change phrase lacks a magnitude")
    value = _parse_integer(magnitude, allow_negative=False)
    if normalized == "an increase":
        return value
    if normalized == "a decrease":
        return -value
    raise ArithmeticContractError("change direction is invalid")


_WEEK_RE = re.compile(
    r"\bIn the week ending "
    r"(?P<date>[A-Za-z]+\.? \d{1,2}(?:,? \d{4})?)\b",
    re.IGNORECASE,
)
_INITIAL_RE = re.compile(
    r"advance figure for seasonally adjusted initial claims was "
    r"(?P<current>[0-9][0-9,]*), "
    r"(?P<direction>an increase|a decrease|unchanged)"
    r"(?: of (?P<magnitude>[0-9][0-9,]*))? from the previous week's "
    r"(?P<status>revised|unrevised) (?:level|figure)"
    r"(?: of (?P<comparison>[0-9][0-9,]*))?",
    re.IGNORECASE,
)
_AVERAGE_RE = re.compile(
    r"(?:The )?4-week moving average was "
    r"(?P<current>[0-9][0-9,]*), "
    r"(?P<direction>an increase|a decrease|unchanged)"
    r"(?: of (?P<magnitude>[0-9][0-9,]*))? from the previous week's "
    r"(?P<status>revised|unrevised) average"
    r"(?: of (?P<comparison>[0-9][0-9,]*))?",
    re.IGNORECASE,
)
_PRIOR_REVISION_RE = re.compile(
    r"previous week's (?:level|figure) was revised"
    r"(?: (?P<direction>up|down) by (?P<magnitude>[0-9][0-9,]*))?"
    r" from (?P<before>[0-9][0-9,]*) to (?P<after>[0-9][0-9,]*)",
    re.IGNORECASE,
)
_AVERAGE_REVISION_RE = re.compile(
    r"previous week's average was revised"
    r"(?: (?P<direction>up|down) by (?P<magnitude>[0-9][0-9,]*))?"
    r" from (?P<before>[0-9][0-9,]*) to (?P<after>[0-9][0-9,]*)",
    re.IGNORECASE,
)


def _revision_endpoints(
    text: str,
    pattern: re.Pattern[str],
) -> tuple[int | None, int | None]:
    match = pattern.search(text)
    if match is None:
        return None, None
    before = _parse_integer(match.group("before"), allow_negative=False)
    after = _parse_integer(match.group("after"), allow_negative=False)
    direction = match.group("direction")
    magnitude = match.group("magnitude")
    if direction is not None:
        expected = after - before
        disclosed = _parse_integer(magnitude, allow_negative=False)
        if direction.lower() == "down":
            disclosed = -disclosed
        if expected != disclosed:
            raise ArithmeticContractError("revision endpoints do not reconcile")
    return before, after


def parse_national_teaser(
    raw: bytes,
    *,
    release_date: date,
    node_id: int,
    path: str,
    source_page_sha256: str,
) -> NationalRelease:
    text = _normalize_html_fragment(raw, label="national teaser")
    week_match = _WEEK_RE.search(text)
    initial_match = _INITIAL_RE.search(text)
    average_match = _AVERAGE_RE.search(text)
    if week_match is None or initial_match is None or average_match is None:
        raise NewsroomError("national teaser grammar changed")

    week_ending = _parse_observation_date(
        week_match.group("date"),
        floor=release_date,
    )
    current = _parse_integer(
        initial_match.group("current"),
        allow_negative=False,
    )
    signed_change = _signed_phrase(
        initial_match.group("direction"),
        initial_match.group("magnitude"),
    )
    prior_status = initial_match.group("status").lower()
    comparison_literal = initial_match.group("comparison")
    comparison = (
        _parse_integer(comparison_literal, allow_negative=False)
        if comparison_literal is not None
        else None
    )
    prior_before, revision_after = _revision_endpoints(
        text,
        _PRIOR_REVISION_RE,
    )
    if comparison is not None and revision_after is not None:
        if comparison != revision_after:
            raise ArithmeticContractError("prior comparison disagrees with revision")
    prior_after = comparison if comparison is not None else revision_after
    if prior_after is None:
        raise NewsroomError("prior comparison value is unavailable")
    if current - prior_after != signed_change:
        raise ArithmeticContractError("initial-claims change does not reconcile")

    current_average = _parse_integer(
        average_match.group("current"),
        allow_negative=False,
    )
    signed_average_change = _signed_phrase(
        average_match.group("direction"),
        average_match.group("magnitude"),
    )
    average_comparison_literal = average_match.group("comparison")
    average_comparison = (
        _parse_integer(average_comparison_literal, allow_negative=False)
        if average_comparison_literal is not None
        else None
    )
    prior_average_before, average_revision_after = _revision_endpoints(
        text,
        _AVERAGE_REVISION_RE,
    )
    if average_comparison is not None and average_revision_after is not None:
        if average_comparison != average_revision_after:
            raise ArithmeticContractError("average comparison disagrees with revision")
    prior_average_after = (
        average_comparison
        if average_comparison is not None
        else average_revision_after
    )
    if prior_average_after is None:
        raise NewsroomError("prior four-week comparison value is unavailable")
    if current_average - prior_average_after != signed_average_change:
        raise ArithmeticContractError("four-week change does not reconcile")

    if NEWSROOM_PATH_RE.fullmatch(path) is None:
        raise NewsroomError("newsroom path is noncanonical")
    if not HEX64_RE.fullmatch(source_page_sha256):
        raise NewsroomError("source-page digest is invalid")
    return NationalRelease(
        node_id=node_id,
        path=path,
        release_date=release_date,
        availability_at_utc=historical_availability(release_date),
        week_ending=week_ending,
        current_initial_claims=current,
        signed_change=signed_change,
        prior_status=prior_status,
        prior_before=prior_before,
        prior_after=prior_after,
        current_four_week_average=current_average,
        signed_average_change=signed_average_change,
        prior_average_before=prior_average_before,
        prior_average_after=prior_average_after,
        teaser_sha256=sha256_bytes(raw),
        source_page_sha256=source_page_sha256,
        grammar_variant=(
            "explicit_revision_endpoints"
            if prior_before is not None and prior_average_before is not None
            else "comparison_only"
        ),
    )


_RECORD_START_RE = re.compile(
    rb"<div\b(?=[^>]*\bdata-history-node-id=\"[0-9]+\")"
    rb"(?=[^>]*\babout=\"[^\"]+\")"
    rb"(?=[^>]*\bclass=\"left-teaser-text\")[^>]*>",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(
    rb"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)=\"(?P<value>[^\"]*)\""
)
_DATE_FRAGMENT_RE = re.compile(
    rb"<p\b[^>]*\bclass=\"dol-date-text\"[^>]*>(?P<value>.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_FRAGMENT_RE = re.compile(
    rb"<h3\b[^>]*>\s*<span\b[^>]*>(?P<value>.*?)</span>\s*</h3>",
    re.IGNORECASE | re.DOTALL,
)
_BODY_FRAGMENT_RE = re.compile(
    rb"<div\b[^>]*\bclass=\"[^\"]*field--name-field-press-body[^\"]*\""
    rb"[^>]*>\s*<p\b[^>]*>(?P<value>.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
_ANCHOR_RE = re.compile(rb"<a\b(?P<tag>[^>]*)>", re.IGNORECASE | re.DOTALL)


def _attributes(tag: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _ATTRIBUTE_RE.finditer(tag):
        name = match.group("name").decode("ascii").lower()
        if name in result:
            raise NewsroomError("HTML tag duplicates an attribute")
        try:
            value = match.group("value").decode("ascii")
        except UnicodeDecodeError as exc:
            raise NewsroomError("structural attribute is not ASCII") from exc
        result[name] = html.unescape(value)
    return result


def _parse_next_page(raw: bytes, *, expected_page: int) -> int | None:
    candidates: list[dict[str, str]] = []
    for match in _ANCHOR_RE.finditer(raw):
        attrs = _attributes(match.group("tag"))
        rel = attrs.get("rel")
        if rel is not None and "next" in rel.split():
            candidates.append(attrs)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise NewsroomError("newsroom page has ambiguous next links")
    attrs = candidates[0]
    if attrs.get("rel") != "next":
        raise NewsroomError("next-link relation is not exact")
    expected = expected_page + 1
    if attrs.get("href") != f"?page={expected}":
        raise NewsroomError("next-link page chain is noncanonical")
    return expected


def parse_newsroom_page(raw: bytes, *, expected_page: int) -> NewsroomPage:
    if type(expected_page) is not int or expected_page < 0:
        raise NewsroomError("expected page number is invalid")
    if len(raw) > HTML_BODY_CAP:
        raise NewsroomError("newsroom page exceeds the frozen cap")
    _validate_utf8_bytes(raw, label="newsroom page")
    raw_sha = sha256_bytes(raw)
    starts = list(_RECORD_START_RE.finditer(raw))
    if not starts or len(starts) > 40:
        raise NewsroomError("newsroom record count is outside the frozen boundary")

    releases: list[NationalRelease] = []
    displayed_dates: list[date] = []
    node_ids: set[int] = set()
    paths: set[str] = set()
    release_dates: set[date] = set()
    opaque_count = 0
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(raw)
        block = raw[start.start() : end]
        attrs = _attributes(start.group(0))
        try:
            node_id = int(attrs["data-history-node-id"])
            path = attrs["about"]
        except (KeyError, ValueError) as exc:
            raise NewsroomError("newsroom identity attributes are invalid") from exc
        path_match = NEWSROOM_PATH_RE.fullmatch(path)
        if path_match is None:
            raise NewsroomError("newsroom about path is noncanonical")
        date_fragment = _DATE_FRAGMENT_RE.search(block)
        title_fragment = _TITLE_FRAGMENT_RE.search(block)
        if date_fragment is None or title_fragment is None:
            raise NewsroomError("newsroom record lacks date or title")
        displayed = _parse_full_date(
            _normalize_html_fragment(
                date_fragment.group("value"),
                label="displayed release date",
            )
        )
        title = _normalize_html_fragment(
            title_fragment.group("value"),
            label="newsroom title",
        )
        path_date = datetime.strptime(path_match.group("stamp"), "%Y%m%d").date()
        if path_date != displayed:
            raise NewsroomError("newsroom path date disagrees with displayed date")
        if node_id in node_ids or path in paths:
            raise NewsroomError("newsroom page duplicates node or path")
        node_ids.add(node_id)
        paths.add(path)
        displayed_dates.append(displayed)

        if title not in AUTHORIZED_TITLES:
            continue
        if displayed in release_dates:
            raise NewsroomError("weekly release date is duplicated")
        release_dates.add(displayed)
        body_match = _BODY_FRAGMENT_RE.search(block)
        if body_match is None:
            raise NewsroomError("weekly release lacks teaser-body span")
        if displayed >= SOURCE_END:
            opaque_count += 1
            continue
        if displayed < SOURCE_START:
            continue
        releases.append(
            parse_national_teaser(
                body_match.group("value"),
                release_date=displayed,
                node_id=node_id,
                path=path,
                source_page_sha256=raw_sha,
            )
        )

    if any(current > previous for previous, current in zip(displayed_dates, displayed_dates[1:])):
        raise NewsroomError("newsroom page is not reverse chronological")
    return NewsroomPage(
        page_number=expected_page,
        raw_sha256=raw_sha,
        record_count=len(starts),
        opaque_postwindow_count=opaque_count,
        selected_count=len(releases),
        minimum_displayed_date=min(displayed_dates),
        maximum_displayed_date=max(displayed_dates),
        next_page=_parse_next_page(raw, expected_page=expected_page),
        releases=tuple(releases),
    )


def validate_newsroom_chain(pages: Sequence[NewsroomPage]) -> None:
    if not pages or pages[0].page_number != 0:
        raise NewsroomError("newsroom chain does not start at page zero")
    for index, page in enumerate(pages):
        if page.page_number != index:
            raise NewsroomError("newsroom page sequence is not contiguous")
        if index + 1 < len(pages):
            following = pages[index + 1]
            if page.next_page != following.page_number:
                raise NewsroomError("newsroom next-link chain changed")
            if following.maximum_displayed_date > page.minimum_displayed_date:
                raise NewsroomError("newsroom chronology overlaps across pages")
            if page.minimum_displayed_date < SOURCE_START:
                raise NewsroomError("newsroom crawl continued after crossing 2012")
    if pages[-1].minimum_displayed_date >= SOURCE_START:
        raise NewsroomError("newsroom chain ended before crossing 2012")


class _LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        values = dict(attrs)
        href = values.get("href")
        if href is not None:
            self.links.append(href)


def parse_state_inventory(raw: bytes, *, requested_year: int) -> StateInventory:
    if requested_year not in EXPECTED_STATE_TABLE_COUNTS:
        raise StateInventoryError("state inventory year is unauthorized")
    if len(raw) > HTML_BODY_CAP:
        raise StateInventoryError("state inventory exceeds the frozen cap")
    text = _strict_utf8(raw, label="state inventory")
    parser = _LinkParser()
    parser.feed(text)
    parser.close()

    prefix = f"/unemploy/page8/{requested_year}/"
    page8_references = [
        value
        for value in parser.links
        if "page8" in urllib.parse.unquote(value).lower()
    ]
    if any(
        not value.startswith("/unemploy/page8/")
        for value in page8_references
    ):
        raise StateInventoryError("state inventory has a non-relative page8 link")
    all_page8 = page8_references
    if any(not value.startswith(prefix) for value in all_page8):
        raise StateInventoryError("state inventory links outside requested year")
    relevant = all_page8
    valid_dates: dict[str, date] = {}
    malformed: list[str] = []
    for value in relevant:
        match = STATE_PATH_RE.fullmatch(value)
        if match is None:
            malformed.append(value)
            continue
        try:
            parsed_date = datetime.strptime(match.group("stamp"), "%m%d%y").date()
        except ValueError:
            malformed.append(value)
            continue
        if (
            int(match.group("year")) != requested_year
            or parsed_date.year != requested_year
            or parsed_date.weekday() != 5
        ):
            malformed.append(value)
            continue
        valid_dates[value] = parsed_date
    expected_malformed = list(FROZEN_MALFORMED_PATHS.get(requested_year, ()))
    if malformed != expected_malformed:
        raise StateInventoryError("state inventory malformed-link set changed")

    artifacts: list[StateArtifact] = []
    seen_paths: set[str] = set()
    seen_dates: set[date] = set()
    for path in relevant:
        if path not in valid_dates:
            continue
        week = valid_dates[path]
        if path in seen_paths or week in seen_dates:
            raise StateInventoryError("state inventory duplicates a selected artifact")
        seen_paths.add(path)
        seen_dates.add(week)
        artifacts.append(
            StateArtifact(
                year=requested_year,
                week_ending=week,
                path=path,
                url=f"https://oui.doleta.gov{path}",
            )
        )
    artifacts.sort(key=lambda row: (row.week_ending, row.path))
    expected_count = EXPECTED_STATE_TABLE_COUNTS[requested_year]
    if len(artifacts) != expected_count:
        raise StateInventoryError("state inventory count changed")
    structural = {
        "artifacts": [
            {
                "path": row.path,
                "week_ending": row.week_ending.isoformat(),
            }
            for row in artifacts
        ],
        "malformed_link_count": len(malformed),
        "year": requested_year,
    }
    return StateInventory(
        year=requested_year,
        artifacts=tuple(artifacts),
        malformed_link_count=len(malformed),
        raw_sha256=sha256_bytes(raw),
        structural_sha256=sha256_bytes(
            canonical_json_bytes(structural, newline=False)
        ),
    )


class _TableParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.all_text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized = tag.lower()
        if normalized == "tr":
            if self._row is not None:
                raise StateTableError("state table nests rows")
            self._row = []
        elif normalized in {"th", "td"} and self._row is not None:
            if self._cell is not None:
                raise StateTableError("state table nests cells")
            self._cell = []

    def handle_data(self, data: str) -> None:
        self.all_text.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"th", "td"} and self._cell is not None:
            assert self._row is not None
            value = _SPACE_RE.sub(
                " ",
                html.unescape("".join(self._cell)).replace("\u00a0", " "),
            ).strip()
            self._row.append(value)
            self._cell = None
        elif normalized == "tr" and self._row is not None:
            if self._cell is not None:
                raise StateTableError("state table row closes inside a cell")
            self.rows.append(self._row)
            self._row = None

    def close(self) -> None:
        super().close()
        if self._row is not None or self._cell is not None:
            raise StateTableError("state table ends with an open row or cell")


_INITIAL_HEADING_RE = re.compile(
    r"INITIAL CLAIMS FILED DURING WEEK ENDED "
    r"(?P<date>[A-Za-z]+ \d{1,2}(?:,? \d{4})?)",
    re.IGNORECASE,
)
_INSURED_HEADING_RE = re.compile(
    r"INSURED UNEMPLOYMENT FOR WEEK ENDED "
    r"(?P<date>[A-Za-z]+ \d{1,2}(?:,? \d{4})?)",
    re.IGNORECASE,
)


def _parse_rate_tenths(value: str) -> int:
    if RATE_RE.fullmatch(value.strip()) is None:
        raise StateTableError("insured-unemployment rate is noncanonical")
    try:
        scaled = Decimal(value.strip()) * Decimal(10)
    except InvalidOperation as exc:
        raise StateTableError("insured-unemployment rate is invalid") from exc
    if scaled != scaled.to_integral_value():
        raise StateTableError("insured-unemployment rate has excess precision")
    result = int(scaled)
    if not 0 <= result <= 1_000:
        raise StateTableError("insured-unemployment rate is outside [0, 100]")
    return result


def _parse_state_values(values: Sequence[str]) -> tuple[int, ...]:
    if len(values) != 12:
        raise StateTableError("state row does not contain twelve numeric cells")
    parsed: list[int] = []
    for index, value in enumerate(values):
        if index == RATE_COLUMN:
            parsed.append(_parse_rate_tenths(value))
            continue
        try:
            number = _parse_integer(value, allow_negative=True)
        except SourceContractError as exc:
            raise StateTableError("state numeric cell is invalid") from exc
        if index in NONNEGATIVE_COLUMNS and number < 0:
            raise StateTableError("state count/level is negative")
        parsed.append(number)
    return tuple(parsed)


def parse_state_table(raw: bytes, *, path: str) -> StateTableSummary:
    if len(raw) > HTML_BODY_CAP:
        raise StateTableError("state table exceeds the frozen cap")
    text = _strict_utf8(raw, label="state table")
    path_match = STATE_PATH_RE.fullmatch(path)
    if path_match is None:
        raise StateTableError("state table path is noncanonical")
    try:
        path_week = datetime.strptime(path_match.group("stamp"), "%m%d%y").date()
    except ValueError as exc:
        raise StateTableError("state table path date is invalid") from exc
    if path_week.year != int(path_match.group("year")) or path_week.weekday() != 5:
        raise StateTableError("state table path date is not an in-year Saturday")

    parser = _TableParser()
    parser.feed(text)
    parser.close()
    normalized_text = _SPACE_RE.sub(
        " ",
        " ".join(parser.all_text).replace("\u00a0", " "),
    ).strip()
    initial_match = _INITIAL_HEADING_RE.search(normalized_text)
    insured_match = _INSURED_HEADING_RE.search(normalized_text)
    if initial_match is None or insured_match is None:
        raise StateTableError("state table headings changed")
    initial_week = _parse_observation_date(
        initial_match.group("date"),
        floor=path_week + timedelta(days=1),
    )
    if initial_week != path_week:
        raise StateTableError("state table path and heading dates disagree")
    insured_week = _parse_observation_date(
        insured_match.group("date"),
        floor=initial_week,
    )
    if insured_week != initial_week - timedelta(days=7):
        raise StateTableError("insured-unemployment week is not seven days earlier")

    rows: dict[str, tuple[int, ...]] = {}
    totals: tuple[int, ...] | None = None
    for cells in parser.rows:
        if not cells:
            continue
        name = _SPACE_RE.sub(" ", cells[0]).strip()
        if name not in JURISDICTION_SET and name.lower() != "totals":
            continue
        numeric = [value for value in cells[1:] if value.strip()]
        parsed = _parse_state_values(numeric)
        if name.lower() == "totals":
            if totals is not None:
                raise StateTableError("state table duplicates totals")
            totals = parsed
        else:
            if name in rows:
                raise StateTableError("state table duplicates a jurisdiction")
            rows[name] = parsed
    if set(rows) != JURISDICTION_SET or len(rows) != 53:
        raise StateTableError("state table jurisdiction set changed")
    if totals is None:
        raise StateTableError("state table lacks totals")
    for column in SUMMED_COLUMNS:
        if sum(values[column] for values in rows.values()) != totals[column]:
            raise StateTableError("state table totals do not reconcile")

    canonical_rows = [
        {
            "name": name,
            "values": list(rows[name]),
        }
        for name in JURISDICTIONS
    ]
    structural = {
        "initial_week": initial_week.isoformat(),
        "insured_week": insured_week.isoformat(),
        "rows": canonical_rows,
        "totals": list(totals),
    }
    explicit_year = bool(
        re.search(r"\b\d{4}\b", initial_match.group("date"))
    )
    return StateTableSummary(
        week_ending=initial_week,
        insured_week_ending=insured_week,
        path=path,
        raw_sha256=sha256_bytes(raw),
        structural_sha256=sha256_bytes(
            canonical_json_bytes(structural, newline=False)
        ),
        jurisdiction_count=len(rows),
        arithmetic_columns_reconciled=len(SUMMED_COLUMNS),
        grammar_variant="explicit_year" if explicit_year else "path_year",
    )


def _quarter(value: date) -> str:
    return f"{value.year}-Q{((value.month - 1) // 3) + 1}"


def _coverage_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 6)


def evaluate_support(
    national: Sequence[NationalRelease],
    state_tables: Sequence[StateTableSummary],
    *,
    inventory_counts: Mapping[int, int],
    malformed_link_counts: Mapping[int, int],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    ordered = sorted(national, key=lambda row: (row.release_date, row.path))
    checks["national_total_at_least_700"] = len(ordered) >= 700
    checks["national_unique_identity"] = (
        len({row.node_id for row in ordered}) == len(ordered)
        and len({row.path for row in ordered}) == len(ordered)
        and len({row.release_date for row in ordered}) == len(ordered)
        and len({row.teaser_sha256 for row in ordered}) == len(ordered)
    )
    checks["national_window_exact"] = all(
        SOURCE_START <= row.release_date < SOURCE_END for row in ordered
    )
    checks["national_clock_exact"] = all(
        row.availability_at_utc == historical_availability(row.release_date)
        for row in ordered
    )
    checks["national_week_precedes_release"] = all(
        timedelta(days=1)
        <= row.release_date - row.week_ending
        <= timedelta(days=14)
        for row in ordered
    )

    year_counts = Counter(row.release_date.year for row in ordered)
    checks["national_2012_2024_year_counts"] = all(
        50 <= year_counts[year] <= 54 for year in range(2012, 2025)
    )
    checks["national_2025_exact_count"] = year_counts[2025] == 46
    quarter_counts = Counter(_quarter(row.release_date) for row in ordered)
    checks["national_2012_2024_quarter_counts"] = all(
        quarter_counts[f"{year}-Q{quarter}"] >= 12
        for year in range(2012, 2025)
        for quarter in range(1, 5)
    )
    checks["national_2025_quarter_counts"] = (
        all(quarter_counts[f"2025-Q{quarter}"] >= 12 for quarter in (1, 2, 3))
        and quarter_counts["2025-Q4"] >= 7
    )
    gaps: list[int] = []
    gap_valid = True
    for previous, current in zip(ordered, ordered[1:]):
        gap = (current.release_date - previous.release_date).days
        gaps.append(gap)
        if 5 <= gap <= 10:
            continue
        if (
            previous.release_date == date(2025, 9, 25)
            and current.release_date == date(2025, 11, 20)
        ):
            continue
        gap_valid = False
    checks["national_release_gaps"] = gap_valid

    endpoint_count = sum(
        row.prior_before is not None and row.prior_average_before is not None
        for row in ordered
    )
    status_count = sum(row.prior_status in {"revised", "unrevised"} for row in ordered)
    average_count = sum(type(row.signed_average_change) is int for row in ordered)
    checks["national_prior_status_coverage"] = (
        status_count >= (9 * len(ordered) + 9) // 10
    )
    checks["national_revision_endpoint_coverage"] = (
        endpoint_count >= (3 * len(ordered) + 3) // 4
    )
    checks["national_average_change_coverage"] = (
        average_count >= (9 * len(ordered) + 9) // 10
    )

    frozen_inventory = dict(sorted(EXPECTED_STATE_TABLE_COUNTS.items()))
    supplied_inventory = {
        int(year): int(count)
        for year, count in sorted(inventory_counts.items())
    }
    checks["state_inventory_counts_exact"] = supplied_inventory == frozen_inventory
    expected_malformed_counts = {
        year: len(FROZEN_MALFORMED_PATHS.get(year, ()))
        for year in range(2012, 2026)
    }
    supplied_malformed_counts = {
        int(year): int(count)
        for year, count in sorted(malformed_link_counts.items())
    }
    checks["state_malformed_links_exact"] = (
        supplied_malformed_counts == expected_malformed_counts
    )
    checks["state_table_total_exact"] = len(state_tables) == sum(
        EXPECTED_STATE_TABLE_COUNTS.values()
    )
    checks["state_table_unique_week"] = (
        len({row.week_ending for row in state_tables}) == len(state_tables)
    )
    checks["state_table_unique_artifacts"] = (
        len({row.path for row in state_tables}) == len(state_tables)
        and len({row.raw_sha256 for row in state_tables}) == len(state_tables)
        and len({row.structural_sha256 for row in state_tables})
        == len(state_tables)
    )
    checks["state_table_structure"] = all(
        row.jurisdiction_count == 53
        and row.arithmetic_columns_reconciled == len(SUMMED_COLUMNS)
        and row.insured_week_ending == row.week_ending - timedelta(days=7)
        for row in state_tables
    )

    national_by_week: dict[date, list[NationalRelease]] = {}
    for row in ordered:
        national_by_week.setdefault(row.week_ending, []).append(row)
    represented_weeks = {row.week_ending for row in state_tables}
    checks["state_to_national_unique_match"] = all(
        len(national_by_week.get(week, [])) == 1 for week in represented_weeks
    )
    matched = sum(
        len(national_by_week.get(week, [])) == 1 for week in represented_weeks
    )
    matched_national = {
        national_by_week[week][0].release_date
        for week in represented_weeks
        if len(national_by_week.get(week, [])) == 1
    }
    checks["full_state_coverage_at_least_90pct"] = (
        matched >= (9 * len(ordered) + 9) // 10
    )
    coverage_by_year: dict[int, float] = {}
    for year in range(2012, 2026):
        denominator = year_counts[year]
        numerator = sum(value.year == year for value in matched_national)
        coverage_by_year[year] = _coverage_percent(numerator, denominator)
    checks["normal_year_state_coverage"] = all(
        coverage_by_year[year] >= 98.0
        for year in range(2012, 2025)
        if year != 2019
    )
    checks["state_2019_coverage"] = coverage_by_year[2019] >= 40.0
    checks["state_2025_coverage"] = coverage_by_year[2025] >= 90.0

    decision = (
        "SOURCE_SUPPORT_PASS" if checks and all(checks.values()) else "TERMINAL_REJECT"
    )
    return {
        "checks": dict(sorted(checks.items())),
        "decision": decision,
        "national_counts_by_year": {
            str(year): year_counts[year] for year in range(2012, 2026)
        },
        "national_release_count": len(ordered),
        "national_teasers_with_full_revision_endpoints": endpoint_count,
        "state_coverage_percent_by_national_year": {
            str(year): coverage_by_year[year] for year in range(2012, 2026)
        },
        "state_missing_event_count": len(ordered) - matched,
        "state_table_count": len(state_tables),
    }


def _strict_url(
    value: str,
    *,
    allowed_query: str | None,
) -> urllib.parse.SplitResult:
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
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.netloc != parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or parsed.query != (allowed_query or "")
        or not parsed.path.startswith("/")
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise TransportError("URL is outside the frozen authority")
    return parsed


def make_get_intent(url: str) -> RequestIntent:
    if url == SCHEDULE_URL:
        _strict_url(url, allowed_query=None)
    elif url == NEWSROOM_URL:
        _strict_url(url, allowed_query=None)
    elif url.startswith(f"{NEWSROOM_URL}?page="):
        parsed = urllib.parse.urlsplit(url)
        match = re.fullmatch(r"page=(0|[1-9]\d*)", parsed.query, re.ASCII)
        if match is None:
            raise TransportError("newsroom page query is noncanonical")
        _strict_url(url, allowed_query=parsed.query)
    elif url.startswith("https://oui.doleta.gov/unemploy/page8/"):
        parsed = _strict_url(url, allowed_query=None)
        if STATE_PATH_RE.fullmatch(parsed.path) is None:
            raise TransportError("state-table URL path is noncanonical")
    else:
        raise TransportError("GET endpoint is unauthorized")
    return RequestIntent(
        method="GET",
        url=url,
        body=b"",
        headers=REQUEST_HEADERS,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
    )


def make_inventory_intent(year: int) -> RequestIntent:
    if year not in EXPECTED_STATE_TABLE_COUNTS:
        raise TransportError("inventory year is unauthorized")
    _strict_url(ARCHIVE_URL, allowed_query=None)
    body = f"report=page8&year={year}&submit=Submit".encode("ascii")
    return RequestIntent(
        method="POST",
        url=ARCHIVE_URL,
        body=body,
        headers=(
            *REQUEST_HEADERS,
            ("Content-Type", "application/x-www-form-urlencoded"),
        ),
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
    )


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
    *,
    body_cap: int,
) -> None:
    if payload.intent != intent:
        raise TransportError("transport changed the exact intent")
    if payload.status != 200:
        raise TransportError("HTTP status is not 200")
    if payload.response_completed_at_utc < payload.request_started_at_utc:
        raise TransportError("HTTP receipt clock moved backward")
    if len(payload.raw) > body_cap:
        raise TransportError("HTTP body exceeds the frozen cap")
    content_type = _single_header(
        payload.headers,
        "Content-Type",
        required=True,
    )
    assert content_type is not None
    if content_type.split(";", 1)[0].strip().lower() != "text/html":
        raise TransportError("HTTP content type is not text/html")
    content_encoding = _single_header(
        payload.headers,
        "Content-Encoding",
        required=False,
    )
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        raise TransportError("HTTP content encoding is not identity")
    if _single_header(payload.headers, "Content-Range", required=False) is not None:
        raise TransportError("partial HTTP response is forbidden")
    content_length = _single_header(
        payload.headers,
        "Content-Length",
        required=False,
    )
    if content_length is not None:
        if not content_length.isdigit() or int(content_length) != len(payload.raw):
            raise TransportError("HTTP content length disagrees with body")


class Transport(Protocol):
    def request(self, intent: RequestIntent) -> HttpPayload: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
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
            data=intent.body if intent.method == "POST" else None,
            headers=dict(intent.headers),
            method=intent.method,
        )
        try:
            with self._opener.open(
                request,
                timeout=intent.timeout_seconds,
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
    body_cap: int,
    sleep: Callable[[float], None] = time.sleep,
) -> HttpPayload:
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            payload = transport.request(intent)
        except RetryableTransportError as exc:
            last_error = exc
            if attempt >= 2:
                break
            sleep(RETRY_DELAYS[attempt])
            continue
        if payload.intent != intent:
            raise TransportError("transport changed the exact intent")
        if payload.status in RETRYABLE_STATUSES:
            last_error = RetryableTransportError("retryable HTTP status")
            if attempt >= 2:
                break
            sleep(RETRY_DELAYS[attempt])
            continue
        _validate_success_payload(payload, intent, body_cap=body_cap)
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
    _exclusive_write(paths.sentinel, sentinel, mode=0o400)
    _exclusive_write(paths.manifest, b"", mode=0o600)
    paths.raw_dir.mkdir(mode=0o700)
    manifest_metadata = paths.manifest.stat(follow_symlinks=False)
    return AttemptGuard(
        paths=paths,
        sentinel_sha256=sha256_bytes(sentinel),
        manifest_device=manifest_metadata.st_dev,
        manifest_inode=manifest_metadata.st_ino,
    )


def _deduplicating_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("manifest JSON duplicates a key")
        result[key] = value
    return result


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
    if len(raw) > 64 * 1024 * 1024:
        raise ProtocolError("manifest exceeds the frozen cap")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ProtocolError("manifest is empty")

    previous_hash = "0" * 64
    pending_intent = False
    raw_artifacts: set[str] = set()
    response_hashes: list[str] = []
    final_event = ""
    for sequence, line in enumerate(lines):
        if not line.endswith(b"\n"):
            raise ProtocolError("manifest line lacks newline")
        try:
            record = json.loads(
                line.decode("ascii"),
                object_pairs_hook=_deduplicating_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ProtocolError(f"invalid manifest constant: {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("manifest line is not canonical JSON") from exc
        if not isinstance(record, dict):
            raise ProtocolError("manifest record is not an object")
        if canonical_json_bytes(record) != line:
            raise ProtocolError("manifest record is not canonical")
        record_hash = record.get("record_hash")
        if not isinstance(record_hash, str) or HEX64_RE.fullmatch(record_hash) is None:
            raise ProtocolError("manifest record hash is invalid")
        unsigned = dict(record)
        unsigned.pop("record_hash")
        if sha256_bytes(
            canonical_json_bytes(unsigned, newline=False)
        ) != record_hash:
            raise ProtocolError("manifest record hash disagrees")
        if unsigned.get("sequence") != sequence:
            raise ProtocolError("manifest sequence is not contiguous")
        if unsigned.get("previous_hash") != previous_hash:
            raise ProtocolError("manifest previous hash disagrees")
        if unsigned.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError("manifest protocol version changed")
        event = unsigned.get("event")
        payload = unsigned.get("payload")
        if not isinstance(event, str) or not isinstance(payload, dict):
            raise ProtocolError("manifest event or payload is invalid")
        if event == "request_intent":
            if pending_intent:
                raise ProtocolError("manifest overlaps request intents")
            pending_intent = True
        elif event == "response_committed":
            if not pending_intent:
                raise ProtocolError("manifest response lacks request intent")
            pending_intent = False
            name = payload.get("raw_artifact")
            expected_sha = payload.get("body_sha256")
            expected_bytes = payload.get("body_bytes")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or name in raw_artifacts
                or not isinstance(expected_sha, str)
                or HEX64_RE.fullmatch(expected_sha) is None
                or type(expected_bytes) is not int
                or expected_bytes < 0
            ):
                raise ProtocolError("manifest raw-artifact binding is invalid")
            artifact = raw_dir / name
            if artifact.is_symlink() or not artifact.is_file():
                raise ProtocolError("manifest raw artifact is not regular")
            artifact_raw = artifact.read_bytes()
            if (
                len(artifact_raw) != expected_bytes
                or sha256_bytes(artifact_raw) != expected_sha
            ):
                raise ProtocolError("manifest raw artifact disagrees")
            raw_artifacts.add(name)
            response_hashes.append(expected_sha)
        elif event in {"source_support_complete", "terminal_reject"}:
            if pending_intent and event == "source_support_complete":
                raise ProtocolError("completed manifest has a pending intent")
        else:
            raise ProtocolError("manifest event is unauthorized")
        previous_hash = record_hash
        final_event = event
    if require_complete and (
        pending_intent or final_event != "source_support_complete"
    ):
        raise ProtocolError("manifest is not a completed source audit")
    actual_artifacts = {
        entry.name
        for entry in raw_dir.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    if actual_artifacts != raw_artifacts or any(
        entry.is_symlink() or not entry.is_file() for entry in raw_dir.iterdir()
    ):
        raise ProtocolError("raw directory and manifest artifact sets disagree")
    return {
        "final_record_hash": previous_hash,
        "raw_artifact_count": len(raw_artifacts),
        "record_count": len(lines),
        "response_hashes_sha256": sha256_bytes(
            canonical_json_bytes(response_hashes, newline=False)
        ),
    }


def _persist_payload(
    guard: AttemptGuard,
    payload: HttpPayload,
    *,
    kind: str,
    key: str,
) -> Path:
    digest = sha256_bytes(payload.raw)
    safe_kind = re.sub(r"[^a-z0-9_-]", "_", kind.lower())
    filename = f"{guard.sequence:05d}-{safe_kind}-{digest[:16]}.html"
    path = guard.paths.raw_dir / filename
    _exclusive_write(path, payload.raw, mode=0o400)
    guard.append(
        "response_committed",
        {
            "body_bytes": len(payload.raw),
            "body_sha256": digest,
            "key": key,
            "kind": kind,
            "raw_artifact": filename,
            "status": payload.status,
        },
    )
    return path


def _atomic_publish(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PublicationError("report path already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    _exclusive_write(temporary, payload, mode=0o400)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PublicationError("report path raced into existence") from exc
        except OSError as exc:
            raise PublicationError("report no-clobber publication failed") from exc
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _outcome_boundary() -> dict[str, bool]:
    return {
        "market_or_price_opened": False,
        "model_tokenizer_adapter_or_prompt_opened": False,
        "portfolio_reward_checkpoint_or_performance_opened": False,
        "private_forecast_or_consensus_opened": False,
        "mutable_current_claims_series_opened": False,
        "postwindow_teaser_body_opened": False,
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
    schedule_sha256: str,
    newsroom_pages_sha256: str,
    inventories_sha256: str,
    state_tables_sha256: str,
    support: Mapping[str, Any],
) -> dict[str, Any]:
    decision = str(support["decision"])
    report = {
        "bindings": {
            "boundary_path": str(BOUNDARY_PATH),
            "boundary_sha256": BOUNDARY_SHA256,
            "manifest_sha256": manifest_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "runner_git_blob": runner_blob,
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "sentinel_sha256": sentinel_sha256,
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_file(TEST_PATH),
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
            "inventories_sha256": inventories_sha256,
            "newsroom_pages_sha256": newsroom_pages_sha256,
            "schedule_sha256": schedule_sha256,
            "state_tables_sha256": state_tables_sha256,
        },
        "support": dict(support),
    }
    return _finalize_report(report)


def _fingerprint(values: Sequence[Any]) -> str:
    normalized = [
        value.private_dict() if hasattr(value, "private_dict") else value
        for value in values
    ]
    return sha256_bytes(canonical_json_bytes(normalized, newline=False))


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
    if sha256_file(BOUNDARY_PATH) != BOUNDARY_SHA256:
        raise ProtocolError("boundary digest changed")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProtocolError("production source audit requires a clean worktree")
    verifier_commit = _git("rev-parse", "HEAD")
    if HEX40_RE.fullmatch(verifier_commit) is None:
        raise ProtocolError("HEAD is not a full Git commit")
    entries: dict[str, str] = {}
    for path in (BOUNDARY_PATH, SCRIPT_PATH, TEST_PATH):
        output = _git("ls-tree", "HEAD", "--", str(path))
        fields = output.split()
        if len(fields) < 4 or fields[1] != "blob":
            raise ProtocolError(f"committed protocol file missing: {path}")
        entries[str(path)] = fields[2]
    if any(HEX40_RE.fullmatch(value) is None for value in entries.values()):
        raise ProtocolError("protocol Git blob identity is invalid")
    for path in (BOUNDARY_PATH, SCRIPT_PATH, TEST_PATH):
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
    used = usage.total - usage.free
    if used >= DISK_USED_LIMIT:
        raise DiskGuardError("filesystem use is at or above 300 GiB")
    if usage.free < DISK_FREE_FLOOR:
        raise DiskGuardError("filesystem free space is below 8 GiB")
    return used, usage.free


def _request_and_store(
    *,
    guard: AttemptGuard,
    transport: Transport,
    intent: RequestIntent,
    kind: str,
    key: str,
) -> Path:
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
    payload = request_with_retries(
        transport,
        intent,
        body_cap=HTML_BODY_CAP,
    )
    return _persist_payload(guard, payload, kind=kind, key=key)


def _production_report_failure(
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
                "manifest_sha256": sha256_file(guard.paths.manifest),
                "protocol_version": PROTOCOL_VERSION,
                "runner_git_blob": runner_blob,
                "script_path": str(SCRIPT_PATH),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "sentinel_sha256": guard.sentinel_sha256,
                "test_path": str(TEST_PATH),
                "test_sha256": sha256_file(TEST_PATH),
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


def run_production_audit() -> dict[str, Any]:
    verifier_commit, runner_blob = assert_protocol_committed()
    _disk_guard()
    guard = reserve_attempt(
        paths=PRODUCTION_PATHS,
        verifier_commit=verifier_commit,
        runner_blob=runner_blob,
    )
    transport = StdlibTransport()
    stage = "schedule"
    try:
        schedule_path = _request_and_store(
            guard=guard,
            transport=transport,
            intent=make_get_intent(SCHEDULE_URL),
            kind="schedule",
            key="schedule",
        )
        schedule = parse_schedule_page(schedule_path.read_bytes())

        stage = "newsroom"
        newsroom_paths: list[Path] = []
        pages: list[NewsroomPage] = []
        national: list[NationalRelease] = []
        page_number = 0
        while True:
            url = NEWSROOM_URL if page_number == 0 else f"{NEWSROOM_URL}?page={page_number}"
            path = _request_and_store(
                guard=guard,
                transport=transport,
                intent=make_get_intent(url),
                kind="newsroom",
                key=f"page-{page_number}",
            )
            page = parse_newsroom_page(path.read_bytes(), expected_page=page_number)
            newsroom_paths.append(path)
            pages.append(page)
            national.extend(page.releases)
            if page.minimum_displayed_date < SOURCE_START:
                break
            if page.next_page is None:
                raise NewsroomError("newsroom chain ended before crossing 2012")
            page_number = page.next_page
            if page_number > 200:
                raise NewsroomError("newsroom pagination exceeds the frozen cap")
        validate_newsroom_chain(pages)

        stage = "state_inventory"
        inventory_paths: list[Path] = []
        inventories: list[StateInventory] = []
        for year in range(2012, 2026):
            path = _request_and_store(
                guard=guard,
                transport=transport,
                intent=make_inventory_intent(year),
                kind="state_inventory",
                key=str(year),
            )
            inventory_paths.append(path)
            inventories.append(
                parse_state_inventory(path.read_bytes(), requested_year=year)
            )

        stage = "state_tables"
        table_paths: list[tuple[StateArtifact, Path]] = []
        tables: list[StateTableSummary] = []
        raw_total = sum(path.stat().st_size for path in (
            [schedule_path, *newsroom_paths, *inventory_paths]
        ))
        for inventory in inventories:
            for artifact in inventory.artifacts:
                _disk_guard()
                path = _request_and_store(
                    guard=guard,
                    transport=transport,
                    intent=make_get_intent(artifact.url),
                    kind="state_table",
                    key=f"{artifact.year}-{artifact.week_ending:%m%d}",
                )
                raw_total += path.stat().st_size
                if raw_total > TOTAL_RAW_SOURCE_CAP:
                    raise DiskGuardError("raw source bytes exceed 256 MiB")
                parsed = parse_state_table(path.read_bytes(), path=artifact.path)
                if parsed.week_ending != artifact.week_ending:
                    raise StateTableError("inventory and table week disagree")
                table_paths.append((artifact, path))
                tables.append(parsed)

        stage = "replay"
        if parse_schedule_page(schedule_path.read_bytes()) != schedule:
            raise SourceContractError("schedule replay changed")
        replay_pages = [
            parse_newsroom_page(path.read_bytes(), expected_page=index)
            for index, path in enumerate(newsroom_paths)
        ]
        validate_newsroom_chain(replay_pages)
        replay_national = [
            row for page in replay_pages for row in page.releases
        ]
        replay_inventories = [
            parse_state_inventory(path.read_bytes(), requested_year=year)
            for year, path in zip(range(2012, 2026), inventory_paths)
        ]
        replay_tables = [
            parse_state_table(path.read_bytes(), path=artifact.path)
            for artifact, path in table_paths
        ]
        if _fingerprint(replay_national) != _fingerprint(national):
            raise SourceContractError("national replay fingerprint changed")
        if _fingerprint(replay_tables) != _fingerprint(tables):
            raise SourceContractError("state replay fingerprint changed")
        if replay_inventories != inventories:
            raise SourceContractError("inventory replay changed")

        stage = "support"
        inventory_counts = {
            row.year: len(row.artifacts) for row in inventories
        }
        malformed_counts = {
            row.year: row.malformed_link_count for row in inventories
        }
        support = evaluate_support(
            national,
            tables,
            inventory_counts=inventory_counts,
            malformed_link_counts=malformed_counts,
        )
        if support["decision"] != "SOURCE_SUPPORT_PASS":
            raise SupportError("frozen DOL source support gates failed")

        guard.append(
            "source_support_complete",
            {
                "decision": support["decision"],
                "national_release_count": support["national_release_count"],
                "state_table_count": support["state_table_count"],
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
        ):
            raise ProtocolError("manifest replay disagrees with writer state")
        report = build_report(
            execution_authority="production_one_shot",
            source_audit_authoritative=True,
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            manifest_sha256=sha256_file(guard.paths.manifest),
            sentinel_sha256=guard.sentinel_sha256,
            schedule_sha256=schedule["raw_sha256"],
            newsroom_pages_sha256=_fingerprint(
                [page.aggregate_dict() for page in pages]
            ),
            inventories_sha256=_fingerprint(
                [
                    {
                        "raw_sha256": row.raw_sha256,
                        "structural_sha256": row.structural_sha256,
                        "year": row.year,
                    }
                    for row in inventories
                ]
            ),
            state_tables_sha256=_fingerprint(tables),
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
        report = _production_report_failure(
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
        "DOL_WCRV_ISOLATED_CHILD_TOKEN": token,
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen DOL weekly-claims source-only audit."
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
        or os.environ.get("DOL_WCRV_ISOLATED_CHILD_TOKEN")
        != args._isolated_child_token
    ):
        raise ProtocolError("production requires the isolated CLI child")
    report = run_production_audit()
    print(f"DOL-WCRV source audit: {report['decision']}")
    return 0 if report["decision"] == "SOURCE_SUPPORT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
