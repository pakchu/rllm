"""Build an outcome-blind Daily Treasury Statement source snapshot.

The builder opens only official Fiscal Data metadata, announcements, and DTS
PDF reports through 2023-12-29.  It does not import crypto market data,
returns, labels, or portfolio outcomes.  Current metadata may mention later
dates, but no post-cap report URL or API value row is admitted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo


FISCAL_DATA_HOST = "fiscaldata.treasury.gov"
DATASET_PAGE_DATA_URL = (
    "https://fiscaldata.treasury.gov/page-data/datasets/"
    "daily-treasury-statement/page-data.json"
)
ANNOUNCEMENTS_URL = (
    "https://fiscaldata.treasury.gov/static-data/published-reports/dts/"
    "DailyTreasuryStatement_Announcements.xlsx"
)
REPORT_URL_TEMPLATE = (
    "https://fiscaldata.treasury.gov/static-data/published-reports/dts/"
    "DailyTreasuryStatement_{date_compact}.pdf"
)
USER_AGENT = "rllm-dffb-source-freeze/1.0"
SCHEMA_VERSION = 1
PARSER_VERSION = 1
MIN_REPORT_DATE = date(2019, 1, 2)
MAX_REPORT_DATE = date(2023, 12, 29)
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
PDF_DATE_PATTERN = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday),\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
NUMBER_TOKEN = re.compile(r"^-?(?:0|[1-9]\d{0,2}(?:,\d{3})*)$")
FOOTNOTE_TOKEN = re.compile(r"^(?:\d+/|\*+|[a-z]/)$", re.IGNORECASE)
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
XLSX_REL_NS = {
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}
ANNOUNCEMENT_HEADERS = {
    "DTS Changes": {
        "A": "Date of DTS Change",
        "B": "Table Name",
        "C": "Table Section",
        "D": "Change",
    },
    "Full - Table": {
        "A": "DTS Date",
        "B": "Table Number",
        "C": "Table Section",
        "D": "Entity",
        "E": "Change",
    },
}

OUTPUT_COLUMNS = (
    "record_date",
    "source_available_not_before_utc",
    "earliest_execution_time_utc",
    "research_stage",
    "table_id",
    "side",
    "parent_section",
    "raw_category_label",
    "normalized_category_label",
    "today_amount_usd_millions",
    "month_to_date_amount_usd_millions",
    "fiscal_year_to_date_amount_usd_millions",
    "footnote_markers",
    "missing_value_tokens",
    "row_kind",
    "page_number",
    "source_order",
    "source_pdf_sha256",
)
TABLE_I_OUTPUT_COLUMNS = (
    "record_date",
    "source_available_not_before_utc",
    "earliest_execution_time_utc",
    "research_stage",
    "raw_category_label",
    "normalized_category_label",
    "published_value_count",
    "published_values_usd_millions_json",
    "missing_value_tokens_json",
    "footnote_markers",
    "schema_variant",
    "page_number",
    "source_order",
    "source_pdf_sha256",
)


@dataclass(frozen=True)
class BuildConfig:
    start_date: str = MIN_REPORT_DATE.isoformat()
    end_date: str = MAX_REPORT_DATE.isoformat()
    output_dir: str = "data/daily_treasury_fiscal_flow_2019_2023"
    max_workers: int = 6
    retries: int = 5
    timeout_seconds: int = 60
    request_pace_seconds: float = 0.10


@dataclass(frozen=True)
class PublishedReport:
    record_date: date
    url: str


@dataclass(frozen=True)
class RedirectHop:
    status: int
    target_url: str


@dataclass(frozen=True)
class FetchReceipt:
    url: str
    status: int
    final_url: str
    retrieved_at_utc: str
    etag: str
    last_modified: str
    redirect_chain: tuple[RedirectHop, ...]
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class PdfTextCell:
    page_number: int
    source_order: int
    x: float
    y: float
    text: str
    font_name: str
    font_size: float


@dataclass(frozen=True)
class DtsRow:
    record_date: str
    source_available_not_before_utc: str
    earliest_execution_time_utc: str
    research_stage: str
    table_id: str
    side: str
    parent_section: str
    raw_category_label: str
    normalized_category_label: str
    today_amount_usd_millions: str
    month_to_date_amount_usd_millions: str
    fiscal_year_to_date_amount_usd_millions: str
    footnote_markers: str
    missing_value_tokens: str
    row_kind: str
    page_number: int
    source_order: int
    source_pdf_sha256: str


@dataclass(frozen=True)
class DtsTableIRow:
    record_date: str
    source_available_not_before_utc: str
    earliest_execution_time_utc: str
    research_stage: str
    raw_category_label: str
    normalized_category_label: str
    published_value_count: int
    published_values_usd_millions_json: str
    missing_value_tokens_json: str
    footnote_markers: str
    schema_variant: str
    page_number: int
    source_order: int
    source_pdf_sha256: str


@dataclass(frozen=True)
class ParsedDtsPdf:
    record_date: date
    creation_metadata_raw: tuple[str, ...]
    table_ids_found: tuple[str, ...]
    cells: tuple[PdfTextCell, ...]
    table_i_rows: tuple[DtsTableIRow, ...]
    rows: tuple[DtsRow, ...]


@dataclass(frozen=True)
class Announcement:
    effective_date: date
    table_name: str
    table_section: str
    entity: str
    change: str
    sheet_name: str
    source_row: int


@dataclass(frozen=True)
class _PdfObject:
    number: int
    generation: int
    offset: int
    body: bytes
    dictionary: bytes
    stream_start: int | None
    stream_end_hint: int | None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def canonical_csv(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(columns),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _validate_official_url(url: str, *, allow_metadata: bool = False) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != FISCAL_DATA_HOST:
        raise ValueError(f"non-official Fiscal Data URL: {url}")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError(f"unsafe Fiscal Data URL authority: {url}")
    if allow_metadata:
        if url not in {DATASET_PAGE_DATA_URL, ANNOUNCEMENTS_URL}:
            raise ValueError(f"unapproved metadata URL: {url}")
        return
    match = re.fullmatch(
        r"/static-data/published-reports/dts/"
        r"DailyTreasuryStatement_(\d{8})\.pdf",
        parsed.path,
    )
    if match is None or parsed.query or parsed.fragment:
        raise ValueError(f"unapproved DTS report URL: {url}")
    record_date = datetime.strptime(match.group(1), "%Y%m%d").date()
    if not MIN_REPORT_DATE <= record_date <= MAX_REPORT_DATE:
        raise ValueError(f"DTS report date is outside the frozen cap: {record_date}")


def report_url(record_date: date) -> str:
    if not MIN_REPORT_DATE <= record_date <= MAX_REPORT_DATE:
        raise ValueError(f"DTS report date is outside the frozen cap: {record_date}")
    url = REPORT_URL_TEMPLATE.format(date_compact=record_date.strftime("%Y%m%d"))
    _validate_official_url(url)
    return url


def _observed_fixed_holiday(month: int, day: int, year: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    current = next_month - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def federal_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(1, 1, year),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(7, 4, year),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _observed_fixed_holiday(11, 11, year),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(12, 25, year),
    }
    if year >= 2021:
        holidays.add(_observed_fixed_holiday(6, 19, year))
    if year % 4 == 1:
        inauguration = date(year, 1, 20)
        if inauguration.weekday() == 6:
            inauguration += timedelta(days=1)
        holidays.add(inauguration)
    observed_next_new_year = _observed_fixed_holiday(1, 1, year + 1)
    if observed_next_new_year.year == year:
        holidays.add(observed_next_new_year)
    return {holiday for holiday in holidays if holiday.year == year}


def is_federal_business_day(value: date) -> bool:
    return value.weekday() < 5 and value not in federal_holidays(value.year)


def next_federal_business_day(value: date) -> date:
    candidate = value + timedelta(days=1)
    while not is_federal_business_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def source_clock(record_date: date) -> tuple[datetime, datetime, str]:
    publish_date = next_federal_business_day(record_date)
    available_local = datetime.combine(
        publish_date, wall_time(16, 0), tzinfo=NEW_YORK
    )
    execution_local = available_local + timedelta(minutes=5)
    available_utc = available_local.astimezone(UTC)
    execution_utc = execution_local.astimezone(UTC)
    if execution_utc < datetime(2021, 1, 1, tzinfo=UTC):
        stage = "warmup"
    elif execution_utc < datetime(2023, 1, 1, tzinfo=UTC):
        stage = "train"
    elif execution_utc < datetime(2024, 1, 1, tzinfo=UTC):
        stage = "selection"
    else:
        stage = "boundary_quarantine"
    return available_utc, execution_utc, stage


def parse_published_reports(
    payload: bytes, *, start: date, end: date
) -> list[PublishedReport]:
    if start < MIN_REPORT_DATE or end > MAX_REPORT_DATE or start > end:
        raise ValueError("published-report range violates the frozen source cap")
    try:
        document = json.loads(payload.decode("utf-8"))
        raw_reports = document["result"]["pageContext"]["config"][
            "publishedReports"
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Fiscal Data page-data has an invalid schema") from exc
    if not isinstance(raw_reports, list):
        raise ValueError("Fiscal Data publishedReports must be a list")

    selected: list[PublishedReport] = []
    seen: set[date] = set()
    for raw in raw_reports:
        if not isinstance(raw, dict):
            raise ValueError("Fiscal Data publishedReports contains a non-object")
        path = raw.get("path")
        if not isinstance(path, str):
            continue
        match = re.fullmatch(
            r"/static-data/published-reports/dts/"
            r"DailyTreasuryStatement_(\d{8})\.pdf",
            path,
        )
        if match is None:
            continue
        record_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        if not start <= record_date <= end:
            continue
        raw_date = raw.get("report_date")
        if not isinstance(raw_date, str):
            raise ValueError("retained Fiscal Data report row is missing report_date")
        if raw_date[:15] != record_date.strftime("%a %b %d %Y"):
            raise ValueError(f"published report date/path mismatch: {path}")
        if record_date in seen:
            raise ValueError(f"duplicate published DTS report date: {record_date}")
        seen.add(record_date)
        url = urllib.parse.urljoin(
            f"https://{FISCAL_DATA_HOST}/", path.lstrip("/")
        )
        _validate_official_url(url)
        selected.append(PublishedReport(record_date=record_date, url=url))
    selected.sort(key=lambda row: row.record_date)
    if not selected:
        raise ValueError("Fiscal Data index yielded no frozen DTS reports")
    return selected


def _excel_serial_to_date(value: str) -> date:
    if not re.fullmatch(r"\d+(?:\.0+)?", value):
        raise ValueError(f"unsupported Excel date serial: {value!r}")
    serial = int(float(value))
    if serial < 1:
        raise ValueError(f"invalid Excel date serial: {value!r}")
    return date(1899, 12, 30) + timedelta(days=serial)


def _xlsx_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except (KeyError, ElementTree.ParseError) as exc:
        raise ValueError("announcement workbook has no valid shared strings") from exc
    return [
        unicodedata.normalize(
            "NFC", "".join(node.text or "" for node in item.findall(".//m:t", XLSX_NS))
        )
        for item in root.findall("m:si", XLSX_NS)
    ]


def _xlsx_sheet_rows(
    archive: ZipFile, sheet_path: str, shared: Sequence[str]
) -> list[tuple[int, dict[str, str]]]:
    try:
        root = ElementTree.fromstring(archive.read(sheet_path))
    except (KeyError, ElementTree.ParseError) as exc:
        raise ValueError(f"announcement workbook has no valid {sheet_path}") from exc
    rows: list[tuple[int, dict[str, str]]] = []
    for row in root.findall(".//m:sheetData/m:row", XLSX_NS):
        row_number = int(row.attrib["r"])
        values: dict[str, str] = {}
        for cell in row.findall("m:c", XLSX_NS):
            reference = cell.attrib.get("r", "")
            column_match = re.match(r"[A-Z]+", reference)
            if column_match is None:
                raise ValueError(f"invalid XLSX cell reference: {reference!r}")
            value_node = cell.find("m:v", XLSX_NS)
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                try:
                    value = shared[int(value)]
                except (ValueError, IndexError) as exc:
                    raise ValueError("invalid XLSX shared-string index") from exc
            values[column_match.group(0)] = unicodedata.normalize("NFC", value)
        rows.append((row_number, values))
    return rows


def _xlsx_announcement_sheets(archive: ZipFile) -> dict[str, str]:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
    except (KeyError, ElementTree.ParseError) as exc:
        raise ValueError("announcement workbook has invalid workbook relationships") from exc
    targets = {
        node.attrib["Id"]: node.attrib["Target"]
        for node in relationships.findall("p:Relationship", XLSX_REL_NS)
        if node.attrib.get("Type", "").endswith("/worksheet")
    }
    sheets: dict[str, str] = {}
    relationship_key = (
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )
    for node in workbook.findall("m:sheets/m:sheet", XLSX_NS):
        name = node.attrib.get("name", "")
        relationship_id = node.attrib.get(relationship_key, "")
        target = targets.get(relationship_id)
        if name not in ANNOUNCEMENT_HEADERS:
            continue
        if not target or target.startswith("/") or ".." in Path(target).parts:
            raise ValueError(f"unsafe announcement worksheet target: {target!r}")
        sheets[name] = f"xl/{target}"
    if set(sheets) != set(ANNOUNCEMENT_HEADERS):
        raise ValueError(
            "announcement workbook sheet names changed: "
            f"expected={sorted(ANNOUNCEMENT_HEADERS)}, actual={sorted(sheets)}"
        )
    return sheets


def _validate_announcement_header(
    sheet_name: str, rows: Sequence[tuple[int, dict[str, str]]]
) -> None:
    if not rows or rows[0][0] != 1:
        raise ValueError(f"announcement sheet {sheet_name} has no header")
    actual = {key: " ".join(value.split()) for key, value in rows[0][1].items()}
    expected = ANNOUNCEMENT_HEADERS[sheet_name]
    if actual != expected:
        raise ValueError(
            f"announcement sheet {sheet_name} header changed: "
            f"expected={expected}, actual={actual}"
        )


def parse_announcements(payload: bytes, *, cap: date = MAX_REPORT_DATE) -> list[Announcement]:
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            shared = _xlsx_shared_strings(archive)
            sheet_paths = _xlsx_announcement_sheets(archive)
            sheets = {
                sheet_name: _xlsx_sheet_rows(archive, path, shared)
                for sheet_name, path in sheet_paths.items()
            }
    except BadZipFile as exc:
        raise ValueError("announcement workbook is not a valid XLSX archive") from exc

    announcements: list[Announcement] = []
    for sheet_name, rows in sheets.items():
        _validate_announcement_header(sheet_name, rows)
        for row_number, values in rows[1:]:
            raw_date = values.get("A", "").strip()
            if not raw_date:
                continue
            effective = _excel_serial_to_date(raw_date)
            if effective > cap:
                continue
            if sheet_name == "DTS Changes":
                table_name = values.get("B", "")
                table_section = values.get("C", "")
                entity = ""
                change = values.get("D", "")
            else:
                table_name = values.get("B", "")
                table_section = values.get("C", "")
                entity = values.get("D", "")
                change = values.get("E", "")
            if not change.strip():
                raise ValueError(
                    f"announcement {sheet_name}!{row_number} has no change text"
                )
            announcements.append(
                Announcement(
                    effective_date=effective,
                    table_name=" ".join(table_name.split()),
                    table_section=" ".join(table_section.split()),
                    entity=" ".join(entity.split()),
                    change=" ".join(change.split()),
                    sheet_name=sheet_name,
                    source_row=row_number,
                )
            )
    announcements.sort(
        key=lambda row: (
            row.effective_date,
            row.sheet_name,
            row.source_row,
            row.change,
        )
    )
    if not announcements:
        raise ValueError("announcement workbook yielded no pre-cap rows")
    return announcements


_OBJECT_HEADER = re.compile(
    rb"(?:\A|[\r\n])[ \t]*(\d+)\s+(\d+)\s+obj[ \t]*(?:\r\n|\r|\n)"
)
_STREAM_HEADER = re.compile(rb"\bstream[ \t]*(?:\r\n|\r|\n)")
_Matrix = tuple[float, float, float, float, float, float]


def _parse_pdf_objects(payload: bytes) -> dict[int, _PdfObject]:
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
        raise ValueError("DTS response is not a complete PDF")
    objects: dict[int, _PdfObject] = {}
    for match in _OBJECT_HEADER.finditer(payload):
        number = int(match.group(1))
        generation = int(match.group(2))
        body_start = match.end()
        stream_match = _STREAM_HEADER.search(
            payload, body_start, min(len(payload), body_start + 16_384)
        )
        plain_end = payload.find(b"endobj", body_start)
        if plain_end < 0:
            raise ValueError(f"PDF object {number} has no endobj marker")
        if stream_match is not None and stream_match.start() < plain_end:
            stream_end_hint = payload.find(b"endstream", stream_match.end())
            if stream_end_hint < 0:
                raise ValueError(f"PDF stream object {number} has no endstream marker")
            object_end = payload.find(b"endobj", stream_end_hint + len(b"endstream"))
            if object_end < 0:
                raise ValueError(f"PDF stream object {number} has no endobj marker")
            dictionary = payload[body_start : stream_match.start()].strip()
            body = payload[body_start:object_end]
            parsed = _PdfObject(
                number=number,
                generation=generation,
                offset=match.start(1),
                body=body,
                dictionary=dictionary,
                stream_start=stream_match.end(),
                stream_end_hint=stream_end_hint,
            )
        else:
            body = payload[body_start:plain_end].strip()
            parsed = _PdfObject(
                number=number,
                generation=generation,
                offset=match.start(1),
                body=body,
                dictionary=body,
                stream_start=None,
                stream_end_hint=None,
            )
        previous = objects.get(number)
        if previous is None or parsed.offset > previous.offset:
            objects[number] = parsed
    if not objects:
        raise ValueError("DTS PDF contains no parseable objects")
    if any(b"/Encrypt" in item.dictionary for item in objects.values()):
        raise ValueError("encrypted DTS PDFs are unsupported")
    return objects


def _resolve_stream_length(item: _PdfObject, objects: dict[int, _PdfObject]) -> int:
    indirect = re.search(rb"/Length\s+(\d+)\s+(\d+)\s+R\b", item.dictionary)
    if indirect is not None:
        reference = int(indirect.group(1))
        generation = int(indirect.group(2))
        target = objects.get(reference)
        if target is None or target.generation != generation or target.stream_start:
            raise ValueError(f"invalid indirect PDF stream length for object {item.number}")
        scalar = target.body.strip()
        if re.fullmatch(rb"\d+", scalar) is None:
            raise ValueError(f"non-integer PDF stream length for object {item.number}")
        return int(scalar)
    direct = re.search(rb"/Length\s+(\d+)\b", item.dictionary)
    if direct is None:
        raise ValueError(f"PDF stream object {item.number} has no length")
    return int(direct.group(1))


def _decode_pdf_stream(
    payload: bytes, item: _PdfObject, objects: dict[int, _PdfObject]
) -> bytes:
    if item.stream_start is None:
        raise ValueError(f"PDF object {item.number} is not a stream")
    length = _resolve_stream_length(item, objects)
    stream_end = item.stream_start + length
    if stream_end > len(payload):
        raise ValueError(f"PDF stream object {item.number} exceeds the file")
    suffix = payload[stream_end : stream_end + 32].lstrip(b"\r\n \t")
    if not suffix.startswith(b"endstream"):
        raise ValueError(f"PDF stream length mismatch for object {item.number}")
    filters: list[str] = []
    array_match = re.search(rb"/Filter\s*\[([^]]*)\]", item.dictionary, re.DOTALL)
    if array_match is not None:
        filters = [
            value.decode("ascii")
            for value in re.findall(rb"/([A-Za-z0-9]+)", array_match.group(1))
        ]
    else:
        single_match = re.search(rb"/Filter\s*/([A-Za-z0-9]+)", item.dictionary)
        if single_match is not None:
            filters = [single_match.group(1).decode("ascii")]
    if b"/DecodeParms" in item.dictionary:
        raise ValueError(f"PDF stream object {item.number} uses unsupported DecodeParms")
    raw = payload[item.stream_start:stream_end]
    if not filters:
        return raw
    if filters != ["FlateDecode"]:
        raise ValueError(
            f"PDF stream object {item.number} uses unsupported filters: {filters}"
        )
    try:
        return zlib.decompress(raw)
    except zlib.error as exc:
        raise ValueError(f"PDF stream object {item.number} is not valid zlib") from exc


def _page_content_references(item: _PdfObject) -> list[int]:
    array_match = re.search(rb"/Contents\s*\[([^]]*)\]", item.dictionary, re.DOTALL)
    if array_match is not None:
        references = [
            int(value)
            for value in re.findall(rb"(\d+)\s+\d+\s+R\b", array_match.group(1))
        ]
    else:
        single = re.search(rb"/Contents\s+(\d+)\s+\d+\s+R\b", item.dictionary)
        references = [] if single is None else [int(single.group(1))]
    if not references:
        raise ValueError(f"DTS PDF page object {item.number} has no content streams")
    return references


def _pdf_literal_string(payload: bytes, start: int) -> tuple[bytes, int]:
    if start >= len(payload) or payload[start] != ord("("):
        raise ValueError("PDF literal string must start with '('")
    index = start + 1
    depth = 1
    output = bytearray()
    escape_map = {
        ord("n"): ord("\n"),
        ord("r"): ord("\r"),
        ord("t"): ord("\t"),
        ord("b"): ord("\b"),
        ord("f"): ord("\f"),
        ord("("): ord("("),
        ord(")"): ord(")"),
        ord("\\"): ord("\\"),
    }
    while index < len(payload):
        value = payload[index]
        index += 1
        if value == ord("\\"):
            if index >= len(payload):
                raise ValueError("unterminated PDF literal-string escape")
            escaped = payload[index]
            index += 1
            if escaped in escape_map:
                output.append(escape_map[escaped])
            elif ord("0") <= escaped <= ord("7"):
                digits = bytearray([escaped])
                for _ in range(2):
                    if index < len(payload) and ord("0") <= payload[index] <= ord("7"):
                        digits.append(payload[index])
                        index += 1
                    else:
                        break
                decoded = int(digits.decode("ascii"), 8)
                if decoded > 255:
                    raise ValueError("PDF octal escape exceeds one byte")
                output.append(decoded)
            elif escaped == ord("\r"):
                if index < len(payload) and payload[index] == ord("\n"):
                    index += 1
            elif escaped == ord("\n"):
                pass
            else:
                output.append(escaped)
        elif value == ord("("):
            depth += 1
            output.append(value)
        elif value == ord(")"):
            depth -= 1
            if depth == 0:
                return bytes(output), index
            output.append(value)
        else:
            output.append(value)
    raise ValueError("unterminated PDF literal string")


def _pdf_hex_string(payload: bytes, start: int) -> tuple[bytes, int]:
    if payload[start : start + 2] == b"<<":
        raise ValueError("PDF dictionaries are unsupported in content streams")
    end = payload.find(b">", start + 1)
    if end < 0:
        raise ValueError("unterminated PDF hexadecimal string")
    compact = re.sub(rb"\s+", b"", payload[start + 1 : end])
    if re.fullmatch(rb"[0-9A-Fa-f]*", compact) is None:
        raise ValueError("invalid PDF hexadecimal string")
    if len(compact) % 2:
        compact += b"0"
    return bytes.fromhex(compact.decode("ascii")), end + 1


def _pdf_tokens(payload: bytes) -> Iterator[tuple[str, Any]]:
    index = 0
    delimiters = b"()<>[]{}/%"
    whitespace = b"\x00\t\n\x0c\r "
    while index < len(payload):
        value = payload[index]
        if value in whitespace:
            index += 1
            continue
        if value == ord("%"):
            while index < len(payload) and payload[index] not in b"\r\n":
                index += 1
            continue
        if value == ord("("):
            decoded, index = _pdf_literal_string(payload, index)
            yield "string", decoded
            continue
        if value == ord("<"):
            decoded, index = _pdf_hex_string(payload, index)
            yield "string", decoded
            continue
        if value == ord("["):
            index += 1
            yield "array_start", None
            continue
        if value == ord("]"):
            index += 1
            yield "array_end", None
            continue
        if value == ord("/"):
            end = index + 1
            while end < len(payload) and payload[end] not in whitespace + delimiters:
                end += 1
            yield "name", payload[index:end].decode("ascii")
            index = end
            continue
        end = index
        while end < len(payload) and payload[end] not in whitespace + delimiters:
            end += 1
        if end == index:
            raise ValueError(f"unsupported PDF content delimiter: {chr(value)!r}")
        raw = payload[index:end]
        index = end
        try:
            yield "number", float(raw)
        except ValueError:
            try:
                yield "operator", raw.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("non-ASCII PDF content operator") from exc


def _multiply_matrix(left: _Matrix, right: _Matrix) -> _Matrix:
    a, b, c, d, e, f = left
    g, h, i, j, k, ell = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * ell + e,
        b * k + d * ell + f,
    )


def parse_pdf_content_cells(payload: bytes, *, page_number: int) -> list[PdfTextCell]:
    operands: list[Any] = []
    arrays: list[list[Any]] = []
    ctm: _Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    graphics_stack: list[_Matrix] = []
    text_matrix: _Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line_matrix = text_matrix
    leading = 0.0
    font_name = ""
    font_size = 0.0
    in_text = False
    source_order = 0
    cells: list[PdfTextCell] = []

    def require_numbers(count: int, operator: str) -> list[float]:
        if len(operands) < count or not all(
            isinstance(value, float) for value in operands[-count:]
        ):
            raise ValueError(f"PDF {operator} has invalid operands")
        return operands[-count:]

    def show_text(value: Any) -> None:
        nonlocal source_order
        if not in_text:
            raise ValueError("PDF text-showing operator occurs outside BT/ET")
        parts = value if isinstance(value, list) else [value]
        if not all(isinstance(part, (bytes, float)) for part in parts):
            raise ValueError("PDF text array contains an invalid value")
        chunks: list[bytes] = []
        for part in parts:
            if isinstance(part, bytes):
                chunks.append(part)
            elif part <= -200.0 and chunks and not chunks[-1].endswith(b" "):
                # Legacy DTS PDFs encode Helvetica's ordinary word space as a
                # -278 TJ displacement instead of embedding a space byte.
                chunks.append(b" ")
        raw = b"".join(chunks)
        try:
            text = unicodedata.normalize("NFC", raw.decode("cp1252"))
        except UnicodeDecodeError as exc:
            raise ValueError("DTS PDF text is not valid WinAnsi/cp1252") from exc
        matrix = _multiply_matrix(ctm, text_matrix)
        cells.append(
            PdfTextCell(
                page_number=page_number,
                source_order=source_order,
                x=round(matrix[4], 6),
                y=round(matrix[5], 6),
                text=text,
                font_name=font_name,
                font_size=round(font_size, 6),
            )
        )
        source_order += 1

    for token_type, value in _pdf_tokens(payload):
        if token_type == "array_start":
            arrays.append(operands)
            operands = []
            continue
        if token_type == "array_end":
            if not arrays:
                raise ValueError("unmatched PDF array close")
            array_value = operands
            operands = arrays.pop()
            operands.append(array_value)
            continue
        if token_type in {"number", "name", "string"}:
            operands.append(value)
            continue
        operator = str(value)
        if operator == "q":
            graphics_stack.append(ctm)
        elif operator == "Q":
            if not graphics_stack:
                raise ValueError("unmatched PDF graphics-state restore")
            ctm = graphics_stack.pop()
        elif operator == "cm":
            values = require_numbers(6, operator)
            ctm = _multiply_matrix(ctm, tuple(values))  # type: ignore[arg-type]
        elif operator == "BT":
            if in_text:
                raise ValueError("nested PDF BT operator")
            in_text = True
            text_matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            line_matrix = text_matrix
        elif operator == "ET":
            if not in_text:
                raise ValueError("unmatched PDF ET operator")
            in_text = False
        elif operator == "Tm":
            values = require_numbers(6, operator)
            text_matrix = tuple(values)  # type: ignore[assignment]
            line_matrix = text_matrix
        elif operator in {"Td", "TD"}:
            tx, ty = require_numbers(2, operator)
            if operator == "TD":
                leading = -ty
            line_matrix = _multiply_matrix(
                line_matrix, (1.0, 0.0, 0.0, 1.0, tx, ty)
            )
            text_matrix = line_matrix
        elif operator == "TL":
            leading = require_numbers(1, operator)[0]
        elif operator == "T*":
            line_matrix = _multiply_matrix(
                line_matrix, (1.0, 0.0, 0.0, 1.0, 0.0, -leading)
            )
            text_matrix = line_matrix
        elif operator == "Tf":
            if (
                len(operands) < 2
                or not isinstance(operands[-2], str)
                or not isinstance(operands[-1], float)
            ):
                raise ValueError("PDF Tf has invalid operands")
            font_name = operands[-2]
            font_size = operands[-1]
        elif operator in {"Tj", "TJ"}:
            if not operands:
                raise ValueError(f"PDF {operator} has no text operand")
            show_text(operands[-1])
        elif operator == "'":
            if not operands:
                raise ValueError("PDF quote operator has no text operand")
            line_matrix = _multiply_matrix(
                line_matrix, (1.0, 0.0, 0.0, 1.0, 0.0, -leading)
            )
            text_matrix = line_matrix
            show_text(operands[-1])
        elif operator == '"':
            if len(operands) < 3:
                raise ValueError("PDF double-quote operator has invalid operands")
            line_matrix = _multiply_matrix(
                line_matrix, (1.0, 0.0, 0.0, 1.0, 0.0, -leading)
            )
            text_matrix = line_matrix
            show_text(operands[-1])
        elif operator in {"G", "g", "w", "Tc"}:
            require_numbers(1, operator)
        elif operator in {"m", "l"}:
            require_numbers(2, operator)
        elif operator == "re":
            require_numbers(4, operator)
        elif operator == "d":
            if (
                len(operands) < 2
                or not isinstance(operands[-2], list)
                or not all(isinstance(item, float) for item in operands[-2])
                or not isinstance(operands[-1], float)
            ):
                raise ValueError("PDF d has invalid operands")
        elif operator in {"h", "W", "n", "S", "f"}:
            if operands:
                raise ValueError(f"PDF {operator} unexpectedly has operands")
        elif operator == "Do":
            if (
                not operands
                or not isinstance(operands[-1], str)
                or re.fullmatch(r"/Im\d+", operands[-1]) is None
            ):
                raise ValueError("DTS PDF Do is not a known image invocation")
        elif operator == "BI":
            raise ValueError("inline PDF images are unsupported")
        else:
            raise ValueError(f"unsupported PDF content operator: {operator}")
        operands = []
    if arrays:
        raise ValueError("unclosed PDF content array")
    if graphics_stack:
        raise ValueError("unrestored PDF graphics state")
    if in_text:
        raise ValueError("unclosed PDF text object")
    return cells


def _pdf_page_cells(payload: bytes) -> tuple[PdfTextCell, ...]:
    objects = _parse_pdf_objects(payload)
    pages = sorted(
        (
            item
            for item in objects.values()
            if re.search(rb"/Type\s*/Page\b", item.dictionary)
        ),
        key=lambda item: item.offset,
    )
    if not pages:
        raise ValueError("DTS PDF contains no page objects")
    all_cells: list[PdfTextCell] = []
    for page_number, page in enumerate(pages, start=1):
        streams: list[bytes] = []
        for reference in _page_content_references(page):
            item = objects.get(reference)
            if item is None:
                raise ValueError(f"DTS PDF page references missing object {reference}")
            streams.append(_decode_pdf_stream(payload, item, objects))
        content = b"\n".join(streams)
        cells = parse_pdf_content_cells(content, page_number=page_number)
        all_cells.extend(cells)
    if not any(cell.text.strip() for cell in all_cells):
        raise ValueError("DTS PDF contains no extractable text")
    return tuple(all_cells)


@dataclass(frozen=True)
class _TextLine:
    page_number: int
    y: float
    cells: tuple[PdfTextCell, ...]


def _group_text_lines(cells: Sequence[PdfTextCell]) -> list[_TextLine]:
    lines: list[_TextLine] = []
    for page_number in sorted({cell.page_number for cell in cells}):
        page_cells = sorted(
            (cell for cell in cells if cell.page_number == page_number),
            key=lambda cell: (-cell.y, cell.x, cell.source_order),
        )
        groups: list[list[PdfTextCell]] = []
        for cell in page_cells:
            if not groups or abs(groups[-1][0].y - cell.y) > 0.40:
                groups.append([cell])
            else:
                groups[-1].append(cell)
        for group in groups:
            group.sort(key=lambda cell: (cell.x, cell.source_order))
            lines.append(
                _TextLine(
                    page_number=page_number,
                    y=round(sum(cell.y for cell in group) / len(group), 6),
                    cells=tuple(group),
                )
            )
    return lines


def _line_text(line: _TextLine) -> str:
    text = " ".join(cell.text.strip() for cell in line.cells if cell.text.strip())
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text


def _table_title(text: str) -> str | None:
    canonical = text.upper().replace("–", "-").replace("—", "-").strip()
    if not canonical.startswith("TABLE "):
        return None
    if re.search(r"\bTABLE\s+IIIA\b|\bTABLE\s+III\s*-\s*A\b", canonical):
        return "IIIA"
    if re.search(r"\bTABLE\s+II\b(?!I)", canonical):
        return "II"
    if re.search(r"\bTABLE\s+I\b(?!I)", canonical):
        return "I"
    if re.search(r"\bTABLE\s+III(?:B|C|\s*-\s*[BC])\b", canonical):
        return "OTHER"
    if re.search(r"\bTABLE\s+(?:IV|V|VI)\b", canonical):
        return "OTHER"
    return None


def _report_date_from_cells(cells: Sequence[PdfTextCell]) -> date:
    matches: set[date] = set()
    for line in _group_text_lines(cells):
        for raw in PDF_DATE_PATTERN.findall(_line_text(line)):
            matches.add(datetime.strptime(raw, "%A, %B %d, %Y").date())
    if len(matches) != 1:
        raise ValueError(f"DTS PDF expected one report date, found {sorted(matches)}")
    return next(iter(matches))


def _creation_metadata(payload: bytes) -> tuple[str, ...]:
    values: set[str] = set()
    for match in re.finditer(rb"/CreationDate\s*", payload):
        index = match.end()
        while index < len(payload) and payload[index] in b" \t\r\n":
            index += 1
        if index < len(payload) and payload[index] == ord("("):
            raw, _ = _pdf_literal_string(payload, index)
            values.add(raw.decode("ascii", errors="strict"))
    return tuple(sorted(values))


def _mechanical_label_normalization(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    return " ".join(normalized.split())


def _is_header_only(value: str) -> bool:
    words = set(re.findall(r"[A-Z]+", value.upper()))
    return bool(words) and words <= {
        "DEPOSITS",
        "WITHDRAWALS",
        "ISSUES",
        "REDEMPTIONS",
        "TODAY",
        "THIS",
        "MONTH",
        "TO",
        "DATE",
        "FISCAL",
        "YEAR",
    }


def _parse_amount_token(value: str) -> int | None:
    stripped = value.strip()
    if not NUMBER_TOKEN.fullmatch(stripped):
        return None
    return int(stripped.replace(",", ""))


def _parse_amount_cell(value: str) -> tuple[bool, int | None, str, str]:
    stripped = " ".join(value.strip().split())
    match = re.fullmatch(
        r"(?:\$\s*)?(?:(\d+/|\*+|[a-z]/)\s*)?"
        r"(-?(?:0|[1-9]\d{0,2}(?:,\d{3})*)|[-–—*])",
        stripped,
        re.IGNORECASE,
    )
    if match is None:
        return False, None, "", ""
    footnote = match.group(1) or ""
    raw_amount = match.group(2)
    amount = _parse_amount_token(raw_amount)
    missing = "" if amount is not None else raw_amount
    return True, amount, footnote, missing


def _is_table_i_header(value: str) -> bool:
    words = set(re.findall(r"[A-Z]+", value.upper()))
    return bool(words) and words <= {
        "TYPE",
        "OF",
        "ACCOUNT",
        "CLOSING",
        "BALANCE",
        "TODAY",
        "OPENING",
        "THIS",
        "MONTH",
        "FISCAL",
        "YEAR",
        "CASH",
        "DETAILS",
    }


def _extract_table_i_rows(
    cells: Sequence[PdfTextCell], *, record_date: date, source_sha256: str
) -> tuple[DtsTableIRow, ...]:
    lines = _group_text_lines(cells)
    titles = [
        (line.page_number, line.y, table)
        for line in lines
        if max((cell.font_size for cell in line.cells), default=0.0) >= 9.0
        for table in [_table_title(_line_text(line))]
        if table is not None
    ]
    table_i_titles = [item for item in titles if item[2] == "I"]
    if len(table_i_titles) != 1:
        raise ValueError(
            f"DTS PDF expected one full-size Table I title, found {len(table_i_titles)}"
        )
    page_number, start_y, _ = table_i_titles[0]
    lower_titles = [
        y for page, y, _ in titles if page == page_number and y < start_y
    ]
    end_y = max(lower_titles) if lower_titles else 0.0
    segment = sorted(
        (
            line
            for line in lines
            if line.page_number == page_number and end_y < line.y < start_y
        ),
        key=lambda line: -line.y,
    )
    available, execution, stage = source_clock(record_date)
    pending: list[str] = []
    pending_footnotes: list[str] = []
    output: list[DtsTableIRow] = []
    for line in segment:
        text = _line_text(line)
        if not text or _is_table_i_header(text):
            continue
        labels: list[str] = []
        footnotes: list[str] = []
        values: list[tuple[int | None, str]] = []
        for cell in line.cells:
            token = cell.text.strip()
            if not token or token == "$":
                continue
            recognized, amount, embedded_footnote, missing = _parse_amount_cell(token)
            if recognized:
                if embedded_footnote:
                    footnotes.append(embedded_footnote)
                values.append((amount, missing))
            elif FOOTNOTE_TOKEN.fullmatch(token):
                footnotes.append(token)
            else:
                labels.append(token)
        label_text = _mechanical_label_normalization(" ".join(labels))
        if not values:
            if label_text:
                pending.append(label_text)
            pending_footnotes.extend(footnotes)
            continue
        if len(values) not in {3, 4}:
            raise ValueError(
                f"DTS {record_date} Table I row {text!r} has "
                f"{len(values)} amount cells"
            )
        label_parts = [*pending]
        pending = []
        footnotes = [*pending_footnotes, *footnotes]
        pending_footnotes = []
        if label_text:
            label_parts.append(label_text)
        raw_label = _mechanical_label_normalization(" ".join(label_parts))
        if not raw_label:
            raise ValueError(f"DTS {record_date} Table I has an unlabeled row")
        published_values = [value for value, _ in values]
        missing_tokens = [
            {"column_index": index, "literal": literal}
            for index, (_, literal) in enumerate(values)
            if literal
        ]
        output.append(
            DtsTableIRow(
                record_date=record_date.isoformat(),
                source_available_not_before_utc=available.isoformat(),
                earliest_execution_time_utc=execution.isoformat(),
                research_stage=stage,
                raw_category_label=raw_label,
                normalized_category_label=_mechanical_label_normalization(raw_label),
                published_value_count=len(values),
                published_values_usd_millions_json=canonical_json(
                    published_values
                ).decode("utf-8").strip(),
                missing_value_tokens_json=canonical_json(missing_tokens)
                .decode("utf-8")
                .strip(),
                footnote_markers=",".join(footnotes),
                schema_variant=(
                    "legacy_four_column" if len(values) == 4 else "modern_three_column"
                ),
                page_number=page_number,
                source_order=min(cell.source_order for cell in line.cells),
                source_pdf_sha256=source_sha256,
            )
        )
    if not output:
        raise ValueError("DTS PDF yielded no Table I operating-cash rows")
    value_counts = {row.published_value_count for row in output}
    if len(value_counts) != 1:
        raise ValueError(
            f"DTS {record_date} Table I mixes published column counts: {value_counts}"
        )
    return tuple(output)


def _extract_table_rows(
    cells: Sequence[PdfTextCell], *, record_date: date, source_sha256: str
) -> tuple[DtsRow, ...]:
    lines = _group_text_lines(cells)
    titles: list[tuple[int, float, str]] = []
    for line in lines:
        if max((cell.font_size for cell in line.cells), default=0.0) < 9.0:
            continue
        table = _table_title(_line_text(line))
        if table is not None:
            titles.append((line.page_number, line.y, table))
    if not any(table == "II" for _, _, table in titles):
        raise ValueError("DTS PDF is missing a full-size Table II title")
    if not any(table == "IIIA" for _, _, table in titles):
        raise ValueError("DTS PDF is missing a full-size Table IIIA title")

    segments: list[tuple[str, int, float, float]] = []
    for page_number in sorted({page for page, _, _ in titles}):
        page_titles = sorted(
            ((y, table) for page, y, table in titles if page == page_number),
            reverse=True,
        )
        for index, (start_y, table) in enumerate(page_titles):
            if table not in {"II", "IIIA"}:
                continue
            end_y = page_titles[index + 1][0] if index + 1 < len(page_titles) else 0.0
            segments.append((table, page_number, start_y, end_y))
    segments.sort(key=lambda item: (item[1], -item[2]))

    available, execution, stage = source_clock(record_date)
    pending: dict[tuple[str, str], list[str]] = {}
    section_stacks: dict[tuple[str, str], list[tuple[float, str]]] = {}
    output: list[DtsRow] = []
    center_x = 304.5
    null_tokens = {"-", "–", "—", "*"}

    for table, page_number, start_y, end_y in segments:
        segment_lines = [
            line
            for line in lines
            if line.page_number == page_number and end_y < line.y < start_y
        ]
        segment_lines.sort(key=lambda line: -line.y)
        side_names = (
            ("deposit", "withdrawal") if table == "II" else ("issue", "redemption")
        )
        for line in segment_lines:
            for side_index, side in enumerate(side_names):
                side_cells = tuple(
                    cell
                    for cell in line.cells
                    if (cell.x < center_x) == (side_index == 0)
                )
                if not side_cells:
                    continue
                text = _line_text(
                    _TextLine(page_number=line.page_number, y=line.y, cells=side_cells)
                )
                if not text or _is_header_only(text) or text == "See Footnote":
                    continue
                if _table_title(text) is not None or text.startswith("Table continued"):
                    continue

                labels: list[str] = []
                footnotes: list[str] = []
                values: list[tuple[int | None, str]] = []
                for cell in side_cells:
                    token = cell.text.strip()
                    if not token or token == "$":
                        continue
                    recognized, amount, embedded_footnote, missing = _parse_amount_cell(
                        token
                    )
                    if recognized:
                        if embedded_footnote:
                            footnotes.append(embedded_footnote)
                        values.append((amount, missing))
                    elif token in null_tokens:
                        numeric_threshold = 160.0 if side_index == 0 else 430.0
                        if cell.x >= numeric_threshold:
                            values.append((None, token))
                        else:
                            labels.append(token)
                    elif FOOTNOTE_TOKEN.fullmatch(token):
                        footnotes.append(token)
                    else:
                        labels.append(token)

                key = (table, side)
                label_text = _mechanical_label_normalization(" ".join(labels))
                if not values:
                    if not label_text:
                        continue
                    if label_text.endswith(":"):
                        x = min(cell.x for cell in side_cells if cell.text.strip())
                        stack = section_stacks.setdefault(key, [])
                        while stack and stack[-1][0] >= x - 0.5:
                            stack.pop()
                        stack.append((x, label_text.removesuffix(":")))
                        pending[key] = []
                    else:
                        pending.setdefault(key, []).append(label_text)
                    continue
                if len(values) != 3:
                    raise ValueError(
                        f"DTS {record_date} {table}/{side} row {text!r} has "
                        f"{len(values)} amount cells"
                    )
                label_parts = pending.pop(key, [])
                if label_text:
                    label_parts.append(label_text)
                raw_label = _mechanical_label_normalization(" ".join(label_parts))
                if not raw_label:
                    raise ValueError(
                        f"DTS {record_date} {table}/{side} has an unlabeled numeric row"
                    )
                stack = section_stacks.get(key, [])
                parent_section = " > ".join(value for _, value in stack)
                missing = [
                    f"{name}={literal}"
                    for name, (_, literal) in zip(
                        ("today", "month_to_date", "fiscal_year_to_date"), values
                    )
                    if literal
                ]
                row_kind = (
                    "total"
                    if re.match(r"^(?:Total|Net Change|Change in Balance)", raw_label)
                    else "detail"
                )
                output.append(
                    DtsRow(
                        record_date=record_date.isoformat(),
                        source_available_not_before_utc=available.isoformat(),
                        earliest_execution_time_utc=execution.isoformat(),
                        research_stage=stage,
                        table_id=table,
                        side=side,
                        parent_section=parent_section,
                        raw_category_label=raw_label,
                        normalized_category_label=_mechanical_label_normalization(
                            raw_label
                        ),
                        today_amount_usd_millions=(
                            "" if values[0][0] is None else str(values[0][0])
                        ),
                        month_to_date_amount_usd_millions=(
                            "" if values[1][0] is None else str(values[1][0])
                        ),
                        fiscal_year_to_date_amount_usd_millions=(
                            "" if values[2][0] is None else str(values[2][0])
                        ),
                        footnote_markers=",".join(footnotes),
                        missing_value_tokens=",".join(missing),
                        row_kind=row_kind,
                        page_number=page_number,
                        source_order=min(cell.source_order for cell in side_cells),
                        source_pdf_sha256=source_sha256,
                    )
                )
    if not any(row.table_id == "II" and row.side == "deposit" for row in output):
        raise ValueError("DTS PDF yielded no Table II deposit rows")
    if not any(row.table_id == "II" and row.side == "withdrawal" for row in output):
        raise ValueError("DTS PDF yielded no Table II withdrawal rows")
    if not any(row.table_id == "IIIA" and row.side == "issue" for row in output):
        raise ValueError("DTS PDF yielded no Table IIIA issue rows")
    if not any(row.table_id == "IIIA" and row.side == "redemption" for row in output):
        raise ValueError("DTS PDF yielded no Table IIIA redemption rows")
    output.sort(
        key=lambda row: (row.page_number, row.source_order, row.table_id, row.side)
    )
    return tuple(output)


def parse_dts_pdf(payload: bytes, *, expected_record_date: date) -> ParsedDtsPdf:
    if not MIN_REPORT_DATE <= expected_record_date <= MAX_REPORT_DATE:
        raise ValueError("expected DTS report date violates the frozen source cap")
    cells = _pdf_page_cells(payload)
    record_date = _report_date_from_cells(cells)
    if record_date != expected_record_date:
        raise ValueError(
            f"DTS filename/report date mismatch: expected={expected_record_date}, "
            f"parsed={record_date}"
        )
    table_ids = sorted(
        {
            table
            for line in _group_text_lines(cells)
            for table in [_table_title(_line_text(line))]
            if table in {"I", "II", "IIIA"}
        }
    )
    source_hash = sha256_bytes(payload)
    table_i_rows = _extract_table_i_rows(
        cells, record_date=record_date, source_sha256=source_hash
    )
    rows = _extract_table_rows(
        cells, record_date=record_date, source_sha256=source_hash
    )
    return ParsedDtsPdf(
        record_date=record_date,
        creation_metadata_raw=_creation_metadata(payload),
        table_ids_found=tuple(table_ids),
        cells=cells,
        table_i_rows=table_i_rows,
        rows=rows,
    )


class _SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, allow_metadata: bool) -> None:
        super().__init__()
        self._allow_metadata = allow_metadata
        self.redirect_chain: list[RedirectHop] = []

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != FISCAL_DATA_HOST:
            raise ValueError(f"unsafe Fiscal Data redirect: {newurl}")
        _validate_official_url(newurl, allow_metadata=self._allow_metadata)
        self.redirect_chain.append(RedirectHop(status=int(code), target_url=newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_official_bytes(
    url: str,
    *,
    allow_metadata: bool,
    retries: int,
    timeout_seconds: int,
) -> tuple[bytes, FetchReceipt]:
    _validate_official_url(url, allow_metadata=allow_metadata)
    redirect_handler = _SameHostRedirectHandler(allow_metadata=allow_metadata)
    opener = urllib.request.build_opener(redirect_handler)
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": (
                        "application/json,application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                        if allow_metadata
                        else "application/pdf"
                    ),
                },
            )
            with opener.open(request, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                parsed_final = urllib.parse.urlsplit(final_url)
                if (
                    parsed_final.scheme != "https"
                    or parsed_final.hostname != FISCAL_DATA_HOST
                ):
                    raise ValueError(f"unsafe terminal Fiscal Data URL: {final_url}")
                _validate_official_url(final_url, allow_metadata=allow_metadata)
                payload = response.read()
                receipt = FetchReceipt(
                    url=url,
                    status=int(getattr(response, "status", 200)),
                    final_url=final_url,
                    retrieved_at_utc=datetime.now(UTC).isoformat(),
                    etag=response.headers.get("ETag", ""),
                    last_modified=response.headers.get("Last-Modified", ""),
                    redirect_chain=tuple(redirect_handler.redirect_chain),
                    byte_length=len(payload),
                    sha256=sha256_bytes(payload),
                )
                return payload, receipt
        except (urllib.error.URLError, TimeoutError, OSError, ssl.SSLError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from last_error


def _freeze_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current != payload:
            raise ValueError(f"frozen source bytes changed at {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _receipt_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".receipt.json")


def _load_bound_receipt(
    target: Path, *, url: str, payload: bytes, allow_metadata: bool
) -> FetchReceipt:
    path = _receipt_path(target)
    if not path.exists():
        raise ValueError(f"unbound cached source bytes are not trusted: {target}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        redirects = tuple(
            RedirectHop(status=int(row["status"]), target_url=str(row["target_url"]))
            for row in raw["redirect_chain"]
        )
        receipt = FetchReceipt(
            url=str(raw["url"]),
            status=int(raw["status"]),
            final_url=str(raw["final_url"]),
            retrieved_at_utc=str(raw["retrieved_at_utc"]),
            etag=str(raw["etag"]),
            last_modified=str(raw["last_modified"]),
            redirect_chain=redirects,
            byte_length=int(raw["byte_length"]),
            sha256=str(raw["sha256"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source receipt sidecar: {path}") from exc
    if (
        receipt.url != url
        or receipt.status != 200
        or receipt.byte_length != len(payload)
        or receipt.sha256 != sha256_bytes(payload)
    ):
        raise ValueError(f"source receipt does not bind cached bytes: {target}")
    _validate_official_url(receipt.final_url, allow_metadata=allow_metadata)
    for redirect in receipt.redirect_chain:
        _validate_official_url(redirect.target_url, allow_metadata=allow_metadata)
    return receipt


def _load_or_fetch(
    *,
    url: str,
    target: Path,
    allow_metadata: bool,
    config: BuildConfig,
) -> tuple[bytes, FetchReceipt]:
    if target.exists():
        payload = target.read_bytes()
        return payload, _load_bound_receipt(
            target, url=url, payload=payload, allow_metadata=allow_metadata
        )
    if config.request_pace_seconds > 0:
        time.sleep(config.request_pace_seconds)
    payload, receipt = fetch_official_bytes(
        url,
        allow_metadata=allow_metadata,
        retries=config.retries,
        timeout_seconds=config.timeout_seconds,
    )
    if allow_metadata:
        if url == DATASET_PAGE_DATA_URL and not payload.lstrip().startswith(b"{"):
            raise ValueError("Fiscal Data page-data response is not JSON")
        if url == ANNOUNCEMENTS_URL and not payload.startswith(b"PK"):
            raise ValueError("Fiscal Data announcement response is not XLSX")
    elif not payload.startswith(b"%PDF-"):
        raise ValueError(f"DTS report response is not PDF: {url}")
    _freeze_bytes(target, payload)
    _freeze_bytes(_receipt_path(target), canonical_json(asdict(receipt)))
    return payload, receipt


def _timezone_file_sha256() -> str:
    candidates = (
        Path("/usr/share/zoneinfo/America/New_York"),
        Path("/usr/share/lib/zoneinfo/America/New_York"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return sha256_file(candidate)
    raise ValueError("cannot bind America/New_York timezone data")


def _weekday_gap_audit(
    reports: Sequence[PublishedReport], *, start: date, end: date
) -> list[str]:
    available = {report.record_date for report in reports}
    gaps: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in available:
            if current not in federal_holidays(current.year):
                gaps.append(current.isoformat())
        current += timedelta(days=1)
    return gaps


def _announcement_rows(announcements: Sequence[Announcement]) -> list[dict[str, Any]]:
    return [
        {
            "effective_date": row.effective_date.isoformat(),
            "table_name": row.table_name,
            "table_section": row.table_section,
            "entity": row.entity,
            "change": row.change,
            "sheet_name": row.sheet_name,
            "source_row": row.source_row,
        }
        for row in announcements
    ]


def build_source(config: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(config.start_date)
    end = date.fromisoformat(config.end_date)
    if start < MIN_REPORT_DATE or end > MAX_REPORT_DATE or start > end:
        raise ValueError("build date range violates the frozen DTS source cap")
    if config.max_workers < 1:
        raise ValueError("max_workers must be positive")

    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"
    raw_reports_dir = raw_dir / "reports"
    raw_reports_dir.mkdir(parents=True, exist_ok=True)

    page_payload, page_receipt = _load_or_fetch(
        url=DATASET_PAGE_DATA_URL,
        target=raw_dir / "page-data.json",
        allow_metadata=True,
        config=config,
    )
    announcement_payload, announcement_receipt = _load_or_fetch(
        url=ANNOUNCEMENTS_URL,
        target=raw_dir / "DailyTreasuryStatement_Announcements.xlsx",
        allow_metadata=True,
        config=config,
    )
    reports = parse_published_reports(page_payload, start=start, end=end)
    announcements = parse_announcements(announcement_payload)

    receipts: list[FetchReceipt] = [page_receipt, announcement_receipt]
    parsed_reports: dict[date, ParsedDtsPdf] = {}
    report_payloads: dict[date, bytes] = {}
    report_receipts: dict[date, FetchReceipt] = {}

    def acquire(report: PublishedReport) -> tuple[date, bytes, FetchReceipt, ParsedDtsPdf]:
        raw_target = raw_reports_dir / f"{report.record_date:%Y%m%d}.pdf"
        payload, receipt = _load_or_fetch(
            url=report.url,
            target=raw_target,
            allow_metadata=False,
            config=config,
        )
        parsed = parse_dts_pdf(payload, expected_record_date=report.record_date)
        return report.record_date, payload, receipt, parsed

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {executor.submit(acquire, report): report for report in reports}
        for future in as_completed(futures):
            record_date, payload, receipt, parsed = future.result()
            parsed_reports[record_date] = parsed
            report_payloads[record_date] = payload
            report_receipts[record_date] = receipt

    all_rows = [
        asdict(row)
        for record_date in sorted(parsed_reports)
        for row in parsed_reports[record_date].rows
    ]
    all_rows.sort(
        key=lambda row: (
            row["record_date"],
            row["page_number"],
            row["source_order"],
            row["table_id"],
            row["side"],
        )
    )
    rows_payload = canonical_csv(all_rows, OUTPUT_COLUMNS)
    rows_path = output_dir / "daily_treasury_fiscal_flow_rows.csv.gz"
    write_gzip(rows_path, rows_payload)

    all_table_i_rows = [
        asdict(row)
        for record_date in sorted(parsed_reports)
        for row in parsed_reports[record_date].table_i_rows
    ]
    all_table_i_rows.sort(
        key=lambda row: (
            row["record_date"],
            row["page_number"],
            row["source_order"],
            row["raw_category_label"],
        )
    )
    table_i_payload = canonical_csv(all_table_i_rows, TABLE_I_OUTPUT_COLUMNS)
    table_i_path = output_dir / "daily_treasury_operating_cash_rows.csv.gz"
    write_gzip(table_i_path, table_i_payload)

    announcement_columns = (
        "effective_date",
        "table_name",
        "table_section",
        "entity",
        "change",
        "sheet_name",
        "source_row",
    )
    announcement_csv = canonical_csv(
        _announcement_rows(announcements), announcement_columns
    )
    announcement_path = output_dir / "precap_schema_announcements.csv.gz"
    write_gzip(announcement_path, announcement_csv)

    parser_source_path = Path(__file__)
    receipts.extend(report_receipts[record_date] for record_date in sorted(report_receipts))
    manifest_path = output_dir / "source_manifest.json"
    report_entries = []
    for report in reports:
        parsed = parsed_reports[report.record_date]
        available, execution, stage = source_clock(report.record_date)
        report_entries.append(
            {
                "record_date": report.record_date.isoformat(),
                "url": report.url,
                "byte_length": len(report_payloads[report.record_date]),
                "sha256": sha256_bytes(report_payloads[report.record_date]),
                "pdf_creation_metadata_raw": list(parsed.creation_metadata_raw),
                "table_ids_found": list(parsed.table_ids_found),
                "table_i_row_count": len(parsed.table_i_rows),
                "row_count": len(parsed.rows),
                "source_available_not_before_utc": available.isoformat(),
                "earliest_execution_time_utc": execution.isoformat(),
                "research_stage": stage,
            }
        )
    content_manifest = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "source_horizon": {"start": start.isoformat(), "end": end.isoformat()},
        "metadata": {
            "page_data_url": DATASET_PAGE_DATA_URL,
            "page_data_sha256": sha256_bytes(page_payload),
            "announcements_url": ANNOUNCEMENTS_URL,
            "announcements_sha256": sha256_bytes(announcement_payload),
            "precap_announcement_count": len(announcements),
        },
        "toolchain": {
            "parser_source_sha256": sha256_file(parser_source_path),
            "python": platform.python_version(),
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "unicode_database": unicodedata.unidata_version,
            "america_new_york_tzfile_sha256": _timezone_file_sha256(),
            "holiday_calendar_source_sha256": sha256_file(parser_source_path),
            "label_normalization_contract_sha256": sha256_bytes(
                b"NFC|dash-to-hyphen|collapse-whitespace|preserve-case-v1"
            ),
            "locale_contract": "LC_ALL=C",
        },
        "reports": report_entries,
        "normalized_rows": {
            "path": rows_path.name,
            "row_count": len(all_rows),
            "uncompressed_sha256": sha256_bytes(rows_payload),
            "file_sha256": sha256_file(rows_path),
        },
        "operating_cash_rows": {
            "path": table_i_path.name,
            "row_count": len(all_table_i_rows),
            "uncompressed_sha256": sha256_bytes(table_i_payload),
            "file_sha256": sha256_file(table_i_path),
        },
        "schema_announcements": {
            "path": announcement_path.name,
            "row_count": len(announcements),
            "uncompressed_sha256": sha256_bytes(announcement_csv),
            "file_sha256": sha256_file(announcement_path),
        },
    }
    _freeze_bytes(manifest_path, canonical_json(content_manifest))

    receipt_lines = b"".join(
        canonical_json(asdict(receipt))
        for receipt in sorted(receipts, key=lambda row: (row.url, row.retrieved_at_utc))
    )
    (output_dir / "receipt_log.jsonl").write_bytes(receipt_lines)

    weekday_gaps = _weekday_gap_audit(reports, start=start, end=end)
    stage_counts: dict[str, int] = {}
    for report in reports:
        stage = source_clock(report.record_date)[2]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    report = {
        "candidate_family": "DFFB",
        "decision": "SOURCE_BUILT_REQUIRES_SCHEMA_AUDIT",
        "calendar_coverage_gate_pass": len(weekday_gaps) == 0,
        "source_quality_gates_evaluated": False,
        "all_source_quality_gates_pass": None,
        "next_stage_authorized": None,
        "report_count": len(reports),
        "normalized_row_count": len(all_rows),
        "operating_cash_row_count": len(all_table_i_rows),
        "stage_report_counts": dict(sorted(stage_counts.items())),
        "first_report_date": reports[0].record_date.isoformat(),
        "last_report_date": reports[-1].record_date.isoformat(),
        "unexplained_weekday_gaps": weekday_gaps,
        "source_manifest_sha256": sha256_file(manifest_path),
        "protocol": {
            "source_only": True,
            "post_2023_report_opened": False,
            "post_2023_api_value_row_opened": False,
            "btc_market_data_opened": False,
            "funding_opened": False,
            "future_return_opened": False,
            "labels_opened": False,
            "pnl_cagr_mdd_opened": False,
            "current_metadata_postcap_rows_used_in_logic": False,
        },
    }
    report_path = output_dir / "source_build_report.json"
    _freeze_bytes(report_path, canonical_json(report))
    return report


def _parse_args(argv: Sequence[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=MIN_REPORT_DATE.isoformat())
    parser.add_argument("--end-date", default=MAX_REPORT_DATE.isoformat())
    parser.add_argument(
        "--output-dir", default="data/daily_treasury_fiscal_flow_2019_2023"
    )
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--request-pace-seconds", type=float, default=0.10)
    args = parser.parse_args(argv)
    return BuildConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        request_pace_seconds=args.request_pace_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["LC_ALL"] = "C"
    report = build_source(_parse_args(argv))
    sys.stdout.buffer.write(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
