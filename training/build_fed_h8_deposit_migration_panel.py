"""Freeze point-in-time Federal Reserve H.8 bank-balance-sheet releases.

The builder downloads the Federal Reserve's exact dated H.8 archive pages and
extracts the two latest Wednesday observations from the large-bank, small-bank,
and all-domestic-bank tables.  Both seasonally adjusted and not-seasonally-
adjusted levels are retained.  No BTC, funding, return, portfolio, or label data
are read.

H.8 changed its HTML layout during 2020.  The parser deliberately supports both
the older split-page tables and the current single-table layout, then validates
the accounting identities before accepting a release.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as wall_time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


RELEASE_DATES_URL = "https://www.federalreserve.gov/releases/h8/releaseDates.json"
ARCHIVE_URL = "https://www.federalreserve.gov/releases/h8/{release_date}/default.htm"
ABOUT_URL = "https://www.federalreserve.gov/releases/h8/about.htm"
DATA_DOWNLOAD_URL = "https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H8"
SOURCE_SNAPSHOT_DATE = "2026-07-18"
START_YEAR = 2017
END_YEAR = 2023
FROZEN_COVERAGE = (365, "2017-01-06", "2023-12-29")
# Filled after the official source snapshot is built and audited.
FROZEN_RELEASE_DATES_SNAPSHOT_SHA256 = (
    "20a7d218ffbe2c4a47508ff4c547fdee7047663f585b31ceba62c6f66b771629"
)
FROZEN_ARCHIVE_SNAPSHOT_SHA256 = (
    "65edb50eb2b2a01785518fe30a92acf1a35f0f8b78cd332a96557ff9bad8601a"
)
FROZEN_PANEL_SHA256 = "c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572"
USER_AGENT = "rllm-fed-h8-deposit-migration-freeze/1.0"
NEW_YORK = ZoneInfo("America/New_York")

GROUPS = ("domestic", "large", "small")
ADJUSTMENTS = ("sa", "nsa")
METRICS = (
    "cash_assets",
    "total_assets",
    "deposits",
    "large_time_deposits",
    "other_deposits",
    "borrowings",
)
LABELS = {
    "cash_assets": "Cash assets",
    "total_assets": "Total assets",
    "deposits": "Deposits",
    "large_time_deposits": "Large time deposits",
    "other_deposits": "Other deposits",
    "borrowings": "Borrowings",
}
BASE_COLUMNS = (
    "release_calendar_key",
    "archive_path_date",
    "release_date",
    "release_time_utc",
    "release_weekday",
    "holiday_shifted_release",
    "prior_week_ending",
    "latest_week_ending",
    "observation_lag_days",
    "archive_url",
    "response_sha256",
    "response_bytes",
    "html_schema",
)
LEVEL_COLUMNS = tuple(
    f"{adjustment}_{group}_{metric}_{point}"
    for adjustment in ADJUSTMENTS
    for group in GROUPS
    for metric in METRICS
    for point in ("prior", "latest")
)
PANEL_COLUMNS = (*BASE_COLUMNS, *LEVEL_COLUMNS)


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "data/fed_h8_deposit_migration_2017_2023"
    retries: int = 5
    timeout_seconds: int = 60
    from_snapshot: bool = False


@dataclass(frozen=True)
class ArchiveRecord:
    release_calendar_key: str
    archive_path_date: str
    url: str
    response_sha256: str
    body: bytes


class H8HTMLParser(HTMLParser):
    """Collect table metadata and normalized cell text from both H.8 layouts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._last_unit = ""
        self._unit_tag: str | None = None
        self._unit_class = ""
        self._unit_text: list[str] = []
        self._table: dict[str, Any] | None = None
        self._row: list[dict[str, str | None]] | None = None
        self._cell: dict[str, Any] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag in {"h5", "span"}:
            self._unit_tag = tag
            self._unit_class = attributes.get("class", "")
            self._unit_text = []
        if tag == "table":
            self._table = {
                "id": attributes.get("id") or None,
                "title": attributes.get("title")
                or attributes.get("summary")
                or "",
                "unit": self._last_unit,
                "rows": [],
            }
        elif self._table is not None and tag == "tr":
            self._row = []
        elif self._table is not None and tag in {"th", "td"}:
            self._cell = {
                "tag": tag,
                "id": attributes.get("id") or None,
                "headers": attributes.get("headers", ""),
                "text_parts": [],
            }

    def handle_data(self, data: str) -> None:
        if self._unit_tag is not None:
            self._unit_text.append(data)
        if self._cell is not None:
            self._cell["text_parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._unit_tag:
            text = _normalize_text(" ".join(self._unit_text))
            if (
                tag == "h5"
                or "tableunit" in self._unit_class
            ) and "adjusted" in text.lower():
                self._last_unit = text
            self._unit_tag = None
            self._unit_class = ""
            self._unit_text = []
        if tag in {"th", "td"} and self._cell is not None:
            cell = {
                "tag": self._cell["tag"],
                "id": self._cell["id"],
                "headers": self._cell["headers"],
                "text": _normalize_text(" ".join(self._cell["text_parts"])),
            }
            if self._row is not None:
                self._row.append(cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._table is not None:
                self._table["rows"].append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json,text/html;q=0.9,*/*;q=0.1",
                    "User-Agent": USER_AGENT,
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if not payload:
                    raise RuntimeError(f"empty H.8 response: {url}")
                return payload
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code == 404:
                break
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch H.8 source: {url}") from error


def parse_release_dates(payload: bytes) -> list[date]:
    try:
        source = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("H.8 releaseDates.json is invalid") from exc
    if not isinstance(source, list):
        raise ValueError("H.8 release-date root changed")
    selected: list[date] = []
    seen: set[date] = set()
    for year_item in source:
        if not isinstance(year_item, Mapping):
            raise ValueError("H.8 release-date year item changed")
        try:
            year = int(year_item["yearValue"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("H.8 release-date year changed") from exc
        months = year_item.get("Months")
        if not isinstance(months, list):
            raise ValueError("H.8 release-date months changed")
        if year < START_YEAR or year > END_YEAR:
            continue
        for month in months:
            if not isinstance(month, Mapping) or not isinstance(month.get("Dates"), list):
                raise ValueError("H.8 release-date month changed")
            for raw in month["Dates"]:
                try:
                    parsed = datetime.strptime(str(raw), "%Y%m%d").date()
                except ValueError as exc:
                    raise ValueError(f"invalid H.8 release date: {raw!r}") from exc
                if parsed.year != year or parsed in seen:
                    raise ValueError("H.8 release dates are duplicated or misfiled")
                seen.add(parsed)
                selected.append(parsed)
    selected.sort()
    expected_count, expected_first, expected_last = FROZEN_COVERAGE
    if (
        len(selected) != expected_count
        or selected[0].isoformat() != expected_first
        or selected[-1].isoformat() != expected_last
    ):
        raise ValueError("H.8 release-date coverage changed")
    if any((current - previous).days < 3 for previous, current in zip(selected, selected[1:])):
        raise ValueError("H.8 release dates contain implausibly close duplicates")
    return selected


def _table_group(title: str) -> str | None:
    if "Large Domestically Chartered Commercial Banks" in title:
        return "large"
    if "Small Domestically Chartered Commercial Banks" in title:
        return "small"
    if "Domestically Chartered Commercial Banks" in title:
        return "domestic"
    return None


def _adjustment(unit: str) -> str | None:
    lowered = unit.lower()
    if lowered.startswith("not seasonally adjusted"):
        return "nsa"
    if lowered.startswith("seasonally adjusted"):
        return "sa"
    return None


_WEEK_RE = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+0?(\d{1,2})$"
)


def _week_labels(rows: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[str, str, str, str]:
    candidates: list[tuple[str, str, str, str]] = []
    for row in rows:
        labels = [str(cell.get("text", "")) for cell in row]
        matched = [label for label in labels if _WEEK_RE.fullmatch(label)]
        if len(matched) == 4:
            candidates.append(tuple(matched))  # type: ignore[arg-type]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(f"H.8 weekly headers changed: {unique!r}")
    return unique[0]


def _resolve_week(label: str, release_date: date) -> date:
    match = _WEEK_RE.fullmatch(label)
    if not match:
        raise ValueError(f"invalid H.8 week label: {label!r}")
    month = datetime.strptime(match.group(1), "%b").month
    day = int(match.group(2))
    candidate = date(release_date.year, month, day)
    if candidate > release_date:
        candidate = date(release_date.year - 1, month, day)
    return candidate


def _decimal_cell(value: str) -> Decimal:
    cleaned = value.replace(",", "").strip()
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"H.8 level is not numeric: {value!r}") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"H.8 level must be positive and finite: {value!r}")
    return parsed


def _metric_key(label: str) -> str | None:
    normalized = _normalize_text(label)
    # Legacy pages append numeric footnote markers to the stub text.  Remove
    # only a trailing marker; prefix matching would wrongly classify rows such
    # as "Borrowings from banks in the U.S." as the aggregate Borrowings row.
    normalized = re.sub(r"(?:\s+\d+)+$", "", normalized)
    for key, expected in LABELS.items():
        if normalized == expected:
            return key
    return None


def _extract_table_values(
    tables: Sequence[Mapping[str, Any]],
    *,
    release_date: date,
) -> tuple[dict[tuple[str, str, str], tuple[Decimal, Decimal]], date, date, str]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    modern = False
    for table in tables:
        title = str(table.get("title", ""))
        group = _table_group(title)
        adjustment = _adjustment(str(table.get("unit", "")))
        if group is None or adjustment is None:
            continue
        grouped.setdefault((adjustment, group), []).append(table)
        modern = modern or bool(table.get("id"))
    expected_groups = {(adjustment, group) for adjustment in ADJUSTMENTS for group in GROUPS}
    if set(grouped) != expected_groups:
        raise ValueError(
            f"H.8 relevant table set changed: {sorted(grouped)}"
        )

    result: dict[tuple[str, str, str], tuple[Decimal, Decimal]] = {}
    common_weeks: tuple[date, date, date, date] | None = None
    for (adjustment, group), physical_tables in sorted(grouped.items()):
        labels_seen: tuple[str, str, str, str] | None = None
        for table in physical_tables:
            rows = table["rows"]
            labels = _week_labels(rows)
            if labels_seen is not None and labels != labels_seen:
                raise ValueError("H.8 split-table weekly headers disagree")
            labels_seen = labels
            for row in rows:
                texts = [str(cell.get("text", "")) for cell in row]
                if len(texts) < 6:
                    continue
                metric = _metric_key(texts[1])
                if metric is None:
                    continue
                numeric = [_decimal_cell(value) for value in texts[-4:]]
                key = (adjustment, group, metric)
                if key in result:
                    raise ValueError(f"duplicate H.8 metric row: {key}")
                result[key] = (numeric[-2], numeric[-1])
        if labels_seen is None:
            raise ValueError("H.8 table has no weekly headers")
        parsed_weeks = tuple(_resolve_week(label, release_date) for label in labels_seen)
        if common_weeks is not None and parsed_weeks != common_weeks:
            raise ValueError("H.8 relevant tables use different week endings")
        common_weeks = parsed_weeks

    expected_keys = {
        (adjustment, group, metric)
        for adjustment in ADJUSTMENTS
        for group in GROUPS
        for metric in METRICS
    }
    if set(result) != expected_keys:
        missing = sorted(expected_keys - set(result))
        extra = sorted(set(result) - expected_keys)
        raise ValueError(f"H.8 metric schema changed; missing={missing}, extra={extra}")
    if common_weeks is None:
        raise ValueError("H.8 page contains no usable week endings")
    if any((current - previous).days != 7 for previous, current in zip(common_weeks, common_weeks[1:])):
        raise ValueError("H.8 week endings are not seven days apart")
    lag_days = (release_date - common_weeks[-1]).days
    if lag_days < 7 or lag_days > 14:
        raise ValueError(f"H.8 latest-week release lag changed: {lag_days} days")
    return result, common_weeks[-2], common_weeks[-1], "modern" if modern else "legacy"


def _validate_accounting(
    values: Mapping[tuple[str, str, str], tuple[Decimal, Decimal]]
) -> None:
    rounding_tolerance = Decimal("0.5")
    for adjustment in ADJUSTMENTS:
        for group in GROUPS:
            for point in (0, 1):
                deposits = values[(adjustment, group, "deposits")][point]
                components = (
                    values[(adjustment, group, "large_time_deposits")][point]
                    + values[(adjustment, group, "other_deposits")][point]
                )
                if abs(deposits - components) > rounding_tolerance:
                    raise ValueError(
                        f"H.8 deposit identity failed for {adjustment}/{group}"
                    )
        for metric in METRICS:
            for point in (0, 1):
                domestic = values[(adjustment, "domestic", metric)][point]
                split = (
                    values[(adjustment, "large", metric)][point]
                    + values[(adjustment, "small", metric)][point]
                )
                if abs(domestic - split) > rounding_tolerance:
                    raise ValueError(
                        f"H.8 large+small identity failed for {adjustment}/{metric}"
                    )


def parse_archive_page(record: ArchiveRecord) -> dict[str, str]:
    archive_path_date = date.fromisoformat(record.archive_path_date)
    if sha256_bytes(record.body) != record.response_sha256:
        raise ValueError("H.8 archive record body hash mismatch")
    try:
        page_text = record.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("H.8 archive page is not UTF-8") from exc
    page_date_match = re.search(
        r"Release Date:\s*([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", page_text
    )
    if page_date_match is None:
        raise ValueError("H.8 archive page release date is missing")
    page_release_date = datetime.strptime(
        " ".join(page_date_match.groups()), "%B %d %Y"
    ).date()
    if abs((page_release_date - archive_path_date).days) > 7:
        raise ValueError("H.8 archive path/page release date disagreement")
    parser = H8HTMLParser()
    try:
        parser.feed(page_text)
    except Exception as exc:
        raise ValueError("H.8 archive HTML parser failed") from exc
    values, prior_week, latest_week, schema = _extract_table_values(
        parser.tables, release_date=page_release_date
    )
    _validate_accounting(values)
    release_local = datetime.combine(
        page_release_date, wall_time(16, 15), tzinfo=NEW_YORK
    )
    row: dict[str, str] = {
        "release_calendar_key": record.release_calendar_key,
        "archive_path_date": record.archive_path_date,
        "release_date": page_release_date.isoformat(),
        "release_time_utc": release_local.astimezone(timezone.utc).isoformat(),
        "release_weekday": page_release_date.strftime("%A"),
        "holiday_shifted_release": "1" if page_release_date.weekday() != 4 else "0",
        "prior_week_ending": prior_week.isoformat(),
        "latest_week_ending": latest_week.isoformat(),
        "observation_lag_days": str((page_release_date - latest_week).days),
        "archive_url": record.url,
        "response_sha256": record.response_sha256,
        "response_bytes": str(len(record.body)),
        "html_schema": schema,
    }
    for adjustment in ADJUSTMENTS:
        for group in GROUPS:
            for metric in METRICS:
                prior, latest = values[(adjustment, group, metric)]
                row[f"{adjustment}_{group}_{metric}_prior"] = format(prior, "f")
                row[f"{adjustment}_{group}_{metric}_latest"] = format(latest, "f")
    if set(row) != set(PANEL_COLUMNS):
        raise ValueError("H.8 normalized panel schema changed")
    return row


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(PANEL_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in PANEL_COLUMNS})
    return output.getvalue().encode()


def _archive_snapshot_bytes(records: Sequence[ArchiveRecord]) -> bytes:
    lines = []
    for record in records:
        item = {
            "archive_path_date": record.archive_path_date,
            "body_b64": base64.b64encode(record.body).decode("ascii"),
            "release_calendar_key": record.release_calendar_key,
            "response_sha256": record.response_sha256,
            "url": record.url,
        }
        lines.append(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return ("\n".join(lines) + "\n").encode()


def _load_archive_snapshot(payload: bytes) -> list[ArchiveRecord]:
    records: list[ArchiveRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            item = json.loads(line)
            body = base64.b64decode(item["body_b64"], validate=True)
            record = ArchiveRecord(
                release_calendar_key=str(
                    item.get("release_calendar_key", item.get("scheduled_release_date"))
                ),
                archive_path_date=str(
                    item.get("archive_path_date", item.get("release_date"))
                ),
                url=str(item["url"]),
                response_sha256=str(item["response_sha256"]),
                body=body,
            )
        except Exception as exc:
            raise ValueError(
                f"invalid H.8 archive snapshot line {line_number}"
            ) from exc
        expected_url = ARCHIVE_URL.format(
            release_date=record.archive_path_date.replace("-", "")
        )
        if record.url != expected_url or sha256_bytes(body) != record.response_sha256:
            raise ValueError("H.8 archive snapshot record failed integrity checks")
        records.append(record)
    expected_count, expected_first, expected_last = FROZEN_COVERAGE
    dates = [record.release_calendar_key for record in records]
    if (
        len(records) != expected_count
        or dates[0] != expected_first
        or dates[-1] != expected_last
        or dates != sorted(set(dates))
    ):
        raise ValueError("H.8 archive snapshot coverage changed")
    return records


def write_gzip(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            handle.write(payload)


def read_gzip(path: str | Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "release_dates": root / "source" / "releaseDates_2026-07-18.json.gz",
        "archive": root / "source" / "h8_archive_pages_2017_2023.jsonl.gz",
        "panel": root / "fed_h8_deposit_migration_2017_2023.csv.gz",
        "manifest": root / "build_manifest.json",
    }


def _validate_frozen_hash(path: Path, expected: str, *, label: str) -> None:
    if expected and sha256_file(path) != expected:
        raise RuntimeError(f"frozen H.8 {label} hash changed")


def _download_records(
    dates: Iterable[date], *, retries: int, timeout: int
) -> list[ArchiveRecord]:
    records: list[ArchiveRecord] = []
    dates_list = list(dates)
    for index, release_date in enumerate(dates_list, start=1):
        actual_date = release_date
        compact = actual_date.strftime("%Y%m%d")
        url = ARCHIVE_URL.format(release_date=compact)
        try:
            body = _fetch_bytes(url, retries=retries, timeout=timeout)
        except RuntimeError:
            if release_date.weekday() != 4:
                raise
            actual_date = release_date - timedelta(days=1)
            compact = actual_date.strftime("%Y%m%d")
            url = ARCHIVE_URL.format(release_date=compact)
            body = _fetch_bytes(url, retries=retries, timeout=timeout)
        records.append(
            ArchiveRecord(
                release_calendar_key=release_date.isoformat(),
                archive_path_date=actual_date.isoformat(),
                url=url,
                response_sha256=sha256_bytes(body),
                body=body,
            )
        )
        if index == 1 or index % 25 == 0 or index == len(dates_list):
            print(
                f"H.8 archive {index}/{len(dates_list)}: {release_date}",
                file=sys.stderr,
                flush=True,
            )
    return records


def build(config: BuildConfig = BuildConfig()) -> dict[str, Any]:
    paths = artifact_paths(config.output_dir)
    if config.from_snapshot:
        _validate_frozen_hash(
            paths["release_dates"],
            FROZEN_RELEASE_DATES_SNAPSHOT_SHA256,
            label="release-date snapshot",
        )
        _validate_frozen_hash(
            paths["archive"], FROZEN_ARCHIVE_SNAPSHOT_SHA256, label="archive snapshot"
        )
        release_dates_raw = read_gzip(paths["release_dates"])
        release_dates = parse_release_dates(release_dates_raw)
        records = _load_archive_snapshot(read_gzip(paths["archive"]))
        network_read = False
    else:
        release_dates_raw = _fetch_bytes(
            RELEASE_DATES_URL,
            retries=config.retries,
            timeout=config.timeout_seconds,
        )
        release_dates = parse_release_dates(release_dates_raw)
        records = _download_records(
            release_dates,
            retries=config.retries,
            timeout=config.timeout_seconds,
        )
        write_gzip(paths["release_dates"], release_dates_raw)
        write_gzip(paths["archive"], _archive_snapshot_bytes(records))
        network_read = True

    expected_dates = [item.isoformat() for item in release_dates]
    calendar_keys = [record.release_calendar_key for record in records]
    if calendar_keys != expected_dates:
        raise ValueError("H.8 archive pages disagree with releaseDates.json")
    # Canonicalize a legacy in-progress snapshot schema before final hashing.
    write_gzip(paths["archive"], _archive_snapshot_bytes(records))
    rows = [parse_archive_page(record) for record in records]
    actual_release_dates = [date.fromisoformat(row["release_date"]) for row in rows]
    if actual_release_dates != sorted(set(actual_release_dates)):
        raise ValueError("H.8 page release dates are duplicated or not increasing")
    if any(
        (current - previous).days < 3 or (current - previous).days > 11
        for previous, current in zip(actual_release_dates, actual_release_dates[1:])
    ):
        raise ValueError("H.8 page release-date cadence changed")
    panel = _csv_bytes(rows)
    write_gzip(paths["panel"], panel)
    _validate_frozen_hash(paths["panel"], FROZEN_PANEL_SHA256, label="panel")
    schemas: dict[str, int] = {}
    for row in rows:
        schemas[row["html_schema"]] = schemas.get(row["html_schema"], 0) + 1
    core: dict[str, Any] = {
        "schema_version": 1,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder": "training/build_fed_h8_deposit_migration_panel.py",
        "config": asdict(config),
        "official_sources": {
            "release_dates": RELEASE_DATES_URL,
            "dated_archive_template": ARCHIVE_URL,
            "about_h8": ABOUT_URL,
            "data_download": DATA_DOWNLOAD_URL,
        },
        "source_contract": {
            "provider": "Board of Governors of the Federal Reserve System",
            "release": "H.8 Assets and Liabilities of Commercial Banks in the United States",
            "point_in_time_dated_archive_pages": True,
            "weekly_observation": "Wednesday level",
            "release_clock": "Friday generally 16:15 America/New_York; holiday Thursday retained",
            "large_bank_definition": "top 25 domestically chartered commercial banks by domestic assets at the previous benchmark Call Report",
            "small_bank_definition": "all other domestically chartered commercial banks",
            "seasonal_adjustment_vintage": "value printed in each dated archive release",
            "network_read": network_read,
            "release_dates_response_sha256": sha256_bytes(release_dates_raw),
            "release_dates_snapshot": str(paths["release_dates"]),
            "release_dates_snapshot_sha256": sha256_file(paths["release_dates"]),
            "archive_snapshot": str(paths["archive"]),
            "archive_snapshot_sha256": sha256_file(paths["archive"]),
            "archive_response_hashes_sha256": canonical_hash(
                [
                    {
                        "archive_path_date": record.archive_path_date,
                        "release_calendar_key": record.release_calendar_key,
                        "url": record.url,
                        "response_sha256": record.response_sha256,
                        "response_bytes": len(record.body),
                    }
                    for record in records
                ]
            ),
            "market_or_funding_rows_read": 0,
        },
        "panel": {
            "path": str(paths["panel"]),
            "sha256": sha256_file(paths["panel"]),
            "rows": len(rows),
            "first_release": rows[0]["release_time_utc"],
            "last_release": rows[-1]["release_time_utc"],
            "html_schema_counts": schemas,
            "columns": list(PANEL_COLUMNS),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--from-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        BuildConfig(
            output_dir=args.output_dir,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            from_snapshot=args.from_snapshot,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
