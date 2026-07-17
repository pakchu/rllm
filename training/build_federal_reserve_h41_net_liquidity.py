"""Freeze a causal Federal Reserve H.4.1 net-liquidity source panel.

The weekly H.4.1 archive preserves each release as it was published.  This
builder reads the consolidated statement in each archived release and records
three values from the same vintage: total Federal Reserve assets, the U.S.
Treasury General Account, and Federal Reserve reverse repurchase agreements.
The derived source-only measure is::

    net_liquidity = total_assets - treasury_general_account - reverse_repos

Every observation becomes available five minutes after the documented 16:30
America/New_York release time.  No crypto price, return, label, or portfolio
outcome is imported or derived.  The source is deliberately capped at 2023 so
the candidate-specific 2024+ outcome period remains sealed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as wall_time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo


INDEX_URL = "https://www.federalreserve.gov/releases/h41/releaseDates.json"
RELEASE_URL_TEMPLATE = "https://www.federalreserve.gov/releases/h41/{release_date}/"
USER_AGENT = "rllm-h41-source-freeze/1.0"
SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_DATE = "2026-07-17"
MAX_RESEARCH_YEAR = 2023
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

FROZEN_YEAR_COVERAGE: dict[int, tuple[int, str, str]] = {
    2018: (52, "20180104", "20181227"),
    2019: (52, "20190103", "20191226"),
    2020: (53, "20200102", "20201231"),
    2021: (52, "20210107", "20211230"),
    2022: (52, "20220106", "20221229"),
    2023: (52, "20230105", "20231228"),
}

OUTPUT_COLUMNS = (
    "release_date",
    "observation_date",
    "available_at_utc",
    "total_assets_usd_millions",
    "treasury_general_account_usd_millions",
    "reverse_repurchase_agreements_usd_millions",
    "net_liquidity_usd_millions",
    "source_format",
    "source_url",
    "source_sha256",
)

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
DATE_PATTERN = re.compile(rf"\b({MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}})\b")


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2018
    end_year: int = 2023
    output_dir: str = "data/federal_reserve_h41_net_liquidity_2018_2023"
    retries: int = 5
    timeout_seconds: int = 60


class _H41ArchiveParser(HTMLParser):
    """Collect legacy PRE reports and modern table cells without dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pre_blocks: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._pre_parts: list[str] | None = None
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "pre":
            self._pre_parts = []
        elif tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._pre_parts is not None:
            self._pre_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._pre_parts is not None:
            self.pre_blocks.append("".join(self._pre_parts))
            self._pre_parts = None
        elif tag in ("td", "th") and self._cell_parts is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell_parts).split()))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            assert self._table is not None
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def release_url(release_date: str) -> str:
    if not re.fullmatch(r"\d{8}", release_date):
        raise ValueError(f"release_date must be YYYYMMDD, got {release_date!r}")
    return RELEASE_URL_TEMPLATE.format(release_date=release_date)


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "text/html,application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def parse_release_dates(payload: bytes, *, start_year: int, end_year: int) -> list[str]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("H.4.1 release-date index is not valid UTF-8 JSON") from exc
    if not isinstance(document, list):
        raise ValueError("H.4.1 release-date index must be a list")

    dates: list[str] = []
    for year_block in document:
        if not isinstance(year_block, dict):
            raise ValueError("H.4.1 year block must be an object")
        try:
            year = int(year_block["yearValue"])
            months = year_block["Months"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("H.4.1 year block has an invalid schema") from exc
        if not start_year <= year <= end_year:
            continue
        if not isinstance(months, list):
            raise ValueError(f"H.4.1 Months must be a list for {year}")
        for month in months:
            if not isinstance(month, dict) or not isinstance(month.get("Dates"), list):
                raise ValueError(f"H.4.1 month block has an invalid schema for {year}")
            for raw_date in month["Dates"]:
                if not isinstance(raw_date, str) or not re.fullmatch(r"\d{8}", raw_date):
                    raise ValueError(f"invalid H.4.1 release date: {raw_date!r}")
                parsed = datetime.strptime(raw_date, "%Y%m%d").date()
                if parsed.year != year:
                    raise ValueError(
                        f"H.4.1 release {raw_date} is outside year block {year}"
                    )
                dates.append(raw_date)

    dates.sort()
    if len(set(dates)) != len(dates):
        raise ValueError("H.4.1 release-date index contains duplicate releases")
    for year in range(start_year, end_year + 1):
        expected = FROZEN_YEAR_COVERAGE.get(year)
        if expected is None:
            raise ValueError(f"no frozen H.4.1 coverage contract for {year}")
        year_dates = [value for value in dates if value.startswith(str(year))]
        actual = (
            len(year_dates),
            year_dates[0] if year_dates else None,
            year_dates[-1] if year_dates else None,
        )
        if actual != expected:
            raise ValueError(
                f"H.4.1 {year} coverage changed: expected={expected}, actual={actual}"
            )
    return dates


def _parse_h41_date(value: str) -> date:
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unrecognized H.4.1 date: {value!r}")


def _parse_usd_millions(value: str, *, field: str) -> int:
    cleaned = value.replace(",", "").strip()
    if not re.fullmatch(r"\d+", cleaned):
        raise ValueError(f"{field} must be non-negative integer millions, got {value!r}")
    return int(cleaned)


def _parse_legacy_pre(report: str) -> dict[str, Any]:
    section_markers = list(
        re.finditer(
            r"(?m)^\d+\. Consolidated Statement of Condition of All Federal Reserve Banks(?: \(continued\))?\s*$",
            report,
        )
    )
    if not section_markers:
        raise ValueError("legacy H.4.1 report is missing the consolidated statement")
    section = report[section_markers[0].start() :]
    next_section = re.search(
        r"\n\d+\. Statement of Condition of Each Federal Reserve Bank", section
    )
    if next_section:
        section = section[: next_section.start()]

    header = section[:2000]
    date_match = DATE_PATTERN.search(header)
    if date_match is None:
        raise ValueError("legacy H.4.1 consolidated statement has no observation date")
    patterns = {
        "total_assets_usd_millions": re.compile(
            r"(?m)^Total assets\s+\(0\)\s+([\d,]+)\b"
        ),
        "reverse_repurchase_agreements_usd_millions": re.compile(
            r"(?m)^\s{2}Reverse repurchase agreements(?: \(\d+\))?\s+([\d,]+)\b"
        ),
        "treasury_general_account_usd_millions": re.compile(
            r"(?m)^\s{4}U\.S\. Treasury, General Account\s+([\d,]+)\b"
        ),
    }
    values: dict[str, int] = {}
    for field, pattern in patterns.items():
        matches = pattern.findall(section)
        if len(matches) != 1:
            raise ValueError(
                f"legacy H.4.1 consolidated statement expected one {field}, "
                f"found {len(matches)}"
            )
        values[field] = _parse_usd_millions(matches[0], field=field)
    return {
        "observation_date": _parse_h41_date(date_match.group(1)),
        **values,
        "source_format": "legacy_pre",
    }


def _find_unique_row(
    tables: list[list[list[str]]],
    *,
    predicate: Callable[[list[str]], bool],
    field: str,
) -> tuple[int, list[str]]:
    matches: list[tuple[int, list[str]]] = []
    for table_index, table in enumerate(tables):
        for row in table:
            if predicate(row):
                matches.append((table_index, row))
    if len(matches) != 1:
        raise ValueError(f"modern H.4.1 expected one {field} row, found {len(matches)}")
    return matches[0]


def _parse_modern_tables(tables: list[list[list[str]]]) -> dict[str, Any]:
    asset_table_index, asset_row = _find_unique_row(
        tables,
        predicate=lambda row: len(row) >= 3
        and row[0] == "Total assets"
        and row[1] == "(0)",
        field="consolidated total assets",
    )
    _, rrp_row = _find_unique_row(
        tables,
        predicate=lambda row: len(row) >= 3
        and row[0].startswith("Reverse repurchase agreements")
        and row[1] == "",
        field="consolidated reverse repurchase agreements",
    )
    _, tga_row = _find_unique_row(
        tables,
        predicate=lambda row: len(row) >= 3
        and row[0] == "U.S. Treasury, General Account"
        and row[1] == "",
        field="consolidated Treasury General Account",
    )

    header_text = " ".join(
        cell for row in tables[asset_table_index][:3] for cell in row
    )
    date_match = DATE_PATTERN.search(header_text)
    if date_match is None:
        raise ValueError("modern H.4.1 consolidated statement has no observation date")
    return {
        "observation_date": _parse_h41_date(date_match.group(1)),
        "total_assets_usd_millions": _parse_usd_millions(
            asset_row[2], field="total_assets_usd_millions"
        ),
        "treasury_general_account_usd_millions": _parse_usd_millions(
            tga_row[2], field="treasury_general_account_usd_millions"
        ),
        "reverse_repurchase_agreements_usd_millions": _parse_usd_millions(
            rrp_row[2], field="reverse_repurchase_agreements_usd_millions"
        ),
        "source_format": "modern_html_tables",
    }


def parse_release_page(payload: bytes, *, release_date: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("H.4.1 archive page is not valid UTF-8") from exc
    parser = _H41ArchiveParser()
    parser.feed(text)
    legacy = [
        block
        for block in parser.pre_blocks
        if "Consolidated Statement of Condition" in block
    ]
    if legacy:
        if len(legacy) != 1:
            raise ValueError("H.4.1 page contains multiple legacy consolidated reports")
        parsed = _parse_legacy_pre(legacy[0])
    else:
        parsed = _parse_modern_tables(parser.tables)

    released = datetime.strptime(release_date, "%Y%m%d").date()
    observed = parsed["observation_date"]
    lag_days = (released - observed).days
    if lag_days < 1 or lag_days > 7:
        raise ValueError(
            f"H.4.1 {release_date} observation lag is not causal: "
            f"observation={observed}, lag_days={lag_days}"
        )
    total = parsed["total_assets_usd_millions"]
    tga = parsed["treasury_general_account_usd_millions"]
    rrp = parsed["reverse_repurchase_agreements_usd_millions"]
    net = total - tga - rrp
    if total <= 0 or tga < 0 or rrp < 0 or net <= 0:
        raise ValueError(
            f"H.4.1 {release_date} consolidated values violate balance constraints"
        )
    available = datetime.combine(
        released,
        wall_time(hour=16, minute=35),
        tzinfo=NEW_YORK,
    ).astimezone(UTC)
    return {
        "release_date": released.isoformat(),
        "observation_date": observed.isoformat(),
        "available_at_utc": available.isoformat(),
        "total_assets_usd_millions": total,
        "treasury_general_account_usd_millions": tga,
        "reverse_repurchase_agreements_usd_millions": rrp,
        "net_liquidity_usd_millions": net,
        "source_format": parsed["source_format"],
    }


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        handle.write(payload)
    return output.getvalue()


def _gunzip_bytes(payload: bytes, *, path: Path) -> bytes:
    try:
        return gzip.decompress(payload)
    except (gzip.BadGzipFile, EOFError) as exc:
        raise ValueError(f"invalid H.4.1 source snapshot: {path}") from exc


def _write_csv_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(OUTPUT_COLUMNS)
                for row in rows:
                    writer.writerow([row[column] for column in OUTPUT_COLUMNS])


def _validate_config(cfg: BuildConfig) -> None:
    if cfg.start_year < min(FROZEN_YEAR_COVERAGE):
        raise ValueError("the frozen H.4.1 source contract starts in 2018")
    if cfg.start_year > cfg.end_year:
        raise ValueError("start_year must not exceed end_year")
    if cfg.end_year > MAX_RESEARCH_YEAR:
        raise ValueError("2024+ is sealed by the current research protocol")
    if cfg.retries < 1 or cfg.timeout_seconds < 1:
        raise ValueError("retries and timeout_seconds must be positive")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid H.4.1 source manifest: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"H.4.1 source manifest must be an object: {path}")
    return document


def _expected_source_hashes(source_manifest: dict[str, Any]) -> dict[str, str]:
    reports = source_manifest.get("reports")
    if not isinstance(reports, list):
        raise ValueError("H.4.1 source manifest reports must be a list")
    output: dict[str, str] = {}
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("H.4.1 source manifest report must be an object")
        release_date = report.get("release_date")
        sha256 = report.get("response_sha256")
        if not isinstance(release_date, str) or not isinstance(sha256, str):
            raise ValueError("H.4.1 source manifest report is missing hash fields")
        output[release_date] = sha256
    if len(output) != len(reports):
        raise ValueError("H.4.1 source manifest contains duplicate releases")
    return output


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
    from_snapshot: bool = False,
) -> dict[str, Any]:
    _validate_config(cfg)
    output_dir = Path(cfg.output_dir)
    raw_dir = output_dir / "raw"
    source_manifest_path = output_dir / "source_manifest.json"
    prior_source_manifest = (
        _load_json(source_manifest_path) if source_manifest_path.exists() else None
    )
    if from_snapshot and prior_source_manifest is None:
        raise ValueError("--from-snapshot requires an existing source_manifest.json")

    index_snapshot_path = raw_dir / "releaseDates.json.gz"
    if from_snapshot:
        assert prior_source_manifest is not None
        index_payload = _gunzip_bytes(index_snapshot_path.read_bytes(), path=index_snapshot_path)
    else:
        index_payload = fetcher(
            INDEX_URL,
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    index_sha256 = hashlib.sha256(index_payload).hexdigest()
    if prior_source_manifest is not None:
        expected_index = prior_source_manifest.get("index_response_sha256")
        if expected_index != index_sha256:
            raise ValueError(
                "H.4.1 release-date index changed from the frozen snapshot: "
                f"expected={expected_index}, actual={index_sha256}"
            )
    release_dates = parse_release_dates(
        index_payload,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
    )
    if not from_snapshot:
        raw_dir.mkdir(parents=True, exist_ok=True)
        index_snapshot_path.write_bytes(_gzip_bytes(index_payload))

    expected_hashes = (
        _expected_source_hashes(prior_source_manifest)
        if prior_source_manifest is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for compact_date in release_dates:
        url = release_url(compact_date)
        raw_path = raw_dir / "releases" / f"{compact_date}.html.gz"
        if from_snapshot:
            payload = _gunzip_bytes(raw_path.read_bytes(), path=raw_path)
        else:
            payload = fetcher(url, retries=cfg.retries, timeout=cfg.timeout_seconds)
        response_sha256 = hashlib.sha256(payload).hexdigest()
        expected = expected_hashes.get(compact_date)
        if expected is not None and expected != response_sha256:
            raise ValueError(
                f"H.4.1 {compact_date} changed from the frozen snapshot: "
                f"expected={expected}, actual={response_sha256}"
            )
        parsed = parse_release_page(payload, release_date=compact_date)
        parsed.update({"source_url": url, "source_sha256": response_sha256})
        rows.append(parsed)
        if not from_snapshot:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(_gzip_bytes(payload))
        reports.append(
            {
                "release_date": compact_date,
                "source_url": url,
                "raw_path": str(raw_path.relative_to(output_dir)),
                "response_sha256": response_sha256,
                "source_format": parsed["source_format"],
                "observation_date": parsed["observation_date"],
            }
        )

    if expected_hashes and set(expected_hashes) != set(release_dates):
        raise ValueError("H.4.1 source manifest release set differs from frozen index")
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "index_url": INDEX_URL,
        "index_raw_path": str(index_snapshot_path.relative_to(output_dir)),
        "index_response_sha256": index_sha256,
        "start_year": cfg.start_year,
        "end_year": cfg.end_year,
        "report_count": len(reports),
        "reports": reports,
    }
    source_manifest_text = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if from_snapshot:
        if source_manifest_text != source_manifest_path.read_text():
            raise ValueError("H.4.1 source snapshots do not reproduce source_manifest.json")
    else:
        source_manifest_path.write_text(source_manifest_text)

    output_path = output_dir / (
        "federal_reserve_h41_net_liquidity_"
        f"{rows[0]['release_date']}_{rows[-1]['release_date']}.csv.gz"
    )
    _write_csv_gzip(output_path, rows)
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": asdict(cfg),
        "protocol": {
            "source": "Board of Governors of the Federal Reserve System H.4.1 archive",
            "measure": (
                "total assets minus U.S. Treasury General Account minus Federal "
                "Reserve reverse repurchase agreements"
            ),
            "value_vintage": "consolidated statement values in each archived weekly release",
            "release_time": "16:30 America/New_York on the archive release date",
            "availability_rule": (
                "16:35 America/New_York on the archive release date; five-minute "
                "operational delay after the documented release time"
            ),
            "observation_rule": "Wednesday/as-of date printed in the same release",
            "future_year_guard": "2024+ fetches are rejected",
            "crypto_market_fields_opened": False,
            "outcomes_opened": False,
        },
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": hashlib.sha256(
            source_manifest_path.read_bytes()
        ).hexdigest(),
        "output": str(output_path),
        "output_sha256": output_sha256,
        "rows": len(rows),
        "first_release_date": rows[0]["release_date"],
        "last_release_date": rows[-1]["release_date"],
        "first_available_at_utc": rows[0]["available_at_utc"],
        "last_available_at_utc": rows[-1]["available_at_utc"],
        "legacy_pre_rows": sum(row["source_format"] == "legacy_pre" for row in rows),
        "modern_html_rows": sum(
            row["source_format"] == "modern_html_tables" for row in rows
        ),
        "columns": list(OUTPUT_COLUMNS),
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=BuildConfig.start_year)
    parser.add_argument("--end-year", type=int, default=BuildConfig.end_year)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument(
        "--timeout-seconds", type=int, default=BuildConfig.timeout_seconds
    )
    parser.add_argument("--from-snapshot", action="store_true")
    args = vars(parser.parse_args())
    from_snapshot = bool(args.pop("from_snapshot"))
    manifest = build(BuildConfig(**args), from_snapshot=from_snapshot)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
