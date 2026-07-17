"""Freeze point-in-time EIA weekly petroleum inventory breadth releases.

Each row comes from the archived Table 1 CSV of the Weekly Petroleum Status
Report (WPSR), preserving the values and weekly differences visible in that
specific issue. No BTC price, return, funding, or existing-alpha input is read.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd


ARCHIVE_URL = "https://www.eia.gov/petroleum/supply/weekly/archive/"
SCHEDULE_URL = "https://www.eia.gov/petroleum/supply/weekly/schedule.php"
WPSR_URL = "https://www.eia.gov/petroleum/supply/weekly/"
AUTOMATION_POLICY_URL = "https://www.eia.gov/about/privacy_security_policy.php"
USER_AGENT = "rllm-causal-research/1.0 (+https://github.com/pakchu/rllm)"
FetchBytes = Callable[[str], bytes]
TARGET_ROWS = {
    "commercial_crude": "Commercial (Excluding SPR)",
    "gasoline": "Total Motor Gasoline",
    "distillate": "Distillate Fuel Oil",
}


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2019
    end_year: int = 2023
    output_dir: str = "data/eia_petroleum_stock_breadth_2019_2023"
    max_workers: int = 4
    timeout_seconds: int = 60
    retries: int = 5
    cache_dir: str = "/tmp/rllm_eia_wpsr_cache"


class _IssueLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def fetch_bytes(url: str, *, timeout_seconds: int = 60, retries: int = 5) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/csv,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to retrieve {url}") from last_error


def parse_archive_index(
    html: str,
    *,
    start_year: int,
    end_year: int,
    require_annual_density: bool = True,
) -> list[dict[str, Any]]:
    parser = _IssueLinkParser()
    parser.feed(html)
    pattern = re.compile(
        r"/archive/(?P<year>\d{4})/(?P<stamp>\d{4}_\d{2}_\d{2})/"
        r"wpsr_(?P=stamp)\.php$"
    )
    issues: dict[str, dict[str, Any]] = {}
    for href in parser.hrefs:
        page_url = urljoin(ARCHIVE_URL, href)
        match = pattern.search(page_url)
        if not match:
            continue
        year = int(match.group("year"))
        if not start_year <= year <= end_year:
            continue
        release_date = datetime.strptime(match.group("stamp"), "%Y_%m_%d").date()
        table_url = page_url.rsplit("/", 1)[0] + "/csv/table1.csv"
        issues[page_url] = {
            "release_date": release_date,
            "archive_page_url": page_url,
            "table1_csv_url": table_url,
        }
    rows = sorted(issues.values(), key=lambda row: row["release_date"])
    if not rows:
        raise ValueError("EIA archive index yielded no WPSR issues")
    if len({row["release_date"] for row in rows}) != len(rows):
        raise ValueError("EIA archive index contains duplicate release dates")
    counts = {
        year: sum(row["release_date"].year == year for row in rows)
        for year in range(start_year, end_year + 1)
    }
    if require_annual_density and any(
        count < 50 or count > 53 for count in counts.values()
    ):
        raise ValueError(f"unexpected EIA issue count by year: {counts}")
    return rows


def _decimal(value: str) -> Decimal:
    cleaned = value.replace(",", "").strip()
    try:
        result = Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f"invalid EIA numeric value: {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"non-finite EIA numeric value: {value!r}")
    return result


def _header_date(value: str, *, release_year: int) -> date:
    month, day, short_year = (int(part) for part in value.strip().split("/"))
    year = 2000 + short_year if short_year < 70 else 1900 + short_year
    result = date(year, month, day)
    if result.year not in {release_year - 1, release_year}:
        raise ValueError(f"EIA table week year is implausible: {value}")
    return result


def parse_table1(payload: bytes, *, release_date: date) -> dict[str, Any]:
    text = payload.decode("cp1252").replace("\x1a", "")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) < 4:
        raise ValueError(f"EIA Table 1 header is incomplete: {release_date}")
    current_week = _header_date(rows[0][1], release_year=release_date.year)
    previous_week = _header_date(rows[0][2], release_year=release_date.year)
    if current_week - previous_week != timedelta(days=7):
        raise ValueError(f"EIA Table 1 weeks are not seven days apart: {release_date}")
    if not current_week < release_date <= current_week + timedelta(days=14):
        raise ValueError(f"EIA release/data-week relationship changed: {release_date}")

    first_section: dict[str, list[str]] = {}
    for row in rows[1:]:
        if row and row[0] == "STUB_1":
            break
        if row:
            first_section[row[0].strip()] = row
    values: dict[str, Decimal] = {}
    for key, label in TARGET_ROWS.items():
        row = first_section.get(label)
        if row is None or len(row) < 4:
            raise ValueError(f"missing EIA Table 1 row {label}: {release_date}")
        current = _decimal(row[1])
        previous = _decimal(row[2])
        published_change = _decimal(row[3])
        arithmetic_change = current - previous
        discrepancy = arithmetic_change - published_change
        values[f"{key}_stock_mmbbl"] = current
        values[f"{key}_change_mmbbl"] = published_change
        values[f"{key}_arithmetic_change_mmbbl"] = arithmetic_change
        values[f"{key}_change_discrepancy_mmbbl"] = discrepancy
    return {
        "data_week_ending": current_week,
        "previous_week_ending": previous_week,
        **values,
    }


def conservative_available_time_utc(release_date: date) -> pd.Timestamp:
    # Historical issue pages bind the release date but do not expose a machine-
    # readable holiday clock. The next UTC day at 13:00 is after the entire U.S.
    # Eastern release date, preventing a false precision or early entry.
    value = datetime.combine(
        release_date + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ) + timedelta(hours=13)
    return cast(pd.Timestamp, pd.Timestamp(value))


def build_panel(
    issues: list[dict[str, Any]], payloads: dict[str, bytes]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for issue in sorted(issues, key=lambda row: row["release_date"]):
        url = issue["table1_csv_url"]
        payload = payloads[url]
        values = parse_table1(payload, release_date=issue["release_date"])
        discrepancy_columns = [
            f"{key}_change_discrepancy_mmbbl" for key in TARGET_ROWS
        ]
        difference_consistent = all(
            abs(values[column]) <= Decimal("0.0011")
            for column in discrepancy_columns
        )
        records.append(
            {
                "release_date": issue["release_date"].isoformat(),
                "available_time_utc": conservative_available_time_utc(
                    issue["release_date"]
                ).isoformat(),
                "data_week_ending": values["data_week_ending"].isoformat(),
                "previous_week_ending": values["previous_week_ending"].isoformat(),
                "commercial_crude_stock_mmbbl": float(
                    values["commercial_crude_stock_mmbbl"]
                ),
                "commercial_crude_change_mmbbl": float(
                    values["commercial_crude_change_mmbbl"]
                ),
                "commercial_crude_arithmetic_change_mmbbl": float(
                    values["commercial_crude_arithmetic_change_mmbbl"]
                ),
                "commercial_crude_change_discrepancy_mmbbl": float(
                    values["commercial_crude_change_discrepancy_mmbbl"]
                ),
                "gasoline_stock_mmbbl": float(values["gasoline_stock_mmbbl"]),
                "gasoline_change_mmbbl": float(values["gasoline_change_mmbbl"]),
                "gasoline_arithmetic_change_mmbbl": float(
                    values["gasoline_arithmetic_change_mmbbl"]
                ),
                "gasoline_change_discrepancy_mmbbl": float(
                    values["gasoline_change_discrepancy_mmbbl"]
                ),
                "distillate_stock_mmbbl": float(values["distillate_stock_mmbbl"]),
                "distillate_change_mmbbl": float(
                    values["distillate_change_mmbbl"]
                ),
                "distillate_arithmetic_change_mmbbl": float(
                    values["distillate_arithmetic_change_mmbbl"]
                ),
                "distillate_change_discrepancy_mmbbl": float(
                    values["distillate_change_discrepancy_mmbbl"]
                ),
                "published_difference_consistent": difference_consistent,
                "archive_page_url": issue["archive_page_url"],
                "table1_csv_url": url,
                "table1_sha256": sha256_bytes(payload),
                "source_complete": difference_consistent,
            }
        )
    frame = pd.DataFrame(records)
    frame["available_time_utc"] = pd.to_datetime(frame["available_time_utc"], utc=True)
    if not frame["available_time_utc"].is_monotonic_increasing:
        raise ValueError("EIA conservative availability clock is not monotonic")
    if frame["available_time_utc"].duplicated().any():
        raise ValueError("EIA conservative availability clock is duplicated")
    return frame


def _fetch_many(
    urls: list[str],
    fetcher: FetchBytes,
    *,
    max_workers: int,
    cache_dir: Path,
) -> dict[str, bytes]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(url: str) -> Path:
        return cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.bin"

    def cached_fetch(url: str) -> bytes:
        path = cache_path(url)
        if path.exists():
            return path.read_bytes()
        payload = fetcher(url)
        path.write_bytes(payload)
        return payload

    if max_workers == 1:
        return {url: cached_fetch(url) for url in urls}
    output: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(cached_fetch, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            output[url] = future.result()
    return output


def build(config: BuildConfig) -> dict[str, Any]:
    if config.start_year < 1980 or config.end_year < config.start_year:
        raise ValueError("invalid EIA source horizon")
    if not 1 <= config.max_workers <= 4:
        raise ValueError("EIA retrieval uses at most four respectful workers")
    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"
    cache_dir = Path(config.cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def configured_fetch(url: str) -> bytes:
        return fetch_bytes(
            url,
            timeout_seconds=config.timeout_seconds,
            retries=config.retries,
        )

    archive_payload = configured_fetch(ARCHIVE_URL)
    archive_text = archive_payload.decode("utf-8", errors="replace")
    issues = parse_archive_index(
        archive_text,
        start_year=config.start_year,
        end_year=config.end_year,
    )
    urls = [issue["table1_csv_url"] for issue in issues]
    payloads = _fetch_many(
        urls,
        configured_fetch,
        max_workers=config.max_workers,
        cache_dir=cache_dir,
    )
    frame = build_panel(issues, payloads)

    output_name = (
        f"eia_petroleum_stock_breadth_{config.start_year}_{config.end_year}.csv.gz"
    )
    output_path = output_dir / output_name
    write_gzip(output_path, frame.to_csv(index=False, lineterminator="\n").encode())

    archive_snapshot = raw_dir / "eia_wpsr_archive_index.html.gz"
    table_snapshot = raw_dir / "eia_wpsr_table1_archives.json.gz"
    write_gzip(archive_snapshot, archive_payload)
    write_gzip(
        table_snapshot,
        canonical_json(
            [
                {
                    "official_url": url,
                    "payload_cp1252": payloads[url]
                    .decode("cp1252")
                    .replace("\x1a", ""),
                    "sha256": sha256_bytes(payloads[url]),
                }
                for url in urls
            ]
        ),
    )

    source_manifest = {
        "protocol_version": "eia_petroleum_stock_breadth_source_v1",
        "official_urls": {
            "archive": ARCHIVE_URL,
            "schedule": SCHEDULE_URL,
            "weekly_report": WPSR_URL,
            "automation_policy": AUTOMATION_POLICY_URL,
        },
        "point_in_time_contract": (
            "each archived issue's own Table 1 current, prior, and difference values"
        ),
        "availability_contract": (
            "next UTC day 13:00 after official archive release date; deliberately "
            "later than the complete U.S. Eastern release date"
        ),
        "archive_index_sha256": sha256_bytes(archive_payload),
        "table1_payload_sha256": {
            url: sha256_bytes(payloads[url]) for url in urls
        },
        "snapshots": {
            str(archive_snapshot): sha256_file(archive_snapshot),
            str(table_snapshot): sha256_file(table_snapshot),
        },
    }
    source_manifest_path = output_dir / "source_manifest.json"
    source_manifest_path.write_bytes(canonical_json(source_manifest))

    year_counts = frame["release_date"].str[:4].value_counts().sort_index()
    build_manifest = {
        "protocol_version": "eia_petroleum_stock_breadth_build_v1",
        "config": {
            key: value for key, value in asdict(config).items() if key != "cache_dir"
        },
        "rows": len(frame),
        "rows_by_release_year": {
            str(year): int(count) for year, count in year_counts.items()
        },
        "first_release_date": str(frame.iloc[0]["release_date"]),
        "last_release_date": str(frame.iloc[-1]["release_date"]),
        "first_available_time_utc": frame.iloc[0]["available_time_utc"].isoformat(),
        "last_available_time_utc": frame.iloc[-1]["available_time_utc"].isoformat(),
        "source_complete_rows": int(frame["source_complete"].sum()),
        "source_quarantined_rows": int((~frame["source_complete"]).sum()),
        "market_or_funding_rows_read": 0,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    build_manifest["manifest_hash"] = sha256_bytes(canonical_json(build_manifest))
    build_manifest_path = output_dir / "build_manifest.json"
    build_manifest_path.write_bytes(canonical_json(build_manifest))
    return build_manifest


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=BuildConfig.start_year)
    parser.add_argument("--end-year", type=int, default=BuildConfig.end_year)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--max-workers", type=int, default=BuildConfig.max_workers)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--cache-dir", default=BuildConfig.cache_dir)
    return BuildConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
