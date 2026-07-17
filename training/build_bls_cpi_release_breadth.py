"""Freeze point-in-time BLS CPI release values and publication clocks.

The signal source is the archived BLS CPI news release, not a revised market
dataset. Historical BLS schedule pages determine the publication date and the
archived release's Table A supplies the headline and core unadjusted 12-month
rates that were public at 08:30 America/New_York.

BLS currently rejects this environment's automated client at its CDN edge.
The builder therefore retrieves a plain-text rendering through r.jina.ai while
binding every payload to its official ``bls.gov`` URL. The published values are
independently checked against the Federal Reserve Bank of St. Louis FRED mirror.
No BTC price, return, funding, or existing-alpha input is read here.
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
from datetime import date, datetime, time as clock_time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


BLS_HOST = "www.bls.gov"
JINA_PREFIX = "https://r.jina.ai/http://"
FRED_SERIES = {
    "headline": "CPIAUCNS",
    "core": "CPILFENS",
}
USER_AGENT = "rllm-causal-research/1.0"
FetchText = Callable[[str], str]


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2019
    end_year: int = 2023
    output_dir: str = "data/bls_cpi_release_breadth_2019_2023"
    max_workers: int = 1
    timeout_seconds: int = 60
    retries: int = 6
    request_pace_seconds: float = 1.25
    cache_dir: str = "/tmp/rllm_bls_cpi_release_cache"


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


def official_schedule_url(year: int) -> str:
    return f"https://{BLS_HOST}/schedule/{year}/home.htm"


def official_release_url(release_date: date) -> str:
    suffix = release_date.strftime("%m%d%Y")
    return f"https://{BLS_HOST}/news.release/archives/cpi_{suffix}.htm"


def transport_url(official_url: str) -> str:
    prefix = f"https://{BLS_HOST}/"
    if not official_url.startswith(prefix):
        raise ValueError(f"non-BLS source URL: {official_url}")
    return JINA_PREFIX + official_url.removeprefix("https://")


def fetch_text(url: str, *, timeout_seconds: int = 60, retries: int = 4) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                payload = response.read()
            return payload.decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < retries:
                retry_after = 0.0
                if isinstance(error, HTTPError) and error.code == 429:
                    raw_retry_after = error.headers.get("Retry-After")
                    if raw_retry_after and raw_retry_after.isdigit():
                        retry_after = float(raw_retry_after)
                    retry_after = max(retry_after, 10.0 * (attempt + 1))
                time.sleep(max(retry_after, 1.0 * (attempt + 1)))
    raise RuntimeError(f"failed to retrieve {url}") from last_error


def _clean_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("**", "")).strip()


def parse_schedule(markdown: str, *, year: int) -> list[dict[str, Any]]:
    expected_url = official_schedule_url(year).replace("https://", "http://")
    if f"URL Source: {expected_url}" not in markdown:
        raise ValueError(f"BLS schedule provenance mismatch for {year}")

    rows: list[dict[str, Any]] = []
    for line in markdown.splitlines():
        if "Consumer Price Index" not in line or not line.lstrip().startswith("|"):
            continue
        cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[2].startswith("Consumer Price Index for "):
            continue
        release_date = datetime.strptime(cells[0], "%A, %B %d, %Y").date()
        if release_date.year != year:
            raise ValueError(f"BLS release year mismatch: {cells[0]}")
        if cells[1] != "08:30 AM":
            raise ValueError(f"unexpected CPI release time: {cells[1]}")
        reference = cells[2].removeprefix("Consumer Price Index for ")
        reference_month = datetime.strptime(reference, "%B %Y").date().replace(day=1)
        rows.append(
            {
                "release_date": release_date,
                "reference_month": reference_month,
                "schedule_url": official_schedule_url(year),
                "release_url": official_release_url(release_date),
            }
        )
    if len(rows) != 12:
        raise ValueError(f"expected 12 CPI releases in {year}, found {len(rows)}")
    if len({row["reference_month"] for row in rows}) != 12:
        raise ValueError(f"duplicate CPI reference month in {year}")
    return sorted(rows, key=lambda row: row["release_date"])


def _table_value(markdown: str, label: str) -> Decimal:
    for line in markdown.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] != label:
            continue
        try:
            return Decimal(cells[-1])
        except InvalidOperation as error:
            raise ValueError(f"invalid BLS Table A value for {label}: {cells[-1]}") from error
    raise ValueError(f"missing BLS Table A row: {label}")


def _signed_decimal(verb: str, raw_value: str) -> Decimal:
    value = Decimal(raw_value)
    if verb.lower() in {"declined", "decreased", "fell"}:
        return -value
    return value


def _prose_yoy_value(markdown: str, *, core: bool) -> Decimal:
    text = re.sub(r"\s+", " ", markdown).lower()
    verb = r"(increased|rose|declined|decreased|fell)"
    if core:
        patterns = [
            rf"(?:the )?(?:index for )?all items less food and energy(?: index)? "
            rf"{verb} ([0-9]+(?:\.[0-9]+)?) percent "
            rf"(?:over|for) (?:the )?(?:last|past) 12 months"
        ]
    else:
        patterns = [
            rf"over the last 12 months, the all items index {verb} "
            rf"([0-9]+(?:\.[0-9]+)?) percent before seasonal adjustment",
            rf"the all items index {verb} ([0-9]+(?:\.[0-9]+)?) percent "
            rf"for the 12 months ending",
        ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _signed_decimal(match.group(1), match.group(2))
    label = "core" if core else "headline"
    raise ValueError(f"missing BLS {label} 12-month prose value")


def _release_value(markdown: str, *, label: str, core: bool) -> Decimal:
    prose = _prose_yoy_value(markdown, core=core)
    try:
        table = _table_value(markdown, label)
    except ValueError:
        return prose
    # Some legacy HTML tables are rendered by the transport with the final
    # 12-month column on a separate line. In that case ``cells[-1]`` is merely
    # the latest one-month change. The release prose is the canonical archived
    # value and is independently checked against FRED below.
    return table if table == prose else prose


def parse_release(
    markdown: str, *, release_date: date, reference_month: date
) -> dict[str, Decimal]:
    expected_url = official_release_url(release_date).replace("https://", "http://")
    if f"URL Source: {expected_url}" not in markdown:
        raise ValueError(f"BLS archive provenance mismatch for {release_date}")

    header = markdown[:1800]
    date_pattern = release_date.strftime("%B %d, %Y").replace(" 0", " ")
    if "8:30 a.m." not in header or date_pattern not in header:
        raise ValueError(f"BLS embargo clock mismatch for {release_date}")
    title_pattern = re.compile(
        rf"CONSUMER PRICE INDEX.{{0,12}}{reference_month.strftime('%B').upper()}\s+"
        rf"{reference_month.year}",
        re.IGNORECASE | re.DOTALL,
    )
    if not title_pattern.search(header):
        raise ValueError(f"BLS reference month mismatch for {release_date}")

    return {
        "headline_yoy_pct": _release_value(markdown, label="All items", core=False),
        "core_yoy_pct": _release_value(
            markdown,
            label="All items less food and energy",
            core=True,
        ),
    }


def release_time_utc(release_date: date) -> pd.Timestamp:
    eastern = ZoneInfo("America/New_York")
    local = datetime.combine(release_date, clock_time(8, 30), tzinfo=eastern)
    return cast(pd.Timestamp, pd.Timestamp(local.astimezone(timezone.utc)))


def fred_url(series_id: str, *, start_year: int, end_year: int) -> str:
    # The first release in start_year describes December of start_year-1, so a
    # 12-month rate needs December of start_year-2.
    start = f"{start_year - 2}-01-01"
    end = f"{end_year}-12-31"
    return (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}&cosd={start}&coed={end}"
    )


def parse_fred_csv(payload: str, *, series_id: str) -> dict[date, Decimal]:
    rows: dict[date, Decimal] = {}
    for raw in csv.DictReader(io.StringIO(payload)):
        value = raw.get(series_id, ".")
        if value in (None, "", "."):
            continue
        month = datetime.strptime(raw["observation_date"], "%Y-%m-%d").date()
        rows[month] = Decimal(value)
    if not rows:
        raise ValueError(f"empty FRED series: {series_id}")
    return rows


def fred_yoy(index: dict[date, Decimal], month: date) -> Decimal:
    prior = month.replace(year=month.year - 1)
    if month not in index or prior not in index:
        raise ValueError(f"missing FRED year-over-year pair for {month}")
    return (index[month] / index[prior] - Decimal(1)) * Decimal(100)


def _fred_crosscheck(release_value: Decimal, calculated: Decimal) -> tuple[bool, float]:
    difference = abs(release_value - calculated)
    return difference <= Decimal("0.051"), float(difference)


def build_panel(
    schedules: list[dict[str, Any]],
    releases: dict[str, str],
    fred: dict[str, dict[date, Decimal]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for schedule in sorted(schedules, key=lambda row: row["release_date"]):
        release_url = schedule["release_url"]
        values = parse_release(
            releases[release_url],
            release_date=schedule["release_date"],
            reference_month=schedule["reference_month"],
        )
        headline_calc = fred_yoy(fred["headline"], schedule["reference_month"])
        core_calc = fred_yoy(fred["core"], schedule["reference_month"])
        headline_match, headline_error = _fred_crosscheck(
            values["headline_yoy_pct"], headline_calc
        )
        core_match, core_error = _fred_crosscheck(values["core_yoy_pct"], core_calc)
        rows.append(
            {
                "reference_month": schedule["reference_month"].isoformat(),
                "release_time_utc": release_time_utc(schedule["release_date"]).isoformat(),
                "headline_yoy_pct": float(values["headline_yoy_pct"]),
                "core_yoy_pct": float(values["core_yoy_pct"]),
                "headline_fred_yoy_pct": float(headline_calc),
                "core_fred_yoy_pct": float(core_calc),
                "headline_fred_abs_error_pct": headline_error,
                "core_fred_abs_error_pct": core_error,
                "fred_crosscheck_passed": headline_match and core_match,
                "schedule_url": schedule["schedule_url"],
                "release_url": release_url,
                "source_complete": True,
            }
        )
    frame = pd.DataFrame(rows)
    frame["release_time_utc"] = pd.to_datetime(frame["release_time_utc"], utc=True)
    if not frame["release_time_utc"].is_monotonic_increasing:
        raise ValueError("CPI release clock is not monotonic")
    if frame["release_time_utc"].duplicated().any():
        raise ValueError("duplicate CPI release clock")
    if not bool(frame["fred_crosscheck_passed"].all()):
        failed = frame.loc[~frame["fred_crosscheck_passed"], "reference_month"].tolist()
        raise ValueError(f"BLS/FRED CPI cross-check failed: {failed}")
    return frame


def _fetch_many(
    urls: list[str],
    fetcher: FetchText,
    max_workers: int,
    *,
    pace_seconds: float,
    cache_dir: Path,
) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(official_url: str) -> Path:
        cache_key = hashlib.sha256(official_url.encode()).hexdigest()
        return cache_dir / f"{cache_key}.txt"

    def cached_fetch(official_url: str) -> str:
        path = cache_path(official_url)
        if path.exists():
            return path.read_text()
        payload = fetcher(transport_url(official_url))
        path.write_text(payload)
        return payload

    output: dict[str, str] = {}
    if max_workers == 1:
        for position, url in enumerate(urls):
            was_cached = cache_path(url).exists()
            output[url] = cached_fetch(url)
            if not was_cached and position + 1 < len(urls):
                time.sleep(pace_seconds)
        return output
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(cached_fetch, url): url for url in urls}
        for future in as_completed(futures):
            official_url = futures[future]
            output[official_url] = future.result()
    return output


def build(config: BuildConfig) -> dict[str, Any]:
    if config.start_year < 1980 or config.end_year < config.start_year:
        raise ValueError("invalid CPI source horizon")
    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"
    cache_dir = Path(config.cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def configured_fetch(url: str) -> str:
        return fetch_text(
            url,
            timeout_seconds=config.timeout_seconds,
            retries=config.retries,
        )

    schedule_urls = [official_schedule_url(year) for year in range(config.start_year, config.end_year + 1)]
    schedule_payloads = _fetch_many(
        schedule_urls,
        configured_fetch,
        config.max_workers,
        pace_seconds=config.request_pace_seconds,
        cache_dir=cache_dir,
    )
    schedules: list[dict[str, Any]] = []
    for year in range(config.start_year, config.end_year + 1):
        schedules.extend(parse_schedule(schedule_payloads[official_schedule_url(year)], year=year))

    release_urls = [row["release_url"] for row in schedules]
    release_payloads = _fetch_many(
        release_urls,
        configured_fetch,
        config.max_workers,
        pace_seconds=config.request_pace_seconds,
        cache_dir=cache_dir,
    )

    fred_payloads: dict[str, str] = {}
    fred_indexes: dict[str, dict[date, Decimal]] = {}
    for name, series_id in FRED_SERIES.items():
        url = fred_url(series_id, start_year=config.start_year, end_year=config.end_year)
        payload = configured_fetch(url) if url.startswith(f"https://{BLS_HOST}") else fetch_text(
            url,
            timeout_seconds=config.timeout_seconds,
            retries=config.retries,
        )
        fred_payloads[name] = payload
        fred_indexes[name] = parse_fred_csv(payload, series_id=series_id)

    frame = build_panel(schedules, release_payloads, fred_indexes)
    output_name = f"bls_cpi_release_breadth_{config.start_year}_{config.end_year}.csv.gz"
    output_path = output_dir / output_name
    csv_payload = frame.to_csv(index=False, lineterminator="\n").encode()
    write_gzip(output_path, csv_payload)

    schedule_snapshot = raw_dir / "bls_schedule_pages.json.gz"
    release_snapshot = raw_dir / "bls_cpi_archived_releases.json.gz"
    write_gzip(
        schedule_snapshot,
        canonical_json(
            [{"official_url": url, "payload": schedule_payloads[url]} for url in schedule_urls]
        ),
    )
    write_gzip(
        release_snapshot,
        canonical_json(
            [{"official_url": url, "payload": release_payloads[url]} for url in release_urls]
        ),
    )
    fred_paths: dict[str, Path] = {}
    for name, payload in fred_payloads.items():
        path = raw_dir / f"fred_{FRED_SERIES[name]}.csv.gz"
        write_gzip(path, payload.encode())
        fred_paths[name] = path

    source_manifest = {
        "protocol_version": "bls_cpi_release_breadth_source_v1",
        "transport_note": (
            "BLS pages were rendered through r.jina.ai; every payload binds its official "
            "bls.gov URL and every release value is cross-checked against FRED."
        ),
        "official_schedule_urls": schedule_urls,
        "official_release_urls": release_urls,
        "fred_urls": {
            name: fred_url(series_id, start_year=config.start_year, end_year=config.end_year)
            for name, series_id in FRED_SERIES.items()
        },
        "snapshots": {
            str(schedule_snapshot): sha256_file(schedule_snapshot),
            str(release_snapshot): sha256_file(release_snapshot),
            **{str(path): sha256_file(path) for path in fred_paths.values()},
        },
    }
    source_manifest_path = output_dir / "source_manifest.json"
    source_manifest_path.write_bytes(canonical_json(source_manifest))

    build_manifest = {
        "protocol_version": "bls_cpi_release_breadth_build_v1",
        "config": {
            key: value for key, value in asdict(config).items() if key != "cache_dir"
        },
        "rows": len(frame),
        "first_reference_month": str(frame.iloc[0]["reference_month"]),
        "last_reference_month": str(frame.iloc[-1]["reference_month"]),
        "first_release_time_utc": frame.iloc[0]["release_time_utc"].isoformat(),
        "last_release_time_utc": frame.iloc[-1]["release_time_utc"].isoformat(),
        "headline_fred_max_abs_error_pct": float(frame["headline_fred_abs_error_pct"].max()),
        "core_fred_max_abs_error_pct": float(frame["core_fred_abs_error_pct"].max()),
        "all_fred_crosschecks_passed": bool(frame["fred_crosscheck_passed"].all()),
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
    parser.add_argument(
        "--request-pace-seconds",
        type=float,
        default=BuildConfig.request_pace_seconds,
    )
    parser.add_argument("--cache-dir", default=BuildConfig.cache_dir)
    return BuildConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
