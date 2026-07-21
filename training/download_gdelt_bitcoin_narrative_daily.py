"""Freeze daily Bitcoin narrative counts from GDELT's official DOC 2.0 API.

The downloader requests four preregistered queries across one frozen historical
window using ``timelinevolraw``.  It stores a deterministic raw-response bundle
plus a daily count table.  No market, return, funding, label, or PnL source is
read.

Official references:
https://www.gdeltproject.org/data.html
https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
OFFICIAL_SOURCES = (
    "https://www.gdeltproject.org/data.html",
    "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
    "https://blog.gdeltproject.org/"
    "doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/",
)
PROTOCOL_VERSION = "gdelt_bitcoin_narrative_daily_source_v1"
BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily.py")
FROZEN_START_DATE = "2020-01-01"
FROZEN_END_DATE_EXCLUSIVE = "2024-01-01"
FROZEN_AVAILABILITY_LAG_HOURS = 48
FROZEN_AVAILABILITY_MINUTE = 15
QUERIES: tuple[tuple[str, str], ...] = (
    ("broad", "(bitcoin OR cryptocurrency)"),
    (
        "failure",
        "(bitcoin OR cryptocurrency) AND "
        "(hack OR hacked OR exploit OR scam OR fraud OR theft OR bankruptcy "
        "OR bankrupt OR collapse OR liquidation OR liquidated)",
    ),
    (
        "constraint",
        "(bitcoin OR cryptocurrency) AND "
        "(ban OR banned OR regulation OR regulator OR crackdown OR lawsuit "
        "OR investigation)",
    ),
    (
        "adoption",
        "(bitcoin OR cryptocurrency) AND "
        "(ETF OR institutional OR adoption OR approval OR approved OR investment)",
    ),
)
UTC = timezone.utc
DAILY_COLUMNS = (
    "date",
    "available_at",
    "global_article_count",
    *(f"{query_id}_article_count" for query_id, _ in QUERIES),
)


@dataclass(frozen=True)
class Config:
    start_date: str = FROZEN_START_DATE
    end_date_exclusive: str = FROZEN_END_DATE_EXCLUSIVE
    availability_lag_hours: int = FROZEN_AVAILABILITY_LAG_HOURS
    availability_minute: int = FROZEN_AVAILABILITY_MINUTE
    cache_dir: str = "data/gdelt_bitcoin_narrative_2020_2023_cache"
    daily_output: str = "data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz"
    raw_bundle_output: str = (
        "data/gdelt_bitcoin_narrative_timeline_raw_2020_2023.jsonl.gz"
    )
    manifest_output: str = (
        "results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json"
    )
    request_pause_seconds: float = 10.0
    timeout_seconds: float = 120.0
    maximum_retries: int = 12
    retry_base_seconds: float = 15.0
    retry_max_seconds: float = 300.0


Fetch = Callable[[str], bytes]


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("GDELT date must use canonical YYYY-MM-DD")
    return parsed


def format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def request_windows(start: date, end: date) -> list[tuple[date, date]]:
    if start >= end:
        raise ValueError("GDELT source interval is empty")
    return [(start, end)]


def validate_config(cfg: Config) -> None:
    if (
        cfg.start_date != FROZEN_START_DATE
        or cfg.end_date_exclusive != FROZEN_END_DATE_EXCLUSIVE
        or cfg.availability_lag_hours != FROZEN_AVAILABILITY_LAG_HOURS
        or cfg.availability_minute != FROZEN_AVAILABILITY_MINUTE
    ):
        raise ValueError("GDELT source interval and availability contract are frozen")
    start = parse_date(cfg.start_date)
    end = parse_date(cfg.end_date_exclusive)
    request_windows(start, end)
    if cfg.availability_lag_hours < 24:
        raise ValueError("GDELT availability lag must be at least 24 hours")
    if not 0 <= cfg.availability_minute <= 59:
        raise ValueError("GDELT availability minute is invalid")
    if cfg.request_pause_seconds < 0 or cfg.timeout_seconds <= 0:
        raise ValueError("GDELT request timing is invalid")
    if cfg.maximum_retries < 0:
        raise ValueError("GDELT maximum_retries is invalid")
    if cfg.retry_base_seconds <= 0 or cfg.retry_max_seconds <= 0:
        raise ValueError("GDELT retry timing is invalid")


def source_contract(cfg: Config) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "endpoint": ENDPOINT,
        "mode": "timelinevolraw",
        "format": "json",
        "queries": [
            {"query_id": query_id, "query": query} for query_id, query in QUERIES
        ],
        "start_date": cfg.start_date,
        "end_date_exclusive": cfg.end_date_exclusive,
        "windowing": "single_full_half_open_with_api_end_at_last_second",
        "required_date_resolution": "day",
        "availability": (
            "source_date UTC midnight + "
            f"{cfg.availability_lag_hours}h{cfg.availability_minute:02d}m"
        ),
    }


def request_url(query: str, start: date, end: date) -> str:
    inclusive_end = datetime.combine(end, datetime.min.time()) - timedelta(seconds=1)
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": inclusive_end.strftime("%Y%m%d%H%M%S"),
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def _http_bytes(cfg: Config, url: str, sleep: Callable[[float], None]) -> bytes:
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "rllm-gdelt-source-research/1.0"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=cfg.timeout_seconds
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read()
            if "json" not in content_type.lower() and not payload.lstrip().startswith(
                b"{"
            ):
                raise RuntimeError("GDELT response is not JSON")
            return payload
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code <= 599
            if not retryable or attempt >= cfg.maximum_retries:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            if delay <= 0:
                delay = min(
                    cfg.retry_max_seconds,
                    cfg.retry_base_seconds * (2**attempt),
                )
            sleep(max(cfg.request_pause_seconds, delay))
        except (TimeoutError, urllib.error.URLError, RuntimeError):
            if attempt >= cfg.maximum_retries:
                raise
            delay = min(
                cfg.retry_max_seconds,
                cfg.retry_base_seconds * (2**attempt),
            )
            sleep(max(cfg.request_pause_seconds, delay))
    raise AssertionError("unreachable GDELT retry loop")


def _expected_dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=index) for index in range((end - start).days)]


def parse_timeline_response(
    payload: bytes,
    *,
    start: date,
    end: date,
    expected_query: str,
) -> tuple[dict[date, tuple[int, int]], dict[str, Any]]:
    parsed = json.loads(payload.decode("utf-8"))
    details = parsed.get("query_details", {})
    if details.get("title") != expected_query:
        raise ValueError("GDELT response query identity changed")
    if details.get("date_resolution") != "day":
        raise ValueError("GDELT response did not preserve daily resolution")
    timeline = parsed.get("timeline")
    if not isinstance(timeline, list) or len(timeline) != 1:
        raise ValueError("GDELT response must contain exactly one timeline")
    if timeline[0].get("series") != "Article Count":
        raise ValueError("GDELT timeline series changed")
    data = timeline[0].get("data")
    if not isinstance(data, list):
        raise ValueError("GDELT timeline data is missing")
    rows: dict[date, tuple[int, int]] = {}
    for item in data:
        raw_date = item.get("date")
        if not isinstance(raw_date, str):
            raise ValueError("GDELT timeline date is missing")
        timestamp = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if timestamp.time() != datetime.min.time():
            raise ValueError("GDELT daily timeline timestamp is not UTC midnight")
        current = timestamp.date()
        if not start <= current < end or current in rows:
            raise ValueError(
                "GDELT timeline date escaped or duplicated its request window"
            )
        value = item.get("value")
        norm = item.get("norm")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("GDELT article count is not a nonnegative integer")
        if isinstance(norm, bool) or not isinstance(norm, int) or norm <= 0:
            raise ValueError("GDELT global article norm is not a positive integer")
        rows[current] = (value, norm)
    if sorted(rows) != _expected_dates(start, end):
        raise ValueError("GDELT daily timeline is incomplete")
    return rows, parsed


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _cache_path(cache_dir: Path, query_id: str, start: date, end: date) -> Path:
    return cache_dir / f"{query_id}_{start:%Y%m%d}_{end:%Y%m%d}.json"


def collect_responses(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    validate_config(cfg)
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = source_contract(cfg)
    contract_hash = canonical_hash(contract)
    state_path = cache_dir / "contract.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state != {"contract": contract, "contract_hash": contract_hash}:
            raise RuntimeError("GDELT resume-cache contract changed")
    else:
        _atomic_json(
            state_path,
            {"contract": contract, "contract_hash": contract_hash},
        )
    start = parse_date(cfg.start_date)
    end = parse_date(cfg.end_date_exclusive)
    records: list[dict[str, Any]] = []
    for query_id, query in QUERIES:
        for window_start, window_end in request_windows(start, end):
            url = request_url(query, window_start, window_end)
            cache = _cache_path(cache_dir, query_id, window_start, window_end)
            if cache.exists():
                response = cache.read_bytes()
            else:
                response = (
                    fetch(url) if fetch is not None else _http_bytes(cfg, url, sleep)
                )
                parse_timeline_response(
                    response,
                    start=window_start,
                    end=window_end,
                    expected_query=query,
                )
                _atomic_bytes(cache, response)
                sleep(cfg.request_pause_seconds)
            rows, parsed = parse_timeline_response(
                response,
                start=window_start,
                end=window_end,
                expected_query=query,
            )
            records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "start": window_start.isoformat(),
                    "end_exclusive": window_end.isoformat(),
                    "request_url": url,
                    "response_sha256": hashlib.sha256(response).hexdigest(),
                    "rows": rows,
                    "payload": parsed,
                }
            )
    return records


def assemble_daily_rows(
    cfg: Config, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    start = parse_date(cfg.start_date)
    end = parse_date(cfg.end_date_exclusive)
    by_query: dict[str, dict[date, tuple[int, int]]] = {
        query_id: {} for query_id, _ in QUERIES
    }
    for record in records:
        query_id = str(record["query_id"])
        if query_id not in by_query:
            raise ValueError("GDELT response has an unknown query identity")
        rows = record["rows"]
        if not isinstance(rows, Mapping):
            raise ValueError("GDELT parsed rows are malformed")
        overlap = set(by_query[query_id]) & set(rows)
        if overlap:
            raise ValueError("GDELT response windows overlap")
        by_query[query_id].update(rows)
    expected = _expected_dates(start, end)
    if any(sorted(rows) != expected for rows in by_query.values()):
        raise ValueError("GDELT query coverage is incomplete")
    output: list[dict[str, Any]] = []
    for current in expected:
        norms = {by_query[query_id][current][1] for query_id, _ in QUERIES}
        if len(norms) != 1:
            raise ValueError("GDELT global article norm differs across queries")
        available = datetime.combine(
            current, datetime.min.time(), tzinfo=UTC
        ) + timedelta(
            hours=cfg.availability_lag_hours,
            minutes=cfg.availability_minute,
        )
        row: dict[str, Any] = {
            "date": current.isoformat(),
            "available_at": format_timestamp(available),
            "global_article_count": norms.pop(),
        }
        for query_id, _ in QUERIES:
            row[f"{query_id}_article_count"] = by_query[query_id][current][0]
        output.append(row)
    return output


def deterministic_gzip_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=DAILY_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in DAILY_COLUMNS})
    return buffer.getvalue()


def deterministic_raw_bundle(records: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    ordered = sorted(records, key=lambda row: (str(row["query_id"]), str(row["start"])))
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as gz:
        for record in ordered:
            public = {
                key: value for key, value in record.items() if key not in {"rows"}
            }
            line = json.dumps(
                public,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            gz.write(line + b"\n")
    return buffer.getvalue()


def _existing_manifest(cfg: Config) -> dict[str, Any] | None:
    manifest_path = Path(cfg.manifest_output)
    daily_path = Path(cfg.daily_output)
    raw_path = Path(cfg.raw_bundle_output)
    existing = [path.exists() for path in (manifest_path, daily_path, raw_path)]
    if not any(existing):
        return None
    if not all(existing):
        raise RuntimeError("GDELT finalized source artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(unhashed):
        raise RuntimeError("GDELT finalized manifest hash changed")
    if manifest.get("contract_hash") != canonical_hash(source_contract(cfg)):
        raise RuntimeError("GDELT finalized source contract changed")
    if manifest.get("outputs", {}).get("daily_sha256") != sha256_file(daily_path):
        raise RuntimeError("GDELT finalized daily source hash changed")
    if manifest.get("outputs", {}).get("raw_bundle_sha256") != sha256_file(raw_path):
        raise RuntimeError("GDELT finalized raw bundle hash changed")
    return manifest


def run(
    cfg: Config = Config(),
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    validate_config(cfg)
    existing = _existing_manifest(cfg)
    if existing is not None:
        return existing
    records = collect_responses(cfg, fetch=fetch, sleep=sleep)
    daily_rows = assemble_daily_rows(cfg, records)
    daily_bytes = deterministic_gzip_csv(daily_rows)
    raw_bytes = deterministic_raw_bundle(records)
    request_hashes = [
        {
            key: record[key]
            for key in ("query_id", "start", "end_exclusive", "response_sha256")
        }
        for record in records
    ]
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "contract": source_contract(cfg),
        "contract_hash": canonical_hash(source_contract(cfg)),
        "config": asdict(cfg),
        "official_sources": list(OFFICIAL_SOURCES),
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "requests": {
            "count": len(records),
            "response_hashes": request_hashes,
        },
        "source_audit": {
            "daily_rows": len(daily_rows),
            "first_date": daily_rows[0]["date"],
            "last_date": daily_rows[-1]["date"],
            "first_available_at": daily_rows[0]["available_at"],
            "last_available_at": daily_rows[-1]["available_at"],
            "date_resolution": "day",
            "global_norm_consistent_across_queries": True,
        },
        "outputs": {
            "daily_path": cfg.daily_output,
            "daily_sha256": hashlib.sha256(daily_bytes).hexdigest(),
            "daily_columns": list(DAILY_COLUMNS),
            "raw_bundle_path": cfg.raw_bundle_output,
            "raw_bundle_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_news_rows_requested": 0,
            "economic_metrics_computed": False,
        },
        "next_action": "freeze source-only narrative feature and support gates",
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    daily_path = Path(cfg.daily_output)
    raw_path = Path(cfg.raw_bundle_output)
    manifest_path = Path(cfg.manifest_output)
    _atomic_bytes(daily_path, daily_bytes)
    _atomic_bytes(raw_path, raw_bytes)
    try:
        _atomic_json(manifest_path, manifest)
    except Exception:
        daily_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pause-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    cfg = Config(
        request_pause_seconds=args.request_pause_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    manifest = run(cfg)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
