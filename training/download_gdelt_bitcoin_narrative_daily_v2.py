"""Acquire frozen GDELT narrative counts with explicit sparse-bin semantics.

GDELT omits daily timeline bins on dates with no monitored articles. This v2
transport preserves the frozen four feature queries and fills an omitted query
bin with zero matches. If all four queries omit a date, its global norm is also
zero; otherwise every available norm must agree. No market source is read.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V1_DEPENDENCY = Path("training/download_gdelt_bitcoin_narrative_daily.py")
V1_DEPENDENCY_SHA256 = (
    "d756990d979e901033891ad6a8c565783dc58e8a4a9e286d6e866929dd74889e"
)


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if sha256_file(V1_DEPENDENCY) != V1_DEPENDENCY_SHA256:
    raise RuntimeError("GDELT v2 frozen v1 dependency changed before import")
v1 = importlib.import_module("training.download_gdelt_bitcoin_narrative_daily")


ENDPOINT = v1.ENDPOINT
OFFICIAL_SOURCES = v1.OFFICIAL_SOURCES
PROTOCOL_VERSION = "gdelt_bitcoin_narrative_daily_source_v2"
BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily_v2.py")
FROZEN_START_DATE = v1.FROZEN_START_DATE
FROZEN_END_DATE_EXCLUSIVE = v1.FROZEN_END_DATE_EXCLUSIVE
FROZEN_AVAILABILITY_LAG_HOURS = v1.FROZEN_AVAILABILITY_LAG_HOURS
FROZEN_AVAILABILITY_MINUTE = v1.FROZEN_AVAILABILITY_MINUTE
QUERIES = v1.QUERIES
DAILY_COLUMNS = v1.DAILY_COLUMNS
UTC = timezone.utc
KNOWN_GLOBAL_OUTAGE_DATES = ("2020-10-20", "2023-03-23")
Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Config:
    start_date: str = FROZEN_START_DATE
    end_date_exclusive: str = FROZEN_END_DATE_EXCLUSIVE
    availability_lag_hours: int = FROZEN_AVAILABILITY_LAG_HOURS
    availability_minute: int = FROZEN_AVAILABILITY_MINUTE
    cache_dir: str = "data/gdelt_bitcoin_narrative_2020_2023_v2_cache"
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


def canonical_hash(payload: Any) -> str:
    return v1.canonical_hash(payload)


def parse_date(value: str) -> date:
    return v1.parse_date(value)


def format_timestamp(value: datetime) -> str:
    return v1.format_timestamp(value)


def request_windows(start: date, end: date) -> list[tuple[date, date]]:
    return v1.request_windows(start, end)


def request_url(query: str, start: date, end: date) -> str:
    return v1.request_url(query, start, end)


def validate_config(cfg: Config) -> None:
    if (
        cfg.start_date != FROZEN_START_DATE
        or cfg.end_date_exclusive != FROZEN_END_DATE_EXCLUSIVE
        or cfg.availability_lag_hours != FROZEN_AVAILABILITY_LAG_HOURS
        or cfg.availability_minute != FROZEN_AVAILABILITY_MINUTE
    ):
        raise ValueError("GDELT v2 source interval and availability are frozen")
    start = parse_date(cfg.start_date)
    end = parse_date(cfg.end_date_exclusive)
    request_windows(start, end)
    if cfg.availability_lag_hours < 24 or not 0 <= cfg.availability_minute <= 59:
        raise ValueError("GDELT v2 availability lag is invalid")
    if cfg.request_pause_seconds < 0 or cfg.timeout_seconds <= 0:
        raise ValueError("GDELT v2 request timing is invalid")
    if cfg.maximum_retries < 0:
        raise ValueError("GDELT v2 retry count is invalid")
    if cfg.retry_base_seconds <= 0 or cfg.retry_max_seconds <= 0:
        raise ValueError("GDELT v2 retry timing is invalid")


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
        "sparse_bin_policy": {
            "omitted_query_bin": "zero matching articles",
            "global_norm": (
                "consensus norm from available feature-query bins; zero only when "
                "all four query bins are absent"
            ),
            "all_query_absence": "global monitoring outage day with all counts zero",
            "subset_validation": "failure/constraint/adoption must not exceed broad",
        },
    }


def _http_bytes(cfg: Config, url: str, sleep: Callable[[float], None]) -> bytes:
    v1_cfg = v1.Config(
        request_pause_seconds=cfg.request_pause_seconds,
        timeout_seconds=cfg.timeout_seconds,
        maximum_retries=cfg.maximum_retries,
        retry_base_seconds=cfg.retry_base_seconds,
        retry_max_seconds=cfg.retry_max_seconds,
    )
    return v1._http_bytes(v1_cfg, url, sleep)


def _expected_dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=index) for index in range((end - start).days)]


def parse_sparse_timeline_response(
    payload: bytes,
    *,
    start: date,
    end: date,
    expected_query: str,
) -> tuple[dict[date, tuple[int, int]], dict[str, Any]]:
    parsed = json.loads(payload.decode("utf-8"))
    details = parsed.get("query_details", {})
    if details.get("title") != expected_query:
        raise ValueError("GDELT v2 response query identity changed")
    if details.get("date_resolution") != "day":
        raise ValueError("GDELT v2 response did not preserve daily resolution")
    timeline = parsed.get("timeline")
    if not isinstance(timeline, list) or len(timeline) != 1:
        raise ValueError("GDELT v2 response must contain exactly one timeline")
    if timeline[0].get("series") != "Article Count":
        raise ValueError("GDELT v2 timeline series changed")
    data = timeline[0].get("data")
    if not isinstance(data, list):
        raise ValueError("GDELT v2 timeline data is missing")
    rows: dict[date, tuple[int, int]] = {}
    observed_dates: list[date] = []
    for item in data:
        raw_date = item.get("date")
        if not isinstance(raw_date, str):
            raise ValueError("GDELT v2 timeline date is missing")
        timestamp = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if timestamp.time() != datetime.min.time():
            raise ValueError("GDELT v2 daily timestamp is not UTC midnight")
        current = timestamp.date()
        if not start <= current < end or current in rows:
            raise ValueError("GDELT v2 timeline date escaped or duplicated its window")
        value = item.get("value")
        norm = item.get("norm")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("GDELT v2 count is not a nonnegative integer")
        if isinstance(norm, bool) or not isinstance(norm, int) or norm <= 0:
            raise ValueError("GDELT v2 available norm is not a positive integer")
        rows[current] = (value, norm)
        observed_dates.append(current)
    if observed_dates != sorted(observed_dates):
        raise ValueError("GDELT v2 timeline dates are not sorted")
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
    expected_state = {"contract": contract, "contract_hash": contract_hash}
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != expected_state:
            raise RuntimeError("GDELT v2 resume-cache contract changed")
    else:
        _atomic_json(state_path, expected_state)
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
                parse_sparse_timeline_response(
                    response,
                    start=window_start,
                    end=window_end,
                    expected_query=query,
                )
                _atomic_bytes(cache, response)
                sleep(cfg.request_pause_seconds)
            rows, parsed = parse_sparse_timeline_response(
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = parse_date(cfg.start_date)
    end = parse_date(cfg.end_date_exclusive)
    expected_queries = {query_id: query for query_id, query in QUERIES}
    if (
        len(records) != len(QUERIES)
        or {str(record.get("query_id")) for record in records} != set(expected_queries)
        or any(
            record.get("query") != expected_queries[str(record.get("query_id"))]
            or record.get("start") != cfg.start_date
            or record.get("end_exclusive") != cfg.end_date_exclusive
            for record in records
        )
    ):
        raise ValueError("GDELT v2 response family or interval changed")
    by_query: dict[str, dict[date, tuple[int, int]]] = {
        query_id: {} for query_id, _ in QUERIES
    }
    for record in records:
        query_id = str(record["query_id"])
        if query_id not in by_query:
            raise ValueError("GDELT v2 response has an unknown query identity")
        rows = record["rows"]
        if not isinstance(rows, Mapping):
            raise ValueError("GDELT v2 parsed rows are malformed")
        if set(by_query[query_id]) & set(rows):
            raise ValueError("GDELT v2 response windows overlap")
        by_query[query_id].update(rows)
    expected = _expected_dates(start, end)
    output: list[dict[str, Any]] = []
    global_outage_dates: list[str] = []
    missing_by_query = {query_id: 0 for query_id, _ in QUERIES}
    for current in expected:
        available_rows = {
            query_id: rows[current]
            for query_id, rows in by_query.items()
            if current in rows
        }
        norms = {row[1] for row in available_rows.values()}
        if len(norms) > 1:
            raise ValueError("GDELT v2 global norm differs across feature queries")
        global_count = norms.pop() if norms else 0
        if not available_rows:
            global_outage_dates.append(current.isoformat())
        available = datetime.combine(
            current, datetime.min.time(), tzinfo=UTC
        ) + timedelta(
            hours=cfg.availability_lag_hours,
            minutes=cfg.availability_minute,
        )
        row: dict[str, Any] = {
            "date": current.isoformat(),
            "available_at": format_timestamp(available),
            "global_article_count": global_count,
        }
        for query_id, _ in QUERIES:
            query_row = available_rows.get(query_id)
            if query_row is None:
                missing_by_query[query_id] += 1
                row[f"{query_id}_article_count"] = 0
            else:
                row[f"{query_id}_article_count"] = query_row[0]
        broad = row["broad_article_count"]
        if any(
            row[f"{query_id}_article_count"] > broad
            for query_id in ("failure", "constraint", "adoption")
        ):
            raise ValueError("GDELT v2 category count exceeds broad count")
        if global_count == 0 and any(
            row[f"{query_id}_article_count"] != 0 for query_id, _ in QUERIES
        ):
            raise ValueError("GDELT v2 zero global norm has nonzero query counts")
        output.append(row)
    sparse_audit = {
        "missing_bins_by_query": missing_by_query,
        "global_outage_dates": global_outage_dates,
        "global_outage_days": len(global_outage_dates),
        "known_global_outage_dates_match": tuple(global_outage_dates)
        == KNOWN_GLOBAL_OUTAGE_DATES,
    }
    if not sparse_audit["known_global_outage_dates_match"]:
        raise ValueError("GDELT v2 global outage date set changed")
    return output, sparse_audit


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
    return v1.deterministic_raw_bundle(records)


def _existing_manifest(cfg: Config) -> dict[str, Any] | None:
    manifest_path = Path(cfg.manifest_output)
    daily_path = Path(cfg.daily_output)
    raw_path = Path(cfg.raw_bundle_output)
    existing = [path.exists() for path in (manifest_path, daily_path, raw_path)]
    if not any(existing):
        return None
    if not all(existing):
        raise RuntimeError("GDELT v2 finalized source artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(unhashed):
        raise RuntimeError("GDELT v2 finalized manifest hash changed")
    if manifest.get("contract_hash") != canonical_hash(source_contract(cfg)):
        raise RuntimeError("GDELT v2 finalized source contract changed")
    if manifest.get("outputs", {}).get("daily_sha256") != sha256_file(daily_path):
        raise RuntimeError("GDELT v2 finalized daily source hash changed")
    if manifest.get("outputs", {}).get("raw_bundle_sha256") != sha256_file(raw_path):
        raise RuntimeError("GDELT v2 finalized raw source hash changed")
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
    daily_rows, sparse_audit = assemble_daily_rows(cfg, records)
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
        "builder": {
            "path": str(BUILDER),
            "sha256": sha256_file(BUILDER),
            "v1_dependency_path": str(V1_DEPENDENCY),
            "v1_dependency_sha256": V1_DEPENDENCY_SHA256,
        },
        "requests": {"count": len(records), "response_hashes": request_hashes},
        "source_audit": {
            "daily_rows": len(daily_rows),
            "first_date": daily_rows[0]["date"],
            "last_date": daily_rows[-1]["date"],
            "first_available_at": daily_rows[0]["available_at"],
            "last_available_at": daily_rows[-1]["available_at"],
            "date_resolution": "day",
            "global_norm_consistent_across_available_queries": True,
            **sparse_audit,
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
        "next_action": "commit outer source seal before opening feature counts",
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
    print(json.dumps(run(cfg), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
