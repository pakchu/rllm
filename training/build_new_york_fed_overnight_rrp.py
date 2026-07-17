"""Freeze point-in-time-safe New York Fed overnight RRP operation results.

The official repo/reverse-repo API includes both the normal daily ON RRP and
occasional morning small-value exercises.  This builder retains exactly one
normal afternoon ON RRP per operation date, assigns a conservative result
availability time fifteen minutes after the published close, and quarantines
rows whose archive ``lastUpdated`` date is later than the operation date.

No crypto market, return, funding, portfolio, or label data is read here.
The research source is hard-capped at 2023.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as wall_time, timedelta
from pathlib import Path
from typing import Any, Callable, cast
from zoneinfo import ZoneInfo


BASE_URL = "https://markets.newyorkfed.org/api/rp/results/search.json"
USER_AGENT = "rllm-overnight-rrp-source-freeze/1.0"
SCHEMA_VERSION = 1
MAX_RESEARCH_YEAR = 2023
SOURCE_SNAPSHOT_DATE = "2026-07-17"
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

FROZEN_YEAR_COVERAGE: dict[int, tuple[int, str, str, int]] = {
    2018: (249, "2018-01-02", "2018-12-31", 2),
    2019: (250, "2019-01-02", "2019-12-31", 1),
    2020: (251, "2020-01-02", "2020-12-31", 2),
    2021: (250, "2021-01-04", "2021-12-31", 3),
    2022: (249, "2022-01-03", "2022-12-30", 0),
    2023: (249, "2023-01-03", "2023-12-29", 1),
}
FROZEN_RESPONSE_SHA256 = {
    2018: "f5347b4688ad6f5dcc82f497687eb980d19c844c7a21b8fdae861b032d47be39",
    2019: "ed34123d7f9a650702dd8e390b5dfac41a9ff10f73f99170712655f083e29a01",
    2020: "ee1c24bbc389260919b6e09c00efb7f6bcc80a288fe7d382dff70579a6f04e83",
    2021: "dfa8bdcdabdb2a181db55c990a639d06e4a3fff1203a081a5008c17bc51d2788",
    2022: "7a04ced971cb6a95b6445459b5a153b5153ccc89b752372af698f9a4d222e400",
    2023: "b0be21b6e04bd05ffa58ecdd00cf17dc4a7566d4f1bda251631a3bd643024d17",
}

OUTPUT_COLUMNS = (
    "operation_id",
    "operation_date",
    "settlement_date",
    "maturity_date",
    "close_time_et",
    "result_available_at_utc",
    "last_updated_et",
    "total_amount_submitted_usd",
    "total_amount_accepted_usd",
    "participating_counterparties",
    "accepted_counterparties",
    "source_complete",
    "quarantine_reason",
)


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2018
    end_year: int = 2023
    output_dir: str = "data/new_york_fed_overnight_rrp_2018_2023"
    retries: int = 5
    timeout_seconds: int = 60
    from_snapshot: bool = False


def annual_url(year: int, *, base_url: str = BASE_URL) -> str:
    params = {
        "startDate": f"{year:04d}-01-01",
        "endDate": f"{year:04d}-12-31",
        "operationTypes": "Reverse Repo",
        "method": "fixed",
        "term": "overnight",
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def _parse_iso_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"{field} is not canonical ISO: {value!r}")
    return parsed


def _parse_time(value: Any, *, field: str) -> wall_time:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    parsed = wall_time.fromisoformat(value)
    if parsed.second or parsed.microsecond:
        raise ValueError(f"{field} must be minute-aligned: {value!r}")
    return parsed


def _parse_last_updated(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("lastUpdated must be text")
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=NEW_YORK)


def _parse_nonnegative_number(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(number)


def _parse_optional_nonnegative_number(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _parse_nonnegative_number(value, field=field)


def _normal_operation_rows(document: dict[str, Any], *, year: int) -> list[dict[str, Any]]:
    repo = document.get("repo")
    if not isinstance(repo, dict) or not isinstance(repo.get("operations"), list):
        raise ValueError("NY Fed response must contain repo.operations")
    kept: list[dict[str, Any]] = []
    for raw in repo["operations"]:
        if not isinstance(raw, dict):
            raise ValueError("NY Fed operation row must be an object")
        operation_date = _parse_iso_date(raw.get("operationDate"), field="operationDate")
        if operation_date.year != year:
            raise ValueError(f"operation outside requested year {year}: {operation_date}")
        close = _parse_time(raw.get("closeTime"), field="closeTime")
        # Morning operations are explicitly separate small-value exercises.
        if close.hour < 12:
            continue
        kept.append(raw)
    kept.sort(key=lambda row: (row["operationDate"], row["operationId"]))
    dates = [row["operationDate"] for row in kept]
    if len(dates) != len(set(dates)):
        raise ValueError("normal afternoon ON RRP must be unique by operation date")
    return kept


def parse_annual_response(payload: bytes, *, year: int) -> list[dict[str, str]]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NY Fed response is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("NY Fed response must be an object")
    rows = _normal_operation_rows(document, year=year)
    expected_rows, first_date, last_date, expected_quarantined = FROZEN_YEAR_COVERAGE[year]
    if len(rows) != expected_rows:
        raise ValueError(f"unexpected {year} row count: {len(rows)} != {expected_rows}")
    if rows[0]["operationDate"] != first_date or rows[-1]["operationDate"] != last_date:
        raise ValueError(f"unexpected {year} coverage")

    normalized: list[dict[str, str]] = []
    for raw in rows:
        operation_date = _parse_iso_date(raw.get("operationDate"), field="operationDate")
        settlement_date = _parse_iso_date(raw.get("settlementDate"), field="settlementDate")
        maturity_date = _parse_iso_date(raw.get("maturityDate"), field="maturityDate")
        close = _parse_time(raw.get("closeTime"), field="closeTime")
        last_updated = _parse_last_updated(raw.get("lastUpdated"))
        if raw.get("operationType") != "Reverse Repo":
            raise ValueError("unexpected operationType")
        if raw.get("operationMethod") != "Fixed Rate":
            raise ValueError("unexpected operationMethod")
        if raw.get("term") != "Overnight":
            raise ValueError("unexpected term")
        if raw.get("auctionStatus") != "Results":
            raise ValueError("operation is not a published result")
        if settlement_date != operation_date or maturity_date <= settlement_date:
            raise ValueError("invalid ON RRP settlement or maturity date")
        details = raw.get("details")
        if not isinstance(details, list) or len(details) != 1:
            raise ValueError("normal ON RRP must expose one Treasury detail")
        detail = details[0]
        if not isinstance(detail, dict) or detail.get("securityType") != "Treasury":
            raise ValueError("normal ON RRP must use Treasury collateral")
        submitted = _parse_nonnegative_number(
            raw.get("totalAmtSubmitted"), field="totalAmtSubmitted"
        )
        accepted = _parse_nonnegative_number(
            raw.get("totalAmtAccepted"), field="totalAmtAccepted"
        )
        detail_submitted = _parse_nonnegative_number(
            detail.get("amtSubmitted"), field="details.amtSubmitted"
        )
        detail_accepted = _parse_nonnegative_number(
            detail.get("amtAccepted"), field="details.amtAccepted"
        )
        if submitted != detail_submitted or accepted != detail_accepted:
            raise ValueError("operation totals do not reconcile to Treasury detail")
        participating = _parse_optional_nonnegative_number(
            raw.get("participatingCpty"), field="participatingCpty"
        )
        accepted_cpty = _parse_optional_nonnegative_number(
            raw.get("acceptedCpty"), field="acceptedCpty"
        )
        complete = last_updated.date() == operation_date
        reason = "" if complete else "archive_last_updated_after_operation_date"
        close_et = datetime.combine(operation_date, close, tzinfo=NEW_YORK)
        available = (close_et + timedelta(minutes=15)).astimezone(UTC)
        normalized.append(
            {
                "operation_id": str(raw.get("operationId", "")),
                "operation_date": operation_date.isoformat(),
                "settlement_date": settlement_date.isoformat(),
                "maturity_date": maturity_date.isoformat(),
                "close_time_et": close.strftime("%H:%M"),
                "result_available_at_utc": available.isoformat(),
                "last_updated_et": last_updated.isoformat(),
                "total_amount_submitted_usd": str(submitted) if complete else "",
                "total_amount_accepted_usd": str(accepted) if complete else "",
                "participating_counterparties": (
                    str(participating) if complete and participating is not None else ""
                ),
                "accepted_counterparties": (
                    str(accepted_cpty) if complete and accepted_cpty is not None else ""
                ),
                "source_complete": "true" if complete else "false",
                "quarantine_reason": reason,
            }
        )
    quarantined = sum(row["source_complete"] == "false" for row in normalized)
    if quarantined != expected_quarantined:
        raise ValueError(
            f"unexpected {year} quarantined rows: {quarantined} != {expected_quarantined}"
        )
    return normalized


def _write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(payload)


def _write_csv_gzip(path: Path, rows: list[dict[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(cast(Any, rows))
    _write_gzip(path, buffer.getvalue().encode())


def _snapshot_path(output_dir: Path, year: int) -> Path:
    return output_dir / "raw" / f"overnight_rrp_{year}.json.gz"


def _read_snapshot(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def build(
    config: BuildConfig = BuildConfig(),
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    if config.start_year < 2018 or config.end_year > MAX_RESEARCH_YEAR:
        raise ValueError("ON RRP source build is restricted to 2018-2023")
    if config.start_year > config.end_year:
        raise ValueError("start_year must not exceed end_year")
    output_dir = Path(config.output_dir)
    all_rows: list[dict[str, str]] = []
    raw_manifest: list[dict[str, Any]] = []
    for year in range(config.start_year, config.end_year + 1):
        url = annual_url(year)
        snapshot = _snapshot_path(output_dir, year)
        if config.from_snapshot:
            payload = _read_snapshot(snapshot)
        else:
            payload = fetcher(
                url, retries=config.retries, timeout=config.timeout_seconds
            )
        payload_hash = _sha256_bytes(payload)
        if payload_hash != FROZEN_RESPONSE_SHA256[year]:
            raise ValueError(f"NY Fed {year} response hash changed: {payload_hash}")
        rows = parse_annual_response(payload, year=year)
        if not config.from_snapshot:
            _write_gzip(snapshot, payload)
        all_rows.extend(rows)
        raw_manifest.append(
            {
                "year": year,
                "url": url,
                "snapshot": str(snapshot),
                "response_sha256": payload_hash,
                "normal_operation_rows": len(rows),
                "quarantined_rows": sum(
                    row["source_complete"] == "false" for row in rows
                ),
                "first_operation_date": rows[0]["operation_date"],
                "last_operation_date": rows[-1]["operation_date"],
            }
        )
    all_rows.sort(key=lambda row: row["operation_date"])
    dates = [row["operation_date"] for row in all_rows]
    if len(dates) != len(set(dates)):
        raise ValueError("combined ON RRP panel contains duplicate dates")
    output = output_dir / (
        f"new_york_fed_overnight_rrp_{config.start_year}-01-01_"
        f"{config.end_year}-12-31.csv.gz"
    )
    _write_csv_gzip(output, all_rows)
    core = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "official_api": BASE_URL,
        "start_year": config.start_year,
        "end_year": config.end_year,
        "rows": len(all_rows),
        "complete_rows": sum(row["source_complete"] == "true" for row in all_rows),
        "quarantined_rows": sum(
            row["source_complete"] == "false" for row in all_rows
        ),
        "first_operation_date": dates[0],
        "last_operation_date": dates[-1],
        "output": str(output),
        "output_sha256": _sha256_file(output),
        "raw": raw_manifest,
        "availability_rule": (
            "normal afternoon operation closeTime in America/New_York plus 15 minutes"
        ),
        "quarantine_rule": (
            "blank values when archive lastUpdated date exceeds operationDate"
        ),
        "market_or_funding_rows_read": 0,
        "builder": "training/build_new_york_fed_overnight_rrp.py",
    }
    manifest = {**core, "manifest_hash": _canonical_hash(core)}
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--start-year", type=int, default=BuildConfig.start_year)
    parser.add_argument("--end-year", type=int, default=BuildConfig.end_year)
    parser.add_argument("--from-snapshot", action="store_true")
    args = parser.parse_args()
    report = build(
        BuildConfig(
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=args.output_dir,
            from_snapshot=args.from_snapshot,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
