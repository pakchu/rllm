"""Build a source-only 2019-2023 NY Fed SOMA securities-lending panel."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


PROTOCOL_VERSION = "new_york_fed_securities_lending_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_new_york_fed_securities_lending.py")
SOURCE_DECISION = Path(
    "docs/new-york-fed-securities-lending-source-axis-decision-2026-07-23.md"
)
SOURCE_DECISION_SHA256 = (
    "be474218fc19fa55023b2712b187fc53fbc1a1079bb8fe15f8340831bd30a795"
)
OPENAPI_URL = "https://markets.newyorkfed.org/static/docs/markets-api.yml"
SEARCH_URL = (
    "https://markets.newyorkfed.org/api/seclending/seclending/"
    "results/details/search.json"
)
START_YEAR = 2019
END_YEAR = 2023
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc

OPERATION_FIELDS = frozenset(
    {
        "operationId",
        "auctionStatus",
        "operationType",
        "operationDate",
        "settlementDate",
        "maturityDate",
        "releaseTime",
        "closeTime",
        "note",
        "lastUpdated",
        "totalParAmtSubmitted",
        "totalParAmtAccepted",
        "totalParAmtExtended",
        "details",
    }
)
OPERATION_REQUIRED_FIELDS = OPERATION_FIELDS - {"totalParAmtExtended"}
DETAIL_FIELDS = frozenset(
    {
        "cusip",
        "securityDescription",
        "parAmtSubmitted",
        "parAmtAccepted",
        "weightedAverageRate",
        "somaHoldings",
        "theoAvailToBorrow",
        "actualAvailToBorrow",
        "outstandingLoans",
    }
)
OPERATION_COLUMNS = (
    "operation_id",
    "operation_date",
    "settlement_date",
    "maturity_date",
    "release_time_et",
    "close_time_et",
    "last_updated_et",
    "available_at_utc",
    "note",
    "total_par_submitted",
    "total_par_accepted",
    "total_par_extended",
)
DETAIL_COLUMNS = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "cusip",
    "security_description",
    "par_submitted",
    "par_accepted",
    "weighted_average_rate",
    "soma_holdings",
    "theoretical_available_to_borrow",
    "actual_available_to_borrow",
    "outstanding_loans",
)


@dataclass(frozen=True)
class Config:
    output_dir: str = "data/new_york_fed_securities_lending_2019_2023"
    fetch: bool = False
    request_timeout_seconds: int = 180


@dataclass(frozen=True)
class FetchResponse:
    body: bytes
    final_url: str
    status: int
    content_type: str


@dataclass(frozen=True)
class Operation:
    operation_id: str
    operation_date: date
    settlement_date: date
    maturity_date: date
    release_time_et: str
    close_time_et: str
    last_updated_et: str
    available_at_utc: datetime
    note: str
    total_par_submitted: Decimal
    total_par_accepted: Decimal
    total_par_extended: Decimal | None


@dataclass(frozen=True)
class Detail:
    operation_id: str
    operation_date: date
    available_at_utc: datetime
    cusip: str
    security_description: str
    par_submitted: Decimal
    par_accepted: Decimal
    weighted_average_rate: Decimal | None
    soma_holdings: Decimal
    theoretical_available_to_borrow: Decimal
    actual_available_to_borrow: Decimal
    outstanding_loans: Decimal


Fetch = Callable[[str, int], FetchResponse]
Clock = Callable[[], datetime]


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("path must remain repository-relative") from exc
    return resolved


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _gzip_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _default_fetch(url: str, timeout: int) -> FetchResponse:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json, application/yaml, text/yaml",
            "User-Agent": "rllm-source-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return FetchResponse(
            body=response.read(),
            final_url=response.geturl(),
            status=int(response.status),
            content_type=str(response.headers.get("Content-Type", "")),
        )


def annual_url(year: int) -> str:
    if year < START_YEAR or year > END_YEAR:
        raise RuntimeError("year lies outside frozen source window")
    query = urllib.parse.urlencode(
        {"startDate": f"{year}-01-01", "endDate": f"{year}-12-31"}
    )
    return f"{SEARCH_URL}?{query}"


def _validate_response(request_url: str, response: FetchResponse, *, json_expected: bool) -> None:
    if response.status != 200:
        raise RuntimeError(f"NY Fed source HTTP status {response.status}")
    expected = urllib.parse.urlsplit(request_url)
    observed = urllib.parse.urlsplit(response.final_url)
    if observed.scheme != "https" or observed.netloc != "markets.newyorkfed.org":
        raise RuntimeError("NY Fed source redirected outside official host")
    if observed.path != expected.path:
        raise RuntimeError("NY Fed source response path changed")
    content_type = response.content_type.lower()
    if json_expected and "json" not in content_type:
        raise RuntimeError("NY Fed annual response is not JSON")
    if not response.body:
        raise RuntimeError("NY Fed source response is empty")


def _ledger_entry(
    *,
    name: str,
    request_url: str,
    response: FetchResponse,
    retrieved_at: datetime,
) -> dict[str, Any]:
    if retrieved_at.tzinfo is None:
        raise RuntimeError("retrieval clock must be timezone-aware")
    return {
        "name": name,
        "request_url": request_url,
        "final_url": response.final_url,
        "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
        "http_status": response.status,
        "content_type": response.content_type,
        "bytes": len(response.body),
        "sha256": sha256_bytes(response.body),
    }


def _raw_paths(root: Path) -> dict[str, Path]:
    paths = {"openapi": root / "raw/markets-api.yml.gz"}
    paths.update(
        {
            f"operations_{year}": root / f"raw/seclending_{year}.json.gz"
            for year in range(START_YEAR, END_YEAR + 1)
        }
    )
    return paths


def acquire_sources(
    cfg: Config,
    *,
    fetcher: Fetch = _default_fetch,
    clock: Clock = lambda: datetime.now(UTC),
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    root = _repository_path(cfg.output_dir)
    ledger_path = root / "raw/fetch_ledger.json"
    raw_paths = _raw_paths(root)
    if all(path.exists() for path in raw_paths.values()) and ledger_path.exists():
        if cfg.fetch:
            raise RuntimeError("frozen source cache exists; refusing refresh")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        by_name = {row["name"]: row for row in ledger}
        payloads: dict[str, bytes] = {}
        for name, path in raw_paths.items():
            payload = gzip.decompress(path.read_bytes())
            if by_name.get(name, {}).get("sha256") != sha256_bytes(payload):
                raise RuntimeError(f"cached NY Fed source hash mismatch: {name}")
            payloads[name] = payload
        return payloads, ledger
    if any(path.exists() for path in raw_paths.values()) or ledger_path.exists():
        raise RuntimeError("partial NY Fed source cache exists")
    if not cfg.fetch:
        raise RuntimeError("source cache absent; rerun with --fetch")

    requests = [("openapi", OPENAPI_URL, False)] + [
        (f"operations_{year}", annual_url(year), True)
        for year in range(START_YEAR, END_YEAR + 1)
    ]
    payloads = {}
    ledger = []
    for name, url, json_expected in requests:
        response = fetcher(url, cfg.request_timeout_seconds)
        _validate_response(url, response, json_expected=json_expected)
        retrieved_at = clock()
        payloads[name] = response.body
        ledger.append(
            _ledger_entry(
                name=name,
                request_url=url,
                response=response,
                retrieved_at=retrieved_at,
            )
        )
    if b"/api/seclending/{operation}/results/{include}/search.{format}" not in payloads["openapi"]:
        raise RuntimeError("official OpenAPI no longer contains securities-lending search")
    for name, path in raw_paths.items():
        _atomic_write(path, _gzip_bytes(payloads[name]))
    _atomic_write(ledger_path, _canonical_json(ledger))
    return payloads, ledger


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise RuntimeError(f"{label} must be text")
    return value


def _date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(_text(value, label))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not an ISO date") from exc


def _clock_text(value: Any, label: str) -> str:
    raw = _text(value, label)
    try:
        parsed = time.fromisoformat(raw)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not an ISO time") from exc
    if parsed.tzinfo is not None:
        raise RuntimeError(f"{label} unexpectedly has a timezone")
    return raw


def _decimal(value: Any, label: str, *, optional: bool = False) -> Decimal | None:
    if optional and (
        value is None
        or (isinstance(value, str) and value.strip().strip('"').upper() == "N/A")
    ):
        return None
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"{label} is not decimal") from exc
    if not result.is_finite() or result < 0:
        raise RuntimeError(f"{label} must be finite and nonnegative")
    return result


def _availability(operation_day: date, last_updated: Any) -> tuple[str, datetime]:
    raw = _text(last_updated, "lastUpdated")
    try:
        local_naive = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise RuntimeError("lastUpdated is not an ISO local timestamp") from exc
    if local_naive.tzinfo is not None:
        raise RuntimeError("lastUpdated unexpectedly has a timezone")
    if local_naive.date() < operation_day:
        raise RuntimeError("lastUpdated predates operation")
    revision = local_naive.replace(tzinfo=NEW_YORK).astimezone(UTC)
    base = datetime.combine(operation_day + timedelta(days=1), time(), UTC)
    return raw, max(base, revision)


def _row_decimal(value: Decimal | None) -> str:
    return "" if value is None else format(value, "f")


def parse_annual_payload(
    payload: bytes, *, expected_year: int
) -> tuple[list[Operation], list[Detail]]:
    try:
        document = json.loads(
            payload.decode("utf-8"), parse_float=Decimal, parse_int=Decimal
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("annual NY Fed response is invalid JSON") from exc
    root = _mapping(document, "document")
    if set(root) != {"seclending"}:
        raise RuntimeError("annual NY Fed top-level schema changed")
    section = _mapping(root["seclending"], "seclending")
    if set(section) != {"operations"}:
        raise RuntimeError("annual NY Fed section schema changed")
    raw_operations = section["operations"]
    if not isinstance(raw_operations, list):
        raise RuntimeError("operations must be an array")
    operations: list[Operation] = []
    details: list[Detail] = []
    seen_operations: set[str] = set()
    seen_details: set[tuple[str, str]] = set()
    for raw_value in raw_operations:
        raw = _mapping(raw_value, "operation")
        fields = set(raw)
        if not OPERATION_REQUIRED_FIELDS.issubset(fields) or not fields.issubset(OPERATION_FIELDS):
            raise RuntimeError("operation schema changed")
        operation_id = _text(raw["operationId"], "operationId")
        if operation_id in seen_operations:
            raise RuntimeError("duplicate operationId")
        seen_operations.add(operation_id)
        if raw["auctionStatus"] != "Results" or raw["operationType"] != "Securities Lending":
            raise RuntimeError("unexpected operation status/type")
        operation_day = _date(raw["operationDate"], "operationDate")
        if operation_day.year != expected_year or not (
            START_YEAR <= operation_day.year <= END_YEAR
        ):
            raise RuntimeError("operation date outside requested frozen year")
        last_updated, available = _availability(operation_day, raw["lastUpdated"])
        total_submitted = _decimal(raw["totalParAmtSubmitted"], "totalParAmtSubmitted")
        total_accepted = _decimal(raw["totalParAmtAccepted"], "totalParAmtAccepted")
        total_extended = _decimal(
            raw.get("totalParAmtExtended"), "totalParAmtExtended", optional=True
        )
        assert isinstance(total_submitted, Decimal)
        assert isinstance(total_accepted, Decimal)
        if total_accepted > total_submitted:
            raise RuntimeError("operation accepted amount exceeds submitted amount")
        operation = Operation(
            operation_id=operation_id,
            operation_date=operation_day,
            settlement_date=_date(raw["settlementDate"], "settlementDate"),
            maturity_date=_date(raw["maturityDate"], "maturityDate"),
            release_time_et=_clock_text(raw["releaseTime"], "releaseTime"),
            close_time_et=_clock_text(raw["closeTime"], "closeTime"),
            last_updated_et=last_updated,
            available_at_utc=available,
            note=_text(raw["note"], "note", allow_empty=True),
            total_par_submitted=total_submitted,
            total_par_accepted=total_accepted,
            total_par_extended=total_extended,
        )
        raw_details = raw["details"]
        if not isinstance(raw_details, list) or not raw_details:
            raise RuntimeError("operation details must be a nonempty array")
        operation_details: list[Detail] = []
        for detail_value in raw_details:
            detail_raw = _mapping(detail_value, "detail")
            if set(detail_raw) != DETAIL_FIELDS:
                raise RuntimeError("security-detail schema changed")
            cusip = _text(detail_raw["cusip"], "cusip")
            identity = (operation_id, cusip)
            if identity in seen_details:
                raise RuntimeError("duplicate operation/CUSIP detail")
            seen_details.add(identity)
            numeric = {
                name: _decimal(detail_raw[source], source)
                for name, source in {
                    "par_submitted": "parAmtSubmitted",
                    "par_accepted": "parAmtAccepted",
                    "soma_holdings": "somaHoldings",
                    "theoretical_available_to_borrow": "theoAvailToBorrow",
                    "actual_available_to_borrow": "actualAvailToBorrow",
                    "outstanding_loans": "outstandingLoans",
                }.items()
            }
            if any(not isinstance(value, Decimal) for value in numeric.values()):
                raise RuntimeError("detail decimal unexpectedly absent")
            if numeric["par_accepted"] > numeric["par_submitted"]:
                raise RuntimeError("detail accepted amount exceeds submitted amount")
            weighted_average_rate = _decimal(
                detail_raw["weightedAverageRate"],
                "weightedAverageRate",
                optional=True,
            )
            if numeric["par_accepted"] > 0 and weighted_average_rate is None:
                raise RuntimeError("accepted detail is missing weightedAverageRate")
            detail = Detail(
                operation_id=operation_id,
                operation_date=operation_day,
                available_at_utc=available,
                cusip=cusip,
                security_description=_text(
                    detail_raw["securityDescription"], "securityDescription"
                ),
                weighted_average_rate=weighted_average_rate,
                **numeric,
            )
            operation_details.append(detail)
        submitted_sum = sum(
            (item.par_submitted for item in operation_details), Decimal(0)
        )
        accepted_sum = sum(
            (item.par_accepted for item in operation_details), Decimal(0)
        )
        if submitted_sum != total_submitted or accepted_sum != total_accepted:
            raise RuntimeError("operation totals do not reconcile to details")
        operations.append(operation)
        details.extend(operation_details)
    return operations, details


def build_panels(payloads: Mapping[str, bytes]) -> tuple[list[Operation], list[Detail]]:
    operations: list[Operation] = []
    details: list[Detail] = []
    for year in range(START_YEAR, END_YEAR + 1):
        name = f"operations_{year}"
        if name not in payloads:
            raise RuntimeError(f"missing annual payload: {name}")
        annual_operations, annual_details = parse_annual_payload(
            payloads[name], expected_year=year
        )
        operations.extend(annual_operations)
        details.extend(annual_details)
    operations.sort(key=lambda row: (row.operation_date, row.operation_id))
    details.sort(key=lambda row: (row.operation_date, row.operation_id, row.cusip))
    if len({row.operation_id for row in operations}) != len(operations):
        raise RuntimeError("operationId duplicated across years")
    if len({(row.operation_id, row.cusip) for row in details}) != len(details):
        raise RuntimeError("operation/CUSIP duplicated across years")
    if not operations or not details:
        raise RuntimeError("NY Fed source panel is empty")
    return operations, details


def _operation_row(row: Operation) -> dict[str, str]:
    return {
        "operation_id": row.operation_id,
        "operation_date": row.operation_date.isoformat(),
        "settlement_date": row.settlement_date.isoformat(),
        "maturity_date": row.maturity_date.isoformat(),
        "release_time_et": row.release_time_et,
        "close_time_et": row.close_time_et,
        "last_updated_et": row.last_updated_et,
        "available_at_utc": row.available_at_utc.isoformat(),
        "note": row.note,
        "total_par_submitted": _row_decimal(row.total_par_submitted),
        "total_par_accepted": _row_decimal(row.total_par_accepted),
        "total_par_extended": _row_decimal(row.total_par_extended),
    }


def _detail_row(row: Detail) -> dict[str, str]:
    return {
        "operation_id": row.operation_id,
        "operation_date": row.operation_date.isoformat(),
        "available_at_utc": row.available_at_utc.isoformat(),
        "cusip": row.cusip,
        "security_description": row.security_description,
        "par_submitted": _row_decimal(row.par_submitted),
        "par_accepted": _row_decimal(row.par_accepted),
        "weighted_average_rate": _row_decimal(row.weighted_average_rate),
        "soma_holdings": _row_decimal(row.soma_holdings),
        "theoretical_available_to_borrow": _row_decimal(
            row.theoretical_available_to_borrow
        ),
        "actual_available_to_borrow": _row_decimal(row.actual_available_to_borrow),
        "outstanding_loans": _row_decimal(row.outstanding_loans),
    }


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_outputs(
    cfg: Config,
    operations: Sequence[Operation],
    details: Sequence[Detail],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = _repository_path(cfg.output_dir)
    operation_path = root / "new_york_fed_securities_lending_operations_2019_2023.csv.gz"
    detail_path = root / "new_york_fed_securities_lending_details_2019_2023.csv.gz"
    operation_payload = _gzip_bytes(
        _csv_bytes(OPERATION_COLUMNS, (_operation_row(row) for row in operations))
    )
    detail_payload = _gzip_bytes(
        _csv_bytes(DETAIL_COLUMNS, (_detail_row(row) for row in details))
    )
    _atomic_write(operation_path, operation_payload)
    _atomic_write(detail_path, detail_payload)
    retrieval_times = sorted(str(row["retrieved_at_utc"]) for row in ledger)
    years = {
        str(year): sum(row.operation_date.year == year for row in operations)
        for year in range(START_YEAR, END_YEAR + 1)
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_decision": {
            "path": str(SOURCE_DECISION),
            "sha256": SOURCE_DECISION_SHA256,
        },
        "builder": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "config": asdict(cfg),
        "source_window": ["2019-01-01", "2023-12-31"],
        "official_urls": {
            "openapi": OPENAPI_URL,
            "annual_search": SEARCH_URL,
        },
        "fetch_ledger": list(ledger),
        "generated_from_latest_retrieval_utc": retrieval_times[-1],
        "operations": {
            "path": str(operation_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_bytes(operation_payload),
            "rows": len(operations),
            "columns": list(OPERATION_COLUMNS),
            "first_operation_date": operations[0].operation_date.isoformat(),
            "last_operation_date": operations[-1].operation_date.isoformat(),
            "rows_by_year": years,
        },
        "details": {
            "path": str(detail_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_bytes(detail_payload),
            "rows": len(details),
            "columns": list(DETAIL_COLUMNS),
            "unique_operation_cusip_rows": len(
                {(row.operation_id, row.cusip) for row in details}
            ),
        },
        "source_checks": {
            "operation_ids_unique": len({row.operation_id for row in operations})
            == len(operations),
            "operation_cusips_unique": len(
                {(row.operation_id, row.cusip) for row in details}
            ) == len(details),
            "all_dates_inside_frozen_window": all(
                START_YEAR <= row.operation_date.year <= END_YEAR
                for row in operations
            ),
            "availability_not_before_next_utc_midnight": all(
                row.available_at_utc
                >= datetime.combine(row.operation_date + timedelta(days=1), time(), UTC)
                for row in operations
            ),
            "operation_detail_totals_reconciled": True,
            "unknown_fields_accepted": False,
        },
        "research_boundary": {
            "candidate_features_computed": [],
            "candidate_incidence_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
        },
        "next_action": "audit source artifact, then preregister one candidate before incidence",
    }
    if not all(core["source_checks"].values()):
        raise RuntimeError("NY Fed source output invariant failed")
    core["manifest_hash"] = canonical_hash(core)
    manifest_path = root / "build_manifest.json"
    _atomic_write(manifest_path, _canonical_json(core))
    return core


def build(
    cfg: Config,
    *,
    fetcher: Fetch = _default_fetch,
    clock: Clock = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("NY Fed source decision hash mismatch")
    payloads, ledger = acquire_sources(cfg, fetcher=fetcher, clock=clock)
    operations, details = build_panels(payloads)
    return write_outputs(cfg, operations, details, ledger)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/new_york_fed_securities_lending_2019_2023",
    )
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(
        Config(
            output_dir=args.output_dir,
            fetch=args.fetch,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )
    print(
        json.dumps(
            {
                "status": "built",
                "operations": report["operations"]["rows"],
                "details": report["details"]["rows"],
                "manifest_hash": report["manifest_hash"],
                "candidate_incidence_opened": report["research_boundary"]
                ["candidate_incidence_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
