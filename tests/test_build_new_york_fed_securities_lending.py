from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from training import build_new_york_fed_securities_lending as source


UTC = timezone.utc


def detail(cusip: str = "912345678", *, submitted: int = 100, accepted: int = 80):
    return {
        "cusip": cusip,
        "securityDescription": "T 1.000 01/01/30",
        "parAmtSubmitted": submitted,
        "parAmtAccepted": accepted,
        "weightedAverageRate": 0.05,
        "somaHoldings": 1000,
        "theoAvailToBorrow": 900,
        "actualAvailToBorrow": 850,
        "outstandingLoans": 50,
    }


def operation(
    day: str = "2020-01-02",
    operation_id: str = "SL 010220 1",
    *,
    details=None,
    last_updated: str | None = None,
):
    rows = details or [detail()]
    return {
        "operationId": operation_id,
        "auctionStatus": "Results",
        "operationType": "Securities Lending",
        "operationDate": day,
        "settlementDate": day,
        "maturityDate": "2020-01-03",
        "releaseTime": "12:00",
        "closeTime": "12:15",
        "note": "",
        "lastUpdated": last_updated or f"{day} 12:15:00",
        "totalParAmtSubmitted": sum(row["parAmtSubmitted"] for row in rows),
        "totalParAmtAccepted": sum(row["parAmtAccepted"] for row in rows),
        "details": rows,
    }


def payload(*operations) -> bytes:
    return json.dumps({"seclending": {"operations": list(operations)}}).encode()


def test_parse_sorts_and_reconciles_exact_decimals() -> None:
    second = operation(
        "2020-01-03",
        "SL 010320 1",
        details=[detail("B", submitted=40, accepted=30), detail("A", submitted=60, accepted=50)],
    )
    first = operation()
    operations, details = source.parse_annual_payload(
        payload(second, first), expected_year=2020
    )
    assert [row.operation_id for row in operations] == ["SL 010320 1", "SL 010220 1"]
    # Annual parser preserves source order; the cross-year panel performs the sort.
    assert operations[0].total_par_submitted == Decimal("100")
    assert operations[0].total_par_accepted == Decimal("80")
    assert len(details) == 3
    assert all(row.weighted_average_rate == Decimal("0.05") for row in details)


def test_build_panels_orders_operations_and_cusips() -> None:
    payloads = {
        f"operations_{year}": payload(
            operation(
                f"{year}-01-02",
                f"SL {year} 1",
                details=[detail("Z"), detail("A", submitted=20, accepted=10)],
            )
        )
        for year in range(source.START_YEAR, source.END_YEAR + 1)
    }
    operations, details = source.build_panels(payloads)
    assert operations[0].operation_date.year == 2019
    assert operations[-1].operation_date.year == 2023
    assert [row.cusip for row in details[:2]] == ["A", "Z"]


def test_availability_uses_next_utc_midnight_or_later_revision() -> None:
    normal, _ = source.parse_annual_payload(payload(operation()), expected_year=2020)
    assert normal[0].available_at_utc == datetime(2020, 1, 3, tzinfo=UTC)
    revised, _ = source.parse_annual_payload(
        payload(operation(last_updated="2020-01-03 03:00:00")),
        expected_year=2020,
    )
    assert revised[0].available_at_utc == datetime(2020, 1, 3, 8, tzinfo=UTC)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.update({"futureField": 1}), "schema changed"),
        (lambda row: row.pop("lastUpdated"), "schema changed"),
        (lambda row: row.update({"operationType": "Repo"}), "status/type"),
        (lambda row: row.update({"operationDate": "2024-01-02"}), "outside requested"),
        (lambda row: row.update({"lastUpdated": "2019-12-31 12:00:00"}), "predates"),
    ],
)
def test_operation_schema_and_clock_drift_fail_closed(mutation, match: str) -> None:
    row = operation()
    mutation(row)
    with pytest.raises(RuntimeError, match=match):
        source.parse_annual_payload(payload(row), expected_year=2020)


def test_detail_schema_and_reconciliation_fail_closed() -> None:
    row = operation()
    row["details"][0]["unknown"] = 1
    with pytest.raises(RuntimeError, match="detail schema changed"):
        source.parse_annual_payload(payload(row), expected_year=2020)
    row = operation()
    row["totalParAmtAccepted"] = 81
    with pytest.raises(RuntimeError, match="do not reconcile"):
        source.parse_annual_payload(payload(row), expected_year=2020)


def test_zero_acceptance_preserves_official_na_rate_as_null() -> None:
    no_award = detail(accepted=0)
    no_award["weightedAverageRate"] = '"N/A"'
    _, details = source.parse_annual_payload(
        payload(operation(details=[no_award])), expected_year=2020
    )
    assert details[0].weighted_average_rate is None
    awarded = operation(details=[no_award])
    awarded["details"][0]["parAmtAccepted"] = 1
    awarded["totalParAmtAccepted"] = 1
    with pytest.raises(RuntimeError, match="missing weightedAverageRate"):
        source.parse_annual_payload(payload(awarded), expected_year=2020)


def test_duplicate_operation_and_cusip_fail_closed() -> None:
    row = operation()
    with pytest.raises(RuntimeError, match="duplicate operationId"):
        source.parse_annual_payload(payload(row, row), expected_year=2020)
    row = operation(details=[detail("A"), detail("A")])
    with pytest.raises(RuntimeError, match="duplicate operation/CUSIP"):
        source.parse_annual_payload(payload(row), expected_year=2020)


def test_annual_urls_are_exact_and_bounded() -> None:
    assert source.annual_url(2019).endswith(
        "startDate=2019-01-01&endDate=2019-12-31"
    )
    with pytest.raises(RuntimeError, match="outside frozen"):
        source.annual_url(2024)


def test_acquire_writes_once_and_replays_cache(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-nyfed-seclending"
    root = source.REPOSITORY_ROOT / relative
    if root.exists():
        import shutil

        shutil.rmtree(root)
    calls: list[str] = []

    def fetcher(url: str, timeout: int) -> source.FetchResponse:
        calls.append(url)
        year = url.split("startDate=")[1][:4] if "startDate=" in url else ""
        body = (
            b"/api/seclending/{operation}/results/{include}/search.{format}"
            if url == source.OPENAPI_URL
            else payload(operation(f"{year}-01-02", f"SL {year}"))
        )
        return source.FetchResponse(
            body=body,
            final_url=url,
            status=200,
            content_type=("text/yaml" if url == source.OPENAPI_URL else "application/json"),
        )

    fixed = lambda: datetime(2026, 7, 23, 1, 2, 3, tzinfo=UTC)
    try:
        cfg = source.Config(output_dir=str(relative), fetch=True)
        first, ledger = source.acquire_sources(cfg, fetcher=fetcher, clock=fixed)
        assert len(calls) == 6
        assert len(ledger) == 6
        replay, replay_ledger = source.acquire_sources(
            source.Config(output_dir=str(relative), fetch=False),
            fetcher=lambda *_: pytest.fail("cache replay fetched network"),
        )
        assert replay == first
        assert replay_ledger == ledger
        with pytest.raises(RuntimeError, match="refusing refresh"):
            source.acquire_sources(cfg, fetcher=fetcher, clock=fixed)
        for path in source._raw_paths(root).values():
            assert gzip.decompress(path.read_bytes())
    finally:
        if root.exists():
            import shutil

            shutil.rmtree(root)


def test_redirect_and_content_type_fail_closed() -> None:
    bad = source.FetchResponse(b"{}", "https://example.com/x", 200, "application/json")
    with pytest.raises(RuntimeError, match="outside official host"):
        source._validate_response(source.annual_url(2020), bad, json_expected=True)
    bad_type = source.FetchResponse(
        b"{}", source.annual_url(2020), 200, "text/html"
    )
    with pytest.raises(RuntimeError, match="not JSON"):
        source._validate_response(
            source.annual_url(2020), bad_type, json_expected=True
        )


def test_source_panels_contain_no_candidate_or_market_fields() -> None:
    forbidden = {"candidate", "side", "hold", "return", "pnl", "cagr", "mdd", "funding"}
    assert not forbidden.intersection(source.OPERATION_COLUMNS)
    assert not forbidden.intersection(source.DETAIL_COLUMNS)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        source._repository_path("../outside")
