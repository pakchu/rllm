from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from training import build_new_york_fed_overnight_rrp as builder


def _operation(
    *,
    operation_id: str,
    operation_date: str,
    close_time: str = "13:15",
    last_updated: str | None = None,
    amount: int = 1_000_000_000,
    security_type: str = "Treasury",
) -> dict[str, object]:
    last_updated = last_updated or f"{operation_date} 13:15:30"
    return {
        "operationId": operation_id,
        "operationDate": operation_date,
        "auctionStatus": "Results",
        "settlementDate": operation_date,
        "maturityDate": str(
            builder.date.fromisoformat(operation_date) + builder.timedelta(days=1)
        ),
        "operationType": "Reverse Repo",
        "operationMethod": "Fixed Rate",
        "settlementType": "Same Day",
        "termCalenderDays": 1,
        "term": "Overnight",
        "closeTime": close_time,
        "totalAmtSubmitted": amount,
        "totalAmtAccepted": amount,
        "participatingCpty": 10,
        "acceptedCpty": 10,
        "lastUpdated": last_updated,
        "note": "",
        "details": [
            {
                "securityType": security_type,
                "amtSubmitted": amount,
                "amtAccepted": amount,
            }
        ],
        "propositions": [],
    }


def _payload(rows: list[dict[str, object]]) -> bytes:
    return json.dumps({"repo": {"operations": rows}}).encode()


def test_annual_url_fixes_reverse_repo_operation_family() -> None:
    url = builder.annual_url(2021)
    assert "operationTypes=Reverse+Repo" in url
    assert "method=fixed" in url
    assert "term=overnight" in url


def test_parser_excludes_morning_exercise_and_quarantines_late_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _operation(
            operation_id="RP 010421 99",
            operation_date="2021-01-04",
            close_time="10:00",
            amount=99_000_000,
        ),
        _operation(
            operation_id="RP 010421 1",
            operation_date="2021-01-04",
            amount=2_000_000_000,
        ),
        _operation(
            operation_id="RP 010521 1",
            operation_date="2021-01-05",
            last_updated="2021-01-08 09:00:00",
            amount=3_000_000_000,
        ),
    ]
    monkeypatch.setitem(
        builder.FROZEN_YEAR_COVERAGE,
        2021,
        (2, "2021-01-04", "2021-01-05", 1),
    )
    parsed = builder.parse_annual_response(_payload(rows), year=2021)
    assert [row["operation_id"] for row in parsed] == [
        "RP 010421 1",
        "RP 010521 1",
    ]
    assert parsed[0]["result_available_at_utc"] == "2021-01-04T18:30:00+00:00"
    assert parsed[0]["total_amount_accepted_usd"] == "2000000000"
    assert parsed[1]["source_complete"] == "false"
    assert parsed[1]["total_amount_accepted_usd"] == ""
    assert parsed[1]["quarantine_reason"]


def test_parser_rejects_duplicate_afternoon_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _operation(operation_id="A", operation_date="2021-01-04"),
        _operation(operation_id="B", operation_date="2021-01-04"),
    ]
    monkeypatch.setitem(
        builder.FROZEN_YEAR_COVERAGE,
        2021,
        (2, "2021-01-04", "2021-01-04", 0),
    )
    with pytest.raises(ValueError, match="unique by operation date"):
        builder.parse_annual_response(_payload(rows), year=2021)


def test_parser_rejects_non_treasury_normal_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _operation(
            operation_id="A",
            operation_date="2021-01-04",
            security_type="Mortgage-Backed",
        )
    ]
    monkeypatch.setitem(
        builder.FROZEN_YEAR_COVERAGE,
        2021,
        (1, "2021-01-04", "2021-01-04", 0),
    )
    with pytest.raises(ValueError, match="Treasury collateral"):
        builder.parse_annual_response(_payload(rows), year=2021)


def test_build_rejects_future_year() -> None:
    with pytest.raises(ValueError, match="restricted to 2018-2023"):
        builder.build(builder.BuildConfig(end_year=2024))


def test_snapshot_gzip_is_deterministic(tmp_path: Path) -> None:
    payload = b'{"repo":{"operations":[]}}'
    left = tmp_path / "left.gz"
    right = tmp_path / "right.gz"
    builder._write_gzip(left, payload)
    builder._write_gzip(right, payload)
    assert left.read_bytes() == right.read_bytes()
    with gzip.open(left, "rb") as handle:
        assert handle.read() == payload
