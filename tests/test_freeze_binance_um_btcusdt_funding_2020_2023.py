from __future__ import annotations

import csv
import io
import urllib.parse

import pytest

from training import freeze_binance_um_btcusdt_funding_2020_2023 as freeze


def _row(time_ms: int, rate: str = "0.00010000") -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": time_ms,
        "fundingRate": rate,
        "markPrice": "10000.00000000",
    }


def test_fetch_records_paginates_with_inclusive_api_without_duplicates() -> None:
    cfg = freeze.FreezeConfig(limit=2, retry_attempts=1)
    responses = [
        [_row(freeze.START_MS), _row(freeze.START_MS + 1)],
        [_row(freeze.START_MS + 2)],
    ]
    cursors: list[int] = []

    def opener(url: str) -> object:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        cursors.append(int(query["startTime"][0]))
        assert int(query["endTime"][0]) == freeze.END_MS
        return responses.pop(0)

    records, pages = freeze.fetch_records(cfg, open_json=opener)
    assert pages == 2
    assert cursors == [freeze.START_MS, freeze.START_MS + 2]
    assert [row["fundingTime"] for row in records] == [
        freeze.START_MS,
        freeze.START_MS + 1,
        freeze.START_MS + 2,
    ]


def test_validate_records_preserves_exact_decimal_strings_and_utc() -> None:
    cfg = freeze.FreezeConfig()
    normalized = freeze.validate_records(
        [_row(freeze.START_MS, "-0.00012359")],
        cfg,
    )
    assert normalized == [
        {
            "funding_time_ms": freeze.START_MS,
            "funding_time_utc": "2020-01-01T00:00:00.000Z",
            "symbol": "BTCUSDT",
            "funding_rate": "-0.00012359",
            "mark_price": "10000.00000000",
        }
    ]
    serialized = freeze._serialize_csv(normalized).decode()
    parsed = list(csv.DictReader(io.StringIO(serialized)))
    assert parsed[0]["funding_rate"] == "-0.00012359"


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_row(freeze.START_MS), _row(freeze.START_MS)], "strictly increasing"),
        ([_row(freeze.END_MS + 1)], "sealed interval"),
        ([_row(freeze.START_MS, "nan")], "not finite"),
        ([], "empty"),
    ],
)
def test_validate_records_rejects_invalid_history(
    records: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        freeze.validate_records(records, freeze.FreezeConfig())


def test_validate_records_rejects_wrong_symbol_or_missing_field() -> None:
    wrong = _row(freeze.START_MS)
    wrong["symbol"] = "ETHUSDT"
    with pytest.raises(ValueError, match="wrong symbol"):
        freeze.validate_records([wrong], freeze.FreezeConfig())

    missing = _row(freeze.START_MS)
    del missing["markPrice"]
    with pytest.raises(ValueError, match="missing a required field"):
        freeze.validate_records([missing], freeze.FreezeConfig())


def test_fetch_records_rejects_scope_or_page_drift() -> None:
    with pytest.raises(ValueError, match="BTCUSDT"):
        freeze.fetch_records(freeze.FreezeConfig(symbol="ETHUSDT"), open_json=lambda _: [])
    with pytest.raises(ValueError, match="calendar 2020-2023"):
        freeze.fetch_records(
            freeze.FreezeConfig(end_ms=freeze.END_MS + 1),
            open_json=lambda _: [],
        )
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        freeze.fetch_records(freeze.FreezeConfig(limit=1_001), open_json=lambda _: [])
