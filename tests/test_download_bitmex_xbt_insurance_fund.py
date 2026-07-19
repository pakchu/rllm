from __future__ import annotations

from dataclasses import replace

import pytest

from training import download_bitmex_xbt_insurance_fund as bitmex


def _rows() -> list[dict[str, object]]:
    return [
        {
            "currency": "XBt",
            "timestamp": "2020-01-01T12:00:00.000Z",
            "walletBalance": 100,
        },
        {
            "symbol": "XBt",
            "timestamp": "2020-01-02T12:00:00.000Z",
            "walletBalance": 101,
        },
        {
            "currency": "XBt",
            "timestamp": "2020-01-03T12:00:00.000Z",
            "walletBalance": 99,
        },
    ]


def _cfg(**changes: object) -> bitmex.Config:
    cfg = bitmex.Config(
        start="2020-01-01",
        end_exclusive="2020-01-04",
        page_size=2,
    )
    return replace(cfg, **changes)


def test_download_paginates_and_normalizes_legacy_and_current_keys() -> None:
    source = _rows()
    seen: list[dict[str, object]] = []

    def fetch(params: dict[str, object]) -> list[dict[str, object]]:
        seen.append(params)
        start = int(params["start"])
        count = int(params["count"])
        return source[start : start + count]

    frame, audit = bitmex.download(_cfg(), fetch=fetch)
    assert frame["wallet_balance_satoshi"].tolist() == [100, 101, 99]
    assert frame["date"].astype(str).tolist() == [
        "2020-01-01 12:00:00",
        "2020-01-02 12:00:00",
        "2020-01-03 12:00:00",
    ]
    assert [request["start"] for request in seen] == [0, 2]
    assert all(request["currency"] == "XBt" for request in seen)
    assert audit["complete_daily_noon_utc_grid"] is True
    assert audit["rows_selected"] == 3


def test_download_rejects_missing_day_instead_of_imputing() -> None:
    source = [_rows()[0], _rows()[2]]
    with pytest.raises(RuntimeError, match="complete daily 12:00 UTC grid"):
        bitmex.download(_cfg(page_size=10), fetch=lambda _: source)


def test_download_rejects_other_currency_and_conflicting_keys() -> None:
    other = _rows()
    other[1] = {**other[1], "symbol": "USDt"}
    with pytest.raises(ValueError, match="another asset"):
        bitmex.download(_cfg(page_size=10), fetch=lambda _: other)

    conflict = _rows()
    conflict[1] = {**conflict[1], "currency": "USDt"}
    with pytest.raises(ValueError, match="conflicting currency keys"):
        bitmex.download(_cfg(page_size=10), fetch=lambda _: conflict)


def test_download_rejects_off_noon_duplicate_and_nonpositive_balance() -> None:
    off_noon = _rows()
    off_noon[1] = {**off_noon[1], "timestamp": "2020-01-02T12:05:00.000Z"}
    with pytest.raises(RuntimeError, match="complete daily 12:00 UTC grid"):
        bitmex.download(_cfg(page_size=10), fetch=lambda _: off_noon)

    duplicate = _rows()
    duplicate[2] = {**duplicate[2], "timestamp": duplicate[1]["timestamp"]}
    with pytest.raises(RuntimeError, match="duplicate timestamps"):
        bitmex.download(_cfg(page_size=10), fetch=lambda _: duplicate)

    nonpositive = _rows()
    nonpositive[1] = {**nonpositive[1], "walletBalance": 0}
    with pytest.raises(RuntimeError, match="must stay positive"):
        bitmex.download(_cfg(page_size=10), fetch=lambda _: nonpositive)


def test_download_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        bitmex.download(_cfg(page_size=501), fetch=lambda _: [])
