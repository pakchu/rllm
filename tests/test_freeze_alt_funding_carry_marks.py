from __future__ import annotations

import pandas as pd
import pytest

from training.freeze_alt_funding_carry_marks import compose_event_marks, download_mark_klines


def kline(open_time_ms: int, price: str) -> list[object]:
    return [open_time_ms, price, price, price, price, "0", open_time_ms + 299_999]


def test_download_mark_klines_paginates_to_exact_grid() -> None:
    start = pd.Timestamp("2023-01-01 00:00")
    end = pd.Timestamp("2023-01-01 00:15")
    base = int(start.tz_localize("UTC").timestamp() * 1_000)
    calls: list[int] = []

    def request(path: str, params: dict[str, object]) -> list[list[object]]:
        cursor = int(params["startTime"])
        calls.append(cursor)
        if cursor == base:
            return [kline(base, "100"), kline(base + 300_000, "101")]
        if cursor == base + 600_000:
            return [kline(base + 600_000, "102")]
        return []

    frame = download_mark_klines("ETHUSDT", start, end, request_json=request, sleep_sec=0)
    assert calls == [base, base + 600_000]
    assert frame["open"].tolist() == [100.0, 101.0, 102.0]


def test_download_mark_klines_allows_exchange_gap_away_from_funding_event() -> None:
    start = pd.Timestamp("2023-01-01 00:00")
    end = pd.Timestamp("2023-01-01 00:15")
    base = int(start.tz_localize("UTC").timestamp() * 1_000)

    def request(path: str, params: dict[str, object]) -> list[list[object]]:
        if int(params["startTime"]) == base:
            return [kline(base, "100"), kline(base + 600_000, "102")]
        return []

    frame = download_mark_klines("ETHUSDT", start, end, request_json=request, sleep_sec=0)
    assert frame["open_time"].tolist() == [start, start + pd.Timedelta(minutes=10)]


def test_compose_event_marks_backfills_missing_and_verifies_overlap() -> None:
    times = pd.to_datetime(["2023-01-01 00:00:00.008", "2023-01-01 08:00:00.001"])
    funding = pd.DataFrame({
        "funding_time": (times.astype("int64") // 1_000_000).astype("int64"),
        "mark_price": [None, 110.0],
    })
    opens = pd.to_datetime([
        "2022-12-31 23:55", "2023-01-01 00:00", "2023-01-01 07:55", "2023-01-01 08:00"
    ])
    klines = pd.DataFrame({
        "open_time": opens,
        "open": [99.0, 100.0, 109.0, 110.0],
        "close": [100.0, 101.0, 110.0, 111.0],
    })
    output, stats = compose_event_marks(funding, klines)
    assert output["causal_mark_price"].tolist() == [100.0, 110.0]
    assert output["mark_source"].tolist() == ["prior_completed_mark_close", "funding_record"]
    assert stats["backfilled_mark_events"] == 1
    assert stats["maximum_recorded_vs_prior_close_error_bp"] == pytest.approx(0.0)


def test_compose_event_marks_rejects_recorded_mark_mismatch() -> None:
    event = pd.Timestamp("2023-01-01 00:00")
    funding = pd.DataFrame({
        "funding_time": [int(event.tz_localize("UTC").timestamp() * 1_000)],
        "mark_price": [101.0],
    })
    klines = pd.DataFrame({
        "open_time": [event - pd.Timedelta(minutes=5), event],
        "open": [100.0, 100.0],
        "close": [100.0, 100.0],
    })
    with pytest.raises(RuntimeError, match="causal funding mark proxy mismatch"):
        compose_event_marks(funding, klines)


def test_compose_event_marks_rejects_gap_at_funding_event() -> None:
    event = pd.Timestamp("2023-01-01 08:00")
    funding = pd.DataFrame({
        "funding_time": [int(event.tz_localize("UTC").timestamp() * 1_000)],
        "mark_price": [None],
    })
    klines = pd.DataFrame({
        "open_time": [event],
        "open": [100.0],
        "close": [100.0],
    })
    with pytest.raises(RuntimeError, match="lacks causal adjacent mark-price bars"):
        compose_event_marks(funding, klines)
