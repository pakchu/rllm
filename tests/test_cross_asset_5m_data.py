import json

import pandas as pd
import pytest

from training import cross_asset_5m_data as source


def _payload(start: str, rows: int = 78) -> dict:
    timestamps = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    prices = [100.0 + index * 0.01 for index in range(rows)]
    return {
        "s": "ok",
        "t": [int(value.timestamp()) for value in timestamps],
        "o": prices,
        "h": [value + 0.1 for value in prices],
        "l": [value - 0.1 for value in prices],
        "c": [value + 0.02 for value in prices],
        "v": [1000 + index for index in range(rows)],
    }


def test_chunk_ranges_are_contiguous_and_split_exclusive() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = pd.Timestamp("2026-04-01T00:00:00Z")
    chunks = source.chunk_ranges(start, end, days=45)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))


def test_extract_jina_payload_returns_canonical_provider_json() -> None:
    raw = b"Title: x\n\nURL Source: x\n\nMarkdown Content:\n{\"s\":\"ok\",\"t\":[1]}\n"
    assert json.loads(source.extract_jina_payload(raw)) == {"s": "ok", "t": [1]}


def test_normalize_provider_rows_accepts_complete_us_session() -> None:
    # 14:30 UTC is 09:30 New York in standard time.
    frame, meta = source.normalize_provider_rows(
        [_payload("2026-01-05T14:30:00Z")],
        symbol="QQQ",
        timezone="America/New_York",
        regular_session=("09:30", "16:00"),
        start_utc=pd.Timestamp("2026-01-05T00:00:00Z"),
        end_utc=pd.Timestamp("2026-01-06T00:00:00Z"),
    )
    assert len(frame) == 78
    assert meta["session_row_count_histogram"] == {"78": 1}


def test_normalize_provider_rows_fails_on_missing_interior_bar() -> None:
    payload = _payload("2026-01-05T14:30:00Z")
    for key in ("t", "o", "h", "l", "c", "v"):
        del payload[key][20]
    with pytest.raises(RuntimeError, match="missing interior"):
        source.normalize_provider_rows(
            [payload],
            symbol="QQQ",
            timezone="America/New_York",
            regular_session=("09:30", "16:00"),
            start_utc=pd.Timestamp("2026-01-05T00:00:00Z"),
            end_utc=pd.Timestamp("2026-01-06T00:00:00Z"),
        )


def test_apply_daily_adjustment_preserves_raw_and_scales_ohlc() -> None:
    frame, _ = source.normalize_provider_rows(
        [_payload("2026-01-05T14:30:00Z")],
        symbol="QQQ",
        timezone="America/New_York",
        regular_session=("09:30", "16:00"),
        start_utc=pd.Timestamp("2026-01-05T00:00:00Z"),
        end_utc=pd.Timestamp("2026-01-06T00:00:00Z"),
    )
    factors = pd.Series([0.9], index=[pd.Timestamp("2026-01-05")])
    adjusted = source.apply_daily_adjustment(frame, factors)
    assert adjusted["raw_close"].iloc[0] == pytest.approx(100.02)
    assert adjusted["close"].iloc[0] == pytest.approx(90.018)


def test_krx_provider_stable_no_bar_gap_is_retained_without_synthesis() -> None:
    payload = _payload("2026-03-04T00:00:00Z", rows=76)
    for key in ("t", "o", "h", "l", "c", "v"):
        del payload[key][28:33]
    frame, meta = source.normalize_provider_rows(
        [payload],
        symbol="069500",
        timezone="Asia/Seoul",
        regular_session=("09:00", "15:20"),
        start_utc=pd.Timestamp("2026-03-04T00:00:00Z"),
        end_utc=pd.Timestamp("2026-03-05T00:00:00Z"),
    )
    assert len(frame) == 71
    assert meta["provider_stable_no_bar_gap_count"] == 1
    assert meta["provider_stable_no_bar_gaps"][0]["gap_minutes"] == 30


def test_krx_unfrozen_interior_gap_fails_closed() -> None:
    payload = _payload("2026-03-05T00:00:00Z", rows=76)
    for key in ("t", "o", "h", "l", "c", "v"):
        del payload[key][28:33]
    with pytest.raises(RuntimeError, match="frozen allowlist"):
        source.normalize_provider_rows(
            [payload],
            symbol="069500",
            timezone="Asia/Seoul",
            regular_session=("09:00", "15:20"),
            start_utc=pd.Timestamp("2026-03-05T00:00:00Z"),
            end_utc=pd.Timestamp("2026-03-06T00:00:00Z"),
        )


def test_action_date_can_bridge_missing_daily_factor_without_price_imputation() -> None:
    frame, _ = source.normalize_provider_rows(
        [_payload("2026-01-05T14:30:00Z")],
        symbol="QQQ",
        timezone="America/New_York",
        regular_session=("09:30", "16:00"),
        start_utc=pd.Timestamp("2026-01-05T00:00:00Z"),
        end_utc=pd.Timestamp("2026-01-06T00:00:00Z"),
    )
    factors = pd.Series(
        [0.9, 0.95],
        index=[pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06")],
    )
    factors.attrs["corporate_action_dates"] = [pd.Timestamp("2026-01-05")]
    adjusted = source.apply_daily_adjustment(frame, factors)
    assert adjusted["adjustment_factor"].iloc[0] == pytest.approx(0.95)
    assert adjusted["adjustment_factor_bridged"].all()
