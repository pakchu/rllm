from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import freeze_binance_um_btcusdt_funding_marks_2020_2023 as freeze


def _kline(open_time: int, price: float) -> list[object]:
    return [
        open_time,
        str(price),
        str(price + 1.0),
        str(price - 1.0),
        str(price),
        "0",
        open_time + freeze.STEP_MS - 1,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


def test_download_requires_a_complete_contiguous_grid() -> None:
    cfg = freeze.FreezeConfig(limit=1_500, sleep_seconds=0.0)
    start_ms = int(freeze.START.tz_localize("UTC").timestamp() * 1_000)
    end_ms = int(freeze.END.tz_localize("UTC").timestamp() * 1_000)
    expected = list(range(start_ms, end_ms, freeze.STEP_MS))

    def request_json(_path: str, params: dict[str, object]) -> list[list[object]]:
        cursor = int(params["startTime"])
        rows = [value for value in expected if value >= cursor][: cfg.limit]
        return [_kline(value, 100.0 + index) for index, value in enumerate(rows)]

    frame, pages = freeze.download_mark_klines(cfg, request_json=request_json)
    assert len(frame) == len(expected) == 4_383
    assert pages == 3
    assert np.array_equal(frame["open_time_ms"].to_numpy(np.int64), expected)


def test_compose_maps_small_funding_jitter_to_the_same_8h_open() -> None:
    start_ms = int(freeze.START.tz_localize("UTC").timestamp() * 1_000)
    funding = pd.DataFrame(
        {
            "funding_time_ms": [start_ms + 7, start_ms + freeze.STEP_MS + 43],
            "funding_time_utc": pd.to_datetime(
                [start_ms + 7, start_ms + freeze.STEP_MS + 43], unit="ms", utc=True
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "funding_rate": [0.0001, -0.0002],
            "mark_price": [100.0, np.nan],
        }
    )
    marks = pd.DataFrame(
        {
            "open_time_ms": [start_ms, start_ms + freeze.STEP_MS],
            "open": [100.00001, 110.0],
        }
    )
    output, stats = freeze.compose_settlement_marks(funding, marks)
    assert output["settlement_mark_price"].tolist() == [100.00001, 110.0]
    assert output["funding_time_offset_ms"].tolist() == [7, 43]
    assert stats["recorded_mark_overlap_events"] == 1
    assert stats["backfilled_events"] == 1


def test_compose_rejects_missing_event_mark() -> None:
    start_ms = int(freeze.START.tz_localize("UTC").timestamp() * 1_000)
    funding = pd.DataFrame(
        {
            "funding_time_ms": [start_ms],
            "funding_time_utc": ["2020-01-01T00:00:00.000000Z"],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.0001],
            "mark_price": [np.nan],
        }
    )
    marks = pd.DataFrame({"open_time_ms": [], "open": []})
    with pytest.raises(ValueError, match="lacks an 8h mark-price open"):
        freeze.compose_settlement_marks(funding, marks)


def test_deterministic_gzip_is_byte_stable(tmp_path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    freeze.deterministic_csv_gz(frame, first)
    freeze.deterministic_csv_gz(frame, second)
    assert first.read_bytes() == second.read_bytes()
