from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from training import export_coinbase_spot_leadership_sources as source


def _row(timestamp: str, close: float = 100.0) -> list[float]:
    epoch = int(pd.Timestamp(timestamp, tz="UTC").timestamp())
    return [epoch, close - 2, close + 2, close - 1, close, 3.0]


def test_request_windows_cover_grid_once_and_stay_below_endpoint_limit() -> None:
    grid = source.expected_grid("2020-01-01", "2020-01-03")
    windows = source.request_windows(grid)
    flattened = pd.DatetimeIndex([value for window in windows for value in window])
    assert flattened.equals(grid)
    assert max(map(len, windows)) == source.MAX_BUCKETS_PER_REQUEST
    assert source.MAX_BUCKETS_PER_REQUEST < 300


def test_payload_parser_sorts_by_timestamp_contract_and_filters_outside() -> None:
    expected = source.expected_grid("2020-01-01", "2020-01-01 00:10")
    payload = [
        _row("2020-01-01 00:10"),
        _row("2020-01-01 00:05", 101.0),
        _row("2020-01-01 00:00", 99.0),
    ]
    parsed, outside = source.parse_coinbase_payload(payload, expected)
    assert list(sorted(parsed)) == list(expected)
    assert outside == 1
    assert parsed[expected[1]][3] == 101.0


def test_payload_parser_rejects_conflicting_duplicates() -> None:
    expected = source.expected_grid("2020-01-01", "2020-01-01 00:05")
    with pytest.raises(RuntimeError, match="conflicting"):
        source.parse_coinbase_payload(
            [_row("2020-01-01", 100.0), _row("2020-01-01", 101.0)], expected
        )


def test_coinbase_frame_keeps_missing_bucket_unimputed() -> None:
    grid = source.expected_grid("2020-01-01", "2020-01-01 00:15")
    parsed, _ = source.parse_coinbase_payload([_row("2020-01-01")], grid)
    frame = source.coinbase_frame(grid, parsed)
    assert frame["source_complete"].tolist() == [1, 0, 0]
    assert frame.loc[1:, ["open", "high", "low", "close", "volume"]].isna().all().all()


def test_range_frame_stops_before_future_non_date_values(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close"])
        writer.writerow(["2022-12-31 23:55:00", "100"])
        writer.writerow(["2023-01-01 00:00:00", "future-value-must-not-parse"])
    frame = source.range_frame(
        path,
        date_column="date",
        start="2022-01-01",
        end="2023-01-01",
        usecols=["date", "close"],
    )
    assert frame["close"].tolist() == ["100"]
    assert frame.attrs["future_non_date_fields_csv_parsed"] == 0
    assert frame.attrs["cutoff_sentinel_date"] == "2023-01-01 00:00:00"


def test_config_refuses_to_open_2023() -> None:
    cfg = source.Config(end="2024-01-01")
    with pytest.raises(RuntimeError, match="before 2023"):
        source.validate_config(cfg)


def test_funding_preserves_small_source_offset_for_asof() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 08:00:00.047"],
                format="mixed",
            ),
            "funding_rate": [0.0001, 0.0002],
        }
    )
    checked = source.validate_funding(frame)
    assert checked.loc[1, "date"] == pd.Timestamp("2020-01-01 08:00:00.047")
    assert checked.attrs["maximum_grid_offset_seconds"] == 0.047


def test_no_silent_fallback_for_missing_raw_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        source.resolve_existing(tmp_path / "missing.csv")
