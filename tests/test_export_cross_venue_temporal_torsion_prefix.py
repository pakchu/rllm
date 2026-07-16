from __future__ import annotations

import pandas as pd
import pytest

from training import export_cross_venue_temporal_torsion_prefix as source


def _row(date: str, *, valid: str = "True") -> dict[str, object]:
    timestamp = pd.Timestamp(date)
    row: dict[str, object] = {
        "date": timestamp,
        "feature_available_time_utc": timestamp + pd.Timedelta("5min"),
        "trade_earliest_time_utc": timestamp + pd.Timedelta("5min"),
        "spot_flow_fraction": 0.2,
        "um_flow_fraction": 0.1,
        "spot_log_return_5m": 0.001,
        "um_log_return_5m": 0.0008,
        "spot_flow_time_centroid": 0.2,
        "um_flow_time_centroid": 0.7,
        "spot_return_time_centroid": 0.6,
        "um_return_time_centroid": 0.3,
        "source_complete": "True",
        "cross_venue_feature_valid": valid,
        "feature_invalid_reason": "ok" if valid == "True" else "bad_support",
    }
    if valid != "True":
        for column in source.SIGNAL_COLUMNS:
            row[column] = None
    return row


def test_validate_features_preserves_availability_clock() -> None:
    grid = pd.date_range("2020-01-01", periods=2, freq="5min")
    raw = pd.DataFrame([_row(str(grid[0])), _row(str(grid[1]), valid="False")])
    frame = source.validate_features(raw, grid)
    assert frame["source_available"].tolist() == [1, 0]
    assert frame["feature_available_time_utc"].tolist() == list(
        grid + pd.Timedelta("5min")
    )
    assert frame["strategy_entry_earliest_time_utc"].tolist() == list(
        grid + pd.Timedelta("10min")
    )
    assert "trade_earliest_time_utc" not in frame


def test_validate_features_blanks_current_and_following_twenty_four() -> None:
    grid = pd.date_range("2020-01-01", periods=35, freq="5min")
    rows = [_row(str(date)) for date in grid]
    rows[3] = _row(str(grid[3]), valid="False")
    frame = source.validate_features(pd.DataFrame(rows), grid)
    assert frame.loc[:2, "source_available"].eq(1).all()
    assert frame.loc[3:27, "source_available"].eq(0).all()
    assert frame.loc[28:, "source_available"].eq(1).all()
    assert frame.loc[3:27, source.SIGNAL_COLUMNS].isna().all().all()
    assert frame.loc[4:27, "selection_invalid_reason"].eq(
        "post_invalid_24bar_quarantine"
    ).all()


def test_validate_features_rejects_future_availability_clock() -> None:
    grid = pd.date_range("2020-01-01", periods=1, freq="5min")
    raw = pd.DataFrame([_row(str(grid[0]))])
    raw.loc[0, "feature_available_time_utc"] += pd.Timedelta("5min")
    with pytest.raises(RuntimeError, match="availability"):
        source.validate_features(raw, grid)


def test_validate_features_rejects_out_of_range_centroid() -> None:
    grid = pd.date_range("2020-01-01", periods=1, freq="5min")
    raw = pd.DataFrame([_row(str(grid[0]))])
    raw.loc[0, "spot_flow_time_centroid"] = 1.1
    with pytest.raises(ValueError, match="centroid"):
        source.validate_features(raw, grid)


def test_validate_features_rejects_signal_on_invalid_row() -> None:
    grid = pd.date_range("2020-01-01", periods=1, freq="5min")
    raw = pd.DataFrame([_row(str(grid[0]), valid="False")])
    raw.loc[0, "spot_flow_fraction"] = 0.2
    with pytest.raises(ValueError, match="invalid CVTT"):
        source.validate_features(raw, grid)


def test_validate_features_rejects_hidden_invalid_descriptor() -> None:
    grid = pd.date_range("2020-01-01", periods=1, freq="5min")
    raw = pd.DataFrame([_row(str(grid[0]), valid="False")])
    raw.loc[0, "spot_flow_fraction"] = "hidden_descriptor"
    with pytest.raises(ValueError, match="raw signal descriptor"):
        source.validate_features(raw, grid)


def test_parse_bool_fails_closed() -> None:
    with pytest.raises(ValueError, match="boolean"):
        source.parse_bool(pd.Series(["yes"]), "source_complete")
