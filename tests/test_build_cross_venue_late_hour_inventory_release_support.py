import pandas as pd

from training import build_cross_venue_late_hour_inventory_release_support as support


def feature_rows() -> pd.DataFrame:
    times = pd.date_range("2023-08-01T00:00:00Z", periods=9, freq="1h")
    frame = pd.DataFrame({
        "decision_time": times, "base_valid": True,
        "bvol_body": 0.01, "dvol_body": -0.02,
        "first_half_return": 0.01, "first_half_q40": 0.002,
        "second_half_return": 0.001, "second_half_q90": 0.02,
        "oi_change": 0.001, "abs_oi_q75": 0.02,
    })
    frame.loc[[1, 8], "first_half_return"] = 0.001
    frame.loc[1, "second_half_return"] = 0.03
    frame.loc[8, "second_half_return"] = -0.03
    frame.loc[[1, 8], "oi_change"] = 0.03
    return frame


def test_clock_follows_two_sided_late_hour_inventory_release():
    clock = support.build_clock(feature_rows())
    assert list(clock["side"]) == [1, -1]
    assert list(clock["entry_time"]) == [pd.Timestamp("2023-08-01T01:05:00Z"), pd.Timestamp("2023-08-01T08:05:00Z")]


def test_same_sign_vol_or_non_tail_oi_blocks():
    frame = feature_rows(); frame["dvol_body"] = 0.02
    assert support.build_clock(frame).empty
    frame = feature_rows(); frame.loc[[1, 8], "oi_change"] = 0.01
    assert support.build_clock(frame).empty


def test_empty_clock_has_schema_and_economics_closed():
    frame = feature_rows(); frame["base_valid"] = False
    assert list(support.build_clock(frame).columns) == list(support.CLOCK_COLUMNS)
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
