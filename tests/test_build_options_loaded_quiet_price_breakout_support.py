import pandas as pd

from training import build_options_loaded_quiet_price_breakout_support as support


def feature_rows() -> pd.DataFrame:
    times = pd.date_range("2023-08-01T00:00:00Z", periods=9, freq="1h")
    frame = pd.DataFrame({
        "decision_time": times, "base_valid": True,
        "bvol_close": 80.0, "bvol_level_tail": 70.0,
        "dvol_close": 75.0, "dvol_level_tail": 65.0,
        "oi_change": 0.01, "oi_tail": 0.02,
        "funding_rate": 0.0001, "funding_neutral_tail": 0.0002,
        "hour_return": 0.003, "quiet_tail": 0.002,
    })
    frame.loc[[1, 8], "oi_change"] = 0.03
    frame.loc[1, "hour_return"] = 0.001
    frame.loc[8, "hour_return"] = -0.001
    return frame


def test_clock_follows_two_sided_quiet_loading_direction():
    clock = support.build_clock(feature_rows())
    assert list(clock["side"]) == [1, -1]
    assert list(clock["entry_time"]) == [pd.Timestamp("2023-08-01T01:05:00Z"), pd.Timestamp("2023-08-01T08:05:00Z")]


def test_extreme_funding_or_nonquiet_price_blocks_primary():
    frame = feature_rows()
    frame.loc[[1, 8], "funding_rate"] = 0.001
    assert support.build_clock(frame).empty
    frame = feature_rows()
    frame.loc[[1, 8], "hour_return"] = 0.01
    assert support.build_clock(frame).empty


def test_empty_clock_has_fixed_schema_and_economics_stay_closed():
    frame = feature_rows()
    frame["base_valid"] = False
    assert list(support.build_clock(frame).columns) == list(support.CLOCK_COLUMNS)
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
