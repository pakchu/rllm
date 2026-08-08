import pandas as pd

from training import build_compressed_volatility_trend_relay_support as support


def feature_rows() -> pd.DataFrame:
    times = pd.date_range("2023-08-01T00:00:00Z", periods=16, freq="1h")
    frame = pd.DataFrame({
        "decision_time": times, "base_valid": True,
        "bvol_body": -0.01, "dvol_body": -0.02,
        "hour_return": 0.003, "return_q40": 0.004, "return_q75": 0.02,
        "oi_change": 0.0, "oi_q35": -0.01, "oi_q65": 0.01,
        "funding_rate": 0.0001, "funding_q50": 0.0002,
    })
    frame.loc[[0, 1], "hour_return"] = 0.01
    frame.loc[[14, 15], "hour_return"] = -0.01
    return frame


def test_clock_follows_two_sided_two_hour_moderate_trends():
    clock = support.build_clock(feature_rows())
    assert list(clock["side"]) == [1, -1]
    assert list(clock["entry_time"]) == [pd.Timestamp("2023-08-01T01:05:00Z"), pd.Timestamp("2023-08-01T15:05:00Z")]


def test_noncontracting_volatility_or_extreme_funding_blocks():
    frame = feature_rows()
    frame.loc[[1, 15], "dvol_body"] = 0.01
    assert support.build_clock(frame).empty
    frame = feature_rows()
    frame.loc[[1, 15], "funding_rate"] = 0.001
    assert support.build_clock(frame).empty


def test_empty_clock_has_fixed_schema_and_economics_stay_closed():
    frame = feature_rows()
    frame["base_valid"] = False
    assert list(support.build_clock(frame).columns) == list(support.CLOCK_COLUMNS)
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
