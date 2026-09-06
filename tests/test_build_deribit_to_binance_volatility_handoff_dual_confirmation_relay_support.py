import pandas as pd

from training import build_deribit_to_binance_volatility_handoff_dual_confirmation_relay_support as support


def rows():
    time = pd.date_range("2023-08-01T00:00:00Z", periods=10, freq="1h")
    frame = pd.DataFrame({
        "decision_time": time, "base_valid": True, "bvol_body": 0.01,
        "dvol_body": 0.01, "hour_return": 0.001,
    })
    frame.loc[1, ["bvol_body", "dvol_body", "hour_return"]] = [-0.01, 0.02, 0.003]
    frame.loc[2, ["bvol_body", "dvol_body", "hour_return"]] = [0.02, 0.01, 0.004]
    frame.loc[8, ["bvol_body", "dvol_body", "hour_return"]] = [-0.02, 0.01, -0.003]
    frame.loc[9, ["bvol_body", "dvol_body", "hour_return"]] = [0.01, 0.02, -0.004]
    return frame


def test_clock_follows_two_sided_ordered_handoff():
    clock = support.clock(rows())
    assert list(clock["side"]) == [1, -1]
    assert list(clock["entry_time"]) == [pd.Timestamp("2023-08-01T02:05:00Z"), pd.Timestamp("2023-08-01T09:05:00Z")]


def test_missing_handoff_or_price_confirmation_blocks():
    frame = rows()
    frame.loc[[1, 8], "bvol_body"] = 0.01
    assert support.clock(frame).empty
    frame = rows()
    frame.loc[2, "hour_return"] = -0.004
    frame.loc[9, "hour_return"] = 0.004
    assert support.clock(frame).empty


def test_schema_and_economics_are_closed():
    frame = rows()
    frame["base_valid"] = False
    assert list(support.clock(frame).columns) == list(support.COLUMNS)
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
