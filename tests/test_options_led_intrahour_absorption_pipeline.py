from pathlib import Path

import pandas as pd

from training import build_options_led_intrahour_absorption_support as support
from training import materialize_options_led_intrahour_absorption_sources as sources


def feature_rows() -> pd.DataFrame:
    times = pd.date_range("2023-08-01T00:00:00Z", periods=9, freq="1h")
    frame = pd.DataFrame({
        "decision_time": times, "base_valid": True,
        "bvol_body": 0.01, "dvol_body": 0.02, "oi_change": -0.01,
        "first_half_return": 0.001, "second_half_return": 0.001,
        "hour_return": 0.001, "first_half_tail": 0.02,
    })
    frame.loc[1, ["first_half_return", "second_half_return", "hour_return"]] = [-0.04, 0.03, 0.01]
    frame.loc[8, ["first_half_return", "second_half_return", "hour_return"]] = [0.04, -0.03, -0.01]
    return frame


def test_clock_follows_two_sided_absorbing_second_half():
    clock = support.build_clock(feature_rows())
    assert list(clock["side"]) == [1, -1]
    assert list(clock["entry_time"]) == [pd.Timestamp("2023-08-01T01:05:00Z"), pd.Timestamp("2023-08-01T08:05:00Z")]


def test_open_reclaim_and_oi_contraction_are_required():
    frame = feature_rows()
    frame.loc[[1, 8], "hour_return"] *= -1
    assert support.build_clock(frame).empty
    frame = feature_rows()
    frame.loc[[1, 8], "oi_change"] = 0.01
    assert support.build_clock(frame).empty


def test_empty_clock_has_fixed_schema_and_outcomes_stay_closed():
    frame = feature_rows()
    frame["base_valid"] = False
    assert list(support.build_clock(frame).columns) == list(support.CLOCK_COLUMNS)
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False


def test_materializer_queries_only_completed_intrahour_features():
    source = Path(sources.__file__).read_text()
    assert "(array_agg(close ORDER BY ts))[30]" in source
    assert "(array_agg(open ORDER BY ts))[31]" in source
    assert "post_entry_return_pnl_or_execution_price_opened" in source
    assert "funding_rates_binance" not in source
