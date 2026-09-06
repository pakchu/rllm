import pandas as pd

from training import build_options_implied_vol_flush_absorption_support as support


def feature_rows() -> pd.DataFrame:
    times = pd.date_range("2023-08-01T00:00:00Z", periods=9, freq="1h")
    oi_change = [0.0] * 9
    hour_return = [0.001] * 9
    oi_change[1] = oi_change[8] = -0.03
    hour_return[1], hour_return[8] = -0.04, 0.04
    return pd.DataFrame(
        {
            "decision_time": times,
            "base_valid": [True] * 9,
            "bvol_body": [0.01] * 9,
            "dvol_body": [0.02] * 9,
            "oi_change": oi_change,
            "oi_floor": [-0.02] * 9,
            "hour_return": hour_return,
            "return_tail": [0.02] * 9,
        }
    )


def test_clock_fades_two_sided_oi_flush_price_shocks():
    clock = support.build_clock(feature_rows())
    assert len(clock) == 2
    assert list(clock["side"]) == [1, -1]
    assert clock.iloc[0]["entry_time"] == pd.Timestamp("2023-08-01T01:05:00Z")
    assert clock.iloc[0]["exit_time"] == pd.Timestamp("2023-08-01T07:05:00Z")


def test_direction_flip_changes_side_without_changing_entries():
    primary = support.build_clock(feature_rows())
    flipped = support.build_clock(feature_rows(), "direction_flip")
    assert list(primary["entry_time"]) == list(flipped["entry_time"])
    assert list(primary["side"]) == [-side for side in flipped["side"]]


def test_missing_flush_or_return_tail_blocks_primary():
    frame = feature_rows()
    frame.loc[[1, 8], "oi_change"] = -0.01
    assert support.build_clock(frame).empty
    assert list(support.build_clock(frame).columns) == list(support.CLOCK_COLUMNS)


def test_source_support_cannot_open_economic_outcomes():
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
