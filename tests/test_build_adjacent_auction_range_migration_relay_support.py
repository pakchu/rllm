import numpy as np
import pandas as pd

from training import build_adjacent_auction_range_migration_relay_support as support


def _frame():
    times = pd.date_range(support.START, periods=24, freq="5min")
    first = np.linspace(100.0, 101.0, 12); second = np.linspace(101.0, 104.0, 12)
    close = np.concatenate([first, second]); open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({
        "bar_time": times, "bar_open": open_, "bar_high": close + 0.2,
        "bar_low": open_ - 0.2, "bar_close": close, "source_rows": 5,
        "distinct_rows": 5, "first_ts": times,
        "last_ts": times + pd.Timedelta(minutes=4), "coherent": True,
    })


def test_rank_and_onset():
    values = pd.Series(range(961), dtype=float)
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[959]) and ranks.iloc[960] == 1.0
    assert support.onset(pd.Series([False, True, True, False, True])).tolist() == [False, True, False, False, True]


def test_complete_adjacent_range_feature():
    raw = _frame(); old_start, old_end = support.START, support.END
    try:
        support.START = raw.bar_time.iloc[0]
        support.END = support.START + pd.Timedelta(hours=3)
        features = support.build_features(raw)
        assert features.iloc[0].source_valid
        assert 0 <= features.iloc[0].overlap_ratio <= 1
        assert features.iloc[0].midpoint_displacement > 0
    finally:
        support.START, support.END = old_start, old_end


def test_primary_follows_midpoint_and_fade_control_flips():
    features = pd.DataFrame({
        "source_valid": [1, 1], "current_range_rank": [0.7, 0.9],
        "overlap_rank": [0.3, 0.1], "midpoint_displacement": [1.0, 1.0],
    })
    active, side = support.active_and_side(features)
    assert active.tolist() == [False, True] and side.tolist() == [1, 1]
    _, faded = support.active_and_side(features, "midpoint_fade")
    assert faded.tolist() == [-1, -1]


def test_outcomes_remain_closed():
    source = open(support.__file__).read()
    assert '"execution_prices_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert '"rv20_opened": False' in source
