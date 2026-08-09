import numpy as np
import pandas as pd
from training import build_volume_stratified_price_control_relay_support as support


def _frame():
    t = pd.date_range(support.START, periods=96, freq="5min")
    o = np.full(96, 100.0)
    # Opposing price cohorts are deterministic because q increases with row number.
    r = np.r_[np.full(18, -0.001), np.zeros(60), np.full(18, 0.002)]
    c = o * np.exp(r)
    return pd.DataFrame(
        {
            "bar_time": t,
            "bar_open": o,
            "bar_close": c,
            "quote_volume": np.arange(1, 97, dtype=float),
            "source_rows": 5,
            "distinct_rows": 5,
            "first_ts": t,
            "last_ts": t + pd.Timedelta(minutes=4),
            "coherent": True,
        }
    )


def test_rank_and_onset():
    x = pd.Series(range(361), dtype=float)
    rank = support.strict_prior_midrank(x)
    assert np.isnan(rank.iloc[359]) and rank.iloc[360] == 1.0
    assert support.onset(pd.Series([False, True, True, False, True])).tolist() == [False, True, False, False, True]


def test_complete_feature_and_lexicographic_strata():
    raw = _frame()
    old_start, old_end = support.START, support.END
    try:
        support.START = raw.bar_time.iloc[0]
        support.END = support.START + pd.Timedelta(hours=12)
        features = support.build_features(raw)
        row = features.iloc[0]
        assert row.source_valid
        assert row.high_volume_return > 0
        assert row.low_volume_return <= 0
        assert row.normalizer > 0
        assert row.high_volume_dominance > 2 / 3
    finally:
        support.START, support.END = old_start, old_end


def test_primary_and_direction_flip():
    frame = pd.DataFrame(
        {
            "source_valid": [True, True, True],
            "high_volume_return": [0.1, 0.1, 0.1],
            "low_volume_return": [-0.01, -0.01, -0.01],
            "high_volume_dominance": [0.9, 0.9, 0.9],
            "stratified_rank": [0.7, 0.9, 0.9],
            "unweighted_return": [0.1, 0.1, 0.1],
            "unweighted_rank": [0.7, 0.9, 0.9],
            "dominant_bar_return": [0.01, 0.01, 0.01],
            "dominant_bar_rank": [0.7, 0.9, 0.9],
            "final_half_return": [0.1, 0.1, 0.1],
            "early_half_return": [-0.01, -0.01, -0.01],
            "temporal_dominance": [0.9, 0.9, 0.9],
            "temporal_rank": [0.7, 0.9, 0.9],
        }
    )
    active, side = support.active_and_side(frame)
    assert active.tolist() == [False, True, False]
    assert side.tolist() == [1, 1, 1]
    _, flipped = support.active_and_side(frame, "direction_flip")
    assert flipped.tolist() == [-1, -1, -1]


def test_outcomes_closed():
    source = open(support.__file__).read()
    assert '"execution_prices_opened":False' in source
    assert '"gross9_rows_opened":False' in source
    assert '"rv20_opened":False' in source
