import pandas as pd

from training import build_high_volatility_premium_index_wick_rejection_relay_support as support


def test_daily_upper_rejection_maps_short():
    day = pd.Timestamp("2023-01-01T00:00:00Z")
    premium = pd.DataFrame([{"source_day": day, "source_rows": 1440, "distinct_timestamps": 1440, "first_ts": day, "last_ts": day + pd.Timedelta(hours=23, minutes=59), "premium_open": 0.0, "premium_close": -0.1, "premium_high": 0.8, "premium_low": -0.2, "premium_coherent": True}])
    btc = pd.DataFrame([{"source_day": day, "bars_5m": 288, "source_rows_1m": 1440, "first_bar": day, "last_bar": day + pd.Timedelta(hours=23, minutes=55), "day_open": 100.0, "day_close": 101.0, "btc_realized_variation": 0.1, "coherent": True}])
    features = support.build_features(premium, btc)
    assert features.rejection_side.tolist() == [-1]
    assert features.upper_wick.iloc[0] > features.lower_wick.iloc[0]


def _feature(**changes):
    row = {"source_day": pd.Timestamp("2023-07-02T00:00:00Z"), "source_valid": True, "rejection_side": -1, "premium_open": 0.0, "premium_close": -0.1, "premium_body": -0.1, "upper_wick": 0.8, "lower_wick": 0.1, "rejection_magnitude": 0.8, "magnitude_rank": 0.8, "btc_realized_variation": 0.1, "btc_variation_rank": 0.8}
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_and_controls_keep_clock_but_change_only_registered_side():
    features = _feature()
    assert support.build_clock(features).side.tolist() == [-1]
    assert support.build_clock(features, "premium_body_direction").side.tolist() == [-1]
    assert support.build_clock(features, "direction_flip").side.tolist() == [1]
    assert support.build_clock(features, "same_clock_forced_long").side.tolist() == [1]


def test_magnitude_tail_is_required_only_by_control_contract():
    features = _feature(magnitude_rank=0.2)
    assert support.build_clock(features).empty
    assert support.build_clock(features, "no_magnitude_tail").side.tolist() == [-1]
