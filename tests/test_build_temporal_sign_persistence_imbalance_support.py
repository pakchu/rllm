import numpy as np
import pandas as pd

from training import build_temporal_sign_persistence_imbalance_support as support


def _day(day="2024-07-02", signs=None):
    if signs is None: signs = np.resize(np.array([1, 1, -1]), 1440)
    opens = np.full(1440, 100.0); closes = opens * np.exp(np.asarray(signs) * 0.001)
    return pd.DataFrame({"ts": pd.date_range(day, periods=1440, freq="1min", tz="UTC"), "open": opens, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes), "close": closes})


def test_run_geometry_squares_every_nonzero_run_and_zero_breaks():
    result = support.run_geometry(np.array([1, 1, 0, 1, -1, -1, -1]))
    assert result == {"positive_run_mass": 5, "negative_run_mass": 9, "maximum_positive_run": 2, "maximum_negative_run": 3}


def test_source_valid_weekly_features_and_primary_side():
    features = support.build_features(_day(signs=np.resize(np.array([1, 1, -1]), 1440)))
    assert features.iloc[0].source_valid
    assert features.iloc[0].decision_time == pd.Timestamp("2024-07-03T00:00Z")
    assert support.signal(features, "primary").iloc[0] == 1
    assert support.signal(features, "direction_flip").iloc[0] == -1


def test_missing_minute_and_zero_only_side_are_ineligible():
    assert not support.build_features(_day().drop(index=10)).iloc[0].source_valid
    assert not support.build_features(_day(signs=np.ones(1440))).iloc[0].source_valid


def test_clock_and_gates_are_frozen():
    features = support.build_features(_day())
    clock = support.build_clock(features)
    assert len(clock) == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-07-03T00:05Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-07-03T12:05Z")
    assert support.MINIMUM == {"train": 8, "test": 12, "eval": 12, "final": 8}


def test_source_evaluator_does_not_open_outcomes_or_gross9():
    source = open(support.__file__).read()
    assert '"execution_prices_opened": False' in source
    assert '"postentry_return_or_pnl_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert '"rv20_opened": False' in source
