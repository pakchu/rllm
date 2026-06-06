import pandas as pd

from training.economic_path_shape_data import PathTemplate, compute_path_shape


def test_compute_path_shape_labels_target_before_stop_for_long():
    market = pd.DataFrame(
        [
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.1},
            {"open": 100.1, "high": 101.3, "low": 99.8, "close": 101.0},
            {"open": 101.0, "high": 101.1, "low": 100.5, "close": 100.8},
        ]
    )
    shape = compute_path_shape(market, 0, PathTemplate(horizon_bars=2, target_pct=1.0, stop_pct=0.6, entry_delay_bars=1))
    assert shape is not None
    assert shape["long_path"]["first_event"] == "TARGET"
    assert shape["long_path"]["grade"] in {"CLEAN_TARGET", "NOISY_TARGET"}
    assert shape["direction_pressure"] in {"LONG_FAVORED", "BOTH_SIDES_VOLATILE"}


def test_compute_path_shape_returns_none_when_future_missing():
    market = pd.DataFrame([{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}])
    assert compute_path_shape(market, 0, PathTemplate(horizon_bars=2)) is None
