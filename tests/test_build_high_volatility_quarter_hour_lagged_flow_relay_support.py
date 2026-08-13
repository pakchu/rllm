import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_quarter_hour_lagged_flow_relay_support as support


def test_rolling_prediction_excludes_current_response() -> None:
    values = pd.Series([float(i * i + 3 * i + 7) for i in range(30)])
    first, intercept = support.causal_rolling_prediction(
        values, lag_count=2, lookback=8, minimum=4
    )
    changed = values.copy()
    changed.iloc[12] = 1_000_000.0
    second, second_intercept = support.causal_rolling_prediction(
        changed, lag_count=2, lookback=8, minimum=4
    )
    assert first.iloc[12] == pytest.approx(second.iloc[12])
    assert intercept.iloc[12] == pytest.approx(second_intercept.iloc[12])
    assert first.iloc[13] != pytest.approx(second.iloc[13])


def test_prediction_excludes_intercept_exactly() -> None:
    values = pd.Series([10.0 + 2.0 * i for i in range(30)])
    prediction, intercept = support.causal_rolling_prediction(
        values, lag_count=1, lookback=10, minimum=4
    )
    index = prediction.last_valid_index()
    assert index is not None
    assert prediction.at[index] + intercept.at[index] == pytest.approx(values.at[index])
    assert abs(intercept.at[index]) > 0


def test_exact_grid_validation_rejects_a_missing_minute() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    times = pd.date_range(start, periods=1500, freq="1min")
    bars = pd.DataFrame(
        {
            "ts": times,
            "open": 100.0,
            "close": 101.0,
            "volume": 10.0,
            "taker_buy_base": 6.0,
        }
    ).drop(index=700)
    with pytest.raises(RuntimeError, match="exact requested one-minute grid"):
        support.validate_bars(bars, start, times[-1] + pd.Timedelta(minutes=1))


def _features() -> pd.DataFrame:
    times = pd.date_range("2024-07-01T00:00:00Z", periods=8, freq="15min")
    return pd.DataFrame(
        {
            "decision_time": times,
            "source_valid": True,
            "opening_imbalance": [0.2, 0.3, -0.2, -0.3, 0.2, -0.2, 0.3, -0.3],
            "shifted_opening_imbalance": [0.2] * 8,
            "lagged_flow_prediction": [0.1, 0.2, 0.3, -0.2, -0.4, -0.5, 0.2, 0.3],
            "lagged_flow_intercept": [0.01] * 8,
            "shifted_lagged_flow_prediction": [-0.1] * 8,
            "model_valid": True,
            "shifted_model_valid": True,
            "flow_strength_rank": [0.96, 0.97, 0.80, 0.98, 0.99, 0.70, 0.97, 0.98],
            "shifted_flow_strength_rank": [0.96] * 8,
            "realized_variation": [0.1] * 8,
            "variation_rank": [0.8] * 8,
        }
    )


def test_onset_uses_previous_model_valid_observation() -> None:
    active, side = support.active_and_side(_features())
    assert active.tolist() == [True, False, False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1, 1]


def test_global_half_open_reservation_and_equal_exit_entry() -> None:
    frame = _features()
    frame.loc[:, "decision_time"] = pd.to_datetime(
        [
            "2024-07-01T00:00:00Z",
            "2024-07-01T00:15:00Z",
            "2024-07-01T00:30:00Z",
            "2024-07-01T03:45:00Z",
            "2024-07-01T04:00:00Z",
            "2024-07-01T04:15:00Z",
            "2024-07-01T04:00:00Z",
            "2024-07-01T04:15:00Z",
        ],
        utc=True,
    )
    clock = support.make_clock(frame)
    assert clock["entry_time"].tolist() == [
        pd.Timestamp("2024-07-01T00:05:00Z"),
        pd.Timestamp("2024-07-01T04:05:00Z"),
    ]
    assert clock.iloc[0]["exit_time"] == clock.iloc[1]["entry_time"]


def test_support_gates_match_frozen_limits() -> None:
    passing = {
        split: {
            "events": support.MINIMUM[split],
            "longs": 1,
            "shorts": 1,
            "minority_side_share": 0.2,
            "max_month_share": 0.45,
        }
        for split in support.SPLITS
    }
    assert all(support.support_checks(passing).values())
    passing["final"]["max_month_share"] = 0.5
    assert support.support_checks(passing)["final_month_concentration"] is False


def test_preregistration_file_hash_is_pinned() -> None:
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
